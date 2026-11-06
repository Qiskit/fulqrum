# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import logging
import time

import cupy
import numpy as np

from .device import DeviceLayout
from .host import HostOperator
from .plan import LayoutPlan, agree

DEFAULT_NUM_SLICES = 4
logger = logging.getLogger(__name__)


class FullStrsHamiltonian:
    """GPU-backed real type-2 Hamiltonian on an explicit set of full determinants.

    Use with ``distributed_primme_eigsh`` to solve eigenvalue problems on one or
    more ranks. Each rank uses its current CUDA device; select it before
    construction and use that device context for subsequent calls.
    The operator must use block spin ordering, with alpha in the
    low bits and beta in the high bits, equal sector widths of 1–64 orbitals,
    fixed sector electron counts and supported spin-conserving terms through
    two-body order. Unsupported operator groups are rejected.

    Input ``keys`` has dtype uint64 and shape ``(n, 2)``. Each row contains the
    alpha word followed by the beta word; bit k is orbital k's occupation in that
    sector. The determinant set need not be a complete tensor-product subspace.
    Inputs may be unsorted or contain duplicates. Construction deduplicates,
    sorts and redistributes keys according to the selected layout. Afterwards,
    ``keys`` gives this rank's final key order, ``n_local`` its owned row count,
    and ``row_start`` its position in the distributed ordering. Local vectors
    and ``diag_local`` use that same order: amplitude i belongs to ``keys[i]``.

    Construction, matvecs and RDM calculations are collective over ``comm``.
    All ranks must participate with matching operator and shared options, while
    supplying their own input keys or local amplitudes. ``shape`` is the global
    square shape, ``dtype`` is float64 and ``supports_device_matvec`` is True.
    The solver uses ``_matvec(v_local, out=buffer)``, ``n_local`` and ``_comm``.
    Its device inputs and optional output buffer are local CuPy float64 arrays.

    Public ``matvec(x)`` requires shape ``(n_local,)``. NumPy input returns a host
    NumPy vector; CuPy input returns a device vector. Device work is asynchronous
    but ordered on the caller's current CUDA stream. RDM calls return full host
    NumPy arrays on every rank, using the supplied amplitudes without normalization.
    RDM return forms and tensor indices follow PySCF ``make_rdm12s(reorder=True)``.

    Args:
        operator: Real type-2 ``QubitOperator`` in the block layout above.
        keys: This rank's input determinant pairs, uint64 shape ``(n, 2)``.
            Input assignment and order may change during construction.
        comm: MPI or ``LocalComm`` communicator for all participating ranks.
            ``None`` uses ``MPI.COMM_SELF``.
        num_slices: Positive number of group-data slices; defaults to 4.
        binned: Select matvec kernels by group-size bins; defaults to True.
            False uses the uniform group kernel.
        residency: ``"resident"`` keeps group slices on the GPU; ``"streamed"``
            transfers slices from host memory as needed; ``"auto"`` (default)
            chooses residency using the device budget.
        device_budget_bytes: Byte budget for automatic device-storage planning.
            ``None`` derives it from free GPU memory minus reserve and buffer costs.
        alpha_kind, beta_kind: Optional group representations, ``"intersection"``
            or ``"union"``, independently for each spin sector. ``None`` chooses
            automatically from the key occupations.
        recut: Request membership-based row repartitioning with True, or ordinary
            row cuts with False. ``None`` selects automatic load balancing for
            distributed row-group layouts. A supplied plan controls the actual cut.
        exchange: ``"stepped"`` (default), ``"grouped"`` or ``"two_level"``
            source-vector exchange. Two-level exchange requires one copy per block.
        copies: Ranks sharing a replicated connection-data block. Defaults to 1;
            ``"auto"`` chooses a layout. An explicit positive count must divide
            communicator size. Vector rows remain partitioned among ranks.
        plan: Optional ``LayoutPlan`` or its dictionary representation. Overrides
            copies, num_slices, exchange and the row-cut policy; all ranks must agree.
        fixed_reduce: Precompute fixed row-reduction plans; defaults to True.
            Atomic reduction disables these plans.
        release_host_copies: Release redundant host group/halo copies after upload
            in distributed layouts; defaults to True. Streamed data is retained.
        domain_size: Optional positive number of contiguous ranks per two-level
            communication domain. ``None`` uses detected node membership.
        groups: ``"by_rows"`` (default) or ``"by_key"`` group ownership. By-key
            grouping requires one copy and uses ordinary row cuts without a plan.
        atomic_reduce: Accumulate contributions with GPU atomic additions;
            defaults to False. Can change floating-point accumulation order.
        vectors: ``"halo"`` (default) exchanges required remote amplitudes;
            ``"full"`` exchanges the full vector internally and enables atomic
            reduction. Full-vector mode requires by-key grouping and one copy.
            Caller inputs still contain only ``n_local`` amplitudes.

    Raises:
        ValueError: Keys, operator terms, storage options or layout combinations
            are unsupported.
        ImportError: CuPy or required native components are unavailable.
        MemoryError: GPU allocations cannot fit in available device memory.
        RuntimeError: Device or communication setup fails.
    """

    def __init__(
        self,
        operator,
        keys,
        *,
        comm=None,
        num_slices=DEFAULT_NUM_SLICES,
        binned=True,
        residency="auto",
        device_budget_bytes=None,
        alpha_kind=None,
        beta_kind=None,
        recut=None,
        exchange="stepped",
        copies=1,
        plan=None,
        fixed_reduce=True,
        release_host_copies=True,
        domain_size=None,
        groups="by_rows",
        atomic_reduce=False,
        vectors="halo",
    ):
        started = time.perf_counter()
        if comm is None:
            from mpi4py import MPI  # noqa: PLC0415

            comm = MPI.COMM_SELF
        self._comm = comm
        self._binned = bool(binned)
        self._block = None
        self._copies = None
        self._fixed_reduce = bool(fixed_reduce)
        self._release_host_copies = bool(release_host_copies)
        self._domain_size = domain_size
        if groups not in ("by_rows", "by_key"):
            raise ValueError(f"groups must be 'by_rows' or 'by_key', got {groups!r}")
        self._groups = groups
        if vectors not in ("halo", "full") or (
            vectors == "full" and groups != "by_key"
        ):
            raise ValueError(
                f"vectors must be 'halo', or 'full' with groups='by_key'; got {vectors!r}"
            )
        self._vectors = vectors
        self._full = None
        self._atomic_reduce = bool(atomic_reduce) or vectors == "full"
        self._y_ext = None
        if plan is not None:
            plan = LayoutPlan.from_dict(
                plan if isinstance(plan, dict) else plan.to_dict()
            )
            plan.check()
            agree(comm, plan)
            copies, num_slices, exchange = plan.copies, plan.num_slices, plan.exchange
            recut = plan.block_cut != "row" and not plan.fixed
        auto_candidates = None
        if copies == "auto" and comm.Get_size() == 1:
            copies = 1
        if copies == "auto":
            from .distribute import plan_copies  # noqa: PLC0415

            copies, auto_candidates = plan_copies(
                comm,
                keys,
                _orbitals(operator),
                int(num_slices),
                alpha_kind=alpha_kind,
                beta_kind=beta_kind,
            )
        copies = int(copies)
        if exchange == "two_level" and copies > 1:
            raise ValueError(
                "the two-level exchange supports one copy per block; use copies=1"
            )
        if groups == "by_key" and copies > 1:
            raise ValueError(
                "groups='by_key' supports one copy per block; use copies=1"
            )
        if groups == "by_key" and plan is None:
            recut = False
        if plan is None:
            if recut is None and comm.Get_size() > 1:
                plan = LayoutPlan(
                    copies=copies,
                    num_slices=int(num_slices),
                    exchange=exchange,
                    block_cut="fitted",
                    shares="balanced",
                    coefficients=[1.0] * 35,
                )
            else:
                plan = LayoutPlan(
                    copies=copies,
                    num_slices=int(num_slices),
                    exchange=exchange,
                    block_cut="membership" if recut and comm.Get_size() > 1 else "row",
                )
            plan.check()
        recut = plan.block_cut != "row" and not plan.fixed
        if auto_candidates is not None:
            plan.measured["copies_candidates"] = {
                str(t): v for t, v in auto_candidates.items()
            }
        self.plan = plan
        if copies > 1:
            self._init_copies(
                operator,
                keys,
                int(num_slices),
                residency,
                device_budget_bytes,
                alpha_kind,
                beta_kind,
                copies,
                exchange,
            )
            agree(comm, self.plan)
            self._log_ready(started)
            return
        if comm.Get_size() > 1:
            self._init_ranks(
                operator,
                keys,
                int(num_slices),
                residency,
                device_budget_bytes,
                alpha_kind,
                beta_kind,
                bool(recut),
                exchange,
            )
            agree(comm, self.plan)
            self._log_ready(started)
            return
        self._is_primary = True
        self.row_start = 0
        self._host = HostOperator(
            operator, keys, alpha_kind, beta_kind, num_slices=int(num_slices)
        )
        self._layout = DeviceLayout(
            self._host,
            residency,
            device_budget_bytes,
            self._fixed_reduce,
            atomic_reduce=self._atomic_reduce,
        )
        self._stream = cupy.cuda.Stream(non_blocking=True)
        self._layout.wait(self._stream)
        self.n_local = int(self._host.num_rows)
        self.plan.share_starts = [0, self.n_local]
        self.shape = (self.n_local, self.n_local)
        self.dtype = np.dtype(np.float64)
        self.supports_device_matvec = True
        self.diag_local = np.array(self._host.diagonal_view(), dtype=np.float64)
        self._log_ready(started)

    def _log_ready(self, started):

        if self._comm.Get_rank() == 0:
            logger.info(
                "Full-string GPU Hamiltonian ready: ranks=%d, rank0_vector_elements=%d, construction=%.3f s",
                self._comm.Get_size(),
                self.n_local,
                time.perf_counter() - started,
                extra={"gpu_rank": 0},
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Full-string GPU layout: %s; device storage (bytes): %s",
                    self.plan.to_dict(),
                    self.device_bytes,
                    extra={"gpu_rank": 0},
                )

    def _init_ranks(
        self,
        operator,
        keys,
        num_slices,
        residency,
        device_budget_bytes,
        alpha_kind,
        beta_kind,
        recut,
        exchange,
    ):


        from fulqrum.gpu.distributed.rank_agreement import run_on_every_rank  # noqa: PLC0415

        from .distribute import build_distributed_groups  # noqa: PLC0415
        from .halo import HaloExchange  # noqa: PLC0415
        from .host import RankOperator, block_reference  # noqa: PLC0415

        comm = self._comm
        num_orbitals = run_on_every_rank(
            comm, "operator width", lambda: _orbitals(operator)
        )
        coefficients = None
        if self.plan.block_cut == "fitted":
            coefficients = (
                self.plan.coefficients
                if self.plan.coefficients is not None
                else [1.0] * 35
            )
        block, receiver, self.recut_info = build_distributed_groups(
            comm,
            keys,
            num_orbitals,
            num_slices,
            alpha_kind,
            beta_kind,
            recut,
            self.plan.share_starts,
            coefficients,
            self.plan.entry_cost,
            fetch_keys=exchange != "two_level",
            groups=self._groups,
            vectors=self._vectors,
        )
        self.plan.measured["groups"] = self._groups
        self.plan.measured["vectors"] = self._vectors
        if self._vectors == "full":
            self._init_full(
                operator, block, receiver, num_orbitals, residency, device_budget_bytes
            )
            return
        self.plan.share_starts = [int(v) for v in block.block_starts]
        self.plan.measured.update(
            {
                "imbalance_before": self.recut_info["imbalance_before"],
                "imbalance_after": self.recut_info["imbalance_after"],
            }
        )
        domains = None
        if exchange == "two_level":
            from .distribute import fetch_halo_keys  # noqa: PLC0415
            from .two_level import exchange_domains, extend_to_domain_union  # noqa: PLC0415

            domain_comm, domains = exchange_domains(comm, self._domain_size)
            extend_to_domain_union(comm, domain_comm, domains, block, receiver)
            fetch_halo_keys(
                comm, block, receiver
            )
        reference = comm.bcast(
            block_reference(block.keys, num_orbitals) if comm.Get_rank() == 0 else None,
            root=0,
        )
        rank_host = run_on_every_rank(
            comm, "table build", lambda: RankOperator(operator, block.keys, reference)
        )
        self._block = block
        self._receiver = receiver
        self._host = (_KeyHost if self._groups == "by_key" else _RankHost)(
            rank_host, receiver, num_orbitals
        )

        if exchange == "two_level":
            from .two_level import TwoLevelExchange  # noqa: PLC0415

            self._halo = TwoLevelExchange(
                comm, block, receiver.halo_row_view(), domains
            )
        else:
            self._halo = HaloExchange(comm, block, receiver.halo_row_view(), exchange)
        tail = getattr(self._halo, "tail_elements", 0)

        y_elements = (
            self._host.num_rows + self._halo.reverse_tail_elements
            if self._groups == "by_key"
            else 0
        )
        reserve = tail + max(0, y_elements - self._host.num_own)
        self._layout = run_on_every_rank(
            comm,
            "device layout",
            lambda: DeviceLayout(
                self._host,
                residency,
                device_budget_bytes,
                self._fixed_reduce,
                reserve_elements=reserve,
                atomic_reduce=self._atomic_reduce,
            ),
        )
        if self._release_host_copies:
            self._layout.release_host_chunks()
            receiver.release_halo()
        self._stream = cupy.cuda.Stream(non_blocking=True)
        self._layout.wait(self._stream)
        self.n_local = block.n_local
        self.row_start = block.row_start
        total = sum(comm.allgather(self.n_local))
        if total != block.num_rows:
            raise ValueError(
                f"the row blocks hold {total} rows, expected {block.num_rows}"
            )
        self.shape = (block.num_rows, block.num_rows)
        self.dtype = np.dtype(np.float64)
        self.supports_device_matvec = True
        self._is_primary = True
        self.diag_local = np.array(rank_host.diagonal_view(), dtype=np.float64)

        self._x_ext = cupy.zeros(self._host.num_rows + tail, dtype=np.float64)
        if y_elements:
            self._y_ext = cupy.zeros(y_elements, dtype=np.float64)
        cupy.cuda.Device().synchronize()

    def _init_full(
        self, operator, block, receiver, num_orbitals, residency, device_budget_bytes
    ):


        from fulqrum.gpu.distributed.rank_agreement import run_on_every_rank  # noqa: PLC0415

        from .full_vectors import FullVectorExchange  # noqa: PLC0415
        from .host import RankOperator, block_reference  # noqa: PLC0415

        comm = self._comm
        size, rank = comm.Get_size(), comm.Get_rank()
        stride = int(np.diff(block.block_starts).max())
        reference = comm.bcast(
            block_reference(block.keys, num_orbitals) if rank == 0 else None, root=0
        )
        rank_host = run_on_every_rank(
            comm, "table build", lambda: RankOperator(operator, block.keys, reference)
        )
        exchange = FullVectorExchange(comm, stride, block.n_local)
        stream = cupy.cuda.Stream(non_blocking=True)


        with stream:
            keys = cupy.zeros(2 * size * stride, dtype=np.uint64)
        keys[2 * rank * stride : 2 * (rank * stride + block.n_local)].set(
            np.ascontiguousarray(block.keys).reshape(-1), stream=stream
        )
        exchange.allgather(keys.view(np.float64), stream, width=2)
        stream.synchronize()
        self._block = block
        self._receiver = receiver
        self._host = _FullHost(rank_host, receiver, num_orbitals, size * stride)
        self._halo = exchange
        reserve = (
            self._host.num_rows - self._host.num_own
        )
        self._layout = run_on_every_rank(
            comm,
            "device layout",
            lambda: DeviceLayout(
                self._host,
                residency,
                device_budget_bytes,
                False,
                reserve_elements=reserve,
                atomic_reduce=True,
                device_keys=keys,
            ),
        )
        if self._release_host_copies:
            self._layout.release_host_chunks()
        self._full = (exchange, stride)
        self._stream = cupy.cuda.Stream(non_blocking=True)
        self._layout.wait(self._stream)
        self.n_local = block.n_local
        self.row_start = block.row_start
        total = sum(comm.allgather(self.n_local))
        if total != block.num_rows:
            raise ValueError(
                f"the row blocks hold {total} rows, expected {block.num_rows}"
            )
        self.shape = (block.num_rows, block.num_rows)
        self.dtype = np.dtype(np.float64)
        self.supports_device_matvec = True
        self._is_primary = True
        self.diag_local = np.array(rank_host.diagonal_view(), dtype=np.float64)
        self._x_ext = cupy.zeros(size * stride, dtype=np.float64)
        self._y_ext = cupy.zeros(size * stride, dtype=np.float64)
        cupy.cuda.Device().synchronize()

    def _init_copies(
        self,
        operator,
        keys,
        num_slices,
        residency,
        device_budget_bytes,
        alpha_kind,
        beta_kind,
        copies,
        exchange,
    ):


        from fulqrum.gpu.distributed.rank_agreement import run_on_every_rank  # noqa: PLC0415

        from .distribute import plan_copy_groups  # noqa: PLC0415
        from .halo import CopyExchange, HaloExchange  # noqa: PLC0415
        from .host import RankOperator, block_reference  # noqa: PLC0415

        comm = self._comm
        num_orbitals = run_on_every_rank(
            comm, "operator width", lambda: _orbitals(operator)
        )
        layout, measured = plan_copy_groups(
            comm, keys, num_orbitals, self.plan, alpha_kind, beta_kind
        )
        self.plan.measured.update(measured)
        reference = comm.bcast(
            block_reference(layout.block_keys, num_orbitals)
            if comm.Get_rank() == 0
            else None,
            root=0,
        )
        rank_host = run_on_every_rank(
            comm,
            "table build",
            lambda: RankOperator(operator, layout.block_keys, reference),
        )
        share = layout.share
        share_begin = share.row_start - layout.block_start
        self._copies = layout
        self._block = share
        self._host = _CopyHost(
            rank_host,
            layout.pieces,
            num_orbitals,
            share_begin,
            share_begin + share.n_local,
        )
        self._layout = run_on_every_rank(
            comm,
            "device layout",
            lambda: DeviceLayout(
                self._host,
                residency,
                device_budget_bytes,
                self._fixed_reduce,
                atomic_reduce=self._atomic_reduce,
            ),
        )
        if self._release_host_copies:
            self._layout.release_host_chunks()
        self._halo = HaloExchange(comm, share, layout.pieces.halo_rows(), exchange)
        if self._release_host_copies:
            layout.pieces.release_halo()
        share_starts = np.array(
            layout.copy_comm.allgather(share_begin) + [len(layout.block_keys)],
            dtype=np.int64,
        )
        self._copy_exchange = CopyExchange(layout.copy_comm, share_starts)
        self._stream = cupy.cuda.Stream(non_blocking=True)
        self._layout.wait(self._stream)
        self.n_local = share.n_local
        self.row_start = share.row_start
        total = sum(comm.allgather(self.n_local))
        if total != share.num_rows:
            raise ValueError(f"the shares hold {total} rows, expected {share.num_rows}")
        self.shape = (share.num_rows, share.num_rows)
        self.dtype = np.dtype(np.float64)
        self.supports_device_matvec = True
        self._is_primary = True
        self.diag_local = np.array(
            rank_host.diagonal_view()[share_begin : share_begin + share.n_local],
            dtype=np.float64,
        )
        self.plan.share_starts = [int(v) for v in share.block_starts]
        self.plan.assignment = layout.plan["assignment"].tolist()
        self.plan.measured["work"] = layout.plan["work"].tolist()
        self._x_ext = cupy.zeros(self._host.num_rows, dtype=np.float64)
        self._y_block = cupy.zeros(self._host.num_own, dtype=np.float64)

    @property
    def keys(self):
        """Return this rank's post-distribution keys as a uint64 ``(n_local, 2)`` copy, alpha then
        beta, sorted by ``(beta, alpha)``."""
        if self._block is not None:
            return self._block.keys.copy()
        return self._host.sorted_keys()

    @property
    def num_orbitals(self):
        """The number of orbitals per spin sector."""
        return self._host.num_orbitals

    @property
    def num_halo(self):
        """The number of remote rows that this rank reads (0 with one rank)."""
        return 0 if self._block is None else self._halo.num_halo

    @property
    def num_streamed_slices(self):
        """The number of key slices streamed from host memory at every matvec."""
        return self._layout.num_streamed

    @property
    def device_bytes(self):
        """The device memory of the stored groups, tables, keys, diagonal and matvec buffers, in bytes."""
        return self._layout.device_bytes()

    def _matvec(self, x, out=None):


        if out is None:
            out = cupy.empty(self.n_local, dtype=np.float64)
        caller = cupy.cuda.get_current_stream()
        self._stream.wait_event(caller.record())
        if self._block is None:
            self._layout.matvec(x, out, self._stream, self._binned)
        elif self._copies is not None:
            block = self._host.num_own
            self._copy_exchange.allgather(x, self._x_ext[:block], self._stream)
            self._halo.run(x, self._x_ext[block:], self._stream)
            self._layout.matvec(self._x_ext, self._y_block, self._stream, self._binned)
            self._copy_exchange.reduce_scatter(self._y_block, out, self._stream)
        elif self._full is not None:
            exchange, stride = self._full
            own = self.n_local
            start = self._comm.Get_rank() * stride
            self._x_ext[start : start + own].data.copy_from_device_async(
                x.data, 8 * own, stream=self._stream
            )
            exchange.allgather(self._x_ext, self._stream)
            with self._stream:
                self._y_ext.fill(0.0)
            self._layout.matvec(
                self._x_ext, self._y_ext, self._stream, self._binned, diagonal=False
            )
            exchange.reduce_scatter(self._y_ext, self._stream)
            with self._stream:
                cupy.add(
                    cupy.multiply(self._layout.diagonal, x),
                    self._y_ext[start : start + own],
                    out=out,
                )
        elif self._y_ext is not None:
            own, rows = self.n_local, self._host.num_rows
            self._x_ext[:own].data.copy_from_device_async(
                x.data, 8 * own, stream=self._stream
            )
            self._halo.run(self._x_ext[:own], self._x_ext[own:], self._stream)
            with self._stream:
                self._y_ext[own:rows].fill(0.0)
            self._layout.matvec(self._x_ext, self._y_ext, self._stream, self._binned)
            self._halo.reverse(self._y_ext[:own], self._y_ext[own:], self._stream)
            out.data.copy_from_device_async(
                self._y_ext.data, 8 * own, stream=self._stream
            )
        else:
            own = self.n_local
            self._x_ext[:own].data.copy_from_device_async(
                x.data, 8 * own, stream=self._stream
            )
            self._halo.run(self._x_ext[:own], self._x_ext[own:], self._stream)
            self._layout.matvec(self._x_ext, out, self._stream, self._binned)
        caller.wait_event(self._stream.record())
        return out

    def _rdm_input(self, x):


        if self._block is None:
            return x
        if self._copies is not None:
            block = self._host.num_own
            self._copy_exchange.allgather(x, self._x_ext[:block], self._stream)
            self._halo.run(x, self._x_ext[block:], self._stream)
            return self._x_ext
        own = self.n_local
        if self._full is not None:
            exchange, stride = self._full
            start = self._comm.Get_rank() * stride
            self._x_ext[start : start + own].data.copy_from_device_async(
                x.data, 8 * own, stream=self._stream
            )
            exchange.allgather(self._x_ext, self._stream)
            return self._x_ext
        self._x_ext[:own].data.copy_from_device_async(
            x.data, 8 * own, stream=self._stream
        )
        self._halo.run(self._x_ext[:own], self._x_ext[own:], self._stream)
        return self._x_ext

    def _rdm_own_keys(self):

        if self._copies is not None:
            first = self._host.share[0]
        elif self._full is not None:
            first = self._comm.Get_rank() * self._full[1]
        else:
            first = 0
        return self._layout.keys[2 * first : 2 * (first + self.n_local)]

    def compute_rdms(self, x, *, rank=1, spin_summed=False):
        """Collectively compute RDMs from this rank's NumPy or CuPy state slice, shape
        ``(n_local,)`` in ``keys`` order. Every rank receives full host NumPy arrays after
        device work and collectives complete; ``rank`` selects the RDM order (1 or 2), and no
        normalization is applied. Spin blocks, spin-summed forms, and conventions follow PySCF
        ``make_rdm12s(reorder=True)``."""
        from .rdm import compute_rdms_full_strs  # noqa: PLC0415

        return compute_rdms_full_strs(self, x, rank=rank, spin_summed=spin_summed)

    def matvec(self, x):
        """Collectively apply the operator to this rank's local vector of shape ``(n_local,)`` in
        ``keys`` order. NumPy input returns NumPy output; CuPy input returns CuPy output."""
        if tuple(x.shape) != (self.n_local,):
            raise ValueError(
                f"x must have shape ({self.n_local},); got {tuple(x.shape)}"
            )
        if isinstance(x, cupy.ndarray):
            return self._matvec(cupy.ascontiguousarray(x, dtype=np.float64))
        d_x = cupy.asarray(np.ascontiguousarray(x, dtype=np.float64))
        y = self._matvec(d_x)
        return cupy.asnumpy(y)


def _orbitals(operator):

    width = int(operator.width)
    if width <= 0 or width % 2:
        raise ValueError(
            f"the operator must have an even, positive qubit count, got {width}"
        )
    return width // 2


class _RankHost:


    def __init__(self, rank_host, receiver, num_orbitals):

        self.rank_host = rank_host
        self.receiver = receiver
        self.num_orbitals = num_orbitals
        self.num_own = rank_host.num_own
        self.num_rows = self.num_own + receiver.halo_row_view().size
        self.num_chunks = receiver.num_chunks()

    def key_words(self):

        return self.rank_host.key_words()

    def halo_key_words(self):

        return self.receiver.halo_key_words()

    def diagonal_view(self):

        return self.rank_host.diagonal_view()

    def table_views(self):

        return self.rank_host.table_views()

    def chunk_views(self, chunk, cls):

        return self.receiver.chunk_views(chunk, cls)

    def release_units(self, units, num_kinds):

        for unit in units:
            self.receiver.release_chunk(unit // num_kinds)


class _KeyHost(_RankHost):


    def chunk_kinds(self):

        return [(0, self.num_rows, 0, self.num_rows, 0, self.num_rows)]

    def chunk_views(self, chunk, cls, kind=0):

        return self.receiver.chunk_views(chunk, cls)


class _FullHost(_RankHost):


    def __init__(self, rank_host, receiver, num_orbitals, num_rows):

        super().__init__(rank_host, receiver, num_orbitals)
        self.num_rows = int(num_rows)

    def chunk_kinds(self):

        return [(0, self.num_rows, 0, self.num_rows, 0, self.num_rows)]

    def chunk_views(self, chunk, cls, kind=0):

        return self.receiver.chunk_views(chunk, cls)


class _CopyHost:


    def __init__(self, rank_host, pieces, num_orbitals, share_begin, share_end):

        self.rank_host = rank_host
        self.pieces = pieces
        self.num_orbitals = num_orbitals
        self.num_own = pieces.num_own
        self.num_rows = self.num_own + pieces.halo_rows().size
        self.num_chunks = pieces.num_chunks()
        self.share = (int(share_begin), int(share_end))
        diagonal = np.zeros(self.num_own, dtype=np.float64)
        diagonal[share_begin:share_end] = rank_host.diagonal_view()[
            share_begin:share_end
        ]
        self.diagonal = diagonal

    def chunk_kinds(self):

        own = tuple(self.pieces.ranges(0)) + self.share
        remote = tuple(self.pieces.ranges(1)) + (0, self.num_own)
        return [own, remote]

    def chunk_views(self, chunk, cls, kind):

        return self.pieces.chunk_views(chunk, cls, kind)

    def release_units(self, units, num_kinds):

        for unit in units:
            self.pieces.release_chunk(unit // num_kinds, unit % num_kinds)

    def key_words(self):

        return self.rank_host.key_words()

    def halo_key_words(self):

        return self.pieces.halo_key_words()

    def diagonal_view(self):

        return self.diagonal

    def table_views(self):

        return self.rank_host.table_views()
