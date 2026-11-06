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


from __future__ import annotations

import json
import logging
import os
import socket
import time
import warnings
from types import SimpleNamespace
from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator

from fulqrum.core.spmv import FulqrumSpMV

from .. import autotune
from ..config import GpuConfig, warn_unknown_names
from ..logging import format_memory
from . import _collective
from .exchange import ExchangeExecutor
from .recorder import MatvecRecorder, RecorderSettings
from .grid import GridComms, TileGrid, discover_sources
from .operator_check import DIRECTION_TOLERANCE, unsupported_groups
from .local_comm import LocalComm, mpi_namespace
from .rank_agreement import raise_on_every_rank, run_on_every_rank
from .exchange_plan import (
    EXCHANGE_RING,
    EXCHANGE_TWO_LEVEL,
    _hierarchical_exchange_plan,
    _mixed_window_schedule,
    _two_level_local_ring_args,
    _two_level_remote_bytes,
    _two_level_round_args,
    check_pairing,
    check_same_settings,
    choose_exchange,
    exchange_mode,
    ring_steps,
    window_steps,
)

from .aabb_direct import full_source_plan, sorted_beta_graph
from .aabb_parents import (
    CLIQUE_MAX_SHARED_BYTES,
    alpha_task_words,
    clique_edge_group_plan,
    clique_size_buckets,
    clique_tile_options,
    degree_sorted_sell,
    orbital_g,
    parent_kind,
    parent_masks,
    parent_options,
    parent_plan,
    remote_alpha_mode,
    remote_beta_options,
    validate_clique_plan,
)
from .aabb_source_timing import source_class
from .aabb_cliques import beta_clique_plan
from .launch import (
    AlphaDoubleClass,
    BetaDoubleClass,
    CliqueCross,
    SourceLaunch,
    build_source_caches,
    cross_ready,
    launch_cross,
    launch_steady_arms,
)
from .arm_plan import (
    CLIQUE as ARM_CLIQUE,
    TRANSPOSED_CLIQUE as ARM_TRANSPOSED_CLIQUE,
    ArmSettings,
    SourceFacts,
    plan_arms,
)
from .arm_pair_tables import (
    ArmPairTablesPayload,
    aa_table,
    bb_table,
    bb_words,
    csr_to_sell_values,
    derive_pair_table,
    merge_pair_tables,
    pair_payload_error,
    pair_table_options,
)
from .tables import (
    TableOptions,
    aabb_coefficients,
    arm_cache_sizes,
    build_tile_tables,
    hbm_fit_error,
    pair_table,
    values_on_edges,
)
from .tile_tables import cross_sector_table, cross_table, sector_tiles, tile_input_error
from .term_tables import (
    _assert_aa_bb_factor_cache_invariance,
    _assert_aaaa_bbbb_proj_invariance,
    _build_aabb_g_dense,
    _build_direct_integral_tables,
    _build_grp_pos_tables,
)
from .diagnostics import (
    _dbg_print,
    _dbg_ring_slot_fingerprint,
    _dbg_v_fingerprint,
    _debug_matvec_enabled,
)
from .dist_matvec import (
    TileClass,
    _default_2d_process_grid,
)
from .tile_discovery import (
    _group_flip_masks,
)

logger = logging.getLogger(__name__)

ARM_CACHES = ("aaaa", "bbbb", "aa_c0", "aa_cpb", "bb_c0", "bb_cpb")
_HBM_FITS_SKIP_LOGGED = False


def _operator_tables(spmv):

    from fulqrum.gpu.common.operator_arrays import to_gpu_arrays  # noqa: PLC0415

    return to_gpu_arrays(spmv, skip_diag=True)


_ALPHA2_SELL_KEYS = (
    "d_alpha2_sell_off",
    "d_alpha2_sell_len",
    "d_alpha2_sell_j",
    "d_alpha2_sell_apid",
    "d_alpha2_sell_asign",
)


def _last_value(array) -> int:

    if array is None or array.size == 0:
        return 0
    return int(array[-1:].get()[0])


def _kernel_param_digest() -> str:

    try:
        from .kernel_launch import param_struct_digest  # noqa: PLC0415
    except ImportError:
        return "unavailable"
    return param_struct_digest()


def _to_device(array, dtype=None):

    import cupy  # noqa: PLC0415

    host = np.ascontiguousarray(array)
    if dtype is not None and host.dtype != np.dtype(dtype):
        host = host.astype(dtype, copy=False)
    return cupy.asarray(host)


def _wait_for_uploads() -> None:

    import cupy  # noqa: PLC0415

    cupy.cuda.Stream.null.synchronize()


def _pack_term_tables(gpu_data) -> tuple:


    inds = np.ascontiguousarray(gpu_data["term_inds"]).astype(np.uint8)
    vals = np.ascontiguousarray(gpu_data["term_vals"]).astype(np.uint8)
    coeffs = np.ascontiguousarray(gpu_data["term_coeffs_re"]).astype(np.float64)
    real_phase = np.ascontiguousarray(gpu_data["term_real_phase"]).astype(np.int8)
    if real_phase.size > 0 and not np.all((real_phase == 1) | (real_phase == -1)):
        raise ValueError("term_real_phase must be +/-1 for the phase fold")
    folded = coeffs * real_phase.astype(np.float64)
    num_ops = np.ascontiguousarray(gpu_data["term_num_ops"]).astype(np.uint8)
    num_proj = np.ascontiguousarray(gpu_data["term_num_proj"]).astype(np.uint8)
    meta = (num_proj.astype(np.uint16) << 8) | num_ops.astype(np.uint16)
    pairs = (vals.astype(np.uint16) << 8) | inds.astype(np.uint16)
    n_terms, stride = pairs.shape
    n_words = (stride + 3) // 4
    padded = np.zeros((n_terms, 4 * n_words), dtype=np.uint16)
    padded[:, :stride] = pairs
    words = np.zeros((n_terms, n_words), dtype=np.uint64)
    for lane in range(4):
        words |= padded[:, lane::4].astype(np.uint64) << np.uint64(16 * lane)
    if folded.size > 0:
        header = np.empty((folded.size, 2), dtype=np.uint64)
        header[:, 0] = np.ascontiguousarray(folded, dtype=np.float64).view(np.uint64)
        header[:, 1] = meta.astype(np.uint64)
    else:
        header = np.zeros((1, 2), dtype=np.uint64)
    return np.ascontiguousarray(header.reshape(-1)), np.ascontiguousarray(words)


def _device_address(array) -> int:

    if array is None:
        return 0
    if hasattr(array, "data") and hasattr(array.data, "ptr"):
        return int(array.data.ptr)
    return int(array)


def _dev_nonnull(a) -> bool:


    return a is not None and a.size > 0


def _empty_pinned(elems: int):

    import cupyx  # noqa: PLC0415

    return cupyx.empty_pinned((elems,), dtype=np.float64)


def _owned_bytes(buffers) -> int:

    if buffers is None:
        return 0
    return sum(int(buffer.nbytes) for buffer in buffers if buffer.base is None)


class TileGPUResidentHamiltonian(LinearOperator):
    """GPU-backed real type-2 Hamiltonian for distributed matrix-free solves.

    Construct one instance on every MPI rank. Each rank uses one CUDA
    device and owns a rectangular tile of the tensor-product state vector.
    Construction, each local-vector product and RDM calculation are collective:
    all ranks must participate in the same order. The Hamiltonian, half strings,
    process grid and shared execution settings must agree across ranks; the
    selected CUDA device may differ by rank.

    Use this operator with ``distributed_primme_eigsh`` for MPI eigenvalue solves.
    Its global LinearOperator shape is ``(N, N)`` and dtype is ``float64``, where
    ``N = len(alpha_strs) * len(beta_strs)``. The distributed solver uses
    ``_matvec(v_local)``, ``n_local`` and ``_comm`` to operate on each rank's
    portion of the vector. Inherited SciPy ``matvec`` checks the global length
    and is not the local-vector interface for a multi-rank solve.

    Alpha and beta strings are sorted independently. The global vector index is
    ``beta_index * len(alpha_strs) + alpha_index``. A local vector contains this
    rank's tile flattened in beta-outer, alpha-inner order: local index
    ``b * n_a_local + a`` identifies the corresponding pair of local strings.
    ``_matvec`` accepts real NumPy-compatible or CuPy input of length ``n_local``.
    Without ``out``, it returns a host NumPy ``float64`` vector. With keyword
    ``out=buffer``, supply a CuPy ``float64`` array of length ``n_local`` on this
    rank's device; the result is a flat device view of that buffer. Required GPU
    work completes before return. ``supports_device_matvec`` exposes this capability.

    The subspace must be a half-string/tensor product with alpha in the low qubits and
    beta in the high qubits. Each sector must have 1–64 orbitals (bits), a fixed
    electron count and operator groups supported by the GPU kernels. Use at
    least two occupied orbitals per half string; lower occupations are outside
    verified GPU support and may give incorrect results. GPU execution requires
    CuPy and the compiled CUDA components.

    Args:
        hsub: Real type-2 ``SubspaceHamiltonian`` on the tensor product subspace.
        alpha_strs, beta_strs: Spin-sector string collections matching ``hsub``.
            Each sector supports 1–64 orbitals, so a string uses at most 64 bits.
            Use one-dimensional integer NumPy arrays (``uint64`` for the full
            64-bit range) or binary-string sequences.
        gpu_ids: Exactly one CUDA device ID for this rank, as a one-element
            sequence. Defaults to ``(0,)``; choose each rank's device explicitly.
        comm: MPI communicator; defaults to ``MPI.COMM_WORLD``. ``LocalComm``
            supports in-process workers, as used by ``GPUHamiltonian``.
        process_grid: ``(p_alpha, p_beta)``, with product equal to communicator
            size (num ranks). ``None`` chooses a grid automatically.
        threads_per_block: CUDA threads per kernel block; defaults to 256.
        comm_backend: ``"auto"``, ``"nccl"``, ``"host_mpi"``, ``"cuda_mpi"`` or
            ``"inproc"``. Automatic selection respects GPU configuration;
            it tries ``nccl`` first and falls back to ``host_mpi`` when conditions
            are not met. Prefer ``nccl`` when available or ``host_mpi`` otherwise.
            ``cuda_mpi`` may work, but is not fully validated yet.
            ``LocalComm`` uses the in-process backend.
        intra_node_size: Optional positive size of contiguous rank groups for
            communication. ``None`` uses detected node membership.
        build_threads: CPU construction threads. Zero derives the count from
            ``OMP_NUM_THREADS``; positive ``FQ_BUILD_THREADS`` takes precedence.
        config: Checked ``GpuConfig`` snapshot. ``None`` uses
            ``GpuConfig.from_environ()`` during construction.

    Raises:
        NotImplementedError: Complex operators or multiple GPU IDs per rank
            are requested.
        ValueError: Device selection, process grid, layout, electron counts,
            operator groups or shared settings are invalid.
        ImportError: The default MPI communicator cannot be obtained because
            mpi4py is unavailable.
        RuntimeError: Collective construction or device-memory checks fail.
    """

    _VALID_BACKENDS = ("nccl", "cuda_mpi", "host_mpi", "inproc", "auto")
    _LEGACY_DISCOVERY_MAX_P = 256
    _SCRATCH_BYTES = 1 << 30
    _HBM_FREE_FRACTION = 0.85

    def __init__(
        self,
        hsub,
        alpha_strs,
        beta_strs,
        *,
        gpu_ids: tuple[int, ...] = (0,),
        comm=None,
        process_grid: Optional[tuple[int, int]] = None,
        threads_per_block: int = 256,
        comm_backend: str = "auto",
        intra_node_size: Optional[int] = None,
        build_threads: int = 0,
        config: Optional[GpuConfig] = None,
    ):

        started = time.perf_counter()
        self._init_run(
            hsub,
            alpha_strs,
            beta_strs,
            gpu_ids=gpu_ids,
            comm=comm,
            threads_per_block=threads_per_block,
            comm_backend=comm_backend,
            intra_node_size=intra_node_size,
            build_threads=build_threads,
            config=config,
        )
        pre_init_elapsed, partition_elapsed = self._init_grid(
            process_grid, comm_backend, started
        )
        self._init_streams()
        self._init_options()
        self._init_device_state()
        self._init_records(pre_init_elapsed, partition_elapsed)
        self._init_exchange()
        self._init_tables()
        self._check_memory_and_log()

    def _init_run(
        self,
        hsub,
        alpha_strs,
        beta_strs,
        *,
        gpu_ids,
        comm,
        threads_per_block,
        comm_backend,
        intra_node_size,
        build_threads,
        config,
    ) -> None:


        self._config = config if config is not None else GpuConfig.from_environ()

        if not hsub.spmv.is_real:
            raise NotImplementedError(
                "TileGPUResidentHamiltonian currently supports real-valued operators only."
            )

        if comm_backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"comm_backend={comm_backend!r} not one of {self._VALID_BACKENDS}"
            )

        if comm is None:
            try:
                from mpi4py import MPI  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "TileGPUResidentHamiltonian needs mpi4py, or an in-process LocalComm."
                ) from exc
            comm = MPI.COMM_WORLD

        gpu_ids = tuple(int(gid) for gid in gpu_ids)
        if len(gpu_ids) > 1:
            raise NotImplementedError(
                "TileGPUResidentHamiltonian is single-GPU-per-rank; "
                f"got gpu_ids={gpu_ids}."
            )

        self._hsub = hsub
        self._alpha_strs = alpha_strs
        self._beta_strs = beta_strs
        self._comm = comm
        self._MPI = mpi_namespace(comm)
        self._rank = comm.Get_rank()
        self._logger = logging.LoggerAdapter(
            logger,
            {"gpu_rank": self._rank, "gpu_device": gpu_ids[0] if gpu_ids else None},
        )
        self._nranks = comm.Get_size()
        self._is_primary = True
        self._primme_comm = comm
        self._gpu_ids = gpu_ids
        if len(gpu_ids) != 1:
            raise ValueError("TileGPUResidentHamiltonian requires one GPU ID per rank")
        try:
            import cupy  # noqa: PLC0415

            cupy.cuda.Device(gpu_ids[0]).use()
        except ImportError:
            pass
        self._threads_per_block = int(threads_per_block)
        self._intra_node_size = intra_node_size
        _bt_env = int(self._config.value("FQ_BUILD_THREADS"))
        if _bt_env > 0:
            self._build_threads = _bt_env
        elif int(build_threads) > 0:
            self._build_threads = int(build_threads)
        else:
            self._build_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))

    def _init_grid(self, process_grid, comm_backend, started) -> tuple:


        if process_grid is None:
            process_grid = _default_2d_process_grid(self._nranks)
        p_alpha, p_beta = int(process_grid[0]), int(process_grid[1])
        if p_alpha * p_beta != self._nranks:
            raise ValueError(
                f"process_grid {process_grid!r} must satisfy "
                f"p_alpha*p_beta == comm.Get_size() = {self._nranks}"
            )
        self._process_grid = (p_alpha, p_beta)
        self._coord_alpha = self._rank // p_beta
        self._coord_beta = self._rank % p_beta

        self._grid_comms = GridComms.split(
            self._comm, TileGrid(p_alpha, p_beta, (), ())
        )
        self._alpha_comm = self._grid_comms.alpha
        self._beta_comm = self._grid_comms.beta
        self._alpha_rank = self._alpha_comm.Get_rank()
        self._beta_rank = self._beta_comm.Get_rank()

        requested_backend = comm_backend
        if comm_backend == "auto" and not isinstance(self._comm, LocalComm):
            requested_backend = self._config.value("FQ_DIST_BACKEND")
        self._backend, self._backend_reason = _collective.backend_for(
            self._comm, requested_backend
        )
        self._backend_name = self._backend.name

        n_a = int(len(self._alpha_strs))
        n_b = int(len(self._beta_strs))
        n_global = n_a * n_b
        self._n_a = n_a
        self._n_b = n_b
        super().__init__(shape=(n_global, n_global), dtype=np.float64)

        self._alpha_uniq, self._beta_uniq = self._resolve_uniq_arrays()
        self._validate_subspace()

        self._cached_gpu_data: dict | None = None

        _pre_init_mpi_elapsed = time.perf_counter() - started
        _t_part = time.perf_counter()
        if isinstance(self._hsub.spmv, FulqrumSpMV):
            try:
                self._cached_gpu_data = _operator_tables(self._hsub.spmv)
            except Exception:  # noqa: BLE001
                self._cached_gpu_data = None
        self._grid = TileGrid.build(p_alpha, p_beta, self._alpha_uniq, self._beta_uniq)
        self._alpha_ranges = list(self._grid.alpha_ranges)
        self._beta_ranges = list(self._grid.beta_ranges)
        _partition_elapsed = time.perf_counter() - _t_part
        self._a0, self._a1 = self._alpha_ranges[self._coord_alpha]
        self._b0, self._b1 = self._beta_ranges[self._coord_beta]
        self._n_a_local = self._a1 - self._a0
        self._n_b_local = self._b1 - self._b0
        self._n_local = self._n_a_local * self._n_b_local

        self._n_a_peer_max = max(a1 - a0 for (a0, a1) in self._alpha_ranges)
        self._n_b_peer_max = max(b1 - b0 for (b0, b1) in self._beta_ranges)

        self._alpha_bra = self._alpha_uniq[self._a0 : self._a1]
        self._beta_bra = self._beta_uniq[self._b0 : self._b1]

        bytes_per_double = 8
        self._d_buffer_bytes = {
            "v_local": self._n_local * bytes_per_double,
            "out_local": self._n_local * bytes_per_double,
            "alpha_ring_total": 2
            * self._n_a_peer_max
            * self._n_b_local
            * bytes_per_double,
            "beta_ring_total": 2
            * self._n_a_local
            * self._n_b_peer_max
            * bytes_per_double,
        }
        return _pre_init_mpi_elapsed, _partition_elapsed

    def _init_streams(self) -> None:

        try:
            import cupy  # noqa: PLC0415

            cupy.cuda.Device(self._gpu_ids[0]).use()
            self._cupy_available = True
            self._d_v_local = None
            self._d_out_local = None
            self._d_alpha_ring = None
            self._d_beta_ring = None
            self._compute_stream = cupy.cuda.Stream(non_blocking=True)


            self._copy_stream = cupy.cuda.Stream(non_blocking=True, priority=-1000)
        except ImportError:
            self._cupy_available = False
            self._d_v_local = None
            self._d_out_local = None
            self._d_alpha_ring = None
            self._d_beta_ring = None
            self._compute_stream = None
            self._copy_stream = None

    def _init_options(self) -> None:


        try:
            self._exchange_mode = exchange_mode(self._config)
            self._exchange_mode_error = None
        except ValueError as error:
            self._exchange_mode = None
            self._exchange_mode_error = str(error)
        self._exchange_name = EXCHANGE_RING
        self._exchange_reason = ""
        self._exchange_digest = ""
        self._hierarchical_exchange_active = False
        self._hierarchical_lean_ring_active = False
        self._matvec_alias_xy_enabled = True
        self._autotune = self._resolve_settings()
        tuned = self._autotune
        self._HBM_FREE_FRACTION = float(tuned.value("hbm_free_fraction"))
        self._diag_recompute_enabled = bool(tuned.value("diag_recompute"))
        self._diag_recompute_explicit = (
            tuned.choices["diag_recompute"].source == "explicit"
        )
        self._exchange_double_buffer_enabled = bool(tuned.value("double_buffer"))
        if self._exchange_mode_error is None:
            self._exchange_mode = tuned.value("exchange")
        self._n_recv_slots = 2 if self._exchange_double_buffer_enabled else 1

        self._aaaa_ycoarsen_k = 4
        self._aa_ycoarsen_k = 4
        self._bb_xcoarsen_k = 4
        self._bbbb_xcoarsen_k = 4
        bb_hoist = bool(tuned.value("bb_prefetch"))
        self._bb_pair_xcoarsen_k = int(tuned.value("bb_alphas")) or self._bb_xcoarsen_k

        aaaa_group = tuned.value("aaaa_group")
        self._aaaa_group_auto = aaaa_group == "auto"
        self._aaaa_group_by_source: dict = {}
        self._arm_order = dict(
            aaaa=0 if self._aaaa_group_auto else int(aaaa_group),
            bbbb=int(self._config.value("FQ_BBBB_ALPHA_BLOCKS_PER_GROUP")),
            bb_pair=int(self._config.value("FQ_BB_PAIR_ALPHA_BLOCKS_PER_GROUP")),
            aa_hoist=bool(tuned.value("aa_prefetch")),
            bb_hoist=bb_hoist,
            parent_group=int(self._config.value("FQ_AABB_OWN_CLIQUES_PER_GROUP")),
        )
        self._aabb_residual_degree_order = bool(
            self._config.value("FQ_AABB_CLIQUE_RESIDUAL_DEGREE_SORT")
        )
        self._aabb_clique_mma = bool(tuned.value("mma"))
        self._aabb_clique_panel = bool(tuned.value("source_panel"))
        self._aabb_clique_panel_state: dict = {}
        self._xterm_scratch_bytes = int(tuned.value("scratch_mib")) << 20
        self._aabb_lbg = bool(tuned.value("label_batching")) and self._aabb_clique_mma
        self._aabb_lbg_state: dict = {}
        self._xterm_scratch = None
        self._xterm_flag = None
        self._bbbb_class_enabled = bool(tuned.value("bbbb_class"))
        self._bbbb_class_min = int(self._config.value("FQ_BBBB_CLASS_MIN_SIZE"))
        self._bbbb_class_alpha_group = int(
            self._config.value("FQ_BBBB_CLASS_ALPHA_CHUNKS_PER_GROUP")
        )
        self._bbbb_class_state: dict = {}
        self._bbbb_class_bytes = 0
        self._aaaa_class_enabled = bool(tuned.value("aaaa_class"))
        self._aaaa_class_min = int(self._config.value("FQ_AAAA_CLASS_MIN_SIZE"))
        self._aaaa_class_slab = int(tuned.value("aaaa_slab"))
        self._aaaa_leftover_xt = bool(
            self._config.value("FQ_AAAA_CLASS_TRANSPOSE_RESIDUAL")
        )
        self._aaaa_slab_identity = None
        self._aaaa_class_state: dict = {}
        self._aaaa_class_scratch = None
        (
            self._aabb_parent_enabled,
            self._aabb_parent_min,
            self._aabb_parent_timing_enabled,
        ) = parent_options(self._config)
        self._aabb_parent_plan = None
        self._aabb_parent_bytes = 0
        self._aabb_parent_kind = parent_kind(self._config)
        self._aabb_clique_output_rows_per_tile, self._aabb_clique_edges_per_batch = (
            clique_tile_options(
                dict(
                    self._config,
                    FQ_AABB_CLIQUE_OUTPUT_ROWS_PER_TILE=str(tuned.value("clique_rows")),
                )
            )
        )
        self._aabb_remote_beta_mode, self._aabb_source_timing_enabled = (
            remote_beta_options(self._config)
        )
        self._aabb_remote_beta_plans = {}
        self._aabb_remote_beta_bytes = 0
        self._aa_pair_enabled, self._bb_pair_enabled = pair_table_options(self._config)
        self._aabb_remote_alpha_mode = remote_alpha_mode(self._config)
        self._aabb_remote_alpha_shared = None
        self._aabb_remote_alpha_plans: dict = {}
        self._aabb_remote_alpha_bytes = 0
        self._aaaa_min_blocks, self._aaaa_identity_slot = 6, True
        self._arm_pair_state: dict = {}
        self._arm_pair_rank: dict = {}
        self._arm_pair_bytes = 0
        self._caches_built = False
        self._d_build_error = None
        self._arm_caches: dict = {}
        self._host_arms = False
        self._host_arm_reason = "operator tables not built"

        self._arm_timing_enabled = (
            self._config.get("FQ_TILE_RESIDENT_ARM_TIMING", "0") == "1"
        )
        self._recorder = None
        self._matvec_call_idx = 0
        self._first_matvec_logged = False

        self._comm_timing_enabled = (
            self._config.get("FQ_COMM_TIMING", "0") == "1"
            or self._aabb_source_timing_enabled
        )
        self._recorder_settings = RecorderSettings(
            comm_timing=self._comm_timing_enabled,
            parent_timing=bool(
                self._aabb_parent_enabled and self._aabb_parent_timing_enabled
            ),
            source_timing=self._aabb_source_timing_enabled,
            arm_timing=self._arm_timing_enabled,
        )

        self._aabb_factored = False
        self._aabb_gate_dead_caches_enabled = True
        self._aabb_csr_bytes = 0
        self._n_beta_pairs = 0

    def _resolve_settings(self):


        device = None
        if self._cupy_available:
            import cupy  # noqa: PLC0415

            bus = cupy.cuda.runtime.deviceGetPCIBusId(cupy.cuda.Device().id)
            sharing = self._comm.allgather((socket.gethostname(), bus))
            device = autotune.device_facts(sharing.count(sharing[self._rank]))
        else:
            self._comm.allgather(None)
        self._autotune_ranks_on_device = (
            device.ranks_on_device if device is not None else 1
        )
        kernels = autotune.kernel_facts()
        resolution = autotune.resolve(
            self._config,
            device,
            kernels,
            probe=autotune.probe_for(device),
            agree=self._comm.allgather,
        )
        raise_on_every_rank(
            self._comm,
            "; ".join(resolution.errors) or None,
            what="GPU setting resolution",
        )
        check_same_settings(
            self._comm, resolution.collective(), "resolved GPU settings"
        )
        for text in resolution.warnings:
            if self._rank == 0:
                self._logger.debug("GPU tuning override: %s", text)
        if self._rank == 0:
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug("GPU configuration resolved: %s", resolution.line())
        return resolution

    def _init_device_state(self) -> None:


        self._d_mixed_slots = None
        self._d_hierarchical_stage = None
        self._d_hierarchical_gather = None
        self._hierarchical_node_comm = None

        self._d_term_header = None
        self._d_term_inds_vals = None
        self._d_term_proj_inds = None
        self._d_term_proj_bits = None
        self._term_inds_stride = 0
        self._term_proj_stride = 0
        self._d_grp_ladder_ptrs = None
        self._ladder_offset = 0
        self._d_oper_elems = None

        self._d_aabb_g = None
        self._d_grp2_ht_keys = None
        self._d_grp2_ht_vals = None
        self._d_grp2_ht_bitset = None
        self._grp2_ht_mask = 0
        self._d_grp4_ht_keys = None
        self._d_grp4_ht_vals = None
        self._d_grp4_ht_bitset = None
        self._grp4_ht_mask = 0
        self._tile_tables = None
        self._d_alpha_uniq = None
        self._d_beta_uniq = None
        self._num_alpha = 0
        self._num_beta = 0
        self._const_energy = 0.0

    def _init_records(self, pre_init_elapsed, partition_elapsed) -> None:


        self._operator_bytes = self._estimate_operator_bytes(self._hsub)

        self._construction_timing: dict = {
            "process_grid": self._process_grid,
            "n_a_local": self._n_a_local,
            "n_b_local": self._n_b_local,
            "n_local": self._n_local,
            "local_tile_bytes": self._n_local * 8,
            "backend": self._backend_name,
            "backend_reason": self._backend_reason,
            "partition": partition_elapsed,
            "pre_init_mpi": pre_init_elapsed,
            "autotune": self._autotune.record(),
        }
        self._last_timing: dict = {}

        self._min_free_bytes = None
        self._hbm_trace: dict = {}

    def _init_exchange(self) -> None:

        _t = time.perf_counter()
        self._flip_masks = self._build_flip_masks()
        self._rank_tiles = self._grid.tiles()
        self._source_set = discover_sources(
            self._grid,
            self._grid_comms,
            self._alpha_uniq,
            self._beta_uniq,
            self._flip_masks,
            self._build_threads,
            self._LEGACY_DISCOVERY_MAX_P,
        )
        self._sources_2d = self._source_set.sources
        self._construction_timing["source_discovery"] = time.perf_counter() - _t

        self._construction_timing["tile_class_counts"] = self._source_set.class_counts()
        self._construction_timing["source_tiles"] = len(self._sources_2d)

        _t = time.perf_counter()
        self._set_mixed_neighbors()
        self._construction_timing["mixed_neighbor_setup"] = time.perf_counter() - _t
        _t = time.perf_counter()
        self._build_hierarchical_exchange_metadata()
        self._construction_timing["hierarchical_setup"] = time.perf_counter() - _t
        self._construction_timing["hierarchical_exchange_active"] = bool(
            self._hierarchical_exchange_active
        )
        self._construction_timing["hierarchical_rounds"] = len(
            getattr(self, "_hierarchical_round_args", ())
        )
        self._construction_timing["hierarchical_lean_ring"] = bool(
            self._hierarchical_lean_ring_active
        )
        self._construction_timing["exchange"] = self._exchange_name
        self._construction_timing["exchange_reason"] = self._exchange_reason
        self._construction_timing["mixed_indegree"] = int(self._mixed_indegree)
        self._construction_timing["mixed_outdegree"] = int(self._mixed_outdegree)

        _t = time.perf_counter()
        self._build_subcomm_exchange_metadata()
        self._construction_timing["subcomm_exchange_metadata"] = (
            time.perf_counter() - _t
        )
        self._construction_timing["mixed_window_hops"] = len(self._mixed_window_hops)
        self._exchange_digest = check_same_settings(
            self._comm, self._exchange_settings()
        )
        self._config_digest = check_same_settings(
            self._comm, self._config.record(), "GPU settings"
        )
        self._construction_timing["config_digest"] = self._config_digest
        self._construction_timing["exchange_digest"] = self._exchange_digest
        self._construction_timing["arm_timing_enabled"] = bool(self._arm_timing_enabled)

    def _init_tables(self) -> None:


        _t = time.perf_counter()
        self._construction_timing["aabb_arms_off"] = ""
        run_on_every_rank(
            self._comm, "operator table build", self._build_and_upload_operator_tables
        )
        if not (
            self._aabb_factored
            and 2 <= self._num_alpha <= 64
            and 2 <= self._num_beta <= 64
        ):
            self._turn_off_factored_aabb_arms(
                "an eligible factored aabb operator with 2 to 64 orbitals in each half"
            )
        self._construction_timing["gpu_upload"] = time.perf_counter() - _t
        self._arm_plans = run_on_every_rank(self._comm, "arm plan", self._plan_arms)
        self._construction_timing["operator_bytes"] = int(self._operator_bytes)
        if self._cupy_available:
            _wait_for_uploads()

    def _check_memory_and_log(self) -> None:

        started = time.perf_counter()
        self.verify_hbm_fits()
        timing = self._construction_timing
        timing["hbm_check"] = time.perf_counter() - started
        timing["predicted_hbm_bytes"] = int(sum(self.predicted_hbm_bytes.values()))
        self._record_free_hbm("post_construction")
        if self._rank != 0:
            return
        if self._config.get("FQ_HBM_LOG", "0") == "1" and self._logger.isEnabledFor(
            logging.DEBUG
        ):
            categories = ", ".join(
                f"{key}={format_memory(value)}"
                for key, value in self.predicted_hbm_bytes.items()
            )
            self._logger.debug(
                "GPU memory after construction: rank=0, predicted=%s, minimum_free_observed=%s. Allocations: %s",
                format_memory(timing["predicted_hbm_bytes"]),
                format_memory(self._min_free_bytes),
                categories,
            )
            peak_rss = None
            try:
                import resource
                import sys

                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                peak_rss = rss if sys.platform == "darwin" else rss * 1024
            except Exception as error:
                self._logger.debug("Unable to read peak host memory: %s", error)
            self._logger.debug(
                "Host memory: rank=0, backend=%s, pinned_staging_required=%s, caller_pinned_buffers=%s, "
                "pinned_staging=%s, process_peak_rss=%s, build_threads=%d",
                self._backend_name,
                bool(self._backend.needs_pinned_staging),
                bool(self._backend.uses_caller_pinned),
                format_memory(self._pinned_staging_bytes()),
                format_memory(peak_rss),
                self._build_threads,
            )
        self._logger.info(
            "GPU Hamiltonian ready: ranks=%d, grid=%dx%d, backend=%s, rank0_vector_elements=%d",
            self._nranks,
            *self._process_grid,
            self._backend_name,
            timing["n_local"],
        )
        self._logger.debug(
            "GPU construction: source_discovery=%.3f s, upload=%.3f s, tile_tables=%.3f s, "
            "tile_classes=%s, mixed_sources=%d, mixed_destinations=%d",
            timing["source_discovery"],
            timing["gpu_upload"],
            timing.get("peer_csr_cache_build", 0.0),
            timing["tile_class_counts"],
            timing["mixed_indegree"],
            timing["mixed_outdegree"],
        )
        warn_unknown_names()
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(
                "GPU run configuration: %s", json.dumps(self.run_record, sort_keys=True)
            )

    @property
    def run_record(self) -> dict:
        """Return this rank's JSON-serializable settings and construction choices.

        Includes the resolved backend, exchange mode, process grid, tuning
        decisions and configuration digests. Uses existing state without GPU
        work or communicator collectives.
        """
        ct = self._construction_timing
        return {
            "settings": self._config.record(),
            "config_digest": self._config_digest,
            "backend": self._backend_name,
            "backend_reason": ct.get("backend_reason", ""),
            "exchange": self._exchange_name,
            "exchange_reason": self._exchange_reason,
            "exchange_digest": self._exchange_digest,
            "aabb_arms_off": ct.get("aabb_arms_off", ""),
            "autotune": self._autotune.record(),
            "process_grid": list(self._process_grid),
            "kernel_params": _kernel_param_digest(),
        }

    @property
    def process_grid(self) -> tuple[int, int]:
        """Process grid ``(p_alpha, p_beta)``; product equals communicator size."""
        return self._process_grid

    @property
    def coord_alpha(self) -> int:
        """This rank's zero-based alpha coordinate in ``process_grid``."""
        return self._coord_alpha

    @property
    def coord_beta(self) -> int:
        """This rank's zero-based beta coordinate in ``process_grid``."""
        return self._coord_beta

    @property
    def comm(self):
        """MPI or ``LocalComm`` communicator used by this rank."""
        return self._comm

    @property
    def backend_name(self) -> str:
        """Resolved backend: ``nccl``, ``host_mpi``, ``cuda_mpi`` or ``inproc``."""
        return self._backend_name

    @property
    def n_local(self) -> int:
        """Local vector length, equal to ``n_a_local * n_b_local``."""
        return self._n_local

    @property
    def supports_device_matvec(self) -> bool:
        """Whether this instance enables CuPy input and ``out`` for local matvecs.

        This reports the selected device-vector code path; it is not a separate
        CUDA hardware or kernel validation test.
        """
        return bool(self._cupy_available)

    @property
    def n_a_local(self) -> int:
        """Number of alpha half strings in this rank's vector tile."""
        return self._n_a_local

    @property
    def n_b_local(self) -> int:
        """Number of beta half strings in this rank's vector tile."""
        return self._n_b_local

    def compute_rdms(self, evec_local, *, rank: int = 1, spin_summed: bool = False):
        """Collectively compute reduced density matrices from local state-vector tiles.

        Every rank supplies its part of the same real state and uses the same
        ``rank`` and ``spin_summed`` options. All ranks receive the full result
        as host NumPy arrays after GPU work and communication complete. Alpha
        and beta must have equal orbital counts.

        Amplitudes are used without normalization. Scaling the full state by a
        real factor scales every RDM by its square. Normalize the full state
        before distributing it if you need RDMs for a normalized state. Spin
        and tensor-index conventions follow PySCF ``make_rdm12s(reorder=True)``.

        Args:
            evec_local: NumPy-compatible or CuPy array of ``n_local`` real
                amplitudes, or shape ``(n_b_local, n_a_local)``. Uses the same
                beta-outer, alpha-inner tile order as the local matvec.
            rank: RDM order, 1 or 2; defaults to 1. This is not an MPI rank.
            spin_summed: Combine alpha and beta spin blocks; defaults to False.

        Returns:
            For ``rank=1``, ``(gamma_a, gamma_b)`` or spin-summed ``gamma``.
            For ``rank=2``, ``((gamma_a, gamma_b), (G_aa, G_ab, G_bb))`` or
            spin-summed ``(gamma, G)``. Each one-particle array has shape
            ``(m, m)`` and each two-particle array ``(m, m, m, m)``, where ``m``
            is the common number of orbitals per sector.

        Raises:
            ValueError: The RDM order, local vector size or sector counts are
                invalid, or alpha and beta have different orbital counts.
        """
        from .rdm_dist import compute_rdms_tile_resident

        return compute_rdms_tile_resident(
            self, evec_local, rank=rank, spin_summed=spin_summed
        )

    @property
    def last_timing(self) -> dict:
        """Return a copy of this rank's latest successful matvec timing dictionary.

        Empty before the first successful product. ``wall``, ``compute`` and
        ``exchange`` are in seconds; ``comm_ms_total`` is in milliseconds, and
        ``*_bytes`` fields are byte counts. The compute interval includes exchange, so these
        durations are not additive. Reading uses existing measurements only.
        """
        return dict(self._last_timing)

    @property
    def construction_timing(self) -> dict:
        """Return a copy of this rank's construction measurements and resolved settings.

        Phase times are in seconds and allocation sizes in bytes. The dictionary
        also includes counts, grid dimensions and execution choices. Reading it
        does not repeat construction or launch GPU work.
        """
        return dict(self._construction_timing)

    @property
    def diag_local(self) -> np.ndarray:
        """Return this rank's Hamiltonian diagonal as a host NumPy ``float64`` vector.

        Includes constant energy, has length ``n_local`` and uses beta-outer,
        alpha-inner tile order. Suitable for the distributed solver's local
        diagonal preconditioner. The array may be cached; treat it as read-only.
        Access may compute the diagonal or copy it from the GPU. Without CuPy,
        this property returns a zero array.
        """
        if not self._cupy_available:
            return np.zeros(self._n_local, dtype=np.float64)
        if self._d_diag_local is not None:
            import cupy

            return cupy.asnumpy(self._d_diag_local).astype(np.float64, copy=False)
        if getattr(self, "_diag_local_host_cache", None) is None:
            self._diag_local_host_cache = self._host_diag_local(
                self._hsub.spmv, self._num_alpha
            )
        return self._diag_local_host_cache

    def _host_diag_local(self, spmv, num_alpha: int) -> np.ndarray:


        if int(self._has_diag):
            if not hasattr(spmv, "diagonal_slice_product_space"):
                raise RuntimeError(
                    "The operator has diagonal terms, but its CPU object has no "
                    "diagonal_slice_product_space method to compute them."
                )
            diag_flat = spmv.diagonal_slice_product_space(
                self._alpha_bra, self._beta_bra, 0, self._n_a_local, num_alpha
            )
            diag = np.array(diag_flat, dtype=np.float64)
        else:
            diag = np.zeros(self._n_local, dtype=np.float64)
        return diag + self._const_energy

    @property
    def source_ranks(self) -> tuple[int, ...]:
        """Sorted communicator ranks whose vector tiles contribute to this rank's result.

        Includes this rank. The tuple describes source communication, not a
        separate set of eigensolver participants.
        """
        return tuple(sorted(self._sources_2d))

    @property
    def assigned_source_ranks(self) -> tuple[int, ...]:
        """Same source-rank tuple as ``source_ranks``."""
        return self.source_ranks

    def run_workers(self) -> None:
        """Compatibility no-op; all ranks participate directly in collective calls."""
        return None

    def stop_workers(self) -> None:
        """Compatibility no-op; there are no persistent workers to stop."""
        return None

    def _predict_arm_cache_bytes(self) -> int:

        if self._tile_tables is None:
            return 0
        if self._host_arms:
            return (
                sum(int(cache.nbytes) for cache in self._arm_caches.values())
                + self._arm_pair_bytes
            )
        total = 0
        for src_rank in self._sources_2d:
            same_alpha, same_beta = self._same_coordinates(src_rank)
            total += 8 * sum(
                self._arm_cache_sizes(src_rank, same_alpha, same_beta).values()
            )
        return total

    def _same_coordinates(self, src_rank: int) -> tuple[bool, bool]:

        _, _, _, _, ca, cb = self._rank_tiles[src_rank]
        return ca == self._coord_alpha, cb == self._coord_beta

    def _arm_cache_sizes(
        self,
        src_rank: int,
        same_alpha: bool,
        same_beta: bool,
        doubles_sell: Optional[bool] = None,
    ) -> dict:


        tables = self._tile_tables.sources[src_rank]
        if doubles_sell is None:
            doubles_sell = tables.alpha.doubles_sell is not None
        lengths = tables.arm_cache_lengths(
            same_alpha, same_beta, doubles_sell, self._aabb_gate_dead_caches_enabled
        )
        return arm_cache_sizes(lengths, self._num_alpha, self._num_beta)

    def _pinned_staging_bytes(self) -> int:


        total = 0
        for attr in (
            "_beta_ring_send_pinned",
            "_beta_ring_recv_pinned",
            "_alpha_ring_send_pinned",
            "_alpha_ring_recv_pinned",
            "_mixed_send_pinned",
            "_mixed_recv_pinned",
        ):
            buf = getattr(self, attr, None)
            if buf is None:
                continue
            if isinstance(buf, list):
                total += sum(int(b.nbytes) for b in buf if b is not None)
            else:
                total += int(buf.nbytes)
        return total

    @property
    def predicted_hbm_bytes(self) -> dict:
        """Return this rank's predicted GPU memory by allocation category, in bytes.

        Covers Hamiltonian buffers, tables and caches. Solver-owned storage is
        separate. These are the estimates used by ``verify_hbm_fits``, not a
        measurement of total device usage. Reading does not allocate GPU memory.
        """
        mixed_recv = 0
        _msl = getattr(self, "_d_mixed_slots", None)
        if _msl is not None:
            mixed_recv += sum(int(s.nbytes) for s in _msl if s.base is None)

        return {
            **self._d_buffer_bytes,
            "operator": self._operator_bytes,
            "scratch": self._SCRATCH_BYTES,
            "mixed_recv": mixed_recv,
            "arm_caches": self._predict_arm_cache_bytes(),
            "aabb_csr": int(self._aabb_csr_bytes),
            "aabb_parent_extra": int(self._aabb_parent_bytes),
            "aabb_remote_beta_metadata": int(self._aabb_remote_beta_bytes),
        }

    def partition_diagnostic(self) -> dict:
        """Return local connection counts and estimates for comparing partition work."""
        a2 = self._alpha2_count
        a4 = self._alpha4_count
        b2 = self._beta2_count
        b4 = self._beta4_count
        na = int(a2.size)
        nb = int(b2.size)
        proxy_alpha_total = int(a2.sum() + a4.sum())
        proxy_beta_total = int(b2.sum() + b4.sum())
        proxy_alpha2_total = int(a2.sum())
        proxy_beta2_total = int(b2.sum())
        proxy_aabb_cross = proxy_alpha2_total * proxy_beta2_total
        proxy_total = (
            nb * proxy_alpha_total + na * proxy_beta_total + proxy_aabb_cross + na * nb
        )
        return {
            "n_a_local": na,
            "n_b_local": nb,
            "alpha2_count": a2,
            "alpha4_count": a4,
            "beta2_count": b2,
            "beta4_count": b4,
            "proxy_alpha_total": proxy_alpha_total,
            "proxy_beta_total": proxy_beta_total,
            "proxy_alpha2_total": proxy_alpha2_total,
            "proxy_beta2_total": proxy_beta2_total,
            "proxy_aabb_cross": int(proxy_aabb_cross),
            "proxy_total": int(proxy_total),
        }

    def verify_hbm_fits(self):
        """Collectively check predicted allocations against available GPU memory.

        Call on every communicator rank. Tries to release unused CuPy device and
        pinned memory-pool blocks before querying free memory, then checks
        estimates with reserved headroom. Existing allocated buffers remain in use.
        Without CuPy, the memory test is skipped; ranks still participate in
        the collective error check.

        Returns:
            None when no rank reports an overrun.

        Raises:
            RuntimeError: At least one rank's predicted allocations do not fit;
                the error is raised on every rank.
        """
        raise_on_every_rank(
            self._comm, self._hbm_fit_error(), RuntimeError, "verify_hbm_fits"
        )
        return None

    def _hbm_fit_error(self) -> Optional[str]:

        global _HBM_FITS_SKIP_LOGGED  # noqa: PLW0603

        if not self._cupy_available:
            if not _HBM_FITS_SKIP_LOGGED:
                self._logger.debug(
                    "GPU memory check skipped because CuPy is unavailable"
                )
                _HBM_FITS_SKIP_LOGGED = True
            return None

        import cupy  # noqa: PLC0415

        try:
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as error:  # noqa: BLE001
            self._logger.debug(
                "freeing the cupy memory pools before the memory check failed: %s",
                error,
            )

        free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
        error = hbm_fit_error(
            self.predicted_hbm_bytes, free_bytes, total_bytes, self._HBM_FREE_FRACTION
        )
        if error is None:
            return None
        return (
            f"{error} backend={self._backend_name} "
            f"process_grid={self._process_grid} "
            f"n_a_local={self._n_a_local} n_b_local={self._n_b_local} "
            f"n_a_peer_max={self._n_a_peer_max} "
            f"n_b_peer_max={self._n_b_peer_max}."
        )

    def _record_free_hbm(self, tag: str) -> None:


        if not self._cupy_available:
            return
        try:
            import cupy  # noqa: PLC0415

            free_bytes, _total = cupy.cuda.runtime.memGetInfo()
        except Exception as error:  # noqa: BLE001
            self._logger.debug(
                "free device memory for %s is not available: %s", tag, error
            )
            return
        free_bytes = int(free_bytes)
        self._hbm_trace[tag] = free_bytes
        if self._min_free_bytes is None or free_bytes < self._min_free_bytes:
            self._min_free_bytes = free_bytes

    @property
    def min_free_hbm_bytes(self):
        """Lowest sampled free memory on this rank's GPU, in bytes, or None.

        Samples are recorded after construction and matvecs. This is a local
        sampled minimum, not a continuously measured peak or a cross-rank minimum.
        """
        return self._min_free_bytes

    def _resolve_uniq_arrays(self) -> tuple[np.ndarray, np.ndarray]:

        alpha = self._alpha_strs
        beta = self._beta_strs
        if isinstance(alpha, np.ndarray) and alpha.dtype.kind in "iu":
            return np.sort(alpha), np.sort(np.asarray(beta))
        from .hash_utils import _pack_half_strs_to_uint64  # noqa: PLC0415

        alpha_arr, beta_arr = _pack_half_strs_to_uint64(alpha, beta)
        return np.sort(alpha_arr), np.sort(beta_arr)

    @staticmethod
    def _estimate_operator_bytes(hsub) -> int:

        spmv = hsub.spmv
        if hasattr(spmv, "estimate_gpu_bytes"):
            return int(spmv.estimate_gpu_bytes())
        try:
            attrs = vars(spmv).values()
        except TypeError:
            return 0
        total = 0
        for attr in attrs:
            if isinstance(attr, np.ndarray):
                total += int(attr.nbytes)
        return total

    def _build_flip_masks(self) -> dict:

        spmv = self._hsub.spmv
        if hasattr(spmv, "flip_masks"):
            return spmv.flip_masks
        if isinstance(spmv, FulqrumSpMV) and hasattr(spmv, "num_alpha"):
            try:
                gpu_data = self._cached_gpu_data
                if gpu_data is None:
                    gpu_data = _operator_tables(spmv)
                grp_oi = np.asarray(gpu_data["grp_offdiag_inds"])
                grp_os = np.asarray(gpu_data["grp_offdiag_sizes"])
                return _group_flip_masks(grp_oi, grp_os, int(spmv.num_alpha))
            except (RuntimeError, ValueError, KeyError) as error:
                warnings.warn(
                    f"Unable to build operator flip masks on rank {self._rank}; "
                    f"source discovery will scan all tiles and may take longer: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return {"alpha2": [], "alpha4": [], "beta2": [], "beta4": []}

    def _set_mixed_neighbors(self) -> None:

        self._mixed_sources = self._source_set.mixed_sources
        self._mixed_destinations = self._source_set.mixed_destinations
        self._mixed_indegree = len(self._mixed_sources)
        self._mixed_outdegree = len(self._mixed_destinations)

    def _build_hierarchical_exchange_metadata(self) -> None:


        self._hierarchical_exchange_active = False
        self._hierarchical_lean_ring_active = False
        self._hierarchical_round_args = []
        self._hierarchical_local_ring_args = ()
        self._hierarchical_plan = None
        self._hierarchical_domains = ()
        self._hierarchical_max_tile_elems = 0
        self._hierarchical_direct_remote_bytes = 0
        self._hierarchical_network_bytes = 0

        raise_on_every_rank(self._comm, self._exchange_mode_error)
        modes = self._comm.allgather(self._exchange_mode)
        raise_on_every_rank(
            self._comm,
            None
            if len(set(modes)) == 1
            else f"FQ_EXCHANGE differs across ranks: {modes}",
        )
        two_level_backend = self._backend_name == "nccl" or bool(
            self._config.value("_FULQRUM_TWO_LEVEL_ANY_BACKEND")
        )
        nccl_on_every_rank = all(
            self._comm.allgather(self._cupy_available and two_level_backend)
        )
        host_mpi_on_every_rank = all(
            self._comm.allgather(
                self._cupy_available and self._backend_name == "host_mpi"
            )
        )
        two_level_possible = nccl_on_every_rank or (
            host_mpi_on_every_rank and self._exchange_mode == EXCHANGE_TWO_LEVEL
        )
        if not two_level_possible or self._exchange_mode == EXCHANGE_RING:
            self._exchange_name, self._exchange_reason = choose_exchange(
                self._exchange_mode, nccl_on_every_rank, None, host_mpi_on_every_rank
            )
            return
        node_comm, node_members, domains = self._exchange_domains()
        self._exchange_name, self._exchange_reason = choose_exchange(
            self._exchange_mode, nccl_on_every_rank, domains, host_mpi_on_every_rank
        )
        if self._exchange_name == EXCHANGE_RING:
            return
        plan = self._two_level_plan(node_members, domains)
        tile_elems = tuple(
            int((a1 - a0) * (b1 - b0)) for a0, a1, b0, b1, _ca, _cb in self._rank_tiles
        )
        round_args = _two_level_round_args(plan, self._rank, tile_elems)
        direct_local, hierarchical_local = _two_level_remote_bytes(
            plan, domains, self._sources_2d, round_args, tile_elems
        )
        direct_global = int(self._comm.allreduce(direct_local, op=self._MPI.SUM))
        hierarchical_global = int(
            self._comm.allreduce(hierarchical_local, op=self._MPI.SUM)
        )

        self._hierarchical_node_comm = node_comm
        self._hierarchical_domains = domains
        self._hierarchical_plan = plan
        self._hierarchical_local_ring_args = _two_level_local_ring_args(
            plan, plan.local_sources, tile_elems
        )
        self._hierarchical_round_args = round_args
        self._hierarchical_max_tile_elems = max(tile_elems, default=0)
        self._hierarchical_direct_remote_bytes = direct_local
        self._hierarchical_network_bytes = hierarchical_local
        self._hierarchical_exchange_active = True
        self._hierarchical_lean_ring_active = True
        self._exchange_reason += (
            f"; domain sizes {[len(domain) for domain in domains]}, "
            f"{len(round_args)} rounds, remote bytes reduced "
            f"{direct_global / hierarchical_global if hierarchical_global > 0 else float('inf'):.2f}x"
        )

    def _exchange_domains(self) -> tuple:


        if self._intra_node_size is None:
            node_comm = self._backend.comm_split_type_node(self._comm)
        else:
            domain_size = int(self._intra_node_size)
            if domain_size <= 0:
                raise ValueError("intra_node_size must be positive")
            node_comm = self._comm.Split(
                color=self._rank // domain_size, key=self._rank
            )
        node_members = tuple(int(r) for r in node_comm.allgather(self._rank))
        leaders = self._comm.allgather(min(node_members))
        grouped: dict[int, list[int]] = {}
        for world_rank, domain_leader in enumerate(leaders):
            grouped.setdefault(int(domain_leader), []).append(world_rank)
        domains = tuple(tuple(grouped[key]) for key in sorted(grouped))
        return node_comm, node_members, domains

    def _two_level_plan(self, node_members, domains):


        sources_by_rank = self._comm.allgather(tuple(sorted(self._sources_2d)))
        plan = _hierarchical_exchange_plan(self._rank, domains, sources_by_rank)
        raise_on_every_rank(
            self._comm,
            None
            if node_members == plan.domain_members
            else "two-level exchange requires node-communicator rank order to match "
            f"domain member order; got node_members={node_members}, "
            f"domain_members={plan.domain_members}",
        )
        check_pairing(
            self._comm,
            [(r.send_dst, r.recv_src) for r in plan.rounds],
            "two-level rounds",
        )
        return plan

    def _validate_subspace(self) -> None:


        spmv = self._hsub.spmv
        if not isinstance(spmv, FulqrumSpMV):
            return
        num_alpha = int(spmv.subspace.num_alpha_qubits)
        widths = {"alpha": num_alpha, "beta": int(spmv.width) - num_alpha}
        for name, width in widths.items():
            if not 0 < width <= 64:
                raise ValueError(
                    f"the GPU operator supports 1 to 64 {name} qubits; got {width}"
                )
        for name, strings in (("alpha", self._alpha_uniq), ("beta", self._beta_uniq)):
            counts = sorted(
                {int(x).bit_count() for x in np.asarray(strings, dtype=np.uint64)}
            )
            if len(counts) > 1:
                raise ValueError(
                    f"the GPU operator needs one electron count in the {name} strings; "
                    f"got counts {counts}"
                )

    def _validate_operator_groups(self, spmv, num_alpha: int, grp4_direct) -> None:


        found = unsupported_groups(spmv, num_alpha, grp4_direct)
        self._construction_timing["operator_groups_checked"] = len(grp4_direct)
        if not found:
            return
        reasons: dict[str, int] = {}
        for _group, _flips, reason in found:
            reasons[reason] = reasons.get(reason, 0) + 1
        examples = "; ".join(
            f"group {group} flips {flips}: {reason}"
            for group, flips, reason in found[:5]
        )
        raise ValueError(
            f"the GPU operator cannot compute {len(found)} operator group(s) exactly "
            f"(reasons {reasons}); first: {examples}. Use the CPU operator for this Hamiltonian."
        )

    def _turn_off_factored_aabb_arms(self, need: str) -> None:


        explicit = {
            "FQ_AABB_CLIQUE_ENABLE": self._aabb_parent_enabled,
            "FQ_AABB_REMOTE_BETA_METHOD": self._aabb_remote_beta_mode != "baseline",
            "FQ_AABB_REMOTE_ALPHA_METHOD": self._aabb_remote_alpha_mode != "baseline",
        }
        requested = [
            name for name, active in explicit.items() if active and name in self._config
        ]
        if requested:
            raise ValueError(f"{', '.join(requested)} require {need}")
        if any(explicit.values()):
            self._construction_timing["aabb_arms_off"] = f"the clique arms need {need}"
            self._logger.debug(
                "Clique optimization disabled: requires %s; using baseline alpha-beta kernels",
                need,
            )
        self._aabb_parent_enabled = False
        self._aabb_remote_beta_mode = "baseline"
        self._aabb_remote_alpha_mode = "baseline"

    def _tile_rows(self, rank: int) -> int:

        a0, a1, b0, b1, _coord_alpha, _coord_beta = self._rank_tiles[rank]
        return b1 - b0

    def _tile_columns(self, rank: int) -> int:

        a0, a1, b0, b1, _coord_alpha, _coord_beta = self._rank_tiles[rank]
        return a1 - a0

    def _exchange_settings(self) -> dict:


        return {
            "exchange": self._exchange_name,
            "double_buffer": bool(self._exchange_double_buffer_enabled),
            "process_grid": list(self._process_grid),
            "intra_node_size": self._intra_node_size,
            "rank_tiles": [list(map(int, tile)) for tile in self._rank_tiles],
        }

    def _build_subcomm_exchange_metadata(self) -> None:

        self._build_exchange_steps()
        self._allocate_pinned_staging()
        if not self._cupy_available:
            self._d_alpha_slot_persistent = None
            self._d_beta_slot_persistent = None
            self._d_exchange_pool = None
            self._d_hierarchical_stage = None
            self._d_hierarchical_gather = None
            self._d_buffer_bytes.update(
                hierarchical_stage=0,
                hierarchical_gather_total=0,
                exchange_pool=0,
                alpha_ring_total=0,
                beta_ring_total=0,
            )
            return
        import cupy  # noqa: PLC0415

        self._d_alpha_slot_persistent = cupy.arange(
            max(int(self._n_a_peer_max), 1), dtype=cupy.int32
        )
        self._d_beta_slot_persistent = cupy.arange(
            max(int(self._n_b_peer_max), 1), dtype=cupy.int32
        )
        self._allocate_exchange_pool()
        if self._hierarchical_exchange_active:
            elems = max(int(self._hierarchical_max_tile_elems), 1)
            self._d_hierarchical_stage = cupy.zeros(elems, dtype=cupy.float64)
            self._d_hierarchical_gather = [cupy.empty(elems, dtype=cupy.float64)]
        else:
            self._d_hierarchical_stage = None
            self._d_hierarchical_gather = None
        stage = (
            None if self._d_hierarchical_stage is None else [self._d_hierarchical_stage]
        )
        self._d_buffer_bytes["hierarchical_stage"] = _owned_bytes(stage)
        self._d_buffer_bytes["hierarchical_gather_total"] = _owned_bytes(
            self._d_hierarchical_gather
        )
        self._d_buffer_bytes["alpha_ring_total"] = _owned_bytes(self._d_alpha_ring)
        self._d_buffer_bytes["beta_ring_total"] = _owned_bytes(self._d_beta_ring)
        self._d_buffer_bytes["exchange_pool"] = _owned_bytes(self._d_exchange_pool)

    def _build_exchange_steps(self) -> None:


        self._alpha_comm_world_ranks = np.asarray(
            self._alpha_comm.allgather(self._rank), dtype=np.int32
        )
        self._beta_comm_world_ranks = np.asarray(
            self._beta_comm.allgather(self._rank), dtype=np.int32
        )
        self._mixed_recv_counts = np.zeros(self._mixed_indegree, dtype=np.int64)
        for i, src in enumerate(self._mixed_sources):
            a0, a1, b0, b1, _, _ = self._rank_tiles[src]
            self._mixed_recv_counts[i] = (a1 - a0) * (b1 - b0)
        self._mixed_recv_total = int(self._mixed_recv_counts.sum())
        self._mixed_slot_elems = (
            int(self._mixed_recv_counts.max()) if self._mixed_indegree > 0 else 0
        )
        p_alpha, p_beta = self._process_grid
        self._mixed_window_hops = _mixed_window_schedule(
            p_alpha,
            p_beta,
            self._coord_alpha,
            self._coord_beta,
            self._mixed_destinations,
            self._mixed_sources,
        )
        check_pairing(self._comm, self._mixed_window_hops, "mixed window")
        self._beta_ring_steps = ring_steps(
            "beta_ring",
            self._alpha_comm_world_ranks,
            self._alpha_rank,
            self._exchange_double_buffer_enabled,
        )
        self._alpha_ring_steps = ring_steps(
            "alpha_ring",
            self._beta_comm_world_ranks,
            self._beta_rank,
            self._exchange_double_buffer_enabled,
        )
        self._mixed_steps = window_steps(
            self._mixed_window_hops, self._exchange_double_buffer_enabled
        )
        self._ringshift_alpha_slot_elems = int(self._n_b_local * self._n_a_peer_max)
        self._ringshift_beta_slot_elems = int(self._n_b_peer_max * self._n_a_local)

    def _allocate_pinned_staging(self) -> None:


        self._beta_ring_send_pinned = None
        self._beta_ring_recv_pinned = None
        self._alpha_ring_send_pinned = None
        self._alpha_ring_recv_pinned = None
        self._mixed_send_pinned = None
        self._mixed_recv_pinned = None
        need_pinned = bool(self._backend.uses_caller_pinned)
        if (
            self._hierarchical_exchange_active
            or not self._cupy_available
            or not need_pinned
        ):
            return
        if self._beta_ring_steps:
            self._beta_ring_send_pinned = _empty_pinned(self._ringshift_beta_slot_elems)
            self._beta_ring_recv_pinned = [
                _empty_pinned(self._ringshift_beta_slot_elems) for _ in range(2)
            ]
        if self._alpha_ring_steps:
            self._alpha_ring_send_pinned = _empty_pinned(
                self._ringshift_alpha_slot_elems
            )
            self._alpha_ring_recv_pinned = [
                _empty_pinned(self._ringshift_alpha_slot_elems) for _ in range(2)
            ]
        mixed_active = self._mixed_indegree > 0 or self._mixed_outdegree > 0
        if mixed_active and int(self._n_local) > 0:
            self._mixed_send_pinned = _empty_pinned(int(self._n_local))
        if mixed_active and int(self._mixed_recv_total) > 0:
            self._mixed_recv_pinned = [
                _empty_pinned(max(int(self._mixed_slot_elems), 1)) for _ in range(2)
            ]

    def _allocate_exchange_pool(self) -> None:


        import cupy  # noqa: PLC0415

        self._d_exchange_pool = None
        self._d_alpha_ring = None
        self._d_beta_ring = None
        ring_exchange = not self._hierarchical_exchange_active
        alpha_ring = ring_exchange and bool(self._alpha_ring_steps)
        beta_ring = ring_exchange and bool(self._beta_ring_steps)
        mixed_window = (
            ring_exchange
            and (self._mixed_indegree > 0 or self._mixed_outdegree > 0)
            and int(self._mixed_recv_total) > 0
        )
        if not (alpha_ring or beta_ring or mixed_window):
            return
        alpha_elems = int(self._ringshift_alpha_slot_elems)
        beta_elems = int(self._ringshift_beta_slot_elems)
        mixed_elems = int(self._mixed_slot_elems)
        pool_elems = max(
            alpha_elems if alpha_ring else 0,
            beta_elems if beta_ring else 0,
            mixed_elems if mixed_window else 0,
            1,
        )
        self._d_exchange_pool = [
            cupy.empty(pool_elems, dtype=cupy.float64)
            for _ in range(self._n_recv_slots)
        ]
        if alpha_ring:
            self._d_alpha_ring = [
                p[:alpha_elems].reshape(self._n_b_local, self._n_a_peer_max)
                for p in self._d_exchange_pool
            ]
        if beta_ring:
            self._d_beta_ring = [
                p[:beta_elems].reshape(self._n_b_peer_max, self._n_a_local)
                for p in self._d_exchange_pool
            ]
        if mixed_window:
            self._d_mixed_slots = [
                p[: max(mixed_elems, 1)] for p in self._d_exchange_pool
            ]

    def _build_and_upload_operator_tables(self) -> None:

        spmv = self._hsub.spmv
        if not self._cupy_available or not isinstance(spmv, FulqrumSpMV):
            return
        from fulqrum.gpu.distributed.tile_discovery import _build_tile_connection_tables  # noqa: PLC0415

        steps = {}
        clock = [time.perf_counter()]

        def step(name):

            now = time.perf_counter()
            steps[name] = now - clock[0]
            clock[0] = now

        gpu_data = self._cached_gpu_data
        if gpu_data is None:
            gpu_data = _operator_tables(spmv)
        self._cached_gpu_data = None
        step("operator_tables")
        grp4_direct = self._read_operator_groups(spmv, gpu_data)
        self._choose_table_source(spmv)
        step("operator_groups")
        self_tables = _build_tile_connection_tables(
            self._alpha_bra,
            self._beta_bra,
            self._alpha_bra,
            self._beta_bra,
            self._flip_masks,
            threads=self._build_threads,
        )
        step("self_connections")
        self._d_alpha_uniq = _to_device(self._alpha_uniq, np.uint64)
        self._d_beta_uniq = _to_device(self._beta_uniq, np.uint64)
        self._upload_diagonal(spmv)
        step("diagonal")
        if not self._host_arms:
            self._upload_term_tables(gpu_data)
        step("term_tables")
        self._upload_aabb_coefficients(spmv, gpu_data, grp4_direct)
        step("aabb_coefficients")
        self._set_bra_self_tables(self_tables)
        self._build_source_tile_tables(self_tables)
        step("source_tile_tables")
        if self._host_arms:
            started = time.perf_counter()
            self._build_host_arm_values()
            self._construction_timing["host_arm_values_build"] = (
                time.perf_counter() - started
            )
        step("host_arm_values")
        self._build_aabb_plans(self_tables)
        step("aabb_plans")
        self._operator_bytes = self._operator_table_bytes()
        for name, seconds in steps.items():
            self._construction_timing["build_" + name] = seconds
        self._logger.debug(
            "Operator construction stages: rank=%d %s",
            self._rank,
            " ".join(f"{name}={seconds:.3f}s" for name, seconds in steps.items()),
        )
        timing = self._construction_timing
        self._logger.debug(
            "Tile tables ready: rank=%d, host_build=%.3fs upload=%.3fs sources=%d alpha_peers=%d "
            "beta_peers=%d operator_bytes=%d",
            self._rank,
            timing["peer_csr_cache_cpu_build"],
            timing["peer_csr_cache_gpu_upload"],
            int(timing["peer_csr_cache_n_peers"]),
            int(timing["peer_csr_cache_n_distinct_alpha_builds"]),
            int(timing["peer_csr_cache_n_distinct_beta_builds"]),
            self._operator_bytes,
        )

    def _read_operator_groups(self, spmv, gpu_data):


        from fulqrum.gpu.distributed.tile_discovery import _group_flip_masks  # noqa: PLC0415

        num_alpha = int(spmv.subspace.num_alpha_qubits)
        num_beta = int(spmv.width) - num_alpha
        self._num_alpha = num_alpha
        self._num_beta = num_beta
        num_groups = int(gpu_data["num_groups"])
        self._has_offdiag = num_groups > 0
        self._const_energy = float(getattr(spmv, "const_energy", 0.0))
        self._has_diag = int(
            bool(gpu_data["has_nonzero_diag"]) or self._const_energy != 0.0
        )
        grp_oi = np.ascontiguousarray(gpu_data["grp_offdiag_inds"])
        grp_os = np.ascontiguousarray(gpu_data["grp_offdiag_sizes"])
        sub = {}
        clock = [time.perf_counter()]

        def step(name):

            now = time.perf_counter()
            sub[name] = now - clock[0]
            clock[0] = now

        self._flip_masks = _group_flip_masks(grp_oi, grp_os, num_alpha)
        step("flip_masks")
        grp4_direct = None
        if num_groups > 0:
            _, grp4_direct = _build_direct_integral_tables(
                gpu_data, grp_oi, grp_os, num_groups, num_alpha
            )
            step("direct_integrals")
            self._validate_operator_groups(spmv, num_alpha, grp4_direct)
            step("validate_groups")
            _assert_aaaa_bbbb_proj_invariance(
                gpu_data, grp_oi, grp_os, num_groups, num_alpha
            )
            step("proj_invariance")
            _assert_aa_bb_factor_cache_invariance(
                gpu_data, grp_oi, grp_os, num_groups, num_alpha
            )
            step("factor_cache_invariance")
            self._aabb_factored = True
        for name, seconds in sub.items():
            self._construction_timing["groups_" + name] = seconds
        self._logger.debug(
            "Operator group processing: rank=%d %s",
            self._rank,
            " ".join(f"{name}={seconds:.3f}s" for name, seconds in sub.items()),
        )
        self._n_beta_pairs = max(num_beta * (num_beta - 1) // 2, 1)
        self._construction_timing["aabb_csr_enabled"] = bool(self._aabb_factored)
        return grp4_direct

    def _choose_table_source(self, spmv) -> None:


        table_source = self._config.value("FQ_ARM_TABLE_SOURCE")
        if table_source == "kernel":
            self._host_arm_reason = "FQ_ARM_TABLE_SOURCE=kernel"
        else:
            self._host_arm_reason = self._host_arm_values_reason(spmv)
        if table_source == "host" and self._host_arm_reason is not None:
            raise ValueError(
                f"FQ_ARM_TABLE_SOURCE=host cannot run: {self._host_arm_reason}"
            )
        self._host_arms = self._host_arm_reason is None
        self._construction_timing["host_arm_values"] = self._host_arm_reason or "on"

    def _upload_diagonal(self, spmv) -> None:


        import cupy  # noqa: PLC0415

        self._d_diag_alpha = None
        self._d_diag_beta = None
        self._d_diag_Mb = None
        recompute = (
            self._diag_recompute_enabled
            and self._has_diag
            and hasattr(spmv, "diag_separable_pieces")
        )
        if not recompute:
            self._d_diag_local = _to_device(
                self._host_diag_local(spmv, self._num_alpha), np.float64
            )
            return
        diag_alpha, diag_beta, pieces, ok = spmv.diag_separable_pieces(
            self._alpha_bra, self._beta_bra, 0, self._n_a_local, self._num_alpha
        )
        if not ok and self._diag_recompute_explicit:
            raise RuntimeError(
                "FQ_DIAG_RECOMPUTE: recompute incompatible "
                "opposite spin terms "
                "for this operator. Unset FQ_DIAG_RECOMPUTE."
            )
        if not ok:
            self._diag_recompute_enabled = False
            self._autotune.choices["diag_recompute"] = autotune.Choice(
                False, "rule", "the operator cannot split its diagonal; stored"
            )
            self._d_diag_local = _to_device(
                self._host_diag_local(spmv, self._num_alpha), np.float64
            )
            return
        self._d_diag_alpha = _to_device(
            np.asarray(diag_alpha, dtype=np.float64) + self._const_energy, np.float64
        )
        self._d_diag_beta = _to_device(diag_beta, np.float64)
        n_beta_qubits = int(pieces.shape[1])
        d_pieces = _to_device(pieces, np.float64)
        d_beta_bra = _to_device(self._beta_bra, np.uint64)
        qubits = cupy.arange(n_beta_qubits, dtype=cupy.uint64).reshape(n_beta_qubits, 1)
        z_beta = 1.0 - 2.0 * (
            (d_beta_bra.reshape(1, -1) >> qubits) & cupy.uint64(1)
        ).astype(cupy.float64)
        self._d_diag_Mb = cupy.ascontiguousarray(d_pieces @ z_beta)
        self._d_diag_local = None

    def _upload_term_tables(self, gpu_data) -> None:


        from fulqrum.gpu.distributed.dist_matvec import _OPER_ELEMS_HOST  # noqa: PLC0415

        if self._has_offdiag:
            (
                g2_keys,
                g2_vals,
                g2_mask,
                g2_bitset,
                g4_keys,
                g4_vals,
                g4_mask,
                g4_bitset,
            ) = _build_grp_pos_tables(
                np.ascontiguousarray(gpu_data["grp_offdiag_inds"]),
                np.ascontiguousarray(gpu_data["grp_offdiag_sizes"]),
                num_alpha=self._num_alpha,
                grp_rowint_length=np.asarray(gpu_data["grp_rowint_length"]).astype(
                    np.uint8
                ),
            )
        else:
            g2_keys = g2_vals = g2_bitset = np.zeros(1, dtype=np.uint32)
            g4_keys = g4_vals = g4_bitset = np.zeros(1, dtype=np.uint64)
            g2_mask = g4_mask = 0
        self._grp2_ht_mask = int(g2_mask)
        self._grp4_ht_mask = int(g4_mask)
        self._ladder_offset = int(gpu_data["ladder_offset"])
        term_header, term_words = _pack_term_tables(gpu_data)
        proj_inds = np.ascontiguousarray(gpu_data["term_proj_inds"])
        self._term_inds_stride = int(np.shape(gpu_data["term_inds"])[1])
        self._term_proj_stride = int(proj_inds.shape[1])
        self._d_term_header = _to_device(term_header, np.uint64)
        self._d_term_inds_vals = _to_device(term_words, np.uint64)
        self._d_term_proj_inds = _to_device(proj_inds)
        self._d_term_proj_bits = _to_device(gpu_data["term_proj_bits"])
        self._d_grp_ladder_ptrs = _to_device(gpu_data["grp_ladder_ptrs"], np.uint32)
        self._d_oper_elems = _to_device(_OPER_ELEMS_HOST, np.int8)
        self._d_grp2_ht_keys = _to_device(g2_keys, np.uint32)
        self._d_grp2_ht_vals = _to_device(g2_vals, np.int32)
        self._d_grp2_ht_bitset = _to_device(g2_bitset, np.uint32)
        self._d_grp4_ht_keys = _to_device(g4_keys, np.uint64)
        self._d_grp4_ht_vals = _to_device(g4_vals, np.int32)
        self._d_grp4_ht_bitset = _to_device(g4_bitset, np.uint64)
        _wait_for_uploads()

    def _upload_aabb_coefficients(self, spmv, gpu_data, grp4_direct) -> None:


        if grp4_direct is None:
            return
        if self._host_arms:
            g_dense = aabb_coefficients(
                cross_sector_table(spmv), self._num_alpha, self._num_beta
            )
        else:
            g_dense = _build_aabb_g_dense(
                np.ascontiguousarray(gpu_data["grp_offdiag_inds"]),
                np.ascontiguousarray(gpu_data["grp_offdiag_sizes"]),
                grp4_direct,
                self._num_alpha,
                self._num_beta,
            )
        self._d_aabb_g = _to_device(g_dense, np.float64)
        self._aabb_csr_bytes += int(self._d_aabb_g.nbytes)

    def _free_term_tables(self) -> None:

        import cupy  # noqa: PLC0415

        cupy.cuda.Device().synchronize()
        self._d_term_header = None
        self._d_term_inds_vals = None
        self._d_term_proj_inds = None
        self._d_term_proj_bits = None
        self._d_grp_ladder_ptrs = None
        self._d_oper_elems = None
        self._d_grp2_ht_keys = None
        self._d_grp2_ht_vals = None
        self._d_grp2_ht_bitset = None
        self._d_grp4_ht_keys = None
        self._d_grp4_ht_vals = None
        self._d_grp4_ht_bitset = None
        cupy.get_default_memory_pool().free_all_blocks()

    def _set_bra_self_tables(self, self_tables) -> None:


        import cupy  # noqa: PLC0415

        self._alpha2_count = np.diff(
            np.ascontiguousarray(self_tables["alpha2"][0], dtype=np.int64)
        )
        self._alpha4_count = np.diff(
            np.ascontiguousarray(self_tables["alpha4"][0], dtype=np.int64)
        )
        self._beta2_count = np.diff(
            np.ascontiguousarray(self_tables["beta2"][0], dtype=np.int64)
        )
        self._beta4_count = np.diff(
            np.ascontiguousarray(self_tables["beta4"][0], dtype=np.int64)
        )
        self._d_alpha_self_slot = cupy.arange(self._n_a_local, dtype=cupy.int32)
        self._d_beta_self_slot = cupy.arange(self._n_b_local, dtype=cupy.int32)

    def _build_source_tile_tables(self, self_tables) -> None:


        started = time.perf_counter()
        alpha_peers, beta_peers, source_coords = self._table_coordinates()
        options = TableOptions(self._num_alpha, self._aabb_factored)
        self._tile_tables, host_seconds, upload_seconds = build_tile_tables(
            self_tables,
            self._alpha_bra,
            self._beta_bra,
            (self._coord_alpha, self._coord_beta),
            alpha_peers,
            beta_peers,
            source_coords,
            self._flip_masks,
            options,
        )
        self._aabb_csr_bytes += self._tile_tables.cross_sector_bytes()
        timing = self._construction_timing
        timing["peer_csr_cache_build"] = time.perf_counter() - started
        timing["peer_csr_cache_cpu_build"] = host_seconds
        timing["peer_csr_cache_gpu_upload"] = upload_seconds
        timing["peer_csr_cache_n_peers"] = float(len(source_coords) - 1)
        timing["peer_csr_cache_n_distinct_alpha_builds"] = float(len(alpha_peers))
        timing["peer_csr_cache_n_distinct_beta_builds"] = float(len(beta_peers))

    def _build_aabb_plans(self, self_tables) -> None:


        if not self._aabb_factored:
            return
        b2_ptr, b2_j, b2_p0, b2_p1 = self_tables["beta2"][:4]
        if self._aabb_parent_enabled:
            self._build_aabb_parent_plan(b2_ptr, b2_j, b2_p0, b2_p1, _to_device)
        if self._aabb_remote_beta_mode != "baseline":
            self._build_aabb_remote_beta_plans(_to_device)
        if self._aabb_remote_alpha_mode == "transposed_clique":
            self._build_aabb_remote_alpha_plans(b2_ptr, b2_j, b2_p0, b2_p1, _to_device)

    def _operator_table_bytes(self) -> int:

        shared = (
            self._d_alpha_uniq,
            self._d_beta_uniq,
            self._d_diag_local,
            self._d_diag_alpha,
            self._d_diag_beta,
            self._d_diag_Mb,
            self._d_term_header,
            self._d_term_inds_vals,
            self._d_term_proj_inds,
            self._d_term_proj_bits,
            self._d_grp_ladder_ptrs,
            self._d_oper_elems,
            self._d_grp2_ht_keys,
            self._d_grp2_ht_vals,
            self._d_grp2_ht_bitset,
            self._d_grp4_ht_keys,
            self._d_grp4_ht_vals,
            self._d_grp4_ht_bitset,
            self._d_alpha_self_slot,
            self._d_beta_self_slot,
        )
        return (
            sum(int(a.nbytes) for a in shared if a is not None)
            + self._tile_tables.connection_bytes()
        )

    def _plan_arms(self) -> dict:


        settings = ArmSettings(
            factored=self._aabb_factored,
            parent=self._aabb_parent_enabled,
            remote_beta=self._aabb_remote_beta_mode,
            remote_alpha=self._aabb_remote_alpha_mode == "transposed_clique",
        )
        if self._tile_tables is None:
            return {}
        import cupy  # noqa: PLC0415

        self._clique_shared_bytes = min(
            int(cupy.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]),
            CLIQUE_MAX_SHARED_BYTES,
        )
        parent = self._aabb_parent_plan
        plans = {}
        for src_rank in set(self._sources_2d) | {self._rank}:
            view = self._tile_tables.sources[src_rank].launch_view
            same_alpha, same_beta = self._same_coordinates(src_rank)
            remote = self._aabb_remote_beta_plans.get(src_rank)
            facts = SourceFacts(
                is_bra=src_rank == self._rank,
                same_alpha=same_alpha,
                same_beta=same_beta,
                singles_edges=int(cupy.asnumpy(view["d_alpha2_ptr"][-1:])[0]) > 0,
                singles_sell=all(
                    _dev_nonnull(view.get(k))
                    for k in (*_ALPHA2_SELL_KEYS, "d_alpha2_sell_pack")
                ),
                doubles_edges=_last_value(view.get("d_alpha4_ptr")) > 0,
                doubles_sell=view.get("d_alpha4_sell_j") is not None,
                parent_built=parent is not None,
                parent_plan=parent is not None and parent["n_beta_edges"] > 0,
                parent_grouped_edges=parent is not None
                and parent.get("grouped_edges") is not None,
                remote_beta_plan=remote is not None,
                remote_beta_grouped_edges=remote is not None
                and remote.get("grouped_edges") is not None,
                remote_alpha_plan=(
                    src_rank in self._aabb_remote_alpha_plans
                    and self._aabb_remote_alpha_shared is not None
                ),
            )
            plans[src_rank] = plan_arms(facts, settings)
            self._validate_clique_plan(src_rank, plans[src_rank])
        if any(
            plan.aabb_path in (ARM_CLIQUE, ARM_TRANSPOSED_CLIQUE)
            for plan in plans.values()
        ):
            from .kernel_launch import prepare_cross_clique  # noqa: PLC0415

            try:
                prepare_cross_clique(self._clique_shared_bytes)
            except RuntimeError as error:
                raise RuntimeError(
                    f"rank {self._rank} cross clique arm: {error}"
                ) from error
        return plans

    def _validate_clique_plan(self, src_rank, arm_plan) -> None:


        if arm_plan.aabb_path not in (ARM_CLIQUE, ARM_TRANSPOSED_CLIQUE):
            return
        n_b_src = self._rank_tiles[src_rank][3] - self._rank_tiles[src_rank][2]
        common = dict(
            n_b_src=n_b_src,
            n_b_out=self._n_b_local,
            tile_rows=self._aabb_clique_output_rows_per_tile,
            eb=self._aabb_clique_edges_per_batch,
            max_shared_bytes=self._clique_shared_bytes,
        )
        if arm_plan.aabb_path == ARM_TRANSPOSED_CLIQUE:
            shared, plan = (
                self._aabb_remote_alpha_shared,
                self._aabb_remote_alpha_plans[src_rank],
            )
            grouped_edges = dict(
                plan["grouped_edges"], bucket_start=shared["clique_bucket_start"]
            )
            validate_clique_plan(
                transposed=True,
                nparents=shared["nparents"],
                residual_edges=shared["n_vres"],
                n_beta_edges=shared["n_vres"],
                n_rows=self._n_a_local,
                orbitals=int(self._num_beta),
                grouped_edges=grouped_edges,
                **common,
            )
            return
        plan = self._aabb_parent_plan
        beta = self._aabb_remote_beta_plans[src_rank] if arm_plan.remote_beta else plan
        validate_clique_plan(
            transposed=False,
            nparents=plan["stats"]["parents"],
            residual_edges=plan["stats"]["residual_edges"],
            n_beta_edges=beta["n_beta_edges"],
            n_rows=self._n_b_local,
            orbitals=int(self._num_alpha),
            grouped_edges=beta.get("grouped_edges")
            if arm_plan.aabb_path == ARM_CLIQUE
            else None,
            **common,
        )

    def _host_arm_values_reason(self, spmv) -> Optional[str]:


        needed = {
            "a factored aabb table": self._aabb_factored,
            "FQ_AA_PAIR_TABLE": self._aa_pair_enabled,
            "FQ_BB_PAIR_TABLE": self._bb_pair_enabled,
        }
        off = [name for name, on in needed.items() if not on]
        if off:
            return "needs " + ", ".join(off)
        return tile_input_error(
            spmv, self._num_alpha, self._alpha_uniq, self._beta_uniq
        )

    def _build_host_arm_values(self) -> None:


        import cupy  # noqa: PLC0415

        spmv = self._hsub.spmv
        threads = self._build_threads
        alpha_bounds = [int(a0) for a0, _ in self._alpha_ranges] + [
            int(self._alpha_ranges[-1][1])
        ]
        beta_bounds = [int(b0) for b0, _ in self._beta_ranges] + [
            int(self._beta_ranges[-1][1])
        ]
        sub = {}
        clock = [time.perf_counter()]

        def step(name):

            now = time.perf_counter()
            sub[name] = sub.get(name, 0.0) + now - clock[0]
            clock[0] = now

        alpha_tiles = sector_tiles(spmv, 0, self._a0, self._a1, alpha_bounds, threads)
        step("alpha_sector_tiles")
        beta_tiles = sector_tiles(spmv, 1, self._b0, self._b1, beta_bounds, threads)
        step("beta_sector_tiles")
        aa_t = cupy.asarray(
            pair_table(
                cross_table(spmv, 0, self._b0, self._b1, threads),
                self._num_alpha,
                DIRECTION_TOLERANCE,
            )
        )
        step("aa_pair_table")
        bb_t = cupy.asarray(
            np.ascontiguousarray(
                pair_table(
                    cross_table(spmv, 1, self._a0, self._a1, threads),
                    self._num_beta,
                    DIRECTION_TOLERANCE,
                ).T
            )
        )
        step("bb_pair_table")
        self._arm_pair_rank = dict(aa=aa_t, bb=bb_t)
        self._arm_pair_bytes = int(aa_t.nbytes) + int(bb_t.nbytes)
        for src_rank in sorted(self._sources_2d):
            same_alpha, same_beta = self._same_coordinates(src_rank)
            view = self._tile_tables.sources[src_rank].launch_view
            payload, keep = ArmPairTablesPayload(), []
            if same_beta:
                tile = alpha_tiles[self._rank_tiles[src_rank][4]]
                self._fill_alpha_arms(src_rank, view, tile, aa_t, payload, keep)
                step("fill_alpha_arms")
            if same_alpha:
                tile = beta_tiles[self._rank_tiles[src_rank][5]]
                self._fill_beta_arms(src_rank, view, tile, bb_t, payload, keep)
                step("fill_beta_arms")
            if same_beta and (self._aaaa_min_blocks or self._aaaa_identity_slot):
                payload.aaaa_min_blocks = int(self._aaaa_min_blocks)
                payload.aaaa_identity_slot = int(bool(self._aaaa_identity_slot))
            enabled = (
                payload.aa_enable
                or payload.bb_enable
                or payload.aaaa_min_blocks
                or payload.aaaa_identity_slot
            )
            self._raise_on_pair_payload_error(src_rank, payload)
            self._arm_pair_state[src_rank] = (
                dict(payload=payload, keep=keep) if enabled else None
            )
            self._arm_pair_bytes += sum(int(array.nbytes) for array in keep)
            step("payload")
        self._caches_built = True
        for name, seconds in sub.items():
            self._construction_timing["arms_" + name] = seconds
        self._logger.debug(
            "Host arm table construction: rank=%d %s",
            self._rank,
            " ".join(f"{name}={seconds:.3f}s" for name, seconds in sub.items()),
        )

    def _fill_alpha_arms(self, src_rank, view, tile, aa_t, payload, keep) -> None:


        import cupy  # noqa: PLC0415

        ptr, columns = (
            cupy.asnumpy(view["d_alpha2_ptr"]),
            cupy.asnumpy(view["d_alpha2_j"]),
        )
        if int(ptr[-1]) > 0:
            sell = [
                view[k]
                for k in (
                    "d_alpha2_sell_off",
                    "d_alpha2_sell_len",
                    "d_alpha2_sell_pack",
                )
            ]
            if not all(_dev_nonnull(table) for table in sell):
                raise ValueError(
                    f"source {src_rank} has alpha single connections but no packed SELL tables"
                )
            signs = cupy.asnumpy(view["d_alpha2_asign"])[: int(ptr[-1])]
            self._check_host_signs(src_rank, ptr, columns, tile, signs)
            c0 = values_on_edges(
                ptr,
                columns,
                tile["single_offsets"],
                tile["columns"],
                tile["single_values"],
            )
            c0_sell = cupy.asarray(
                csr_to_sell_values(
                    c0, ptr, cupy.asnumpy(sell[0]), cupy.asnumpy(sell[1])
                )
            )
            keep.append(c0_sell)
            payload.aa_sell_off, payload.aa_sell_len, payload.aa_sell_pack = (
                int(t.data.ptr) for t in sell
            )
            payload.aa_c0_sell = int(c0_sell.data.ptr)
            payload.aa_t = int(aa_t.data.ptr)
            payload.aa_enable, payload.aa_pairs, payload.aa_t_rows = (
                1,
                int(aa_t.shape[1]),
                int(aa_t.shape[0]),
            )
        ptr4, columns4 = (
            cupy.asnumpy(view["d_alpha4_ptr"]),
            cupy.asnumpy(view["d_alpha4_j"]),
        )
        values = values_on_edges(
            ptr4,
            columns4,
            tile["double_offsets"],
            tile["double_columns"],
            tile["double_values"],
        )
        if view["d_alpha4_sell_j"] is not None:
            values = csr_to_sell_values(
                values,
                ptr4,
                cupy.asnumpy(view["d_alpha4_sell_off"]),
                cupy.asnumpy(view["d_alpha4_sell_len"]),
            )
        if values.size:
            self._arm_caches[("aaaa", src_rank)] = cupy.asarray(values)

    def _fill_beta_arms(self, src_rank, view, tile, bb_t, payload, keep) -> None:


        import cupy  # noqa: PLC0415

        ptr, columns = (
            cupy.asnumpy(view["d_beta2_ptr"]),
            cupy.asnumpy(view["d_beta2_j"]),
        )
        if int(ptr[-1]) > 0:
            words, _pairs, signs = bb_words(
                ptr,
                cupy.asnumpy(view["d_beta2_p0"]),
                cupy.asnumpy(view["d_beta2_p1"]),
                self._beta_bra,
                self._num_beta,
            )
            self._check_host_signs(src_rank, ptr, columns, tile, signs)
            c0 = values_on_edges(
                ptr,
                columns,
                tile["single_offsets"],
                tile["columns"],
                tile["single_values"],
            )
            self._arm_caches[("bb_c0", src_rank)] = cupy.asarray(c0)
            d_words = cupy.asarray(words)
            keep.append(d_words)
            payload.bb_word = int(d_words.data.ptr)
            payload.bb_t = int(bb_t.data.ptr)
            payload.bb_enable, payload.bb_pairs, payload.bb_t_cols = (
                1,
                int(bb_t.shape[0]),
                int(bb_t.shape[1]),
            )
        ptr4, columns4 = (
            cupy.asnumpy(view["d_beta4_ptr"]),
            cupy.asnumpy(view["d_beta4_j"]),
        )
        values = values_on_edges(
            ptr4,
            columns4,
            tile["double_offsets"],
            tile["double_columns"],
            tile["double_values"],
        )
        if values.size:
            self._arm_caches[("bbbb", src_rank)] = cupy.asarray(values)

    @staticmethod
    def _check_host_signs(src_rank, ptr, columns, tile, gpu_signs) -> None:


        own, jw = tile["own_signs"], tile["jw_signs"]
        if not np.all((own == 0) | (own == jw)):
            raise ValueError(
                f"source {src_rank}: an own-sector sign differs from the JW sign"
            )
        host = values_on_edges(
            ptr, columns, tile["single_offsets"], tile["columns"], jw.astype(np.float64)
        )
        if not np.array_equal(host, np.asarray(gpu_signs, dtype=np.float64)):
            raise ValueError(
                f"source {src_rank}: the host JW signs differ from the GPU signs"
            )

    def _table_coordinates(self) -> tuple[dict, dict, dict]:


        alpha_peers, beta_peers = {}, {}
        source_coords = {self._rank: (self._coord_alpha, self._coord_beta)}
        for src_rank in self._sources_2d:
            ca, cb = (int(c) for c in self._rank_tiles[src_rank][4:6])
            source_coords[src_rank] = (ca, cb)
            if ca != self._coord_alpha:
                a0, a1 = self._alpha_ranges[ca]
                alpha_peers[ca] = self._alpha_uniq[a0:a1]
            if cb != self._coord_beta:
                b0, b1 = self._beta_ranges[cb]
                beta_peers[cb] = self._beta_uniq[b0:b1]
        return alpha_peers, beta_peers, source_coords

    def _build_aabb_parent_plan(self, bptr, bj, bp0, bp1, to_device) -> None:

        import cupy

        started = time.perf_counter()
        if not 2 <= self._num_alpha <= 64:
            raise ValueError(
                "the clique aabb arm supports alpha orbital counts in [2,64]"
            )
        graph = sorted_beta_graph(
            bptr, bj, bp0, bp1, self._beta_bra, self._num_beta, self._n_b_local
        )
        stats = dict(parents=0, selected_edges=0, original_edges=0, residual_edges=0)
        g_bytes = 0
        if self._n_a_local and (
            len(graph["words"]) or self._aabb_remote_beta_mode == "clique"
        ):
            view = self._tile_tables.sources[self._rank].launch_view
            tables = (
                view["d_alpha2_sell_off"],
                view["d_alpha2_sell_len"],
                view["d_alpha2_sell_pack"],
            )
            if any(t is None for t in tables):
                raise ValueError(
                    "the clique aabb arm requires packed self-alpha SELL tables"
                )
            host_tables = [cupy.asnumpy(t) for t in tables]
            kind = getattr(self, "_aabb_parent_kind", "union")
            kinds = ("union", "intersection") if kind == "auto" else (kind,)
            plans = [
                parent_plan(
                    self._alpha_bra,
                    *host_tables,
                    self._num_alpha,
                    self._aabb_parent_min,
                    kind=k,
                )
                for k in kinds
            ]
            plan = max(
                plans,
                key=lambda p: (
                    p["stats"]["selected_edges"],
                    p["stats"].get("edge_weighted_size", 0.0),
                    p["stats"]["kind"] == "union",
                ),
            )
            stats = plan["stats"]
            if stats["parents"]:
                masks, ids = parent_masks(plan)
                host = dict(
                    parent_ptr=plan["parent_ptr"],
                    members=plan["members"],
                    masks=masks,
                    ids=ids,
                )
                if stats["residual_edges"]:
                    if self._aabb_residual_degree_order:
                        (
                            host["residual_off"],
                            host["residual_lengths"],
                            host["residual_pack"],
                            host["residual_order"],
                        ) = degree_sorted_sell(
                            plan["residual_off"],
                            plan["residual_lengths"],
                            plan["residual_pack"],
                        )
                    else:
                        for key in (
                            "residual_off",
                            "residual_lengths",
                            "residual_pack",
                        ):
                            host[key] = plan[key]
                if len(graph["words"]):
                    _, tasks = full_source_plan(graph, self._n_a_local)
                    host.update(beta_words=graph["words"], tasks=tasks)
                compact_g = cupy.asnumpy(self._d_aabb_g).reshape(
                    self._num_beta * (self._num_beta - 1) // 2,
                    self._num_alpha * (self._num_alpha - 1) // 2,
                )
                host["orbital_g"] = orbital_g(compact_g, self._num_alpha)
                g_bytes = int(host["orbital_g"].nbytes)
                device = {
                    key: to_device(value, dtype=value.dtype)
                    for key, value in host.items()
                }
                self._aabb_parent_bytes = sum(int(a.nbytes) for a in device.values())
                self._aabb_parent_plan = dict(
                    device=device, stats=stats, n_beta_edges=len(graph["words"])
                )
                bucket_ids, bucket_start = clique_size_buckets(plan["parent_ptr"])
                d_bucket_ids = to_device(bucket_ids, dtype=np.int32)
                self._aabb_parent_bytes += int(d_bucket_ids.nbytes)
                self._aabb_parent_plan["clique_size_buckets"] = dict(
                    ids=d_bucket_ids, start=[int(x) for x in bucket_start]
                )
                if len(graph["words"]):
                    group_started = time.perf_counter()
                    vp = clique_edge_group_plan(
                        graph["words"],
                        graph["rows"],
                        self._n_b_local,
                        self._aabb_clique_output_rows_per_tile,
                    )
                    group_device = dict(
                        clique_words=to_device(vp["words"], dtype=np.uint64),
                        clique_group_ptr=to_device(vp["group_ptr"], dtype=np.int32),
                        clique_group_bpid=to_device(vp["group_bpid"], dtype=np.int32),
                        clique_tile_group_ptr=to_device(
                            vp["tile_group_ptr"], dtype=np.int32
                        ),
                    )
                    group_bytes = sum(int(a.nbytes) for a in group_device.values())
                    group_device["clique_bucket_ids"] = d_bucket_ids
                    self._aabb_parent_bytes += group_bytes
                    self._aabb_parent_plan["grouped_edges"] = dict(
                        device=group_device,
                        bucket_start=[int(x) for x in bucket_start],
                        ntiles=vp["ntiles"],
                        ngroups=vp["ngroups"],
                        nwords=int(len(vp["words"])),
                        distinct_rows=vp["distinct_rows"],
                    )
                    self._logger.debug(
                        f"Clique edge groups ready: rank={self._rank} tile_rows={self._aabb_clique_output_rows_per_tile} "
                        f"eb={self._aabb_clique_edges_per_batch} tiles={vp['ntiles']} groups={vp['ngroups']} "
                        f"mean_group={len(vp['words']) / max(vp['ngroups'], 1):.2f} "
                        f"buckets={[int(b - a) for a, b in zip(bucket_start[:-1], bucket_start[1:])]} "
                        f"metadata={group_bytes / 2**20:.2f} MiB setup={time.perf_counter() - group_started:.3f}s"
                    )
        cupy.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        self._construction_timing["aabb_parent_setup"] = elapsed
        self._construction_timing["aabb_parent_extra_bytes"] = self._aabb_parent_bytes
        self._construction_timing["aabb_parent_selected_edges"] = stats[
            "selected_edges"
        ]
        self._logger.debug(
            f"Local clique plan ready: rank={self._rank} orbitals={self._num_alpha} "
            f"kind={stats.get('kind', 'union')} "
            f"threshold={self._aabb_parent_min} parents={stats['parents']} "
            f"selected_edges={stats['selected_edges']}/{stats['original_edges']} "
            f"residual_edges={stats['residual_edges']} "
            f"extra={self._aabb_parent_bytes / 2**20:.2f} MiB "
            f"orbital_g={g_bytes / 2**20:.2f} MiB scratch=0 B setup={elapsed:.3f}s"
        )

    def _arm_pair_payload(
        self,
        src_rank,
        peer_csr,
        same_alpha,
        same_beta,
        aabb_csr_mode,
        d_alpha_slot,
        d_aa_c0,
        d_aa_cpb,
        d_bb_c0,
        d_bb_cpb,
    ):


        if self._host_arms:
            state = self._arm_pair_state.get(src_rank)
            return None if state is None else state["payload"]
        _aaaa_opts = bool(self._aaaa_min_blocks or self._aaaa_identity_slot)
        if not (self._aa_pair_enabled or self._bb_pair_enabled or _aaaa_opts):
            return None
        if not (same_alpha or same_beta):
            return None
        if src_rank in self._arm_pair_state:
            state = self._arm_pair_state[src_rank]
            return None if state is None else state["payload"]
        import cupy

        started = time.perf_counter()
        self._compute_stream.synchronize()
        keep, notes = [], []
        pay = ArmPairTablesPayload()
        identity_alpha = (
            d_alpha_slot is self._d_alpha_slot_persistent
            or d_alpha_slot is self._d_alpha_self_slot
        )

        def rank_table(kind, d, present, build):


            entry = self._arm_pair_rank.get(kind)
            if entry is None:
                d_t = cupy.asarray(build(d))
                self._arm_pair_rank[kind] = entry = dict(d=d, present=present, d_t=d_t)
                self._arm_pair_bytes += int(d_t.nbytes)
                return entry
            merged = merge_pair_tables(entry["d"], entry["present"], d, present)
            if merged is None:
                return None
            if np.any(merged[1] & ~entry["present"]):
                entry["d_t"][...] = cupy.asarray(build(merged[0]))
                entry["d"], entry["present"] = merged
            return entry

        if (
            self._aa_pair_enabled
            and same_beta
            and _dev_nonnull(d_aa_c0)
            and _dev_nonnull(d_aa_cpb)
            and aabb_csr_mode == 12
            and identity_alpha
            and all(
                _dev_nonnull(peer_csr.get(k))
                for k in (
                    "d_alpha2_ptr",
                    "d_alpha2_j",
                    "d_alpha2_apid",
                    "d_alpha2_asign",
                    "d_alpha2_sell_off",
                    "d_alpha2_sell_len",
                    "d_alpha2_sell_pack",
                )
            )
        ):
            n_pairs = self._num_alpha * (self._num_alpha - 1) // 2
            res = derive_pair_table(
                cupy.asnumpy(d_aa_cpb),
                cupy.asnumpy(peer_csr["d_alpha2_asign"]),
                cupy.asnumpy(peer_csr["d_alpha2_apid"]),
                n_pairs,
                self._num_beta,
            )
            entry = (
                None
                if res is None
                else rank_table(
                    "aa",
                    res[0],
                    res[1],
                    lambda d: aa_table(d, self._beta_bra, self._num_beta),
                )
            )
            if entry is not None:
                c0_sell = cupy.asarray(
                    csr_to_sell_values(
                        cupy.asnumpy(d_aa_c0),
                        cupy.asnumpy(peer_csr["d_alpha2_ptr"]),
                        cupy.asnumpy(peer_csr["d_alpha2_sell_off"]),
                        cupy.asnumpy(peer_csr["d_alpha2_sell_len"]),
                    )
                )
                keep.append(c0_sell)
                self._arm_pair_bytes += int(c0_sell.nbytes)
                pay.aa_sell_off = int(peer_csr["d_alpha2_sell_off"].data.ptr)
                pay.aa_sell_len = int(peer_csr["d_alpha2_sell_len"].data.ptr)
                pay.aa_sell_pack = int(peer_csr["d_alpha2_sell_pack"].data.ptr)
                pay.aa_c0_sell = int(c0_sell.data.ptr)
                pay.aa_t = int(entry["d_t"].data.ptr)
                pay.aa_enable, pay.aa_pairs, pay.aa_t_rows = (
                    1,
                    n_pairs,
                    int(entry["d_t"].shape[0]),
                )
                notes.append("aa=on")
            else:
                notes.append("aa=FALLBACK(verify)")
        if (
            self._bb_pair_enabled
            and same_alpha
            and _dev_nonnull(d_bb_c0)
            and _dev_nonnull(d_bb_cpb)
            and all(
                _dev_nonnull(peer_csr.get(k))
                for k in ("d_beta2_ptr", "d_beta2_j", "d_beta2_p0", "d_beta2_p1")
            )
        ):
            n_pairs = self._num_beta * (self._num_beta - 1) // 2
            words, bpid, bsign = bb_words(
                cupy.asnumpy(peer_csr["d_beta2_ptr"]),
                cupy.asnumpy(peer_csr["d_beta2_p0"]),
                cupy.asnumpy(peer_csr["d_beta2_p1"]),
                self._beta_bra,
                self._num_beta,
            )
            res = (
                derive_pair_table(
                    cupy.asnumpy(d_bb_cpb), bsign, bpid, n_pairs, self._num_alpha
                )
                if len(words)
                else None
            )
            entry = (
                None
                if res is None
                else rank_table(
                    "bb",
                    res[0],
                    res[1],
                    lambda d: bb_table(d, self._alpha_bra, self._num_alpha),
                )
            )
            if entry is not None:
                d_words = cupy.asarray(words)
                keep.append(d_words)
                self._arm_pair_bytes += int(d_words.nbytes)
                pay.bb_word = int(d_words.data.ptr)
                pay.bb_t = int(entry["d_t"].data.ptr)
                pay.bb_enable, pay.bb_pairs, pay.bb_t_cols = (
                    1,
                    n_pairs,
                    int(entry["d_t"].shape[1]),
                )
                notes.append("bb=on")
            else:
                notes.append("bb=FALLBACK(verify)" if len(words) else "bb=no-edges")
        if _aaaa_opts and same_beta:
            pay.aaaa_min_blocks = int(self._aaaa_min_blocks)
            pay.aaaa_identity_slot = int(
                bool(self._aaaa_identity_slot) and identity_alpha
            )
            notes.append(f"aaaa=minb{pay.aaaa_min_blocks}/id{pay.aaaa_identity_slot}")
        cupy.cuda.get_current_stream().synchronize()
        self._raise_on_pair_payload_error(src_rank, pay)
        state = (
            dict(payload=pay, keep=keep)
            if (
                pay.aa_enable
                or pay.bb_enable
                or pay.aaaa_min_blocks
                or pay.aaaa_identity_slot
            )
            else None
        )
        self._arm_pair_state[src_rank] = state
        self._logger.debug(
            f"Pair arm setup: rank={self._rank} src={src_rank} "
            f"{' '.join(notes) or 'not-applicable'} rank_tables="
            f"{self._arm_pair_bytes / 2**20:.1f} MiB setup={time.perf_counter() - started:.3f}s"
        )
        return None if state is None else state["payload"]

    def _raise_on_pair_payload_error(self, src_rank, payload) -> None:


        error = pair_payload_error(payload, self._n_b_local, self._n_a_local)
        if error is not None:
            raise RuntimeError(f"rank {self._rank} source {src_rank}: {error}")

    def _build_aabb_remote_alpha_plans(self, bptr, bj, bp0, bp1, to_device) -> None:


        import cupy

        started = time.perf_counter()
        if not 2 <= self._num_beta <= 64:
            raise ValueError(
                "transposed_clique supports runtime beta orbital counts in [2,64]"
            )
        sources = [
            s
            for s in sorted(self._sources_2d)
            if s != self._rank
            and self._rank_tiles[s][5] == self._coord_beta
            and self._rank_tiles[s][4] != self._coord_alpha
        ]
        graph = sorted_beta_graph(
            bptr, bj, bp0, bp1, self._beta_bra, self._num_beta, self._n_b_local
        )
        stats = dict(kind="-", selected_edges=0, original_edges=len(graph["words"]))
        n_cliques = 0
        if sources and len(graph["words"]) and self._n_a_local:
            bplan = beta_clique_plan(
                self._beta_bra,
                graph,
                self._num_beta,
                self._aabb_parent_min,
                kind=getattr(self, "_aabb_parent_kind", "union"),
            )
            stats = bplan["stats"]
            n_cliques = int(len(bplan["vptr"]) - 1)
            if n_cliques > 0:
                masks, ids = parent_masks(
                    dict(parent_ptr=bplan["vptr"], members=bplan["vmembers"])
                )
                compact_g = cupy.asnumpy(self._d_aabb_g).reshape(
                    self._num_beta * (self._num_beta - 1) // 2,
                    self._num_alpha * (self._num_alpha - 1) // 2,
                )
                host = dict(
                    parent_ptr=bplan["vptr"],
                    members=bplan["vmembers"],
                    masks=masks,
                    ids=ids,
                    orbital_g=orbital_g(
                        np.ascontiguousarray(compact_g.T), self._num_beta
                    ),
                    beta_words=np.ascontiguousarray(
                        bplan["vres_words"], dtype=np.uint64
                    ),
                    tasks=np.ascontiguousarray(bplan["vres_tasks"], dtype=np.int32),
                )
                if not len(host["beta_words"]):
                    del host["beta_words"], host["tasks"]
                device = {
                    key: to_device(value, dtype=value.dtype)
                    for key, value in host.items()
                }
                self._aabb_remote_alpha_bytes += sum(
                    int(a.nbytes) for a in device.values()
                )
                self._aabb_remote_alpha_shared = dict(
                    device=device,
                    nparents=n_cliques,
                    n_vres=int(len(bplan["vres_words"])),
                )
                bucket_ids, bucket_start = clique_size_buckets(bplan["vptr"])
                d_bids = to_device(bucket_ids, dtype=np.int32)
                self._aabb_remote_alpha_bytes += int(d_bids.nbytes)
                self._aabb_remote_alpha_shared["clique_bucket_ids"] = d_bids
                self._aabb_remote_alpha_shared["clique_bucket_start"] = [
                    int(x) for x in bucket_start
                ]
                for src in sources:
                    view = self._tile_tables.sources[src].launch_view
                    tables = [
                        view.get(k)
                        for k in (
                            "d_alpha2_sell_off",
                            "d_alpha2_sell_len",
                            "d_alpha2_sell_pack",
                        )
                    ]
                    if any(t is None for t in tables):
                        continue
                    host_tables = [cupy.asnumpy(t) for t in tables]
                    words, tasks = alpha_task_words(*host_tables, self._n_a_local)
                    if not len(words):
                        continue
                    if self._aabb_residual_degree_order:

                        off, lengths, pack, order = degree_sorted_sell(*host_tables)
                        src_device = dict(
                            residual_off=to_device(off, dtype=np.int64),
                            residual_lengths=to_device(lengths, dtype=np.int32),
                            residual_pack=to_device(pack, dtype=np.uint64),
                            residual_order=to_device(order, dtype=np.int32),
                        )
                        self._aabb_remote_alpha_bytes += sum(
                            int(a.nbytes) for a in src_device.values()
                        )
                    else:
                        src_device = dict(
                            residual_off=tables[0],
                            residual_lengths=tables[1],
                            residual_pack=tables[2],
                        )
                    rows = np.repeat(
                        np.arange(self._n_a_local, dtype=np.int64),
                        (tasks[:, 2] - tasks[:, 1]).astype(np.int64),
                    )
                    vp = clique_edge_group_plan(
                        words,
                        rows,
                        self._n_a_local,
                        self._aabb_clique_output_rows_per_tile,
                    )
                    dv = dict(
                        clique_words=to_device(vp["words"], dtype=np.uint64),
                        clique_group_ptr=to_device(vp["group_ptr"], dtype=np.int32),
                        clique_group_bpid=to_device(vp["group_bpid"], dtype=np.int32),
                        clique_tile_group_ptr=to_device(
                            vp["tile_group_ptr"], dtype=np.int32
                        ),
                        clique_bucket_ids=self._aabb_remote_alpha_shared[
                            "clique_bucket_ids"
                        ],
                    )
                    self._aabb_remote_alpha_bytes += sum(
                        int(a.nbytes) for k, a in dv.items() if k != "clique_bucket_ids"
                    )
                    source_grouped_edges = dict(
                        device=dv,
                        ntiles=vp["ntiles"],
                        ngroups=vp["ngroups"],
                        nwords=int(len(vp["words"])),
                        distinct_rows=vp["distinct_rows"],
                    )
                    self._aabb_remote_alpha_plans[src] = dict(
                        device=src_device, grouped_edges=source_grouped_edges
                    )
        cupy.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        self._construction_timing["aabb_remote_alpha_setup"] = elapsed
        self._construction_timing["aabb_remote_alpha_bytes"] = (
            self._aabb_remote_alpha_bytes
        )
        self._logger.debug(
            f"Remote-alpha clique plans ready: rank={self._rank} mode={self._aabb_remote_alpha_mode} "
            f"candidates={len(sources)} "
            f"sources={len(self._aabb_remote_alpha_plans)} beta_kind={stats.get('kind')} "
            f"beta_min={self._aabb_parent_min} beta_cliques={n_cliques} "
            f"selected_beta_edges={stats.get('selected_edges', 0)}/{stats.get('original_edges', 0)} "
            f"metadata={self._aabb_remote_alpha_bytes / 2**20:.2f} MiB setup={elapsed:.3f}s"
        )

    def _build_aabb_remote_beta_plans(self, to_device) -> None:


        import cupy

        started = time.perf_counter()
        by_coord = {}
        for src in sorted(self._sources_2d):
            a0, a1, b0, b1, ca, cb = self._rank_tiles[src]
            if src == self._rank or ca != self._coord_alpha or cb == self._coord_beta:
                continue
            if (a0, a1) != (self._a0, self._a1):
                raise ValueError(
                    "same-alpha remote source has inconsistent alpha bounds"
                )
            if cb not in by_coord:
                view = self._tile_tables.sources[src].launch_view
                graph = sorted_beta_graph(
                    *(
                        cupy.asnumpy(view[k])
                        for k in (
                            "d_beta2_ptr",
                            "d_beta2_j",
                            "d_beta2_p0",
                            "d_beta2_p1",
                        )
                    ),
                    self._beta_bra,
                    self._num_beta,
                    b1 - b0,
                )
                if not len(graph["words"]) or not self._n_a_local:
                    by_coord[cb] = None
                    continue
                _, tasks = full_source_plan(graph, self._n_a_local)
                device = dict(
                    beta_words=to_device(graph["words"], dtype=np.uint64),
                    tasks=to_device(tasks, dtype=np.int32),
                )
                self._aabb_remote_beta_bytes += sum(
                    int(a.nbytes) for a in device.values()
                )
                entry = dict(
                    device=device, n_beta_edges=len(graph["words"]), n_b_src=b1 - b0
                )
                if (
                    self._aabb_parent_plan is not None
                    and self._aabb_parent_plan.get("clique_size_buckets") is not None
                ):
                    vp = clique_edge_group_plan(
                        graph["words"],
                        graph["rows"],
                        self._n_b_local,
                        self._aabb_clique_output_rows_per_tile,
                    )
                    group_device = dict(
                        clique_words=to_device(vp["words"], dtype=np.uint64),
                        clique_group_ptr=to_device(vp["group_ptr"], dtype=np.int32),
                        clique_group_bpid=to_device(vp["group_bpid"], dtype=np.int32),
                        clique_tile_group_ptr=to_device(
                            vp["tile_group_ptr"], dtype=np.int32
                        ),
                        clique_bucket_ids=self._aabb_parent_plan["clique_size_buckets"][
                            "ids"
                        ],
                    )
                    self._aabb_remote_beta_bytes += sum(
                        int(a.nbytes)
                        for k, a in group_device.items()
                        if k != "clique_bucket_ids"
                    )
                    entry["grouped_edges"] = dict(
                        device=group_device,
                        bucket_start=self._aabb_parent_plan["clique_size_buckets"][
                            "start"
                        ],
                        ntiles=vp["ntiles"],
                        ngroups=vp["ngroups"],
                        nwords=int(len(vp["words"])),
                        distinct_rows=vp["distinct_rows"],
                    )
                by_coord[cb] = entry
            if by_coord[cb] is not None:
                self._aabb_remote_beta_plans[src] = by_coord[cb]
        cupy.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        self._construction_timing["aabb_remote_beta_setup"] = elapsed
        self._construction_timing["aabb_remote_beta_bytes"] = (
            self._aabb_remote_beta_bytes
        )
        self._logger.debug(
            f"Remote-beta clique plans ready: rank={self._rank} mode={self._aabb_remote_beta_mode} "
            f"sources={len(self._aabb_remote_beta_plans)} metadata={self._aabb_remote_beta_bytes} B "
            f"shared_alpha_plan={int(self._aabb_parent_plan is not None)} setup={elapsed:.3f}s"
        )

    def _arm_cache_buffers(self, src_rank: int, sizes: dict) -> dict:


        import cupy  # noqa: PLC0415

        buffers = {}
        for name in ARM_CACHES:
            if not sizes.get(name):
                buffers[name] = None
                continue
            key = (name, src_rank)
            if key not in self._arm_caches:
                with self._compute_stream:
                    self._arm_caches[key] = cupy.zeros(
                        int(sizes[name]), dtype=cupy.float64
                    )
            buffers[name] = self._arm_caches[key]
        return buffers

    def rdm_scratch(self) -> list:


        buffers = [self._d_v_local, self._d_out_local, self._d_hierarchical_stage]
        buffers += list(self._d_hierarchical_gather or [])
        buffers += list(self._d_exchange_pool or [])
        return [b.reshape(-1) for b in buffers if b is not None and b.size > 1]

    def reset_cache(self) -> None:
        """Clear this rank's cached GPU connection data for rebuilding on the next matvec.

        Host-resident caches are retained. For a distributed operator, invoke
        this on all ranks before their next collective product.
        """
        if self._host_arms:
            return
        self._caches_built = False
        self._arm_caches.clear()
        self._bbbb_class_state.clear()
        self._bbbb_class_bytes = 0
        self._aaaa_class_state.clear()

    def _launch_diagonal(
        self,
        d_v_local=None,
        d_out_local=None,
        *,
        stream=None,
    ) -> None:


        if self._d_alpha_uniq is None:
            raise NotImplementedError(
                "diagonal kernel called before operator tables were "
                "uploaded (_build_and_upload_operator_tables); "
                "this happens when hsub.spmv is not a FulqrumSpMV (a test "
                "mock) or cupy is unavailable on this host."
            )
        if d_v_local is None:
            d_v_local = self._d_v_local
        if d_out_local is None:
            d_out_local = self._d_out_local
        if d_v_local is None or d_out_local is None:
            raise RuntimeError(
                "_launch_diagonal: _d_v_local / _d_out_local are None; "
                "construction did not allocate the persistent buffers "
                "(cupy missing?)."
            )
        stream_handle = 0 if stream is None else int(getattr(stream, "ptr", stream))
        self._launch_bra_diagonal(
            d_v_local,
            d_out_local,
            self._d_alpha_self_slot,
            self._d_beta_self_slot,
            self._n_a_local,
            (0, self._n_a_local),
            (0, self._n_b_local),
            stream_handle,
        )

    def _launch_bra_diagonal(
        self,
        d_src_in,
        d_out,
        d_alpha_slot,
        d_beta_slot,
        n_alpha_compact,
        a_range,
        b_range,
        stream_handle,
    ) -> None:


        from .kernel_launch import launch_diagonal  # noqa: PLC0415

        geometry = (
            self._a0,
            self._n_a_local,
            self._b0,
            self._n_b_local,
            self._a0,
            self._n_a_local,
            self._b0,
            self._n_b_local,
            a_range[0],
            a_range[1],
            b_range[0],
            b_range[1],
            int(n_alpha_compact),
            int(self._threads_per_block),
        )
        tables = (
            self._d_diag_local,
            self._d_diag_alpha,
            self._d_diag_beta,
            self._d_diag_Mb,
        )
        try:
            launch_diagonal(
                geometry,
                *(
                    _device_address(a)
                    for a in (d_src_in, d_out, d_alpha_slot, d_beta_slot)
                ),
                *(_device_address(a) for a in tables),
                _device_address(self._d_alpha_uniq),
                int(self._num_alpha),
                int(stream_handle),
            )
        except RuntimeError as error:
            raise RuntimeError(f"diagonal arm: {error}") from error

    def _clique_cross(self, src_rank, path, remote):


        if path == ARM_CLIQUE:
            plan = self._aabb_parent_plan
            remote_plan = self._aabb_remote_beta_plans[src_rank] if remote else None
            device = dict(plan["device"])
            if remote_plan is not None:
                device.update(remote_plan["device"])
            grouped_edges = (remote_plan or plan)["grouped_edges"]
            residual_edges, bucket_start = (
                plan["stats"]["residual_edges"],
                grouped_edges["bucket_start"],
            )
            orbitals, transposed = int(self._num_alpha), False
            parent_group = int(self._arm_order["parent_group"])
        elif path == ARM_TRANSPOSED_CLIQUE:
            shared, plan = (
                self._aabb_remote_alpha_shared,
                self._aabb_remote_alpha_plans[src_rank],
            )
            grouped_edges = plan["grouped_edges"]
            device = dict(shared["device"])
            device.update(plan["device"])
            residual_edges, bucket_start = (
                shared["n_vres"],
                shared["clique_bucket_start"],
            )
            orbitals, transposed = int(self._num_beta), True
            parent_group = 0
        else:
            return None
        device.update(grouped_edges["device"])
        lbg = (
            self._lbg_plan(
                src_rank,
                path,
                remote,
                device,
                bucket_start,
                grouped_edges,
                residual_edges,
            )
            if self._aabb_lbg
            else None
        )
        panel = (
            self._clique_panel(src_rank, path, remote, device, bucket_start)
            if self._aabb_clique_panel and lbg is None
            else None
        )
        return CliqueCross(
            residual_off=device.get("residual_off"),
            residual_lengths=device.get("residual_lengths"),
            residual_pack=device.get("residual_pack"),
            residual_order=device.get("residual_order"),
            beta_words=device.get("beta_words"),
            tasks=device.get("tasks"),
            residual_edges=int(residual_edges),
            edge_words=device["clique_words"],
            group_offsets=device["clique_group_ptr"],
            group_pairs=device["clique_group_bpid"],
            tile_group_offsets=device["clique_tile_group_ptr"],
            parent_offsets=device["parent_ptr"],
            members=device["members"],
            bucket_ids=device["clique_bucket_ids"],
            coefficients=device["orbital_g"],
            bucket_start=tuple(int(x) for x in bucket_start),
            num_tiles=int(grouped_edges["ntiles"]),
            tile_rows=int(self._aabb_clique_output_rows_per_tile),
            edges_per_batch=int(self._aabb_clique_edges_per_batch),
            orbitals=orbitals,
            transposed=transposed,
            parent_group=parent_group,
            mma=self._aabb_clique_mma,
            distinct_rows=bool(grouped_edges.get("distinct_rows", False)),
            **(
                dict(
                    panel=panel[0],
                    nonfinite=panel[1],
                    panel_columns=panel[2],
                    panel_batch=panel[3],
                )
                if panel is not None
                else {}
            ),
            **(
                dict(lbg=lbg[0], lbg_scratch=lbg[1], lbg_flag=lbg[2])
                if lbg is not None
                else {}
            ),
        )

    def _xterm_buffers(self, doubles):


        import cupy  # noqa: PLC0415

        if self._xterm_flag is None:
            self._xterm_flag = cupy.zeros(1, dtype=cupy.int32)
        budget = self._xterm_scratch_bytes // 8
        if budget and doubles <= budget:
            if self._xterm_scratch is None:
                self._xterm_scratch = cupy.empty(budget, dtype=cupy.float64)
            return self._xterm_scratch, self._xterm_flag
        return cupy.empty(max(int(doubles), 1), dtype=cupy.float64), self._xterm_flag

    def _lbg_plan(
        self,
        src_rank,
        path,
        remote,
        device,
        bucket_start,
        grouped_edges,
        residual_edges,
    ):


        if path != ARM_CLIQUE:
            return None
        state = self._aabb_lbg_state.get(src_rank)
        if state is None:
            from . import aabb_lbg  # noqa: PLC0415
            from .kernel_launch import clique_panel_row  # noqa: PLC0415

            if not bool(grouped_edges.get("distinct_rows", False)):
                state = ()
                self._logger.debug(
                    f"Label batching unavailable: rank={self._rank} src={src_rank} reason=duplicate output rows; using the standard clique kernels"
                )
            else:
                probe = SimpleNamespace(
                    edge_words=device["clique_words"],
                    group_offsets=device["clique_group_ptr"],
                    group_pairs=device["clique_group_bpid"],
                    tile_group_offsets=device["clique_tile_group_ptr"],
                    tile_rows=int(self._aabb_clique_output_rows_per_tile),
                    bucket_start=tuple(int(x) for x in bucket_start),
                    transposed=False,
                    residual_edges=int(residual_edges),
                    residual_off=device.get("residual_off"),
                    residual_lengths=device.get("residual_lengths"),
                    residual_pack=device.get("residual_pack"),
                    residual_order=device.get("residual_order"),
                    coefficients=device["orbital_g"],
                    orbitals=int(self._num_alpha),
                    tasks=device.get("tasks"),
                )
                n_src_rows = int(
                    self._rank_tiles[src_rank][3] - self._rank_tiles[src_rank][2]
                )
                t0 = time.perf_counter()
                try:
                    plan = aabb_lbg.build_plan(
                        probe,
                        clique_panel_row,
                        int(self._n_b_local),
                        n_src_rows,
                        int(self._n_a_local),
                        self._xterm_scratch_bytes,
                        self._alpha_bra,
                        int(self._num_alpha),
                    )
                    state = (plan, *self._xterm_buffers(plan.scratch_doubles))
                    self._logger.debug(
                        f"Label-batched clique plan ready: rank={self._rank} src={src_rank} remote={bool(remote)} "
                        f"scratch={plan.scratch_doubles * 8 / 2**30:.2f} GiB build={time.perf_counter() - t0:.2f}s "
                        f"{plan.stats}"
                    )
                except ValueError as error:
                    state = ()
                    self._logger.debug(
                        f"Label batching unavailable: rank={self._rank} src={src_rank} using the standard clique kernels: {error}"
                    )
            self._aabb_lbg_state[src_rank] = state
        return state or None

    def _clique_panel(self, src_rank, path, remote, device, bucket_start):


        if (
            path != ARM_CLIQUE
            or remote
            or src_rank != self._rank
            or not self._aabb_clique_mma
        ):
            return None
        state = self._aabb_clique_panel_state.get(src_rank)
        if state is not None:
            return state or None
        import cupy  # noqa: PLC0415

        from .kernel_launch import clique_panel_row  # noqa: PLC0415

        if not bool(
            self._aabb_parent_plan["grouped_edges"].get("distinct_rows", False)
        ):
            self._aabb_clique_panel_state[src_rank] = ()
            self._logger.debug(
                f"Source panel unavailable: rank={self._rank} src={src_rank} reason=duplicate output rows; "
                "using the standard FP64 clique kernel"
            )
            return None
        words = device["clique_words"]
        columns = (
            int(cupy.asnumpy((words & cupy.uint64(0xFFFFFFFF)).max())) + 1
            if words.size
            else 0
        )
        counts = [
            int(bucket_start[b + 1]) - int(bucket_start[b])
            for b in range(len(bucket_start) - 1)
        ]
        per_parent = [columns * max(clique_panel_row(b), 0) for b in range(len(counts))]
        doubles = max(c * p for c, p in zip(counts, per_parent))
        if doubles <= 0:
            self._aabb_clique_panel_state[src_rank] = ()
            return None
        batch = ()
        if self._xterm_scratch_bytes and doubles * 8 > self._xterm_scratch_bytes:
            cap = self._xterm_scratch_bytes // 8
            batch = tuple(
                max(1, min(c, cap // p)) if p else max(c, 1)
                for c, p in zip(counts, per_parent)
            )
            doubles = max(b * p for b, p in zip(batch, per_parent))
        if self._xterm_scratch_bytes:

            scratch, _ = self._xterm_buffers(doubles)
            state = (scratch, cupy.zeros(1, dtype=cupy.int32), columns, batch)
        else:
            state = (
                cupy.empty(doubles, dtype=cupy.float64),
                cupy.zeros(1, dtype=cupy.int32),
                columns,
                batch,
            )
        self._aabb_clique_panel_state[src_rank] = state
        launches = (
            sum(-(-c // b) for c, b in zip(counts, batch))
            if batch
            else sum(1 for c in counts if c)
        )
        self._logger.debug(
            f"Clique source panel ready: rank={self._rank} src={src_rank} columns={columns} "
            f"scratch={doubles * 8 / 2**30:.2f} GiB launches={launches}"
        )
        return state

    def _launch_cross_clique_arm(
        self, src_rank, d_src_in, d_out, n_alpha_compact
    ) -> None:


        plan = self._arm_plans[src_rank]
        cross = self._clique_cross(src_rank, plan.aabb_path, plan.remote_beta)
        if cross is None:
            raise ValueError(
                f"source {src_rank} takes the aabb path {plan.aabb_path}, not a clique path"
            )
        a0_peer, a1_peer, b0_peer, b1_peer, _, _ = self._rank_tiles[src_rank]
        geometry = (
            self._a0,
            self._n_a_local,
            self._b0,
            self._n_b_local,
            a0_peer,
            a1_peer - a0_peer,
            b0_peer,
            b1_peer - b0_peer,
            0,
            self._n_a_local,
            0,
            self._n_b_local,
            int(n_alpha_compact),
            int(self._threads_per_block),
        )
        launch_cross(
            SourceLaunch(
                geometry=geometry,
                src_in=_device_address(d_src_in),
                out=_device_address(d_out),
                alpha_slot=0,
                beta_slot=0,
                stream=0,
                same_alpha=False,
                same_beta=False,
                view=self._tile_tables.sources[src_rank].launch_view,
                caches={},
                pair=None,
                rows={},
                num_alpha=int(self._num_alpha),
                num_beta=int(self._num_beta),
                alpha_strings=self._d_alpha_uniq,
                beta_strings=self._d_beta_uniq,
                coefficients=self._d_aabb_g,
                cross=cross,
            )
        )

    def _launch_source(
        self,
        src_rank,
        *,
        d_src_in,
        d_out,
        n_alpha_compact,
        same_alpha,
        same_beta,
        add_diag,
        stream_handle,
    ) -> None:


        view = self._tile_tables.sources[src_rank].launch_view
        a0_peer, a1_peer, b0_peer, b1_peer, _, _ = self._rank_tiles[src_rank]
        geometry = (
            self._a0,
            self._n_a_local,
            self._b0,
            self._n_b_local,
            a0_peer,
            a1_peer - a0_peer,
            b0_peer,
            b1_peer - b0_peer,
            0,
            self._n_a_local,
            0,
            self._n_b_local,
            int(n_alpha_compact),
            int(self._threads_per_block),
        )
        caches = self._source_caches(src_rank, same_alpha, same_beta)
        if not self._caches_built and not self._host_arms and (same_alpha or same_beta):
            self._build_source_caches(
                src_rank, geometry, view, caches, same_alpha, same_beta, stream_handle
            )
        arm_plan = self._arm_plans[src_rank]
        pair = self._arm_pair_payload(
            src_rank,
            view,
            same_alpha,
            same_beta,
            arm_plan.aabb_mode,
            self._d_alpha_slot_persistent,
            caches["aa_c0"],
            caches["aa_cpb"],
            caches["bb_c0"],
            caches["bb_cpb"],
        )
        cross = None
        if arm_plan.aabb_mode == 12:
            cross = (
                self._clique_cross(src_rank, arm_plan.aabb_path, arm_plan.remote_beta)
                or "sell"
            )
        built = (
            src_rank not in self._bbbb_class_state,
            src_rank not in self._aaaa_class_state,
        )
        bbbb_class = self._beta_double_class(src_rank, view, caches, same_alpha)
        aaaa_class = self._alpha_double_class(src_rank, view, caches, same_beta)
        if (self._bbbb_class_enabled and same_alpha and built[0]) or (
            self._aaaa_class_enabled and same_beta and built[1]
        ):

            import cupy  # noqa: PLC0415

            cupy.get_default_memory_pool().free_all_blocks()
        launch = SourceLaunch(
            geometry=geometry,
            src_in=_device_address(d_src_in),
            out=_device_address(d_out),
            alpha_slot=_device_address(self._d_alpha_slot_persistent),
            beta_slot=_device_address(self._d_beta_slot_persistent),
            stream=int(stream_handle),
            same_alpha=bool(same_alpha),
            same_beta=bool(same_beta),
            view=view,
            caches=caches,
            pair=pair,
            rows=dict(
                aaaa=int(self._aaaa_ycoarsen_k),
                aa=int(self._aa_ycoarsen_k),
                bb=int(self._bb_xcoarsen_k),
                bb_pair=int(self._bb_pair_xcoarsen_k),
                bbbb=int(self._bbbb_xcoarsen_k),
            ),
            order=self._source_order(src_rank, view, same_beta),
            num_alpha=int(self._num_alpha),
            num_beta=int(self._num_beta),
            alpha_strings=self._d_alpha_uniq,
            beta_strings=self._d_beta_uniq,
            coefficients=self._d_aabb_g,
            cross=cross,
            arm_events=self._recorder.arm_events
            if self._recorder is not None
            else None,
            bbbb_class=bbbb_class,
            aaaa_class=aaaa_class,
        )
        self._add_aabb_events(launch, src_rank, same_alpha, same_beta)
        try:
            if add_diag:
                self._launch_bra_diagonal(
                    d_src_in,
                    d_out,
                    self._d_alpha_slot_persistent,
                    self._d_beta_slot_persistent,
                    n_alpha_compact,
                    (0, self._n_a_local),
                    (0, self._n_b_local),
                    stream_handle,
                )
            launch_steady_arms(launch)
        except RuntimeError as error:
            raise RuntimeError(
                f"rank {self._rank} source {src_rank} ({source_class(same_alpha, same_beta)}): "
                f"{error}"
            ) from error

    def _source_order(self, src_rank, view, same_beta):


        if not (
            self._aaaa_group_auto
            and same_beta
            and view.get("d_alpha4_sell_off") is not None
        ):
            return self._arm_order
        group = self._aaaa_group_by_source.get(src_rank)
        if group is None:
            import cupy  # noqa: PLC0415

            entries = int(cupy.asnumpy(view["d_alpha4_sell_off"][-1:])[0])
            blocks = max(1, -(-self._n_a_local // int(self._threads_per_block)))
            group, reason = autotune.aaaa_block_group(
                entries * 12 / blocks,
                autotune.device_facts(self._autotune_ranks_on_device),
            )
            self._aaaa_group_by_source[src_rank] = group
            self._autotune.candidates[f"aaaa group source {src_rank}"] = (
                f"{group}: {reason}"
            )
        return dict(self._arm_order, aaaa=int(group))

    def _beta_double_class(self, src_rank, view, caches, same_alpha):


        if not (self._bbbb_class_enabled and same_alpha):
            return None
        if src_rank in self._bbbb_class_state:
            return self._bbbb_class_state[src_rank]
        cache = caches.get("bbbb")
        if (
            cache is None
            or view.get("d_beta4_ptr") is None
            or view.get("d_beta4_j") is None
        ):
            self._bbbb_class_state[src_rank] = None
            return None
        import cupy  # noqa: PLC0415

        from .doubles_classes import double_class_tables  # noqa: PLC0415

        started = time.perf_counter()
        self._compute_stream.synchronize()
        _, _, b0_peer, b1_peer, _, _ = self._rank_tiles[src_rank]

        with self._compute_stream:
            tables = double_class_tables(
                view["d_beta4_ptr"][: self._n_b_local + 1],
                view["d_beta4_j"],
                cache,
                self._d_beta_uniq[self._b0 : self._b0 + self._n_b_local],
                self._d_beta_uniq[b0_peer:b1_peer],
                self._bbbb_class_min,
                xp=cupy,
            )
            device = {
                name: getattr(tables, name)
                for name in (
                    "blocks",
                    "block_offsets",
                    "tile_info",
                    "row_members",
                    "col_members",
                )
            }
            device["nonfinite"] = cupy.zeros(1, dtype=cupy.int32)
            residual = tables.stats["residual_edges"] > 0
            if residual:
                device.update(
                    {
                        name: getattr(tables, name)
                        for name in (
                            "residual_offsets",
                            "residual_columns",
                            "residual_values",
                        )
                    }
                )
        self._compute_stream.synchronize()
        if tables.stats["tiles"] == 0:

            self._bbbb_class_state[src_rank] = None
            self._logger.debug(
                f"Beta doubles use the CSR kernel: rank={self._rank} src={src_rank} classes=0 "
                f"setup={time.perf_counter() - started:.3f}s"
            )
            return None
        form = BetaDoubleClass(
            blocks=device["blocks"],
            block_offsets=device["block_offsets"],
            tile_info=device["tile_info"],
            row_members=device["row_members"],
            col_members=device["col_members"],
            nonfinite=device["nonfinite"],
            num_tiles=int(tables.stats["tiles"]),
            alpha_group=self._bbbb_class_alpha_group,
            residual_offsets=device.get("residual_offsets"),
            residual_columns=device.get("residual_columns"),
            residual_values=device.get("residual_values"),
        )
        nbytes = sum(int(array.nbytes) for array in device.values())
        self._bbbb_class_bytes += nbytes
        self._bbbb_class_state[src_rank] = form
        stats = tables.stats
        self._logger.debug(
            f"Beta double classes ready: rank={self._rank} src={src_rank} classes={stats['classes']} "
            f"tiles={stats['tiles']} class_edges={stats['class_edges']} "
            f"residual_edges={stats['residual_edges']} fill={stats['fill']:.3f} "
            f"bytes={nbytes / 2**20:.1f} MiB rank_total={self._bbbb_class_bytes / 2**20:.1f} MiB "
            f"setup={time.perf_counter() - started:.3f}s"
        )
        return form

    def _alpha_double_class(self, src_rank, view, caches, same_beta):


        if not (self._aaaa_class_enabled and same_beta):
            return None
        if src_rank in self._aaaa_class_state:
            return self._aaaa_class_state[src_rank]
        cache = caches.get("aaaa")
        names = (
            "d_alpha4_ptr",
            "d_alpha4_j",
            "d_alpha4_sell_off",
            "d_alpha4_sell_len",
            "d_alpha4_sell_j",
        )
        if cache is None or any(view.get(name) is None for name in names):
            self._aaaa_class_state[src_rank] = None
            return None
        import cupy  # noqa: PLC0415

        from .doubles_classes import double_class_tables, sell_from_csr  # noqa: PLC0415
        from .tile_tables import SELL_SLICE_ROWS  # noqa: PLC0415

        started = time.perf_counter()
        self._compute_stream.synchronize()
        a0_peer, a1_peer, _, _, _, _ = self._rank_tiles[src_rank]
        with self._compute_stream:
            ptr = view["d_alpha4_ptr"][: self._n_a_local + 1].astype(cupy.int64)
            nnz = int(ptr[-1])
            lengths = view["d_alpha4_sell_len"][: self._n_a_local].astype(cupy.int64)
            if not bool(cupy.all(ptr[1:] - ptr[:-1] == lengths)):
                raise ValueError(
                    f"source {src_rank}: alpha double CSR and SELL lengths disagree"
                )

            starts = cupy.zeros(nnz + 1, dtype=cupy.int64)
            cupy.add.at(starts, ptr[1:-1], 1)
            rows = cupy.cumsum(starts[:nnz])
            k = cupy.arange(nnz, dtype=cupy.int64) - ptr[rows]
            off = view["d_alpha4_sell_off"].astype(cupy.int64)
            slot = (
                off[rows // SELL_SLICE_ROWS]
                + k * SELL_SLICE_ROWS
                + rows % SELL_SLICE_ROWS
            )
            values = cache[slot]
            tables = double_class_tables(
                ptr,
                view["d_alpha4_j"],
                values,
                self._d_alpha_uniq[self._a0 : self._a0 + self._n_a_local],
                self._d_alpha_uniq[a0_peer:a1_peer],
                self._aaaa_class_min,
                xp=cupy,
            )
            if tables.stats["tiles"] == 0:
                form = None
            else:
                residual = leftover = None
                if tables.stats["residual_edges"] > 0 and self._aaaa_leftover_xt:


                    leftover = (
                        tables.residual_offsets,
                        tables.residual_columns,
                        tables.residual_values,
                    )
                    if self._aaaa_slab_identity is None:
                        self._aaaa_slab_identity = cupy.arange(
                            self._aaaa_class_slab, dtype=cupy.int32
                        )
                elif tables.stats["residual_edges"] > 0:

                    residual = sell_from_csr(
                        tables.residual_offsets,
                        tables.residual_columns,
                        tables.residual_values,
                        xp=cupy,
                        slice_rows=SELL_SLICE_ROWS,
                    )
                if self._aaaa_class_scratch is None:
                    slab = self._aaaa_class_slab
                    self._aaaa_class_scratch = (
                        cupy.empty(slab * int(self._n_a_peer_max), dtype=cupy.float64),
                        cupy.empty(slab * int(self._n_a_local), dtype=cupy.float64),
                    )
                classes = BetaDoubleClass(
                    blocks=tables.blocks,
                    block_offsets=tables.block_offsets,
                    tile_info=tables.tile_info,
                    row_members=tables.row_members,
                    col_members=tables.col_members,
                    nonfinite=cupy.zeros(1, dtype=cupy.int32),
                    num_tiles=int(tables.stats["tiles"]),
                    alpha_group=self._bbbb_class_alpha_group,
                    residual_offsets=None,
                    residual_columns=None,
                    residual_values=None,
                )
                form = AlphaDoubleClass(
                    classes=classes,
                    xt=self._aaaa_class_scratch[0],
                    yt=self._aaaa_class_scratch[1],
                    slab=self._aaaa_class_slab,
                    n_alpha_src=int(a1_peer - a0_peer),
                    residual=residual,
                    leftover=leftover,
                    slab_identity=self._aaaa_slab_identity,
                )
        self._compute_stream.synchronize()
        self._aaaa_class_state[src_rank] = form
        stats = tables.stats
        nbytes = (
            0
            if form is None
            else sum(
                int(a.nbytes)
                for a in (
                    tables.blocks,
                    tables.block_offsets,
                    tables.tile_info,
                    tables.row_members,
                    tables.col_members,
                )
            )
            + (
                0
                if form.residual is None
                else sum(int(a.nbytes) for a in form.residual)
            )
            + (
                0
                if form is None or form.leftover is None
                else sum(int(a.nbytes) for a in form.leftover)
            )
        )
        scratch = (
            0
            if self._aaaa_class_scratch is None
            else sum(int(a.nbytes) for a in self._aaaa_class_scratch)
        )
        self._logger.debug(
            f"Alpha double class setup: rank={self._rank} src={src_rank} classes={stats['classes']} tiles={stats['tiles']} "
            f"class_edges={stats['class_edges']} residual_edges={stats['residual_edges']} fill={stats['fill']:.3f} "
            f"bytes={nbytes / 2**20:.1f} MiB scratch={scratch / 2**20:.1f} MiB"
            f"{' (SELL arm kept)' if form is None else ''} setup={time.perf_counter() - started:.3f}s"
        )
        return form

    def _source_caches(self, src_rank, same_alpha, same_beta) -> dict:


        if self._host_arms:
            return {name: self._arm_caches.get((name, src_rank)) for name in ARM_CACHES}
        return self._arm_cache_buffers(
            src_rank, self._arm_cache_sizes(src_rank, bool(same_alpha), bool(same_beta))
        )

    def _build_source_caches(
        self, src_rank, geometry, view, caches, same_alpha, same_beta, stream_handle
    ) -> None:


        import cupy  # noqa: PLC0415

        if self._d_build_error is None:
            with self._compute_stream:
                self._d_build_error = cupy.zeros(1, dtype=cupy.int32)
        try:
            build_source_caches(
                geometry,
                self._term_table_values(),
                view,
                caches,
                bool(same_alpha),
                bool(same_beta),
                _device_address(self._d_alpha_slot_persistent),
                _device_address(self._d_beta_slot_persistent),
                _device_address(self._d_build_error),
                int(stream_handle),
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"rank {self._rank} source {src_rank}: {error}"
            ) from error

    def _term_table_values(self) -> tuple:

        addresses = (
            self._d_term_header,
            self._d_term_inds_vals,
            self._d_term_proj_inds,
            self._d_term_proj_bits,
            self._d_grp_ladder_ptrs,
            self._d_oper_elems,
            self._d_grp2_ht_keys,
            self._d_grp2_ht_vals,
            self._d_grp2_ht_bitset,
            self._d_grp4_ht_keys,
            self._d_grp4_ht_vals,
            self._d_grp4_ht_bitset,
            self._d_alpha_uniq,
            self._d_beta_uniq,
        )
        return (
            *(_device_address(a) for a in addresses),
            int(self._grp4_ht_mask),
            int(self._grp2_ht_mask),
            int(self._ladder_offset),
            int(self._term_inds_stride),
            int(self._term_proj_stride),
            int(self._num_alpha),
            int(self._num_beta),
        )

    def _add_aabb_events(self, launch, src_rank, same_alpha, same_beta) -> None:


        clique = isinstance(launch.cross, CliqueCross)
        recorder = self._recorder
        if recorder is None or not (
            self._aabb_source_timing_enabled
            or (clique and self._aabb_parent_timing_enabled)
        ):
            return
        import cupy  # noqa: PLC0415

        events = launch.events
        if clique:
            events["parent_start"], events["parent_end"] = (
                cupy.cuda.Event(),
                cupy.cuda.Event(),
            )
            recorder.parent_pairs.append(
                (src_rank, events["parent_start"], events["parent_end"])
            )
            if self._aabb_source_timing_enabled:
                events["residual_end"] = cupy.cuda.Event()
        if not self._aabb_source_timing_enabled:
            return
        if cross_ready(launch):
            events["source_start"], events["source_end"] = (
                cupy.cuda.Event(),
                cupy.cuda.Event(),
            )
        recorder.source_records.append(
            dict(
                src=src_rank,
                kind=source_class(same_alpha, same_beta),
                path=("transposed_clique" if launch.cross.transposed else "clique")
                if clique
                else "baseline",
                start=events.get("source_start"),
                end=events.get("source_end"),
                parent_start=events.get("parent_start"),
                split=events.get("residual_end"),
                parent_end=events.get("parent_end"),
            )
        )

    def _ensure_local_staging(self) -> None:

        if self._d_v_local is not None:
            return
        import cupy

        self._d_v_local = cupy.empty(
            (self._n_b_local, self._n_a_local), dtype=cupy.float64
        )
        self._d_out_local = cupy.empty(
            (self._n_b_local, self._n_a_local), dtype=cupy.float64
        )

    def _matvec(self, v, *, out=None):  # noqa: D401


        v_arr, v_is_device = self._matvec_input(v, out)
        self._matvec_call_idx += 1
        recorder = MatvecRecorder(
            self._recorder_settings,
            self._comm,
            self._rank,
            self._matvec_call_idx,
            self._compute_stream,
            self._copy_stream,
            lambda src: self._rank_tiles[src],
        )
        self._recorder = recorder
        d_v, d_out, alias_ok = self._stage(v_arr, v_is_device, out, recorder)
        building_caches = not self._caches_built
        if building_caches and not self._host_arms and self._d_term_header is None:
            self._upload_term_tables(_operator_tables(self._hsub.spmv))
        _dbg_v_fingerprint(d_v, self._comm, self._rank, self._matvec_call_idx)
        self._debug_input_norm(d_v)
        recorder.compute_started = time.perf_counter()
        self._launch_own_tile(d_v, d_out)
        recorder.checkpoint("diag", d_out)
        self._exchange(d_v, d_out, recorder)
        recorder.compute_done()
        recorder.finish(d_v, d_out)
        result = self._unstage(d_out, out, alias_ok, recorder)
        recorder.finish_arms()
        self._finish_cache_build(building_caches)
        recorder.timing["caches_built_this_call"] = bool(building_caches)
        self._record_matvec(recorder)
        return result

    def _matvec_input(self, v, out):


        v_size = getattr(v, "size", None)
        if v_size is None:
            v = np.ascontiguousarray(v)
            v_size = v.size
        if int(v_size) != self._n_local:
            raise ValueError(
                f"_matvec: expected local vector length {self._n_local}, got {int(v_size)}"
            )
        if not self._cupy_available:
            raise NotImplementedError(
                "_matvec requires cupy and a CUDA GPU; this host has none. "
                "TileGPUResidentHamiltonian is GPU-resident by design and "
                "has no CPU fallback path."
            )
        if self._d_alpha_uniq is None:
            raise NotImplementedError(
                "_matvec called before operator tables were uploaded "
                "(_build_and_upload_operator_tables); this happens "
                "when hsub.spmv is not a FulqrumSpMV (a test mock)."
            )

        import cupy  # noqa: PLC0415

        v_is_device = isinstance(v, cupy.ndarray)
        v_arr = v.reshape(-1) if v_is_device else np.ascontiguousarray(v).reshape(-1)
        if out is not None and (
            not isinstance(out, cupy.ndarray) or out.size != self._n_local
        ):
            raise ValueError(
                f"_matvec: out= must be a cupy array of length {self._n_local}"
            )
        return v_arr, v_is_device

    def _stage(self, v_arr, v_is_device, out, recorder):


        import cupy  # noqa: PLC0415

        timing = recorder.timing
        v_2d = v_arr.astype(np.float64, copy=False).reshape(
            self._n_b_local, self._n_a_local
        )
        alias_ok = (
            self._matvec_alias_xy_enabled
            and v_is_device
            and out is not None
            and v_2d.dtype == np.float64
            and v_2d.flags.c_contiguous
            and out.dtype == np.float64
            and out.flags.c_contiguous
            and (
                int(out.data.ptr) + int(out.nbytes) <= int(v_2d.data.ptr)
                or int(v_2d.data.ptr) + int(v_2d.nbytes) <= int(out.data.ptr)
            )
        )
        if v_is_device or out is not None:
            producer_done = cupy.cuda.get_current_stream().record()
            self._copy_stream.wait_event(producer_done)
        _t = time.perf_counter()
        if alias_ok:
            d_v = v_2d
            d_out = out.reshape(self._n_b_local, self._n_a_local)
            with self._copy_stream:
                d_out.fill(0.0)
        else:
            self._ensure_local_staging()
            d_v = self._d_v_local
            d_out = self._d_out_local
            with self._copy_stream:
                if v_is_device:
                    cupy.copyto(self._d_v_local, v_2d)
                else:
                    self._d_v_local.set(v_2d)
                self._d_out_local.fill(0.0)
        h2d_event = self._copy_stream.record()
        self._compute_stream.wait_event(h2d_event)
        timing["h2d"] = time.perf_counter() - _t
        timing["h2d_bytes"] = int(v_2d.nbytes)
        timing["alias_xy"] = bool(alias_ok)
        self._copy_stream.synchronize()
        return d_v, d_out, alias_ok

    def _debug_input_norm(self, d_v) -> None:

        if not _debug_matvec_enabled():
            return
        import cupy  # noqa: PLC0415

        self._compute_stream.synchronize()
        _dbg_print(
            self._rank,
            f"Input vector norm: scope=local, norm={float(cupy.sqrt(cupy.sum(d_v * d_v)).get()):.6e}",
        )

    def _run_source(
        self,
        src_rank,
        d_src_in,
        d_out,
        n_a_peer,
        n_b_peer,
        same_alpha,
        same_beta,
        add_diag,
        n_alpha_compact=None,
        label="",
    ):


        import cupy  # noqa: PLC0415

        if src_rank not in self._sources_2d:
            return None
        with self._compute_stream:
            start = cupy.cuda.Event()
            start.record()
        self._launch_source(
            src_rank,
            d_src_in=d_src_in,
            d_out=d_out,
            n_alpha_compact=int(
                n_a_peer if n_alpha_compact is None else n_alpha_compact
            ),
            same_alpha=same_alpha,
            same_beta=same_beta,
            add_diag=add_diag,
            stream_handle=int(self._compute_stream.ptr),
        )
        with self._compute_stream:
            end = cupy.cuda.Event()
            end.record()
        self._recorder.source_launched(src_rank, start, end, d_out, label)
        return end

    def _launch_own_tile(self, d_v, d_out) -> None:

        if not (
            self._rank in self._sources_2d
            and self._sources_2d[self._rank] is TileClass.DIAGONAL
        ):
            return
        a0p, a1p, b0p, b1p, _ca, _cb = self._rank_tiles[self._rank]
        self._run_source(
            self._rank,
            d_v,
            d_out,
            a1p - a0p,
            b1p - b0p,
            same_alpha=1,
            same_beta=1,
            add_diag=1 if bool(self._has_diag) else 0,
            label="diag",
        )

    def _exchange(self, d_v, d_out, recorder) -> None:


        timing = recorder.timing
        d_send_view = d_v.ravel()
        slot_events: list = [None, None]
        executor = ExchangeExecutor(
            self._backend,
            self._comm,
            self._copy_stream,
            self._compute_stream,
            recorder.exchange_begin,
            recorder.exchange_end,
            slot_events,
        )
        if self._hierarchical_exchange_active:
            total_bytes = self._exchange_two_level(
                executor, d_send_view, d_out, recorder
            )
        else:
            self._exchange_beta_ring(
                executor, slot_events, d_v, d_send_view, d_out, recorder
            )
            self._exchange_alpha_ring(
                executor, slot_events, d_v, d_send_view, d_out, recorder
            )
            mixed_bytes = self._exchange_mixed_window(
                executor, d_send_view, d_out, recorder
            )
            bytes_beta_ring = (
                len(self._beta_ring_steps) * self._ringshift_beta_slot_elems * 8
            )
            bytes_alpha_ring = (
                len(self._alpha_ring_steps) * self._ringshift_alpha_slot_elems * 8
            )
            total_bytes = bytes_beta_ring + bytes_alpha_ring + mixed_bytes
        timing["exchange"] = (
            timing["exchange_alpha"]
            + timing["exchange_beta"]
            + timing["exchange_mixed"]
        )
        timing["allgatherv_bytes"] = int(total_bytes)
        timing["exchange_bandwidth_gbps"] = (
            (total_bytes / 1e9) / timing["exchange"] if timing["exchange"] > 0 else 0.0
        )
        timing["hierarchical_path"] = bool(self._hierarchical_exchange_active)
        timing["ringshift_path"] = not self._hierarchical_exchange_active

    def _exchange_two_level(self, executor, d_send_view, d_out, recorder) -> int:


        timing = recorder.timing
        timing["exchange_alpha"] = 0.0
        timing["exchange_beta"] = 0.0
        timing["exchange_mixed"] = 0.0
        started = time.perf_counter()
        logical_bytes = [0]

        def launch_two_level_source(src_rank: int, source_buffer, source_label: str):

            src_rank = int(src_rank)
            if (
                src_rank < 0
                or src_rank == self._rank
                or src_rank not in self._sources_2d
            ):
                return None
            a0p, a1p, b0p, b1p, ca, cb = self._rank_tiles[src_rank]
            count = (a1p - a0p) * (b1p - b0p)
            event = self._run_source(
                src_rank,
                source_buffer[:count].reshape(b1p - b0p, a1p - a0p),
                d_out,
                a1p - a0p,
                b1p - b0p,
                int(ca == self._coord_alpha),
                int(cb == self._coord_beta),
                0,
                label=source_label,
            )
            logical_bytes[0] += count * 8
            return event

        network_bytes = executor.run_two_level(
            d_send_view,
            self._d_hierarchical_stage,
            self._d_hierarchical_gather[0],
            self._hierarchical_node_comm,
            self._hierarchical_plan.domain_rank,
            self._hierarchical_plan.local_sources,
            self._hierarchical_local_ring_args,
            self._hierarchical_round_args,
            launch_two_level_source,
            recorder.fetch_pairs,
            recorder.gather_pairs,
        )
        timing["exchange_hierarchical"] = time.perf_counter() - started
        timing["exchange_mixed"] = timing["exchange_hierarchical"]
        timing["hierarchical_network_bytes"] = int(network_bytes)
        timing["hierarchical_direct_remote_bytes"] = int(
            self._hierarchical_direct_remote_bytes
        )
        recorder.checkpoint("hierarchical", d_out)
        return int(logical_bytes[0])

    def _seed_ring_slot(self, slot_events, fill) -> None:


        if slot_events[0] is not None:
            self._copy_stream.wait_event(slot_events[0])
        with self._copy_stream:
            fill()
        self._copy_stream.synchronize()

    def _exchange_beta_ring(
        self, executor, slot_events, d_v, d_send_view, d_out, recorder
    ) -> None:


        timing = recorder.timing
        double_buffer = self._exchange_double_buffer_enabled
        iteration = self._matvec_call_idx
        timing["exchange_alpha"] = 0.0
        if self._beta_ring_steps:
            if double_buffer:

                def fill():
                    """Clear slot 0 of the beta ring and copy the input tile into its first rows."""
                    self._d_beta_ring[0].fill(0.0)
                    self._d_beta_ring[0][: self._n_b_local, :] = d_v

                self._seed_ring_slot(slot_events, fill)
            started = time.perf_counter()
            executor.run_ring(
                self._beta_ring_steps,
                self._d_beta_ring,
                d_send_view,
                lambda step: self._tile_rows(step.source) * self._n_a_local,
                lambda step, received: (
                    received[: self._tile_rows(step.source), :]
                    if double_buffer
                    else received.reshape(self._tile_rows(step.source), self._n_a_local)
                ),
                lambda step, tile, index: self._run_source(
                    step.source,
                    tile,
                    d_out,
                    self._tile_columns(step.source),
                    self._tile_rows(step.source),
                    1,
                    0,
                    0,
                    label=f"{'ringshift' if double_buffer else 'df'}_beta_ring[step={index}]",
                ),
                send_pinned=self._beta_ring_send_pinned if double_buffer else None,
                receive_pinned=self._beta_ring_recv_pinned if double_buffer else None,
                after_receive=(
                    (
                        lambda step, index, received: _dbg_ring_slot_fingerprint(
                            received,
                            self._comm,
                            self._rank,
                            "ringshift_beta",
                            index,
                            step.source,
                            iteration,
                        )
                    )
                    if double_buffer
                    else None
                ),
            )
            timing["exchange_alpha"] = time.perf_counter() - started
        recorder.checkpoint("ringshift_beta_only", d_out)

    def _exchange_alpha_ring(
        self, executor, slot_events, d_v, d_send_view, d_out, recorder
    ) -> None:


        timing = recorder.timing
        double_buffer = self._exchange_double_buffer_enabled
        iteration = self._matvec_call_idx
        timing["exchange_beta"] = 0.0
        if self._alpha_ring_steps:
            if double_buffer:

                def fill():
                    """Clear slot 0 of the alpha ring and copy the input tile into its first columns."""
                    self._d_alpha_ring[0].fill(0.0)
                    self._d_alpha_ring[0][:, : self._n_a_local] = d_v

                self._seed_ring_slot(slot_events, fill)
            started = time.perf_counter()
            executor.run_ring(
                self._alpha_ring_steps,
                self._d_alpha_ring,
                d_send_view,
                lambda step: self._n_b_local * self._tile_columns(step.source),
                lambda step, received: (
                    received
                    if double_buffer
                    else received.reshape(
                        self._n_b_local, self._tile_columns(step.source)
                    )
                ),
                lambda step, tile, index: self._run_source(
                    step.source,
                    tile,
                    d_out,
                    self._tile_columns(step.source),
                    self._tile_rows(step.source),
                    0,
                    1,
                    0,
                    n_alpha_compact=self._n_a_peer_max if double_buffer else None,
                    label=f"{'ringshift' if double_buffer else 'df'}_alpha_ring[step={index}]",
                ),
                send_pinned=self._alpha_ring_send_pinned if double_buffer else None,
                receive_pinned=self._alpha_ring_recv_pinned if double_buffer else None,
                after_receive=(
                    (
                        lambda step, index, received: _dbg_ring_slot_fingerprint(
                            received,
                            self._comm,
                            self._rank,
                            "ringshift_alpha",
                            index,
                            step.source,
                            iteration,
                        )
                    )
                    if double_buffer
                    else None
                ),
            )
            timing["exchange_beta"] = time.perf_counter() - started
        recorder.checkpoint("ringshift_alpha_only", d_out)

    def _exchange_mixed_window(self, executor, d_send_view, d_out, recorder) -> int:


        timing = recorder.timing
        timing["exchange_mixed"] = 0.0
        mixed_bytes = 0
        if self._mixed_indegree > 0 or self._mixed_outdegree > 0:
            timing["exchange_mixed"], mixed_bytes = executor.run_window(
                self._mixed_steps,
                self._d_mixed_slots,
                d_send_view,
                lambda step: (
                    self._tile_rows(step.source) * self._tile_columns(step.source)
                ),
                lambda step, received: received.reshape(
                    self._tile_rows(step.source), self._tile_columns(step.source)
                ),
                lambda step, tile, index: self._run_source(
                    step.source,
                    tile,
                    d_out,
                    self._tile_columns(step.source),
                    self._tile_rows(step.source),
                    0,
                    0,
                    0,
                    label=f"ringshift_mixed_win[hop={index}]",
                ),
                send_pinned=self._mixed_send_pinned,
                receive_pinned=self._mixed_recv_pinned,
            )
        recorder.checkpoint("ringshift_mixed", d_out)
        return mixed_bytes

    def _unstage(self, d_out, out, alias_ok, recorder):


        import cupy  # noqa: PLC0415

        timing = recorder.timing
        _t = time.perf_counter()
        with self._copy_stream:
            if out is not None:
                if not alias_ok:
                    cupy.copyto(out, d_out.reshape(out.shape))
                out_2d = out
            else:
                out_2d = cupy.asnumpy(d_out)
        self._copy_stream.synchronize()
        timing["d2h"] = time.perf_counter() - _t
        timing["d2h_bytes"] = int(out_2d.nbytes)
        timing["source_tiles"] = len(self._sources_2d)
        timing["kernel_launches"] = recorder.launches
        return out_2d.reshape(-1).astype(np.float64, copy=False)

    def _finish_cache_build(self, building_caches: bool) -> None:

        if not building_caches:
            return
        failed = (
            self._d_build_error is not None and int(self._d_build_error.get()[0]) != 0
        )
        raise_on_every_rank(
            self._comm,
            "an operator group could not be factored over the other sector"
            if failed
            else None,
            RuntimeError,
            "arm cache build",
        )
        self._caches_built = True
        self._free_term_tables()

    def _record_matvec(self, recorder) -> None:

        timing = recorder.timing
        timing["wall"] = time.perf_counter() - recorder.started
        self._last_timing = timing
        self._record_free_hbm(
            "post_first_matvec" if recorder.iteration == 1 else "steady"
        )
        if self._first_matvec_logged or self._rank != 0:
            return
        free = self._hbm_trace.get("post_first_matvec")
        predicted = self._construction_timing.get("predicted_hbm_bytes", 0)
        if self._config.get("FQ_HBM_LOG", "0") == "1" and self._logger.isEnabledFor(
            logging.DEBUG
        ):
            import cupy

            pool = cupy.get_default_memory_pool()
            self._logger.debug(
                "GPU memory after first matvec: rank=0, predicted=%s, predicted_arm_caches=%s, "
                "free_snapshot=%s, pool_used=%s, pool_reserved=%s",
                format_memory(predicted),
                format_memory(self.predicted_hbm_bytes.get("arm_caches", 0)),
                format_memory(free),
                format_memory(pool.used_bytes()),
                format_memory(pool.total_bytes()),
            )
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(
                "First matvec profile: rank=0, wall=%.4f s, host_to_device=%.4f s, exchange=%.4f s, "
                "compute=%.4f s, synchronization=%.4f s, device_to_host=%.4f s, source_tiles=%d, "
                "kernel_launches=%d, exchange_data=%s, predicted_memory=%s, free_memory=%s",
                timing["wall"],
                timing["h2d"],
                timing["exchange"],
                timing["compute"],
                timing["sync"],
                timing["d2h"],
                timing["source_tiles"],
                timing["kernel_launches"],
                format_memory(timing["allgatherv_bytes"]),
                format_memory(predicted),
                format_memory(free),
            )
        self._first_matvec_logged = True
