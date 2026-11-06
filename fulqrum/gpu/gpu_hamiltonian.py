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

import logging
import os
import sys
import warnings

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..exceptions import FulqrumError
from .distributed.dist_matvec import _default_2d_process_grid
from .distributed.local_comm import LocalComm, run_ranks

logger = logging.getLogger(__name__)


def _check_gpu_preconditions(hsub, classname: str) -> None:

    if hsub.off_H.type != 2 or hsub.diag_H.type != 2:
        raise FulqrumError(
            f"{classname} requires operator type=2 "
            f"(got off_H.type={hsub.off_H.type}, diag_H.type={hsub.diag_H.type})."
        )
    subspace = hsub.spmv.subspace
    if not getattr(subspace, "half_strs_mode", 0):
        raise FulqrumError(
            f"{classname} requires a half-string Subspace. "
            "Construct it with: fq.Subspace([alpha_strs, beta_strs])."
        )
    a_arr, b_arr = subspace.to_alpha_beta_arrays()
    low_a = a_arr.size and min(int(x).bit_count() for x in a_arr) < 2
    low_b = b_arr.size and min(int(x).bit_count() for x in b_arr) < 2
    if low_a or low_b:
        warnings.warn(
            f"{classname}: subspace contains alpha or beta half-strings with Hamming weight < 2. "
            "GPU kernels require a fixed particle count in each spin sector and at least two "
            "occupied orbitals in every half-string. GPU results may be incorrect; use a "
            "supported subspace or the CPU operator.",
            RuntimeWarning,
            stacklevel=3,
        )


class GPUHamiltonian(LinearOperator):
    """GPU-backed real type-2 Hamiltonian for matrix-free matvec and eigensolving.

    This SciPy ``LinearOperator`` represents the full tensor-product subspace.
    It can be used with ``operator @ vector`` and LinearOperator-compatible
    eigensolvers such as ``scipy.sparse.linalg.eigsh`` and ``primme.eigsh``.
    Each entry in ``gpu_ids`` creates one worker; the wrapper partitions the
    calculation and combines the results. MPI is not required for this wrapper.
    For multiple processes, use ``DistributedMPIGPUHamiltonian``.

    Supported inputs use ``fq.Subspace([alpha_strs, beta_strs])``, with alpha in
    the low qubits and beta in the high qubits. Each spin-sector must contain 1–64
    orbitals (bits), have a fixed electron count, and pass the GPU operator checks.
    Verified GPU support requires at least two occupied orbitals in every half
    string. Smaller occupations trigger ``RuntimeWarning`` and may produce
    incorrect results.

    The LinearOperator has shape ``(N, N)`` and dtype ``float64``, where
    ``N = n_alpha * n_beta`` and each count is the number of half strings.
    Half strings are sorted independently. Vector index
    ``beta_index * n_alpha + alpha_index`` identifies the corresponding product
    state (full determinant). ``matvec`` accepts a full real host vector of shape
    ``(N,)`` or ``(N, 1)`` and returns a NumPy ``float64`` array of the same shape.

    Args:
        hsub: Real type-2 ``SubspaceHamiltonian`` on the half-string/tensor product
            subspace described above.
        gpu_ids: Sequence of CUDA device IDs, one per worker. Repeated IDs share
            a device. ``None`` uses the current CUDA device; the sequence must
            be nonempty.
        threads_per_block: CUDA threads per kernel block; defaults to 256.
        verbose: Emit a DEBUG log describing wrapper construction. Configure
            the GPU logger at DEBUG level to display it.

    Raises:
        NotImplementedError: The Hamiltonian has complex values.
        FulqrumError: The Hamiltonian is not Fulqrum's type 2 operator or the
            subspace is not a half-string/tensor product.
        ValueError: Device selection, sector layout, electron counts or operator
            groups are unsupported.
        ImportError: CuPy or a required compiled GPU component is unavailable.
        RuntimeError: Construction checks or predicted device-memory checks fail.
    """

    def __init__(self, hsub, gpu_ids=None, threads_per_block=256, verbose=False):

        if not hsub.spmv.is_real:
            raise NotImplementedError(
                "GPUHamiltonian currently supports real-valued operators only."
            )
        _check_gpu_preconditions(hsub, "GPUHamiltonian")
        import cupy  # noqa: PLC0415

        from .distributed.dist_matvec_tile_resident import TileGPUResidentHamiltonian  # noqa: PLC0415

        if gpu_ids is None:
            gpu_ids = (int(cupy.cuda.runtime.getDevice()),)
        self._gpu_ids = tuple(int(gid) for gid in gpu_ids)
        if not self._gpu_ids:
            raise ValueError("GPUHamiltonian needs at least one GPU id")
        alpha, beta = hsub.spmv.subspace.to_alpha_beta_arrays()
        alpha = np.sort(np.asarray(alpha, dtype=np.uint64))
        beta = np.sort(np.asarray(beta, dtype=np.uint64))
        self._n_a, self._n_b = int(alpha.size), int(beta.size)
        super().__init__(
            shape=(self._n_a * self._n_b, self._n_a * self._n_b), dtype=np.float64
        )
        ranks = len(self._gpu_ids)
        threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
        build_threads = 0 if ranks == 1 else max(1, threads // ranks)
        grid = _default_2d_process_grid(ranks)
        self._comms = LocalComm.create(ranks)

        def build(comm):

            gpu_id = self._gpu_ids[comm.Get_rank()]
            with cupy.cuda.Device(gpu_id):
                return TileGPUResidentHamiltonian(
                    hsub,
                    alpha,
                    beta,
                    gpu_ids=(gpu_id,),
                    comm=comm,
                    process_grid=grid,
                    threads_per_block=threads_per_block,
                    comm_backend="inproc",
                    build_threads=build_threads,
                )

        self._ranks = run_ranks(self._comms, build)
        self._tiles = [(op._a0, op._a1, op._b0, op._b1) for op in self._ranks]
        if verbose:
            logger.debug(
                "GPU wrapper constructed: devices=%s, ranks=%d, grid=%s, construction=%s",
                self._gpu_ids,
                ranks,
                grid,
                self._ranks[0].construction_timing,
                extra={"gpu_rank": 0},
            )

    @property
    def gpu_ids(self) -> tuple:
        """CUDA device IDs in worker order, returned as an immutable tuple."""
        return self._gpu_ids

    @property
    def last_timing(self) -> list:
        """Return one timing-dictionary copy per worker, in ``gpu_ids`` order.

        Dictionaries are empty before the first successful matvec. ``wall``,
        ``compute`` and ``exchange`` are in seconds; other fields follow
        ``TileGPUResidentHamiltonian.last_timing``. Reading this property uses
        existing measurements and does not launch GPU work.
        """
        return [op.last_timing for op in self._ranks]

    def reset_cache(self) -> None:
        """Clear cached GPU connection data on every worker.

        The next matvec rebuilds the GPU caches it needs. Host-resident caches
        are retained.
        """
        for op in self._ranks:
            op.reset_cache()

    def dump_profile_summary(self, stream=None) -> str:
        """Write the latest matvec timings for each worker and return the text.

        Reports ``wall``, ``compute`` and ``exchange`` in seconds, with worker
        and device IDs. The compute interval includes exchange, so the reported
        times are not additive. Uses existing measurements without launching
        GPU work or enabling additional profiling.

        Args:
            stream: Text stream with a ``write`` method; defaults to stdout.

        Returns:
            Summary text without a trailing newline. Returns an empty string
            and writes nothing if no worker has a matvec timing record.
        """
        lines = []
        for rank, (gpu_id, timing) in enumerate(zip(self._gpu_ids, self.last_timing)):
            if not timing:
                continue
            lines.append(
                f"Last matvec profile: rank={rank}, device={gpu_id}, wall={timing.get('wall', 0.0):.4f} s, "
                f"compute={timing.get('compute', 0.0):.4f} s, exchange={timing.get('exchange', 0.0):.4f} s"
            )
        text = "\n".join(lines)
        if text:
            (sys.stdout if stream is None else stream).write(text + "\n")
        return text

    def _matvec(self, v):

        import cupy  # noqa: PLC0415

        full = np.asarray(v, dtype=np.float64).reshape(self._n_b, self._n_a)
        out = np.empty((self._n_b, self._n_a), dtype=np.float64)

        def product(comm):

            rank = comm.Get_rank()
            a0, a1, b0, b1 = self._tiles[rank]
            with cupy.cuda.Device(self._gpu_ids[rank]):
                tile = self._ranks[rank]._matvec(
                    np.ascontiguousarray(full[b0:b1, a0:a1]).ravel()
                )
            out[b0:b1, a0:a1] = np.asarray(tile).reshape(b1 - b0, a1 - a0)

        run_ranks(self._comms, product)
        return out.reshape(-1)

    def compute_rdms(self, evec, *, rank: int = 1, spin_summed: bool = False):
        """Compute one- and optionally two-particle reduced density matrices.

        Pass the full real state vector in the same sorted product order as
        ``matvec``. The wrapper distributes it to every worker and returns the
        combined result as host NumPy arrays after GPU work completes. Alpha
        and beta must have equal orbital counts for this operation.

        Amplitudes are used without normalization: scaling the vector by a
        real factor scales every RDM by its square. Normalize the vector before
        calling if you need RDMs for a normalized state. Spin and tensor-index
        conventions follow PySCF ``make_rdm12s(reorder=True)``.

        Args:
            evec: NumPy-compatible or CuPy array containing ``n_alpha * n_beta``
                real amplitudes, in beta-outer, alpha-inner order.
            rank: RDM order, 1 or 2; defaults to 1. This is not a worker index.
            spin_summed: Combine alpha and beta spin blocks; defaults to False.

        Returns:
            For ``rank=1``, ``(gamma_a, gamma_b)`` or spin-summed ``gamma``.
            For ``rank=2``, ``((gamma_a, gamma_b), (G_aa, G_ab, G_bb))`` or
            spin-summed ``(gamma, G)``. Each one-particle array has shape
            ``(m, m)`` and each two-particle array ``(m, m, m, m)``, where ``m``
            is the common number of orbitals per sector.

        Raises:
            ValueError: The RDM order or vector length is invalid, or alpha and
                beta have different orbital counts.
        """
        import cupy  # noqa: PLC0415

        if rank not in (1, 2):
            raise ValueError(f"rank must be 1 or 2, got {rank!r}")
        values = (
            cupy.asnumpy(evec) if isinstance(evec, cupy.ndarray) else np.asarray(evec)
        )
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != self._n_a * self._n_b:
            raise ValueError(
                f"evec has {values.size} values, the operator has {self._n_a * self._n_b} rows"
            )
        full = values.reshape(self._n_b, self._n_a)

        def rdms(comm):

            index = comm.Get_rank()
            a0, a1, b0, b1 = self._tiles[index]
            with cupy.cuda.Device(self._gpu_ids[index]):
                return self._ranks[index].compute_rdms(
                    np.ascontiguousarray(full[b0:b1, a0:a1]).ravel(),
                    rank=rank,
                    spin_summed=spin_summed,
                )

        return run_ranks(self._comms, rdms)[0]
