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

from enum import IntEnum

import numpy as np


class TileClass(IntEnum):


    DIAGONAL = 0
    ALPHA_ONLY = 1
    BETA_ONLY = 2
    MIXED = 3


_OPER_ELEMS_HOST = np.array(
    [
        1,
        0,
        0,
        -1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        -1,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
    ],
    dtype=np.int8,
)


def _split_ranges_even(n: int, parts: int) -> list[tuple[int, int]]:

    base, rem = divmod(int(n), int(parts))
    ranges = []
    start = 0
    for i_part in range(int(parts)):
        stop = start + base + (1 if i_part < rem else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def _default_2d_process_grid(size: int) -> tuple[int, int]:

    best = (1, int(size))
    best_gap = int(size) - 1
    root = int(np.sqrt(size))
    for p_alpha in range(1, root + 1):
        if size % p_alpha:
            continue
        p_beta = size // p_alpha
        gap = abs(p_beta - p_alpha)
        if gap < best_gap:
            best = (p_alpha, p_beta)
            best_gap = gap
    return best


def DistributedMPIGPUHamiltonian(
    hsub,
    layout: str = "tile_resident",
    **kwargs,
):
    """Create a GPU-backed Hamiltonian for distributed MPI eigenvalue solves.

    Returns a ``TileGPUResidentHamiltonian`` using the alpha and beta strings
    stored in ``hsub``. Construct it on every communicator rank with matching
    Hamiltonian and shared settings; each rank may select a different GPU.
    Construction, matvecs and RDM calls are collective. Use the returned operator
    with ``distributed_primme_eigsh``; it supplies this rank's local vector tile,
    while its SciPy ``shape`` describes the full problem.

    The subspace must be ``fq.Subspace([alpha_strs, beta_strs])``, a tensor product,
    subspace with alpha in the low qubits and beta in the high qubits. Each spin-sector
    supports 1–64 orbitals and must have a fixed electron count. Use at least two occupied
    orbitals per half string; smaller occupations are outside verified GPU support and may
    give incorrect results. Operator groups must pass the GPU compatibility checks.

    Args:
        hsub: Real type-2 ``SubspaceHamiltonian`` on the half-string product.
        layout: Must be ``"tile_resident"`` (default).
        **kwargs: Optional constructor settings listed below. Other keyword
            names raise TypeError. Sector strings are supplied from ``hsub``.

    Keyword Args:
        gpu_ids: One-element sequence selecting this rank's CUDA device;
            defaults to ``(0,)``. Exactly one device is required per rank.
        comm: MPI communicator; defaults to ``MPI.COMM_WORLD``. ``LocalComm``
            can instead coordinate in-process workers.
        process_grid: ``(p_alpha, p_beta)``, whose product must equal communicator
            size. ``None`` chooses a grid automatically.
        threads_per_block: CUDA threads per kernel block; defaults to 256.
        comm_backend: ``"auto"`` (default), ``"nccl"``, ``"host_mpi"``,
            ``"cuda_mpi"`` or ``"inproc"``. For MPI, ``auto`` first consults
            ``FQ_DIST_BACKEND``. If that is also auto, multiple ranks use NCCL
            when every rank supports it and uses a distinct GPU; otherwise
            host MPI is selected. CUDA-aware MPI must be requested explicitly.
            ``LocalComm`` uses ``inproc``. Failed NCCL initialization raises
            rather than retrying with host MPI.
        intra_node_size: Optional positive number of contiguous ranks per
            communication domain. ``None`` uses detected node membership.
        build_threads: CPU threads for tile construction. Zero uses
            ``OMP_NUM_THREADS``; a positive ``FQ_BUILD_THREADS`` overrides this
            argument. Defaults to 0.
        config: Checked ``GpuConfig`` snapshot. ``None`` uses
            ``GpuConfig.from_environ()`` during construction.

    Returns:
        A ``TileGPUResidentHamiltonian`` with global square shape, float64 dtype
        and local length ``n_local``. Local vectors use beta-outer, alpha-inner
        order within this rank's tile; see the class for the local-vector protocol.

    Raises:
        ValueError: The layout is unsupported, the subspace is not a half-string
            product, or a forwarded constructor option is invalid.
        TypeError: An unrecognized keyword option is supplied.

    Additional constructor errors follow ``TileGPUResidentHamiltonian``.
    """
    if layout != "tile_resident":
        raise ValueError(f"layout={layout!r} not supported")

    subspace = hsub.spmv.subspace
    if not getattr(subspace, "half_strs_mode", 0):
        raise ValueError(
            "DistributedMPIGPUHamiltonian requires a half-string Subspace. "
            "Construct it with: fq.Subspace([alpha_strs, beta_strs])."
        )

    from .dist_matvec_tile_resident import (  # noqa: PLC0415
        TileGPUResidentHamiltonian,
    )

    return TileGPUResidentHamiltonian(
        hsub, subspace.alpha_strs, subspace.beta_strs, **kwargs
    )
