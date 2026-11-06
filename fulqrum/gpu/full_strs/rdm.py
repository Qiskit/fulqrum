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
import time

import numpy as np

from ..distributed.local_comm import mpi_namespace
from ..distributed.rank_agreement import raise_on_every_rank, run_on_every_rank

logger = logging.getLogger(__name__)
__all__ = ["compute_rdms_full_strs"]

MAX_ORBITALS = 64
ARRAY_NAMES = (
    "gamma_alpha",
    "gamma_beta",
    "same_alpha",
    "same_beta",
    "other_alpha",
    "other_beta",
    "canonical_alpha",
    "canonical_beta",
    "mixed",
    "pairs_alpha",
    "pairs_beta",
    "pairs_mixed",
)


def _array_shapes(m):

    return {
        "gamma_alpha": m * m,
        "gamma_beta": m * m,
        "same_alpha": m**3,
        "same_beta": m**3,
        "other_alpha": m**3,
        "other_beta": m**3,
        "canonical_alpha": m**4,
        "canonical_beta": m**4,
        "mixed": m**4,
        "pairs_alpha": m * m,
        "pairs_beta": m * m,
        "pairs_mixed": m * m,
    }


def _check_memory(cupy, comm, need):

    from .device import DEVICE_RESERVE_BYTES  # noqa: PLC0415

    free = int(cupy.cuda.runtime.memGetInfo()[0]) + int(
        cupy.get_default_memory_pool().free_bytes()
    )
    error = None
    if need > free - DEVICE_RESERVE_BYTES:
        error = (
            f"needs {need / 2**30:.2f} GiB of GPU memory, {max(0, free - DEVICE_RESERVE_BYTES) / 2**30:.2f} GiB "
            "is free"
        )
    raise_on_every_rank(comm, error, MemoryError, "full-string RDM memory check")


def _half_visits(ham):


    layout = ham._layout
    if not all(kind[0] == kind[2] and kind[1] == kind[3] for kind in layout.kinds):
        return 0
    return 2 if ham._block is None or ham._full is not None else 1


def _launch_pairs(ham, x_ext, arrays, stream, half=0):

    from . import kernel_launch  # noqa: PLC0415
    from .device import NUM_SLOTS  # noqa: PLC0415

    layout = ham._layout
    pointers = {name: int(arrays[name].data.ptr) for name in ARRAY_NAMES[:9]}
    halo = layout.halo_keys.data.ptr if layout.halo_keys.size else 0
    streamed = 0
    for chunk in range(len(layout.views)):
        classes, slot = layout._classes(chunk, stream, streamed % NUM_SLOTS)
        streamed += 0 if slot is None else 1
        ranges = tuple(layout.kinds[layout.unit_kind[chunk]][:4])
        for cls, view in enumerate(classes):
            if not view.num_members:
                continue
            for bucket in range(len(view.bucket_start) - 1):
                first = view.bucket_start[bucket]
                count = view.bucket_start[bucket + 1] - first
                if count > 0:
                    kernel_launch.launch_rdm_groups(
                        layout.keys.data.ptr,
                        halo,
                        layout.key_rows,
                        ham.num_orbitals,
                        view.members,
                        view.offsets,
                        first,
                        count,
                        cls,
                        ranges,
                        x_ext.data.ptr,
                        pointers,
                        stream.ptr,
                        bucket,
                        half,
                    )
        if slot is not None:
            slot.released = stream.record()


def _launch_diagonal(ham, x, arrays, stream):


    from . import kernel_launch  # noqa: PLC0415

    keys = ham._rdm_own_keys()
    for sectors, name in enumerate(("pairs_alpha", "pairs_beta", "pairs_mixed")):
        kernel_launch.launch_pair_occupation(
            keys.data.ptr,
            ham.n_local,
            x.data.ptr,
            arrays[name].data.ptr,
            sectors,
            ham.num_orbitals,
            stream.ptr,
        )
    stream.synchronize()


def _add_mirror(raw, m):


    for name in ("gamma_alpha", "gamma_beta"):
        g = raw[name].reshape(m, m)
        raw[name] = (g + g.T).reshape(-1)
    for name in ("same_alpha", "same_beta", "other_alpha", "other_beta"):
        d = raw[name].reshape(m, m, m)
        raw[name] = (d + d.transpose(1, 0, 2)).reshape(-1)
    for name in ("canonical_alpha", "canonical_beta", "mixed"):
        c = raw[name].reshape(m, m, m, m)
        raw[name] = (c + c.transpose(1, 0, 3, 2)).reshape(-1)


def _expand(raw, m):


    eye = np.arange(m)
    a, b, k = np.meshgrid(eye, eye, eye, indexing="ij")
    u, v = np.meshgrid(eye, eye, indexing="ij")
    off = u != v
    gammas, blocks = [], []
    for sector in ("alpha", "beta"):
        pairs = raw["pairs_" + sector].reshape(m, m)
        gamma = raw["gamma_" + sector].reshape(m, m).copy()
        gamma[eye, eye] += pairs[eye, eye]
        gammas.append(gamma)
        c = raw["canonical_" + sector].reshape(m, m, m, m)
        block = (
            c
            - c.transpose(0, 3, 2, 1)
            - c.transpose(2, 1, 0, 3)
            + c.transpose(2, 3, 0, 1)
        )
        d = raw["same_" + sector].reshape(m, m, m)
        np.add.at(block, (a, b, k, k), d)
        np.add.at(block, (a, k, k, b), -d)
        np.add.at(block, (k, b, a, k), -d)
        np.add.at(block, (k, k, a, b), d)
        np.add.at(block, (u[off], u[off], v[off], v[off]), pairs[off])
        np.add.at(block, (u[off], v[off], v[off], u[off]), -pairs[off])
        blocks.append(block)
    mixed = raw["mixed"].reshape(m, m, m, m).copy()
    np.add.at(mixed, (a, b, k, k), raw["other_alpha"].reshape(m, m, m))
    np.add.at(mixed, (k, k, a, b), raw["other_beta"].reshape(m, m, m))
    np.add.at(mixed, (u, u, v, v), raw["pairs_mixed"].reshape(m, m))
    return (gammas[0], gammas[1]), (blocks[0], mixed, blocks[1])


def compute_rdms_full_strs(ham, x, *, rank=1, spin_summed=False):
    """Compute reduced density matrices from distributed full-determinant subspace.

    This is the implementation used by ``FullStrsHamiltonian.compute_rdms``.
    Every rank in ``ham._comm`` must supply its local part of the same real
    state and use the same ``rank`` and ``spin_summed`` options. All ranks receive
    the full result as host NumPy ``float64`` arrays after GPU work and collective
    communication complete.

    Amplitude ``x[i]`` belongs to determinant ``ham.keys[i]``, whose two uint64
    words contain the alpha and beta occupations. Use this post-distribution
    local key order. Pass only this rank's owned amplitudes; the operator handles
    source-vector exchanges. The selected determinants need not form a complete
    tensor-product subspace. The operator supports 1–64 orbitals per spin sector.

    No normalization is applied. Scaling all amplitudes by a real factor scales
    every RDM by its square. For normalized RDMs, normalize the global state before
    distributing it. Spin and tensor-index conventions follow PySCF
    ``make_rdm12s(reorder=True)``.

    Args:
        ham: Constructed ``FullStrsHamiltonian`` supplying the determinant keys,
            local partition, communicator and GPU layout.
        x: Real NumPy or CuPy array of shape ``(ham.n_local,)`` in ``ham.keys``
            order. CuPy input must be on this rank's GPU. Pass one local
            eigenvector column when using distributed eigensolver results.
        rank: RDM order to return, 1 or 2; defaults to 1. This is not an MPI rank.
        spin_summed: Combine alpha and beta spin blocks; defaults to False.

    Returns:
        For ``rank=1``, ``(gamma_a, gamma_b)`` or spin-summed ``gamma``.
        For ``rank=2``, ``((gamma_a, gamma_b), (G_aa, G_ab, G_bb))`` or
        spin-summed ``(gamma, G)``. Each one-particle array has shape ``(m, m)``
        and each two-particle array ``(m, m, m, m)``, where ``m = ham.num_orbitals``
        is the common orbital count per spin sector. Results are replicated on
        every communicator rank.

    Raises:
        ImportError: CuPy or a required compiled CUDA component is unavailable.
        ValueError: The RDM order, local vector shape or orbital count is invalid.
        MemoryError: There is insufficient device memory for the RDM buffers.
        RuntimeError: GPU execution or collective communication fails.
    """
    import cupy  # noqa: PLC0415

    started = time.perf_counter()
    if rank not in (1, 2):
        raise ValueError(f"rank must be 1 or 2, got {rank!r}")
    m = int(ham.num_orbitals)
    if m > MAX_ORBITALS:
        raise ValueError(
            f"the RDM supports at most {MAX_ORBITALS} orbitals per sector, got {m}"
        )
    if tuple(x.shape) != (ham.n_local,):
        raise ValueError(f"x must have shape ({ham.n_local},); got {tuple(x.shape)}")
    comm = ham._comm
    mpi = mpi_namespace(comm)
    stream = ham._stream
    shapes = _array_shapes(m)
    _check_memory(cupy, comm, 8 * (sum(shapes.values()) + ham.n_local))
    half = _half_visits(ham)

    def allocate():

        with stream:
            state = cupy.ascontiguousarray(cupy.asarray(x, dtype=cupy.float64))
            arrays = {
                name: cupy.zeros(size, cupy.float64) for name, size in shapes.items()
            }
        stream.synchronize()
        return state, arrays

    state, arrays = run_on_every_rank(comm, "full-string RDM buffers", allocate)
    caller = cupy.cuda.get_current_stream()
    stream.wait_event(caller.record())
    x_ext = ham._rdm_input(state)

    def launch():

        _launch_pairs(ham, x_ext, arrays, stream, half)
        _launch_diagonal(ham, state, arrays, stream)

    run_on_every_rank(comm, "full-string RDM kernels", launch)
    raw = run_on_every_rank(
        comm,
        "full-string RDM copy",
        lambda: {
            name: np.ascontiguousarray(cupy.asnumpy(array))
            for name, array in arrays.items()
        },
    )
    for name in ARRAY_NAMES:
        comm.Allreduce(mpi.IN_PLACE, [raw[name], mpi.DOUBLE])
    if half:
        _add_mirror(raw, m)
    (gamma_a, gamma_b), (g_aa, g_ab, g_bb) = _expand(raw, m)
    if comm.Get_rank() == 0:
        logger.info(
            "RDM calculation completed: order=%d, elapsed=%.2f s",
            rank,
            time.perf_counter() - started,
            extra={"gpu_rank": 0},
        )
    if rank == 1:
        return (gamma_a + gamma_b) if spin_summed else (gamma_a, gamma_b)
    if spin_summed:
        return gamma_a + gamma_b, g_aa + g_bb + g_ab + g_ab.transpose(2, 3, 0, 1)
    return (gamma_a, gamma_b), (g_aa, g_ab, g_bb)
