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
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

BACKEND_HOST_MPI = "host_mpi"
BACKEND_CUDA_MPI = "cuda_mpi"
BACKEND_NCCL = "nccl"
BACKEND_INPROC = "inproc"

_VALID_ENV_VALUES = (BACKEND_HOST_MPI, BACKEND_CUDA_MPI, BACKEND_NCCL, "auto")

_active: Optional["_Backend"] = None
_init_logged: bool = False
_backend_reason: str = ""
_backends: dict = {}


class _Backend:


    name: str = "abstract"

    needs_pinned_staging: bool = False

    uses_caller_pinned: bool = False

    def init(self, comm) -> None:  # noqa: D401


        return None

    def finalize(self) -> None:

        return None

    def sendrecv_replace_device(self, buf_dev, peer: int, tag: int, comm) -> None:


        raise NotImplementedError

    def allreduce_device(self, buf_dev, op, comm) -> None:


        raise NotImplementedError

    def allgather_device(
        self, send_dev, recv_dev, count: int, comm, stream=None
    ) -> None:


        raise NotImplementedError

    def broadcast_device(self, buf_dev, root: int, comm, stream=None) -> None:


        raise NotImplementedError

    def allgatherv_device(
        self,
        send_dev,
        recv_dev,
        counts,
        displs,
        comm,
    ) -> None:


        raise NotImplementedError

    def ring_shift_device(
        self,
        send_dev,
        recv_dev,
        dst_peer: int,
        src_peer: int,
        comm,
        tag: int = 0,
        stream=None,
        send_pinned=None,
        recv_pinned=None,
    ) -> None:


        raise NotImplementedError

    def grouped_sendrecv_device(
        self,
        send_dev,
        send_counts,
        send_displs,
        dst_ranks,
        recv_dev,
        recv_counts,
        recv_displs,
        src_ranks,
        comm,
        stream=None,
        send_pinned=None,
        recv_pinned=None,
    ) -> None:


        raise NotImplementedError

    def barrier(self, comm) -> None:


        comm.Barrier()

    def comm_split_type_node(self, comm):


        from mpi4py import MPI  # noqa: PLC0415

        return comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.Get_rank())


def _overlap_disabled() -> bool:


    import os  # noqa: PLC0415

    return os.environ.get("_FULQRUM_NO_OVERLAP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


MPI_CHUNK = 1 << 28
"""The largest element count of one MPI message; longer messages are sent in pieces (int32 counts)."""

_INT32_MAX = 2**31 - 1


def _chunks(count: int):

    chunk = MPI_CHUNK
    for start in range(0, int(count), chunk):
        yield start, min(chunk, int(count) - start)


def _chunked_sendrecv(comm, send, dst: int, recv, src: int, tag: int) -> None:


    if send.size <= MPI_CHUNK and recv.size <= MPI_CHUNK:
        comm.Sendrecv(
            sendbuf=send, dest=dst, sendtag=tag, recvbuf=recv, source=src, recvtag=tag
        )
        return
    if send.size != recv.size:
        raise ValueError(
            f"a Sendrecv over {MPI_CHUNK} elements needs equal sizes, got {send.size} and {recv.size}"
        )
    for start, size in _chunks(send.size):
        comm.Sendrecv(
            sendbuf=send[start : start + size],
            dest=dst,
            sendtag=tag,
            recvbuf=recv[start : start + size],
            source=src,
            recvtag=tag,
        )


def _chunked_sendrecv_replace(comm, buf, peer: int, tag: int) -> None:

    for start, size in _chunks(buf.size):
        comm.Sendrecv_replace(
            buf[start : start + size], dest=peer, source=peer, sendtag=tag, recvtag=tag
        )


def _chunked_allreduce(comm, buf, op, in_place) -> None:

    for start, size in _chunks(buf.size):
        comm.Allreduce(in_place, buf[start : start + size], op=op)


def _post_chunked(
    comm, sending: bool, buf, displ: int, count: int, peer: int, double
) -> list:


    requests = []
    for start, size in _chunks(count):
        piece = [buf[displ + start : displ + start + size], size, double]
        if sending:
            requests.append(comm.Isend(buf=piece, dest=peer, tag=0))
        else:
            requests.append(comm.Irecv(buf=piece, source=peer, tag=0))
    return requests


def _chunked_allgatherv(comm, send, recv, counts, displs, double) -> None:


    counts = [int(c) for c in counts]
    displs = [int(d) for d in displs]
    ends = [d + c for c, d in zip(counts, displs)]
    if max(counts, default=0) <= MPI_CHUNK and max(ends, default=0) <= _INT32_MAX:
        comm.Allgatherv(send, [recv, counts, displs, double])
        return
    rank = comm.Get_rank()
    for root, (count, displ) in enumerate(zip(counts, displs)):
        for start, size in _chunks(count):
            piece = recv[displ + start : displ + start + size]
            if root == rank:
                piece[...] = send[start : start + size]
            comm.Bcast(piece, root=root)


def _empty_pinned(shape, dtype):


    try:
        import cupyx  # noqa: PLC0415

        return cupyx.empty_pinned(shape, dtype)
    except Exception as error:  # noqa: BLE001
        logger.debug("Pinned host allocation failed; using pageable memory: %s", error)
        return np.empty(shape, dtype=dtype)


def _fresh_recv_host(recv_dev, pinned=False, pinned_buf=None):


    try:
        import cupy  # noqa: PLC0415

        recv_is_device = isinstance(recv_dev, cupy.ndarray)
    except ImportError:
        recv_is_device = False
    if recv_is_device:
        if pinned:
            if pinned_buf is not None:
                return pinned_buf, True
            return _empty_pinned(recv_dev.shape, recv_dev.dtype), True
        return np.empty(recv_dev.shape, dtype=recv_dev.dtype), True
    return recv_dev, False


def _sync_device() -> None:


    try:
        import cupy  # noqa: PLC0415
    except ImportError:
        return
    cupy.cuda.runtime.deviceSynchronize()


def _resolve_backend_name(comm, requested=None) -> str:


    global _backend_reason
    if requested is None:
        from ..config import read  # noqa: PLC0415

        requested = read("FQ_DIST_BACKEND")
    env = str(requested).strip().lower()
    if env not in _VALID_ENV_VALUES:
        raise ValueError(f"FQ_DIST_BACKEND={env!r} is not one of {_VALID_ENV_VALUES}")
    _backend_reason = f"FQ_DIST_BACKEND={env}"
    if env == BACKEND_NCCL:
        try:
            import cupy.cuda.nccl as _nccl_probe  # noqa: PLC0415,F401
        except ImportError as e:  # noqa: F841
            raise RuntimeError(
                "FQ_DIST_BACKEND=nccl requested but cupy.cuda.nccl "
                "is not importable. Reinstall cupy with NCCL support "
                "(the default cupy build bundles NCCL -- this is the "
                "rare hand-built-without-NCCL case)."
            ) from e
        return BACKEND_NCCL
    if env == BACKEND_HOST_MPI:
        return BACKEND_HOST_MPI
    if env == BACKEND_CUDA_MPI:
        if not _probe_cuda_aware_mpi(comm):
            raise RuntimeError(
                "FQ_DIST_BACKEND=cuda_mpi was requested but the "
                "cuda-aware MPI probe failed (cupy unavailable, mpi4py "
                "rejected the device pointer, or the post-Allreduce "
                "value was wrong). Either install/enable a cuda-aware "
                "MPI build or use "
                "FQ_DIST_BACKEND=auto for silent fallback to "
                "host_mpi."
            )
        return BACKEND_CUDA_MPI
    name, _backend_reason = choose_auto_backend(comm.allgather(_auto_backend_facts()))
    return name


def _auto_backend_facts():


    import socket  # noqa: PLC0415

    host = socket.gethostname()
    try:
        import cupy  # noqa: PLC0415
        import cupy.cuda.nccl  # noqa: PLC0415,F401

        device = cupy.cuda.runtime.getDevice()
        uuid = bytes(cupy.cuda.runtime.getDeviceProperties(device)["uuid"]).hex()
    except Exception as error:  # noqa: BLE001
        logger.debug("NCCL is unavailable on this rank: %s", error)
        return False, host, ""
    return True, host, uuid


def choose_auto_backend(facts):


    if len(facts) == 1:
        return BACKEND_HOST_MPI, "one rank: no tile exchange"
    if not all(importable and uuid for importable, _host, uuid in facts):
        return BACKEND_HOST_MPI, "NCCL or a CUDA device is not available on every rank"
    devices = [(host, uuid) for _importable, host, uuid in facts]
    if len(set(devices)) != len(devices):
        return (
            BACKEND_HOST_MPI,
            "two or more ranks share one GPU, which NCCL does not support",
        )
    return BACKEND_NCCL, "NCCL is available and every rank has its own GPU"


def _mpi_advertises_cuda() -> bool:


    try:
        from mpi4py import MPI  # noqa: PLC0415
    except ImportError:
        return False
    try:
        ver = MPI.Get_library_version().lower()
    except Exception as error:  # noqa: BLE001
        logger.debug("Unable to read the MPI library version: %s", error)
        ver = ""
    if "cuda" in ver or "gpu" in ver:
        return True
    if os.environ.get("OMPI_MCA_opal_cuda_support", "").lower() in ("1", "true"):
        return True
    if os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "0") == "1":
        return True
    return False


def _probe_cuda_aware_mpi(comm) -> bool:


    from mpi4py import MPI  # noqa: PLC0415

    try:
        import cupy  # noqa: PLC0415
    except ImportError:
        cupy = None
    if not all(comm.allgather(cupy is not None and _mpi_advertises_cuda())):
        return False
    try:
        d = cupy.ones(1, dtype=cupy.float64)
        comm.Allreduce(MPI.IN_PLACE, d, op=MPI.SUM)
        cupy.cuda.runtime.deviceSynchronize()
        passed = int(d.get()[0]) == comm.Get_size()
    except Exception as error:  # noqa: BLE001
        logger.debug("CUDA-aware MPI probe failed: %s", error)
        passed = False
    return all(comm.allgather(passed))


def _make_backend(name: str) -> _Backend:


    if name == BACKEND_HOST_MPI:
        from .backends.host_mpi import _HostMpiBackend  # noqa: PLC0415

        return _HostMpiBackend()
    if name == BACKEND_CUDA_MPI:
        from .backends.cuda_mpi import _CudaMpiBackend  # noqa: PLC0415

        return _CudaMpiBackend()
    if name == BACKEND_NCCL:
        from .backends.nccl import _NcclBackend  # noqa: PLC0415

        return _NcclBackend()
    if name == BACKEND_INPROC:
        from .backends.inproc import _InProcessBackend  # noqa: PLC0415

        return _InProcessBackend()
    raise ValueError(f"unknown backend name {name!r}")


def _current_device() -> int:

    try:
        import cupy  # noqa: PLC0415

        return int(cupy.cuda.runtime.getDevice())
    except Exception:  # noqa: BLE001
        return -1


def backend_for(comm, requested=None):


    global _active, _init_logged
    from .local_comm import LocalComm  # noqa: PLC0415

    if isinstance(comm, LocalComm):
        if requested not in (None, "auto", BACKEND_INPROC):
            raise ValueError(
                f"in-process ranks use the {BACKEND_INPROC} backend, not {requested!r}"
            )
        return _make_backend(BACKEND_INPROC), "in-process ranks (one thread per GPU)"
    if requested == BACKEND_INPROC:
        raise ValueError(
            f"the {BACKEND_INPROC} backend needs in-process ranks (a LocalComm), not an MPI communicator"
        )
    name = _resolve_backend_name(comm, requested)
    reason = _backend_reason
    key = (name, id(comm), _current_device())
    entry = _backends.get(key)
    if entry is None:
        backend = _make_backend(name)
        backend.init(comm)
        _backends[key] = (comm, backend)
        if not _init_logged and comm.Get_rank() == 0:
            logger.debug(
                "GPU communication backend: %s. Selection reason: %s",
                name,
                reason,
                extra={"gpu_rank": 0},
            )
            _init_logged = True
    else:
        backend = entry[1]
        check_device = getattr(backend, "_check_device", None)
        if check_device is not None:
            check_device()
    _active = backend
    return backend, reason


def init(comm, requested=None) -> str:


    return backend_for(comm, requested)[0].name


def get_backend_name() -> str:


    if _active is None:
        raise RuntimeError("_collective.init(comm) must be called first")
    return _active.name


def needs_pinned_staging() -> bool:


    return _active is not None and bool(_active.needs_pinned_staging)


def get_backend_reason() -> str:


    return _backend_reason


def is_initialized() -> bool:


    return _active is not None


def reset_for_testing() -> None:


    global _active, _init_logged
    _active = None
    _init_logged = False
    _backends.clear()


def sendrecv_replace_device(buf_dev, peer: int, tag: int, comm) -> None:


    if _active is None:
        raise RuntimeError("_collective.init(comm) must be called first")
    _active.sendrecv_replace_device(buf_dev, peer, tag, comm)


def allreduce_device(buf_dev, op, comm) -> None:


    if _active is None:
        raise RuntimeError("_collective.init(comm) must be called first")
    _active.allreduce_device(buf_dev, op, comm)


def allgather_device(send_dev, recv_dev, count: int, comm, stream=None) -> None:


    if _active is None:
        raise RuntimeError("_collective.init(comm) must be called first")
    _active.allgather_device(send_dev, recv_dev, count, comm, stream=stream)


def has_allgather_device() -> bool:


    if _active is None:
        return False
    return type(_active).allgather_device is not _Backend.allgather_device


def broadcast_device(buf_dev, root: int, comm, stream=None) -> None:


    if _active is None:
        raise RuntimeError("_collective.init(comm) must be called first")
    _active.broadcast_device(buf_dev, root, comm, stream=stream)


def has_broadcast_device() -> bool:


    if _active is None:
        return False
    return type(_active).broadcast_device is not _Backend.broadcast_device


def allgatherv_device(send_dev, recv_dev, counts, displs, comm) -> None:


    if _active is None:
        raise RuntimeError("_collective.init(comm) must be called first")
    _active.allgatherv_device(send_dev, recv_dev, counts, displs, comm)


def ring_shift_device(
    send_dev,
    recv_dev,
    dst_peer: int,
    src_peer: int,
    comm,
    tag: int = 0,
    stream=None,
    send_pinned=None,
    recv_pinned=None,
) -> None:


    if _active is None:
        raise RuntimeError("_collective.init(comm) must be called first")
    _active.ring_shift_device(
        send_dev,
        recv_dev,
        dst_peer,
        src_peer,
        comm,
        tag,
        stream=stream,
        send_pinned=send_pinned,
        recv_pinned=recv_pinned,
    )


def grouped_sendrecv_device(
    send_dev,
    send_counts,
    send_displs,
    dst_ranks,
    recv_dev,
    recv_counts,
    recv_displs,
    src_ranks,
    comm,
    stream=None,
    send_pinned=None,
    recv_pinned=None,
) -> None:


    if _active is None:
        raise RuntimeError("_collective.init(comm) must be called first")
    _active.grouped_sendrecv_device(
        send_dev,
        send_counts,
        send_displs,
        dst_ranks,
        recv_dev,
        recv_counts,
        recv_displs,
        src_ranks,
        comm,
        stream=stream,
        send_pinned=send_pinned,
        recv_pinned=recv_pinned,
    )


def barrier(comm) -> None:


    if _active is None:
        raise RuntimeError("_collective.init(comm) must be called first")
    _active.barrier(comm)


def comm_split_type_node(comm):


    if _active is None:
        raise RuntimeError("_collective.init(comm) must be called first")
    return _active.comm_split_type_node(comm)
