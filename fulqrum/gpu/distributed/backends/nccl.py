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

import numpy as np

from .._collective import BACKEND_NCCL, _Backend, _sync_device

logger = logging.getLogger(__name__)


class _NcclBackend(_Backend):


    name = BACKEND_NCCL

    _NCCL_DOUBLE = None
    _NCCL_FLOAT = None
    _NCCL_INT32 = None
    _NCCL_UINT64 = None

    def __init__(self):


        self._world_comm = None
        self._world_nccl = None
        self._device_id = None
        self._sub_cache = {}
        self._bootstrap_seconds = None
        self._exchange_sync = False

    @classmethod
    def _import_nccl(cls):

        import cupy.cuda.nccl as nccl  # noqa: PLC0415

        if cls._NCCL_DOUBLE is None:
            cls._NCCL_DOUBLE = nccl.NCCL_FLOAT64
            cls._NCCL_FLOAT = nccl.NCCL_FLOAT32
            cls._NCCL_INT32 = nccl.NCCL_INT32
            cls._NCCL_UINT64 = nccl.NCCL_UINT64
        return nccl

    def init(self, comm) -> None:


        import time as _time  # noqa: PLC0415

        import cupy  # noqa: PLC0415

        from ...config import read  # noqa: PLC0415

        self._exchange_sync = bool(read("FQ_EXCHANGE_SYNC"))

        nccl = self._import_nccl()
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        cupy.cuda.Device().use()
        self._device_id = int(cupy.cuda.runtime.getDevice())

        _t_start = _time.perf_counter()

        if rank == 0:
            unique_id = nccl.get_unique_id()
        else:
            unique_id = tuple([0] * 128)
        unique_id = comm.bcast(unique_id, root=0)

        self._world_nccl = nccl.NcclCommunicator(world_size, unique_id, rank)
        self._world_comm = comm
        self._bootstrap_seconds = _time.perf_counter() - _t_start

        if rank == 0:
            try:
                nccl_version = nccl.get_version()
            except Exception:  # noqa: BLE001
                nccl_version = "unknown"

            gpudirect_hint = "unknown"
            try:
                if os.path.exists("/sys/module/nvidia_peermem"):
                    gpudirect_hint = "nvidia_peermem-loaded"
                elif os.path.exists("/sys/kernel/mm/nv_peer_mem"):
                    gpudirect_hint = "nv_peer_mem-loaded"
                else:
                    gpudirect_hint = "no-peermem-module"
            except Exception:  # noqa: BLE001
                gpudirect_hint = "check-failed"

            logger.debug(
                "NCCL initialized: version=%s, ranks=%s, bootstrap=%.3f s, peer_memory_hint=%s",
                nccl_version,
                world_size,
                self._bootstrap_seconds,
                gpudirect_hint,
                extra={"gpu_rank": 0},
            )

    def finalize(self) -> None:

        for _comm, sub in list(self._sub_cache.values()):
            try:
                sub.destroy()
            except Exception as error:  # noqa: BLE001
                logger.debug("destroying an NCCL sub-communicator failed: %s", error)
        self._sub_cache.clear()
        if self._world_nccl is not None:
            try:
                self._world_nccl.destroy()
            except Exception as error:  # noqa: BLE001
                logger.debug("destroying the NCCL world communicator failed: %s", error)
            self._world_nccl = None

    def _stream_ptr(self):

        import cupy  # noqa: PLC0415

        return int(cupy.cuda.get_current_stream().ptr)

    def _check_device(self) -> None:

        import cupy  # noqa: PLC0415

        current = int(cupy.cuda.runtime.getDevice())
        if self._device_id is not None and current != self._device_id:
            raise RuntimeError(
                f"NCCL communicator belongs to GPU {self._device_id}, "
                f"but current GPU is {current}"
            )

    def _get_nccl_for(self, comm):


        if self._world_nccl is None:
            raise RuntimeError("NCCL backend used before init()")
        self._check_device()
        if comm == self._world_comm:
            return self._world_nccl
        if (
            comm.Get_size() == self._world_comm.Get_size()
            and comm.Get_rank() == self._world_comm.Get_rank()
        ):
            pass

        key = id(comm)
        cached = self._sub_cache.get(key)
        if cached is not None:
            return cached[1]

        nccl = self._import_nccl()
        sub_size = comm.Get_size()
        sub_rank = comm.Get_rank()

        if sub_rank == 0:
            unique_id = nccl.get_unique_id()
        else:
            unique_id = tuple([0] * 128)
        unique_id = comm.bcast(unique_id, root=0)

        sub_nccl = nccl.NcclCommunicator(sub_size, unique_id, sub_rank)
        self._sub_cache[key] = (comm, sub_nccl)
        return sub_nccl

    def allgatherv_device(
        self,
        send_dev,
        recv_dev,
        counts,
        displs,
        comm,
    ) -> None:


        import cupy  # noqa: PLC0415

        self._import_nccl()
        sub_nccl = self._get_nccl_for(comm)
        sub_size = comm.Get_size()
        sub_rank = comm.Get_rank()

        counts_arr = np.asarray(counts, dtype=np.int64)
        displs_arr = np.asarray(displs, dtype=np.int64)

        if sub_size == 1:
            n = int(counts_arr[0])
            d = int(displs_arr[0])
            if n > 0:
                cupy.copyto(recv_dev[d : d + n], send_dev[:n])
            return

        if counts_arr.size != sub_size or displs_arr.size != sub_size:
            raise ValueError(
                f"allgatherv_device: counts/displs len ({counts_arr.size}, "
                f"{displs_arr.size}) must equal subgroup size {sub_size}"
            )

        if recv_dev.dtype != np.float64 or send_dev.dtype != np.float64:
            raise NotImplementedError(
                f"allgatherv_device: only float64 supported; got "
                f"send={send_dev.dtype}, recv={recv_dev.dtype}"
            )

        n_max = int(counts_arr.max())
        my_n = int(counts_arr[sub_rank])
        stream = self._stream_ptr()

        if my_n == n_max:
            pad_send = send_dev[:n_max]
        else:
            pad_send = cupy.empty(n_max, dtype=send_dev.dtype)
            if my_n > 0:
                cupy.copyto(pad_send[:my_n], send_dev[:my_n])
            if my_n < n_max:
                pad_send[my_n:].fill(0)

        pad_recv = cupy.empty(sub_size * n_max, dtype=recv_dev.dtype)

        sub_nccl.allGather(
            int(pad_send.data.ptr),
            int(pad_recv.data.ptr),
            n_max,
            self._NCCL_DOUBLE,
            stream,
        )

        for r in range(sub_size):
            n_r = int(counts_arr[r])
            d_r = int(displs_arr[r])
            if n_r > 0:
                cupy.copyto(
                    recv_dev[d_r : d_r + n_r],
                    pad_recv[r * n_max : r * n_max + n_r],
                )

        _sync_device()

    def allreduce_device(self, buf_dev, op, comm) -> None:


        from mpi4py import MPI  # noqa: PLC0415

        nccl = self._import_nccl()
        sub_nccl = self._get_nccl_for(comm)

        if op == MPI.SUM:
            nccl_op = nccl.NCCL_SUM
        elif op == MPI.MAX:
            nccl_op = nccl.NCCL_MAX
        elif op == MPI.MIN:
            nccl_op = nccl.NCCL_MIN
        elif op == MPI.PROD:
            nccl_op = nccl.NCCL_PROD
        else:
            raise NotImplementedError(f"NCCL allreduce: unsupported MPI op {op}")

        if buf_dev.dtype == np.float64:
            dtype = self._NCCL_DOUBLE
        elif buf_dev.dtype == np.float32:
            dtype = self._NCCL_FLOAT
        else:
            raise NotImplementedError(
                f"NCCL allreduce: unsupported dtype {buf_dev.dtype}"
            )

        ptr = int(buf_dev.data.ptr)
        count = int(buf_dev.size)
        sub_nccl.allReduce(ptr, ptr, count, dtype, nccl_op, self._stream_ptr())
        _sync_device()

    def allgather_device(
        self, send_dev, recv_dev, count: int, comm, stream=None
    ) -> None:


        sub_nccl = self._get_nccl_for(comm)
        if send_dev.dtype != np.float64 or recv_dev.dtype != np.float64:
            raise NotImplementedError(
                f"allgather_device: only float64 supported; got "
                f"send={send_dev.dtype}, recv={recv_dev.dtype}"
            )
        n = int(count)
        if int(send_dev.size) < n:
            raise ValueError(
                f"allgather_device: send buffer holds {send_dev.size} < count {n}"
            )
        need = n * comm.Get_size()
        if int(recv_dev.size) < need:
            raise ValueError(
                f"allgather_device: recv buffer holds {recv_dev.size} < "
                f"count*size {need}"
            )
        if stream is not None:
            stream_ptr = int(stream.ptr) if hasattr(stream, "ptr") else int(stream)
        else:
            stream_ptr = self._stream_ptr()
        sub_nccl.allGather(
            int(send_dev.data.ptr),
            int(recv_dev.data.ptr),
            n,
            self._NCCL_DOUBLE,
            stream_ptr,
        )
        if self._exchange_sync:
            _sync_device()

    def reduce_scatter_device(
        self, send_dev, recv_dev, count: int, comm, stream=None
    ) -> None:


        nccl = self._import_nccl()
        sub_nccl = self._get_nccl_for(comm)
        if send_dev.dtype != np.float64 or recv_dev.dtype != np.float64:
            raise NotImplementedError(
                f"reduce_scatter_device: only float64 supported; got "
                f"send={send_dev.dtype}, recv={recv_dev.dtype}"
            )
        n = int(count)
        need = n * comm.Get_size()
        if int(send_dev.size) < need:
            raise ValueError(
                f"reduce_scatter_device: send buffer holds {send_dev.size} < count*size {need}"
            )
        if int(recv_dev.size) < n:
            raise ValueError(
                f"reduce_scatter_device: recv buffer holds {recv_dev.size} < count {n}"
            )
        if stream is not None:
            stream_ptr = int(stream.ptr) if hasattr(stream, "ptr") else int(stream)
        else:
            stream_ptr = self._stream_ptr()
        sub_nccl.reduceScatter(
            int(send_dev.data.ptr),
            int(recv_dev.data.ptr),
            n,
            self._NCCL_DOUBLE,
            nccl.NCCL_SUM,
            stream_ptr,
        )
        if self._exchange_sync:
            _sync_device()

    def broadcast_device(self, buf_dev, root: int, comm, stream=None) -> None:


        sub_nccl = self._get_nccl_for(comm)
        if buf_dev.dtype != np.float64:
            raise NotImplementedError(
                f"broadcast_device: only float64 supported; got {buf_dev.dtype}"
            )
        if stream is not None:
            stream_ptr = int(stream.ptr) if hasattr(stream, "ptr") else int(stream)
        else:
            stream_ptr = self._stream_ptr()
        sub_nccl.bcast(
            int(buf_dev.data.ptr),
            int(buf_dev.size),
            self._NCCL_DOUBLE,
            int(root),
            stream_ptr,
        )
        if self._exchange_sync:
            _sync_device()

    def sendrecv_replace_device(self, buf_dev, peer: int, tag: int, comm) -> None:


        raise NotImplementedError(
            "NCCL sendrecv_replace_device is not implemented; ring-shift uses "
            "ring_shift_device with separate send/recv buffers."
        )

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


        del send_pinned, recv_pinned

        nccl = self._import_nccl()
        sub_nccl = self._get_nccl_for(comm)

        if send_dev.dtype != np.float64 or recv_dev.dtype != np.float64:
            raise NotImplementedError(
                f"ring_shift_device: only float64 supported; got "
                f"send={send_dev.dtype}, recv={recv_dev.dtype}"
            )
        if stream is not None:
            stream_ptr = int(stream.ptr) if hasattr(stream, "ptr") else int(stream)
        else:
            stream_ptr = self._stream_ptr()
        send_ptr = int(send_dev.data.ptr)
        recv_ptr = int(recv_dev.data.ptr)
        send_count = int(send_dev.size)
        recv_count = int(recv_dev.size)

        nccl.groupStart()
        sub_nccl.send(send_ptr, send_count, self._NCCL_DOUBLE, dst_peer, stream_ptr)
        sub_nccl.recv(recv_ptr, recv_count, self._NCCL_DOUBLE, src_peer, stream_ptr)
        nccl.groupEnd()
        if self._exchange_sync:
            _sync_device()

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


        del send_pinned, recv_pinned

        nccl = self._import_nccl()
        sub_nccl = self._get_nccl_for(comm)

        if send_dev.dtype != np.float64 or recv_dev.dtype != np.float64:
            raise NotImplementedError("grouped_sendrecv_device: only float64 supported")

        send_counts = np.asarray(send_counts, dtype=np.int64)
        send_displs = np.asarray(send_displs, dtype=np.int64)
        recv_counts = np.asarray(recv_counts, dtype=np.int64)
        recv_displs = np.asarray(recv_displs, dtype=np.int64)

        if stream is not None:
            stream_ptr = int(stream.ptr) if hasattr(stream, "ptr") else int(stream)
        else:
            stream_ptr = self._stream_ptr()
        send_ptr = int(send_dev.data.ptr)
        recv_ptr = int(recv_dev.data.ptr)

        nccl.groupStart()
        for j in range(len(src_ranks)):
            cnt = int(recv_counts[j])
            if cnt == 0:
                continue
            displ = int(recv_displs[j])
            sub_nccl.recv(
                recv_ptr + displ * 8,
                cnt,
                self._NCCL_DOUBLE,
                int(src_ranks[j]),
                stream_ptr,
            )
        for i in range(len(dst_ranks)):
            cnt = int(send_counts[i])
            if cnt == 0:
                continue
            displ = int(send_displs[i])
            sub_nccl.send(
                send_ptr + displ * 8,
                cnt,
                self._NCCL_DOUBLE,
                int(dst_ranks[i]),
                stream_ptr,
            )
        nccl.groupEnd()
        if self._exchange_sync:
            _sync_device()
