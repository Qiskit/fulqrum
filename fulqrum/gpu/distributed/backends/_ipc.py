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

import socket

import numpy as np

from ..rank_agreement import raise_on_every_rank

_HANDLE_BYTES = 64
_DESCRIPTOR = np.dtype(
    [("handle", np.uint8, _HANDLE_BYTES), ("offset", np.int64), ("nbytes", np.int64)]
)
_DESCRIPTOR_TAG = 1 << 20
_DONE_TAG = 2 << 20


class IpcPeers:


    def __init__(self, enabled: bool, node_of, device_of, reason: str):

        self.enabled = enabled
        self.node_of = node_of
        self.device_of = device_of
        self.reason = reason
        self._opened = {}
        self._world = None

    @classmethod
    def discover(cls, comm) -> "IpcPeers":


        from ...config import read  # noqa: PLC0415

        want = bool(read("FQ_HOST_MPI_USE_CUDA_IPC"))
        wants = comm.allgather(want)
        if not any(wants):
            return cls(False, None, None, "FQ_HOST_MPI_USE_CUDA_IPC is off")
        raise_on_every_rank(
            comm,
            None
            if all(wants)
            else f"FQ_HOST_MPI_USE_CUDA_IPC differs across ranks: {wants}",
            what="host_mpi IPC settings",
        )
        import cupy  # noqa: PLC0415
        from mpi4py import MPI  # noqa: PLC0415

        if comm.Get_size() != MPI.COMM_WORLD.Get_size():
            return cls(
                False,
                None,
                None,
                "the backend communicator is not the world communicator",
            )
        tag_ub = MPI.COMM_WORLD.Get_attr(MPI.TAG_UB)
        if tag_ub is None or int(tag_ub) < _DONE_TAG + (1 << 16):
            return cls(False, None, None, f"MPI_TAG_UB {tag_ub} is below the IPC tags")
        rt = cupy.cuda.runtime
        device = int(rt.getDevice())
        error, handle, probe = None, None, None
        try:
            probe = cupy.full(1, float(comm.Get_rank()), dtype=cupy.float64)
            rt.deviceSynchronize()
            handle = bytes(rt.ipcGetMemHandle(probe.data.ptr))
        except Exception as exc:  # noqa: BLE001
            error = f"ipcGetMemHandle: {exc}"
        node = socket.gethostname()
        facts = comm.allgather((node, device, handle, error))
        node_of = [f[0] for f in facts]
        device_of = [f[1] for f in facts]
        me = comm.Get_rank()
        problems = [f"rank {r}: {f[3]}" for r, f in enumerate(facts) if f[3]]
        if not problems:
            same = [r for r in range(len(facts)) if node_of[r] == node and r != me]
            devices_here = [device_of[r] for r in same] + [device]
            if len(set(devices_here)) != len(devices_here):
                problems.append(
                    f"rank {me}: ranks of node {node} share a GPU {sorted(devices_here)}"
                )
            for peer in same:
                if problems:
                    break
                try:
                    if not rt.deviceCanAccessPeer(device, device_of[peer]):
                        problems.append(
                            f"rank {me}: no peer access to GPU {device_of[peer]}"
                        )
                        break
                    ptr = rt.ipcOpenMemHandle(
                        facts[peer][2], rt.cudaIpcMemLazyEnablePeerAccess
                    )
                    got = cupy.empty(1, dtype=cupy.float64)
                    rt.memcpyPeer(got.data.ptr, device, ptr, device_of[peer], 8)
                    rt.ipcCloseMemHandle(ptr)
                    if float(got.get()[0]) != float(peer):
                        problems.append(
                            f"rank {me}: wrong data through IPC from rank {peer}"
                        )
                except Exception as exc:  # noqa: BLE001
                    problems.append(
                        f"rank {me}: open or copy from rank {peer}: {type(exc).__name__}: {exc}"
                    )
        problems = sorted({p for ps in comm.allgather(problems) for p in ps})
        comm.Barrier()
        del probe
        if problems:
            return cls(False, None, None, "probe failed: " + "; ".join(problems[:3]))
        return cls(
            True,
            node_of,
            device_of,
            f"{len(set(node_of))} node(s), peer copies through NVLink/P2P",
        )

    def report(self) -> str:

        return f"ipc {'on' if self.enabled else 'off'} ({self.reason})"

    def world_ranks(self, comm, peers):

        from mpi4py import MPI  # noqa: PLC0415

        if self._world is None:
            self._world = MPI.COMM_WORLD.Get_group()
        return comm.Get_group().Translate_ranks(list(peers), self._world)

    def same_node(self, world_peer: int) -> bool:

        from mpi4py import MPI  # noqa: PLC0415

        return (
            self.enabled
            and self.node_of[world_peer] == self.node_of[MPI.COMM_WORLD.Get_rank()]
        )

    def open(self, world_peer: int, handle: bytes) -> int:

        import cupy  # noqa: PLC0415

        rt = cupy.cuda.runtime
        cached = self._opened.get(world_peer)
        if cached is not None and cached[0] == handle:
            return cached[1]
        if cached is not None:
            rt.ipcCloseMemHandle(cached[1])
        ptr = int(rt.ipcOpenMemHandle(handle, rt.cudaIpcMemLazyEnablePeerAccess))
        self._opened[world_peer] = (handle, ptr)
        return ptr

    def close_all(self) -> None:

        import cupy  # noqa: PLC0415

        for _handle, ptr in self._opened.values():
            try:
                cupy.cuda.runtime.ipcCloseMemHandle(ptr)
            except Exception:  # noqa: BLE001
                pass
        self._opened.clear()


_LIBCUDA = None


def _allocation_base(address: int) -> int:


    import ctypes  # noqa: PLC0415

    global _LIBCUDA  # noqa: PLW0603
    if _LIBCUDA is None:
        _LIBCUDA = ctypes.CDLL("libcuda.so.1")
        _LIBCUDA.cuMemGetAddressRange_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_uint64,
        ]
    base, size = ctypes.c_uint64(0), ctypes.c_size_t(0)
    status = _LIBCUDA.cuMemGetAddressRange_v2(
        ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(address)
    )
    if status != 0:
        raise RuntimeError(
            f"cuMemGetAddressRange failed with CUresult {status} for address {address:#x}"
        )
    return int(base.value)


def describe(buffer, first: int, count: int) -> np.ndarray:

    import cupy  # noqa: PLC0415

    rt = cupy.cuda.runtime
    address = int(buffer.data.ptr) + first * buffer.itemsize
    base = _allocation_base(address)
    record = np.zeros(1, dtype=_DESCRIPTOR)
    record["handle"][0] = np.frombuffer(bytes(rt.ipcGetMemHandle(base)), dtype=np.uint8)
    record["offset"][0] = address - base
    record["nbytes"][0] = count * buffer.itemsize
    return record
