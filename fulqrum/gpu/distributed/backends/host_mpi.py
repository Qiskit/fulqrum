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

import sys
import time

import numpy as np

from .. import _collective as _col
from .._collective import (
    BACKEND_HOST_MPI,
    _Backend,
    _chunked_allgatherv,
    _chunked_allreduce,
    _chunked_sendrecv_replace,
    _overlap_disabled,
)
from ..rank_agreement import raise_on_every_rank

DEFAULT_CHUNK_MB = 256
"""Size of one piece (``FQ_HOST_MPI_CHUNK_MiB``); every rank must use the same value."""

DEFAULT_TIMEOUT_S = 900.0
"""Seconds without progress before a transfer raises (``FQ_HOST_MPI_TIMEOUT``; 0 waits forever)."""

SLOTS = 2
"""Pieces in flight per peer and direction: one copies while the other is on the network."""

POOL_CAP_BYTES = 8 << 30
"""Most pinned memory one call may stage through; with many peers each peer gets fewer, smaller pieces."""

_ALIGN = 256
_HOST_TO_DEVICE = 1
_DEVICE_TO_HOST = 2


class _Pool:


    def __init__(self):
        self.memory = None
        self.uploads = None
        self.streams = {}

    def take(self, nbytes: int):

        import cupyx  # noqa: PLC0415

        if self.uploads is not None:
            self.uploads.synchronize()
            self.uploads = None
        if self.memory is None or self.memory.nbytes < nbytes:
            self.memory = None
            self.memory = cupyx.empty_pinned(
                (max(int(nbytes), _ALIGN),), dtype=np.uint8
            )
        return self.memory

    def copy_streams(self):

        import cupy  # noqa: PLC0415

        device = int(cupy.cuda.runtime.getDevice())
        if device not in self.streams:
            self.streams[device] = (
                cupy.cuda.Stream(non_blocking=True),
                cupy.cuda.Stream(non_blocking=True),
            )
        return self.streams[device]


_POOL = _Pool()


def _is_device(buffer) -> bool:

    try:
        import cupy  # noqa: PLC0415
    except ImportError:
        return False
    return isinstance(buffer, cupy.ndarray)


class _Lane:


    def __init__(self, sending: bool, peer: int):
        self.sending = sending
        self.peer = peer
        self.pieces = []
        self.posted = 0
        self.sent = 0
        self.done = 0
        self.slots = []
        self.device = False
        self.requests = []

    def state(self) -> str:

        waiting = sorted({slot[2] for slot in self.slots if slot[2] != "free"})
        return (
            f"{'send to' if self.sending else 'receive from'} rank {self.peer}: "
            f"{self.done}/{len(self.pieces)} pieces done, waiting on {', '.join(waiting) or 'nothing'}"
        )


class _HostMpiBackend(_Backend):


    name = BACKEND_HOST_MPI
    needs_pinned_staging = True
    uses_caller_pinned = False

    def __init__(self):

        self._piece_bytes = DEFAULT_CHUNK_MB << 20
        self._timeout = DEFAULT_TIMEOUT_S
        self._ipc = None

    def init(self, comm) -> None:


        from ...config import read  # noqa: PLC0415

        error = None
        try:
            chunk_mb = float(read("FQ_HOST_MPI_CHUNK_MiB"))
            timeout = float(read("FQ_HOST_MPI_TIMEOUT"))
            self._piece_bytes = max(
                _ALIGN, int(chunk_mb * (1 << 20)) // _ALIGN * _ALIGN
            )
            self._timeout = timeout
        except ValueError as exc:
            error = str(exc)
        raise_on_every_rank(comm, error, what="host_mpi settings")
        sizes = comm.allgather(self._piece_bytes)
        raise_on_every_rank(
            comm,
            None
            if len(set(sizes)) == 1
            else f"FQ_HOST_MPI_CHUNK_MiB differs across ranks: "
            f"{sorted(set(sizes))} bytes",
            what="host_mpi settings",
        )
        from ._ipc import IpcPeers  # noqa: PLC0415

        self._ipc = IpcPeers.discover(comm)

    def ipc_report(self) -> str:

        return "ipc off" if self._ipc is None else self._ipc.report()

    @staticmethod
    def _to_host(buf_dev):


        if not _is_device(buf_dev):
            return np.asarray(buf_dev), False
        import cupy  # noqa: PLC0415

        cupy.cuda.runtime.deviceSynchronize()
        return cupy.asnumpy(buf_dev), True

    @staticmethod
    def _from_host(buf_dev, host, is_device):


        if not is_device:
            return
        import cupy  # noqa: PLC0415

        buf_dev.set(host)
        cupy.cuda.runtime.deviceSynchronize()

    def sendrecv_replace_device(self, buf_dev, peer: int, tag: int, comm) -> None:


        host, is_device = self._to_host(buf_dev)
        _chunked_sendrecv_replace(comm, host, peer, tag)
        self._from_host(buf_dev, host, is_device)

    def allreduce_device(self, buf_dev, op, comm) -> None:


        from mpi4py import MPI  # noqa: PLC0415

        host, is_device = self._to_host(buf_dev)
        _chunked_allreduce(comm, host, op, MPI.IN_PLACE)
        self._from_host(buf_dev, host, is_device)

    def allgatherv_device(
        self,
        send_dev,
        recv_dev,
        counts,
        displs,
        comm,
    ) -> None:


        from mpi4py import MPI  # noqa: PLC0415

        send_host, _ = self._to_host(send_dev)
        recv_host = (
            np.empty(recv_dev.shape, dtype=recv_dev.dtype)
            if _is_device(recv_dev)
            else recv_dev
        )
        _chunked_allgatherv(comm, send_host, recv_host, counts, displs, MPI.DOUBLE)
        self._from_host(recv_dev, recv_host, _is_device(recv_dev))

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
        if np.dtype(send_dev.dtype) != np.dtype(recv_dev.dtype):
            raise TypeError(
                f"host_mpi moves bytes: the send ({send_dev.dtype}) and receive ({recv_dev.dtype}) "
                "buffers need the same dtype"
            )
        self._transfer(
            comm,
            [(int(dst_peer), send_dev, 0, int(send_dev.size))],
            [(int(src_peer), recv_dev, 0, int(recv_dev.size))],
            stream,
            int(tag),
        )

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
        if np.dtype(send_dev.dtype) != np.dtype(recv_dev.dtype):
            raise TypeError(
                f"host_mpi moves bytes: the send ({send_dev.dtype}) and receive ({recv_dev.dtype}) "
                "buffers need the same dtype"
            )
        sends = [
            (int(peer), send_dev, int(displ), int(count))
            for peer, displ, count in zip(dst_ranks, send_displs, send_counts)
            if int(count) > 0
        ]
        receives = [
            (int(peer), recv_dev, int(displ), int(count))
            for peer, displ, count in zip(src_ranks, recv_displs, recv_counts)
            if int(count) > 0
        ]
        self._transfer(comm, sends, receives, stream, 0)

    def _transfer(self, comm, sends, receives, stream, tag) -> None:


        import cupy  # noqa: PLC0415

        if _overlap_disabled():
            stream = None
        if not sends and not receives:
            return
        d2h, h2d = _POOL.copy_streams()
        if stream is not None:
            if not hasattr(stream, "record"):
                stream = cupy.cuda.ExternalStream(int(stream))
            start = stream.record()
            d2h.wait_event(start)
            h2d.wait_event(start)
        else:
            cupy.cuda.runtime.deviceSynchronize()
        if (
            self._ipc is not None
            and self._ipc.enabled
            and not self._overlaps(sends, receives)
        ):


            if stream is not None:
                start.synchronize()
            sends, receives = self._ipc_transfer(comm, sends, receives, d2h, tag)
        if self._overlaps(sends, receives):
            sends = self._stage_sends(sends, d2h)
        piece_bytes = min(self._piece_bytes, int(_col.MPI_CHUNK) * 8)
        lanes = self._lanes(sends, True, piece_bytes) + self._lanes(
            receives, False, piece_bytes
        )
        memory = self._assign_slots(lanes)
        self._progress(comm, lanes, memory, d2h, h2d, tag)
        if stream is not None:
            stream.wait_event(d2h.record())
            uploads = h2d.record()
            stream.wait_event(uploads)
            _POOL.uploads = uploads
        else:
            d2h.synchronize()
            h2d.synchronize()

    def _ipc_transfer(self, comm, sends, receives, d2h, tag):


        import cupy  # noqa: PLC0415
        from mpi4py import MPI  # noqa: PLC0415

        from ._ipc import _DESCRIPTOR, _DESCRIPTOR_TAG, _DONE_TAG, describe  # noqa: PLC0415

        ipc = self._ipc
        rt = cupy.cuda.runtime

        def split(blocks):
            peers = [peer for peer, _b, _f, _c in blocks]
            world = ipc.world_ranks(comm, peers) if peers else []
            near, far = [], []
            for block, wpeer in zip(blocks, world):
                (near if _is_device(block[1]) and ipc.same_node(wpeer) else far).append(
                    (block, wpeer)
                )
            return near, [b for b, _w in far]

        near_sends, far_sends = split(sends)
        near_receives, far_receives = split(receives)
        if not near_sends and not near_receives:
            return sends, receives
        requests, done, keep = [], [], []
        for (peer, buffer, first, count), _wpeer in near_receives:
            record = describe(buffer, first, count)
            requests.append(
                comm.Isend(
                    [record.view(np.uint8), MPI.BYTE],
                    dest=peer,
                    tag=tag + _DESCRIPTOR_TAG,
                )
            )
            keep.append(record)
            flag = np.zeros(1, dtype=np.uint8)
            done.append(
                (flag, comm.Irecv([flag, MPI.BYTE], source=peer, tag=tag + _DONE_TAG))
            )
        pending = []
        for (peer, buffer, first, count), wpeer in near_sends:
            record = np.zeros(1, dtype=_DESCRIPTOR)
            pending.append(
                [
                    peer,
                    wpeer,
                    buffer,
                    first,
                    count,
                    record,
                    comm.Irecv(
                        [record.view(np.uint8), MPI.BYTE],
                        source=peer,
                        tag=tag + _DESCRIPTOR_TAG,
                    ),
                    None,
                    None,
                ]
            )
        device = int(rt.getDevice())
        start = time.monotonic()
        last = start
        flags_out = []
        while (
            pending
            or any(not r.Test() for r in requests)
            or any(not req.Test() for _f, req in done)
        ):
            moved = False
            for item in pending:
                peer, wpeer, buffer, first, count, record, request, event, _sent = item
                if event is None and request.Test():
                    nbytes = count * buffer.itemsize
                    if int(record["nbytes"][0]) != nbytes:
                        raise RuntimeError(
                            f"host_mpi ipc: rank {comm.Get_rank()} sends {nbytes} bytes to rank "
                            f"{peer}, which expects {int(record['nbytes'][0])}"
                        )
                    base = ipc.open(wpeer, bytes(record["handle"][0]))
                    rt.memcpyPeerAsync(
                        base + int(record["offset"][0]),
                        ipc.device_of[wpeer],
                        int(buffer.data.ptr) + first * buffer.itemsize,
                        device,
                        nbytes,
                        d2h.ptr,
                    )
                    item[7] = cupy.cuda.Event(disable_timing=True)
                    item[7].record(d2h)
                    moved = True
                elif event is not None and event.done:
                    flag = np.ones(1, dtype=np.uint8)
                    flags_out.append(flag)
                    requests.append(
                        comm.Isend([flag, MPI.BYTE], dest=peer, tag=tag + _DONE_TAG)
                    )
                    item[8] = True
                    moved = True
            pending = [item for item in pending if not item[8]]
            now = time.monotonic()
            if moved:
                last = now
            elif self._timeout and now - last > self._timeout:
                raise RuntimeError(
                    f"host_mpi ipc: rank {comm.Get_rank()} made no progress for {now - last:.0f} s; "
                    f"{len(pending)} IPC sends wait for a descriptor or a copy"
                )
        return far_sends, far_receives

    @staticmethod
    def _overlaps(sends, receives) -> bool:


        def ranges(blocks):
            return [
                (
                    int(b.data.ptr) + first * b.itemsize,
                    int(b.data.ptr) + (first + count) * b.itemsize,
                )
                for _peer, b, first, count in blocks
                if _is_device(b)
            ]

        sent = ranges(sends)
        return any(
            lo < s_hi and s_lo < hi
            for lo, hi in ranges(receives)
            for s_lo, s_hi in sent
        )

    @staticmethod
    def _stage_sends(sends, d2h):

        import cupy  # noqa: PLC0415

        staged = []
        for peer, buffer, first, count in sends:
            if _is_device(buffer):
                host = np.empty(count, dtype=buffer.dtype)
                cupy.cuda.runtime.memcpyAsync(
                    host.ctypes.data,
                    int(buffer.data.ptr) + first * buffer.itemsize,
                    count * buffer.itemsize,
                    _DEVICE_TO_HOST,
                    d2h.ptr,
                )
                staged.append((peer, host, 0, count))
            else:
                staged.append((peer, buffer, first, count))
        d2h.synchronize()
        return staged

    @staticmethod
    def _lanes(blocks, sending: bool, piece_bytes: int) -> list:


        lanes = {}
        for peer, buffer, first, count in blocks:
            device = _is_device(buffer)
            if not buffer.flags.c_contiguous:
                raise ValueError("host_mpi transfers need C-contiguous buffers")
            if peer in lanes and lanes[peer].device != device:
                raise ValueError(
                    f"host_mpi: one call mixes host and device blocks with rank {peer}"
                )
            lane = lanes.setdefault(peer, _Lane(sending, peer))
            lane.device = device
            itemsize = buffer.itemsize
            step = max(1, piece_bytes // itemsize)
            flat = buffer.reshape(-1)
            for start in range(first, first + count, step):
                n = min(step, first + count - start)
                if device:
                    lane.pieces.append(
                        (int(buffer.data.ptr) + start * itemsize, n * itemsize)
                    )
                else:
                    lane.pieces.append((flat[start : start + n], n * itemsize))
        return list(lanes.values())

    @staticmethod
    def _assign_slots(lanes):


        import cupy  # noqa: PLC0415

        device = [lane for lane in lanes if lane.device]
        capacity = {
            id(lane): -(-max(size for _where, size in lane.pieces) // _ALIGN) * _ALIGN
            for lane in device
        }
        counts = {id(lane): 1 for lane in device}
        used = sum(capacity.values())
        for _extra in range(1, SLOTS):
            for lane in device:
                if (
                    counts[id(lane)] < len(lane.pieces)
                    and used + capacity[id(lane)] <= POOL_CAP_BYTES
                ):
                    counts[id(lane)] += 1
                    used += capacity[id(lane)]
        memory = _POOL.take(used) if used else None
        offset = 0
        for lane in device:
            for _ in range(min(counts[id(lane)], len(lane.pieces))):
                lane.slots.append(
                    [
                        offset,
                        capacity[id(lane)],
                        "free",
                        None,
                        cupy.cuda.Event(disable_timing=True),
                        None,
                    ]
                )
                offset += capacity[id(lane)]
        return memory

    def _progress(self, comm, lanes, memory, d2h, h2d, tag) -> None:

        import cupy  # noqa: PLC0415
        from mpi4py import MPI  # noqa: PLC0415

        runtime = cupy.cuda.runtime
        base = memory.ctypes.data if memory is not None else 0
        last = time.monotonic()
        idle = 0
        status = MPI.Status()
        for lane in lanes:
            if not lane.device:
                for where, nbytes in lane.pieces:
                    buf = [where, nbytes, MPI.BYTE]
                    lane.requests.append(
                        comm.Isend(buf, dest=lane.peer, tag=tag)
                        if lane.sending
                        else comm.Irecv(buf, source=lane.peer, tag=tag)
                    )
                lane.posted = len(lane.pieces)
        while True:
            moved = False
            for lane in lanes:
                if lane.device:
                    moved |= self._advance_device(
                        comm, lane, memory, base, runtime, d2h, h2d, tag, status
                    )
                else:
                    moved |= self._advance_host(lane, status)
            if all(lane.done == len(lane.pieces) for lane in lanes):
                return
            if moved:
                last, idle = time.monotonic(), 0
                continue

            idle += 1
            if (
                idle & 1023 == 0
                and self._timeout
                and time.monotonic() - last > self._timeout
            ):
                self._fail(comm, lanes, time.monotonic() - last)

    @staticmethod
    def _advance_device(
        comm, lane, memory, base, runtime, d2h, h2d, tag, status
    ) -> bool:

        from mpi4py import MPI  # noqa: PLC0415

        moved = False
        for slot in lane.slots:
            offset, _capacity, state, piece, event, request = slot
            if state == "free" and lane.posted < len(lane.pieces):
                where, nbytes = lane.pieces[lane.posted]
                slot[3] = lane.posted
                lane.posted += 1
                if lane.sending:
                    runtime.memcpyAsync(
                        base + offset, where, nbytes, _DEVICE_TO_HOST, d2h.ptr
                    )
                    event.record(d2h)
                    slot[2] = "the copy to the host"
                else:
                    slot[5] = comm.Irecv(
                        [memory[offset : offset + nbytes], nbytes, MPI.BYTE],
                        source=lane.peer,
                        tag=tag,
                    )
                    slot[2] = "MPI"
                moved = True
            elif state == "the copy to the host" and piece == lane.sent and event.done:

                lane.sent += 1
                nbytes = lane.pieces[piece][1]
                slot[5] = comm.Isend(
                    [memory[offset : offset + nbytes], nbytes, MPI.BYTE],
                    dest=lane.peer,
                    tag=tag,
                )
                slot[2] = "MPI"
                moved = True
            elif state == "MPI" and request.Test(status):
                where, nbytes = lane.pieces[piece]
                slot[5] = None
                if lane.sending:
                    slot[2] = "free"
                else:
                    got = status.Get_count(MPI.BYTE)
                    if got != nbytes:
                        raise RuntimeError(
                            f"host_mpi: rank {comm.Get_rank()} expected a piece of {nbytes} bytes "
                            f"from rank {lane.peer} and got {got}; the two ranks disagree on a count"
                        )
                    runtime.memcpyAsync(
                        where, base + offset, nbytes, _HOST_TO_DEVICE, h2d.ptr
                    )
                    event.record(h2d)
                    slot[2] = "the copy to the device"
                lane.done += 1
                moved = True
            elif state == "the copy to the device" and event.done:
                slot[2] = "free"
                moved = True
        return moved

    @staticmethod
    def _advance_host(lane, status) -> bool:

        from mpi4py import MPI  # noqa: PLC0415

        moved = False
        while lane.done < len(lane.requests) and lane.requests[lane.done].Test(status):
            if not lane.sending:
                nbytes = lane.pieces[lane.done][1]
                got = status.Get_count(MPI.BYTE)
                if got != nbytes:
                    raise RuntimeError(
                        f"host_mpi: expected a piece of {nbytes} bytes from rank {lane.peer} and "
                        f"got {got}; the two ranks disagree on a count"
                    )
            lane.done += 1
            moved = True
        return moved

    def _fail(self, comm, lanes, waited) -> None:

        for lane in lanes:
            if lane.sending:
                continue
            requests = [slot[5] for slot in lane.slots] + lane.requests[lane.done :]
            for request in requests:
                if request is None:
                    continue
                try:
                    request.Cancel()
                    request.Wait()
                except Exception:  # noqa: BLE001
                    pass
        message = (
            f"Host MPI transfer stalled on rank {comm.Get_rank()} for {waited:.0f} s "
            f"(timeout: {self._timeout:g} s). Pending transfers: "
            + "; ".join(lane.state() for lane in lanes if lane.done < len(lane.pieces))
        )
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
        raise RuntimeError(message)
