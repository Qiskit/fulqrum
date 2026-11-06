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


import cupy
import cupyx
import numpy as np

from fulqrum.gpu.distributed import _collective
from fulqrum.gpu.distributed.exchange_plan import check_pairing

from .distribute import packing_lists

MODES = ("stepped", "grouped")
COPY_STREAM_PRIORITY = -1000


class HaloExchange:


    def __init__(self, comm, block, halo_rows, mode="stepped"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
        self.mode = mode
        self.comm = comm
        self.backend = _collective.backend_for(comm)[0]
        send_rows, halo_starts = packing_lists(comm, block, halo_rows)
        size, rank = comm.Get_size(), comm.Get_rank()
        sends = [p for p in range(size) if p != rank and send_rows[p].size]
        receives = [
            q for q in range(size) if q != rank and halo_starts[q + 1] > halo_starts[q]
        ]
        self.send_counts = np.array([send_rows[p].size for p in sends], dtype=np.int64)
        self.send_displs = np.concatenate(
            ([0], np.cumsum(self.send_counts)[:-1])
        ).astype(np.int64)
        self.dst_ranks = np.array(sends, dtype=np.int32)
        self.recv_counts = np.array(
            [halo_starts[q + 1] - halo_starts[q] for q in receives], dtype=np.int64
        )
        self.recv_displs = np.array([halo_starts[q] for q in receives], dtype=np.int64)
        self.src_ranks = np.array(receives, dtype=np.int32)
        index = (
            np.concatenate([send_rows[p] for p in sends])
            if sends
            else np.zeros(0, dtype=np.int64)
        )
        self.num_send = int(index.size)
        self.num_halo = int(halo_starts[-1])
        self.send_index = cupy.asarray(
            index.astype(np.int32)
        )
        self.send_buffer = cupy.empty(self.num_send, dtype=np.float64)
        self.send_pinned = None
        self.recv_pinned = None
        if self.backend.uses_caller_pinned:
            self.send_pinned = cupyx.empty_pinned((self.num_send,), dtype=np.float64)
            self.recv_pinned = cupyx.empty_pinned((self.num_halo,), dtype=np.float64)
        send_start = dict(zip(sends, self.send_displs.tolist()))
        self.steps = []
        for distance in range(1, size):
            dst, src = (rank + distance) % size, (rank - distance) % size
            send = (
                (dst, int(send_start[dst]), int(send_rows[dst].size))
                if dst in send_start
                else None
            )
            receive = (
                (
                    src,
                    int(halo_starts[src]),
                    int(halo_starts[src + 1] - halo_starts[src]),
                )
                if src in receives
                else None
            )
            if send is not None or receive is not None:
                self.steps.append((send, receive))
        check_pairing(
            comm,
            [(s[0] if s else -1, r[0] if r else -1) for s, r in self.steps],
            "halo exchange steps",
        )
        self.copy_stream = cupy.cuda.Stream(
            non_blocking=True, priority=COPY_STREAM_PRIORITY
        )

    def run(self, x_own, x_halo, stream):


        if self.num_send:
            with stream:
                cupy.take(x_own, self.send_index, out=self.send_buffer)
        if self.mode == "grouped":
            self._run_grouped(x_halo, stream)
        else:
            self._run_stepped(x_halo, stream)

    def _run_stepped(self, x_halo, stream):


        self.copy_stream.wait_event(stream.record())
        empty64 = np.zeros(0, dtype=np.int64)
        empty32 = np.zeros(0, dtype=np.int32)
        zero = np.zeros(1, dtype=np.int64)
        for send, receive in self.steps:
            send_view = self.send_buffer[:0]
            send_pinned = None
            if send is not None:
                send_view = self.send_buffer[send[1] : send[1] + send[2]]
                send_pinned = (
                    None
                    if self.send_pinned is None
                    else self.send_pinned[send[1] : send[1] + send[2]]
                )
            recv_view = x_halo[:0]
            recv_pinned = None
            if receive is not None:
                recv_view = x_halo[receive[1] : receive[1] + receive[2]]
                recv_pinned = (
                    None
                    if self.recv_pinned is None
                    else self.recv_pinned[receive[1] : receive[1] + receive[2]]
                )
            with self.copy_stream:
                self.backend.grouped_sendrecv_device(
                    send_view,
                    np.array([send[2]], dtype=np.int64) if send else empty64,
                    zero if send else empty64,
                    np.array([send[0]], dtype=np.int32) if send else empty32,
                    recv_view,
                    np.array([receive[2]], dtype=np.int64) if receive else empty64,
                    zero if receive else empty64,
                    np.array([receive[0]], dtype=np.int32) if receive else empty32,
                    self.comm,
                    stream=self.copy_stream,
                    send_pinned=send_pinned,
                    recv_pinned=recv_pinned,
                )
        stream.wait_event(self.copy_stream.record())

    @property
    def reverse_tail_elements(self):

        return self.num_send

    def reverse(self, y_own, y_halo, stream):


        import cupyx  # noqa: PLC0415

        if self.backend.needs_pinned_staging:
            raise ValueError(
                "the reverse halo exchange needs a device backend (NCCL), not a host-staged one"
            )
        if y_halo.size < self.num_halo + self.num_send:
            raise ValueError(
                f"y_halo needs {self.num_halo + self.num_send} values (halo and receive region), "
                f"got {y_halo.size}"
            )
        self.copy_stream.wait_event(stream.record())
        empty64 = np.zeros(0, dtype=np.int64)
        empty32 = np.zeros(0, dtype=np.int32)
        zero = np.zeros(1, dtype=np.int64)
        received = y_halo[self.num_halo : self.num_halo + self.num_send]
        for send, receive in reversed(self.steps):
            send_view = (
                y_halo[:0]
                if receive is None
                else y_halo[receive[1] : receive[1] + receive[2]]
            )
            recv_view = (
                received[:0] if send is None else received[send[1] : send[1] + send[2]]
            )
            with self.copy_stream:
                self.backend.grouped_sendrecv_device(
                    send_view,
                    np.array([receive[2]], dtype=np.int64) if receive else empty64,
                    zero if receive else empty64,
                    np.array([receive[0]], dtype=np.int32) if receive else empty32,
                    recv_view,
                    np.array([send[2]], dtype=np.int64) if send else empty64,
                    zero if send else empty64,
                    np.array([send[0]], dtype=np.int32) if send else empty32,
                    self.comm,
                    stream=self.copy_stream,
                )
        with self.copy_stream:
            for displ, count in zip(
                self.send_displs.tolist(), self.send_counts.tolist()
            ):
                if count:
                    cupyx.scatter_add(
                        y_own,
                        self.send_index[displ : displ + count],
                        received[displ : displ + count],
                    )
        stream.wait_event(self.copy_stream.record())

    def _run_grouped(self, x_halo, stream):

        self.backend.grouped_sendrecv_device(
            self.send_buffer,
            self.send_counts,
            self.send_displs,
            self.dst_ranks,
            x_halo,
            self.recv_counts,
            self.recv_displs,
            self.src_ranks,
            self.comm,
            stream=stream,
            send_pinned=self.send_pinned,
            recv_pinned=self.recv_pinned,
        )


class CopyExchange:


    def __init__(self, copy_comm, share_starts):
        self.comm = copy_comm
        self.backend = _collective.backend_for(copy_comm)[0]
        self.copies = copy_comm.Get_size()
        self.copy = copy_comm.Get_rank()
        self.starts = [int(v) for v in share_starts]
        own = self.starts[self.copy + 1] - self.starts[self.copy]
        self.contributions = cupy.empty((self.copies, own), dtype=np.float64)
        self.block_pinned = None
        self.share_pinned = None
        if self.backend.uses_caller_pinned:
            self.block_pinned = cupyx.empty_pinned((self.starts[-1],), dtype=np.float64)
            self.share_pinned = cupyx.empty_pinned((self.copies, own), dtype=np.float64)

    def _segment(self, copy):

        return self.starts[copy], self.starts[copy + 1]

    def _step(
        self, send, send_to, send_pinned, receive, receive_from, receive_pinned, stream
    ):

        self.backend.grouped_sendrecv_device(
            send,
            np.array([send.size], dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.array([send_to], dtype=np.int32),
            receive,
            np.array([receive.size], dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.array([receive_from], dtype=np.int32),
            self.comm,
            stream=stream,
            send_pinned=send_pinned,
            recv_pinned=receive_pinned,
        )

    def allgather(self, x_share, x_block, stream):


        begin, end = self._segment(self.copy)
        x_block[begin:end].data.copy_from_device_async(
            x_share.data, 8 * (end - begin), stream=stream
        )
        for distance in range(1, self.copies):
            send_to = (self.copy + distance) % self.copies
            receive_from = (self.copy - distance) % self.copies
            low, high = self._segment(receive_from)
            pinned = None if self.block_pinned is None else self.block_pinned[low:high]
            share_pinned = (
                None if self.block_pinned is None else self.block_pinned[begin:end]
            )
            self._step(
                x_share,
                send_to,
                share_pinned,
                x_block[low:high],
                receive_from,
                pinned,
                stream,
            )

    def reduce_scatter(self, y_block, out, stream):


        begin, end = self._segment(self.copy)
        own = self.contributions[self.copy]
        own.data.copy_from_device_async(
            y_block[begin:end].data, 8 * (end - begin), stream=stream
        )
        for distance in range(1, self.copies):
            send_to = (self.copy + distance) % self.copies
            receive_from = (self.copy - distance) % self.copies
            low, high = self._segment(send_to)
            send_pinned = (
                None if self.block_pinned is None else self.block_pinned[low:high]
            )
            receive_pinned = (
                None if self.share_pinned is None else self.share_pinned[receive_from]
            )
            self._step(
                y_block[low:high],
                send_to,
                send_pinned,
                self.contributions[receive_from],
                receive_from,
                receive_pinned,
                stream,
            )
        with stream:
            out[...] = self.contributions[0]
            for copy in range(1, self.copies):
                out += self.contributions[copy]
