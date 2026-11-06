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

import numpy as np

from .._collective import BACKEND_INPROC, _Backend


class _InProcessBackend(_Backend):


    name = BACKEND_INPROC
    needs_pinned_staging = False

    def init(self, comm) -> None:


        return None

    def comm_split_type_node(self, comm):

        return comm.Split_type(None, key=comm.Get_rank())

    @staticmethod
    def _send(comm, destination: int, data, stream) -> None:


        import cupy  # noqa: PLC0415

        with stream if stream is not None else cupy.cuda.Stream.null:
            snapshot = cupy.array(data, copy=True)
        (stream or cupy.cuda.Stream.null).synchronize()
        comm.post(destination, snapshot)

    @staticmethod
    def _receive(comm, source: int, recv, count, stream) -> None:


        import cupy  # noqa: PLC0415

        snapshot = comm.take(source)
        count = int(snapshot.size) if count is None else int(count)
        if not recv.flags.c_contiguous:
            raise ValueError(
                f"inproc receive from rank {source}: the receive buffer is not contiguous"
            )
        if snapshot.size != count or recv.size < count:
            raise ValueError(
                f"inproc receive from rank {source}: message of {snapshot.size} values, "
                f"expected {count}, buffer of {recv.size}"
            )
        target = stream if stream is not None else cupy.cuda.Stream.null
        if count:
            recv.data.copy_from_device_async(
                snapshot.data, snapshot.nbytes, stream=target
            )
        target.synchronize()

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


        if dst_peer >= 0:
            self._send(comm, dst_peer, send_dev.ravel(), stream)
        if src_peer >= 0:
            self._receive(comm, src_peer, recv_dev, None, stream)

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


        send_flat = send_dev.ravel()
        for count, displ, rank in zip(
            np.asarray(send_counts, dtype=np.int64),
            np.asarray(send_displs, dtype=np.int64),
            np.asarray(dst_ranks),
        ):
            if int(count) > 0:
                self._send(
                    comm,
                    int(rank),
                    send_flat[int(displ) : int(displ) + int(count)],
                    stream,
                )
        recv_flat = recv_dev.ravel()
        for count, displ, rank in zip(
            np.asarray(recv_counts, dtype=np.int64),
            np.asarray(recv_displs, dtype=np.int64),
            np.asarray(src_ranks),
        ):
            if int(count) > 0:
                self._receive(
                    comm, int(rank), recv_flat[int(displ) :], int(count), stream
                )
