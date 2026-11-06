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

from .._collective import (
    BACKEND_CUDA_MPI,
    _Backend,
    _chunked_allgatherv,
    _chunked_allreduce,
    _chunked_sendrecv,
    _chunked_sendrecv_replace,
    _post_chunked,
    _sync_device,
)


def _wait_before_mpi(stream=None) -> None:


    if stream is None:
        _sync_device()
    elif hasattr(stream, "synchronize"):
        stream.synchronize()
    else:
        import cupy  # noqa: PLC0415

        cupy.cuda.runtime.streamSynchronize(int(stream))


class _CudaMpiBackend(_Backend):


    name = BACKEND_CUDA_MPI

    def init(self, comm) -> None:


        return None

    def sendrecv_replace_device(self, buf_dev, peer: int, tag: int, comm) -> None:


        _wait_before_mpi()
        _chunked_sendrecv_replace(comm, buf_dev, peer, tag)
        _sync_device()

    def allreduce_device(self, buf_dev, op, comm) -> None:


        from mpi4py import MPI  # noqa: PLC0415

        _wait_before_mpi()
        _chunked_allreduce(comm, buf_dev, op, MPI.IN_PLACE)
        _sync_device()

    def allgatherv_device(
        self,
        send_dev,
        recv_dev,
        counts,
        displs,
        comm,
    ) -> None:


        from mpi4py import MPI  # noqa: PLC0415

        _wait_before_mpi()
        _chunked_allgatherv(comm, send_dev, recv_dev, counts, displs, MPI.DOUBLE)
        _sync_device()

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
        _wait_before_mpi(stream)
        _chunked_sendrecv(comm, send_dev, dst_peer, recv_dev, src_peer, tag)
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
        from mpi4py import MPI  # noqa: PLC0415

        send_counts = np.asarray(send_counts, dtype=np.int64)
        send_displs = np.asarray(send_displs, dtype=np.int64)
        recv_counts = np.asarray(recv_counts, dtype=np.int64)
        recv_displs = np.asarray(recv_displs, dtype=np.int64)

        _wait_before_mpi(stream)
        reqs = []
        for j in range(len(src_ranks)):
            reqs += _post_chunked(
                comm,
                False,
                recv_dev,
                int(recv_displs[j]),
                int(recv_counts[j]),
                int(src_ranks[j]),
                MPI.DOUBLE,
            )
        for i in range(len(dst_ranks)):
            reqs += _post_chunked(
                comm,
                True,
                send_dev,
                int(send_displs[i]),
                int(send_counts[i]),
                int(dst_ranks[i]),
                MPI.DOUBLE,
            )
        MPI.Request.Waitall(reqs)
        _sync_device()
