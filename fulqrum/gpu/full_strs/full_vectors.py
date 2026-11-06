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
import numpy as np

from fulqrum.gpu.distributed import _collective
from fulqrum.gpu.distributed.rank_agreement import raise_on_every_rank


class FullVectorExchange:


    def __init__(self, comm, stride, num_own):
        self.comm = comm
        self.backend = _collective.backend_for(comm)[0]
        self.size, self.rank = comm.Get_size(), comm.Get_rank()
        self.stride = int(stride)
        error = (
            "the full-vector form needs a device backend (NCCL), not a host-staged one"
            if self.backend.needs_pinned_staging
            else None
        )
        raise_on_every_rank(comm, error, what="full-vector exchange")
        self.collectives = hasattr(self.backend, "reduce_scatter_device")
        self.num_halo = self.size * self.stride - int(num_own)
        self.num_send = int(num_own)
        self._temporary = None
        received = 8 * (self.size - 1) * self.stride
        self.traffic = {
            "allgather_received": received,
            "reduce_scatter_received": received,
            "peers": self.size - 1,
        }

    def _peers(self):

        return [q for q in range(self.size) if q != self.rank]

    def allgather(self, values, stream, width=1):


        count = self.stride * width
        own = values[self.rank * count : (self.rank + 1) * count]
        if self.collectives:
            self.backend.allgather_device(own, values, count, self.comm, stream=stream)
            return
        peers = self._peers()
        counts = np.full(len(peers), count, dtype=np.int64)
        with stream:
            self.backend.grouped_sendrecv_device(
                values,
                counts,
                np.full(len(peers), self.rank * count, dtype=np.int64),
                np.array(peers, dtype=np.int32),
                values,
                counts,
                np.array([q * count for q in peers], dtype=np.int64),
                np.array(peers, dtype=np.int32),
                self.comm,
                stream=stream,
            )

    def reduce_scatter(self, values, stream):


        count = self.stride
        own = values[self.rank * count : (self.rank + 1) * count]
        if self.collectives:
            self.backend.reduce_scatter_device(
                values, own, count, self.comm, stream=stream
            )
            return
        peers = self._peers()
        if self._temporary is None:
            self._temporary = cupy.empty(max(1, len(peers) * count), dtype=np.float64)
        counts = np.full(len(peers), count, dtype=np.int64)
        with stream:
            self.backend.grouped_sendrecv_device(
                values,
                counts,
                np.array([q * count for q in peers], dtype=np.int64),
                np.array(peers, dtype=np.int32),
                self._temporary,
                counts,
                np.arange(len(peers), dtype=np.int64) * count,
                np.array(peers, dtype=np.int32),
                self.comm,
                stream=stream,
            )
            for index in range(len(peers)):
                own += self._temporary[index * count : (index + 1) * count]
