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

import operator
import threading
from collections import deque
from dataclasses import dataclass
from functools import reduce
from types import SimpleNamespace

import numpy as np

DEFAULT_TIMEOUT_S = 3600.0


@dataclass(frozen=True)
class _Op:


    name: str
    function: object


SUM = _Op("SUM", operator.add)
PROD = _Op("PROD", operator.mul)
MIN = _Op("MIN", min)
MAX = _Op("MAX", max)
_ARRAY_FUNCTIONS = {
    "SUM": np.add,
    "PROD": np.multiply,
    "MIN": np.minimum,
    "MAX": np.maximum,
}

LOCAL_MPI = SimpleNamespace(
    SUM=SUM,
    PROD=PROD,
    MIN=MIN,
    MAX=MAX,
    IN_PLACE=object(),
    DOUBLE=object(),
    COMM_TYPE_SHARED=object(),
    UNDEFINED=-32766,
)


class LocalCommAborted(RuntimeError):
    pass


def mpi_namespace(comm):


    if isinstance(comm, LocalComm):
        return LOCAL_MPI
    from mpi4py import MPI  # noqa: PLC0415

    return MPI


class _Group:


    def __init__(self, size: int, timeout: float):


        self.size = size
        self.timeout = timeout
        self.barrier = threading.Barrier(size, timeout=timeout)
        self.slots: list = [None] * size
        self.condition = threading.Condition()
        self.mailboxes: dict = {}
        self.children: dict = {}
        self.aborted = False

    def abort(self) -> None:

        with self.condition:
            if self.aborted:
                return
            self.aborted = True
            self.condition.notify_all()
            children = list(self.children.values())
        self.barrier.abort()
        for child in children:
            child.abort()

    def wait(self) -> None:

        if self.aborted:
            raise LocalCommAborted("a rank of the in-process group failed")
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError as error:
            self.abort()
            raise LocalCommAborted(
                "a rank of the in-process group failed or timed out"
            ) from error


class LocalComm:


    mpi_constants = LOCAL_MPI

    def __init__(self, group: _Group, rank: int):

        self._group = group
        self._rank = rank
        self._splits = 0

    @classmethod
    def create(cls, size: int, timeout: float = DEFAULT_TIMEOUT_S) -> list:


        if size < 1:
            raise ValueError(
                f"an in-process communicator needs at least one rank, got {size}"
            )
        group = _Group(size, timeout)
        return [cls(group, rank) for rank in range(size)]

    def Get_rank(self) -> int:  # noqa: N802

        return self._rank

    def Get_size(self) -> int:  # noqa: N802

        return self._group.size

    def abort(self) -> None:

        self._group.abort()

    def Abort(self, errorcode: int = 1) -> None:  # noqa: N802

        self._group.abort()
        raise LocalCommAborted(
            f"rank {self._rank} aborted the in-process group (code {errorcode})"
        )

    def Barrier(self) -> None:  # noqa: N802

        self._group.wait()

    def allgather(self, value) -> list:

        group = self._group
        group.slots[self._rank] = value
        group.wait()
        values = list(group.slots)
        group.wait()
        return values

    def allreduce(self, value, op=SUM):

        return reduce(op.function, self.allgather(value))

    def gather(self, value, root: int = 0):

        values = self.allgather(value)
        return values if self._rank == root else None

    def bcast(self, value, root: int = 0):

        return self.allgather(value if self._rank == root else None)[root]

    def Allreduce(self, sendbuf, recvbuf, op=SUM) -> None:  # noqa: N802


        recv = _buffer(recvbuf)
        send = recv if sendbuf is LOCAL_MPI.IN_PLACE else _buffer(sendbuf)
        arrays = self.allgather(np.array(send, copy=True))
        recv[...] = reduce(_ARRAY_FUNCTIONS[op.name], arrays)

    def Bcast(self, buf, root: int = 0) -> None:  # noqa: N802

        array = _buffer(buf)
        source = self.allgather(
            np.array(array, copy=True) if self._rank == root else None
        )[root]
        array[...] = source

    def Alltoall(self, sendbuf, recvbuf) -> None:  # noqa: N802

        send, recv = _buffer(sendbuf), _buffer(recvbuf)
        size = self._group.size
        blocks = self.allgather(np.array(send, copy=True).reshape(size, -1))
        recv_blocks = recv.reshape(size, -1)
        for source in range(size):
            recv_blocks[source] = blocks[source][self._rank]

    def Split(self, color: int = 0, key: int = 0):  # noqa: N802


        self._splits += 1
        members = self.allgather((int(color), int(key), self._rank))
        if int(color) == LOCAL_MPI.UNDEFINED:
            self.allgather(None)
            return None
        same = sorted((k, r) for c, k, r in members if c == int(color))
        index = [r for _, r in same].index(self._rank)
        group = self._group
        with group.condition:
            child = group.children.get((self._splits, int(color)))
            if child is None:
                child = _Group(len(same), group.timeout)
                group.children[(self._splits, int(color))] = child
        self.allgather(None)
        return LocalComm(child, index)

    def Split_type(self, split_type=None, key: int = 0):  # noqa: N802

        return self.Split(0, key)

    def post(self, destination: int, message) -> None:

        group = self._group
        with group.condition:
            group.mailboxes.setdefault((self._rank, int(destination)), deque()).append(
                message
            )
            group.condition.notify_all()

    def take(self, source: int):

        group = self._group
        key = (int(source), self._rank)
        with group.condition:
            ready = group.condition.wait_for(
                lambda: group.aborted or bool(group.mailboxes.get(key)),
                timeout=group.timeout,
            )
            if group.aborted:
                raise LocalCommAborted("a rank of the in-process group failed")
            if not ready:
                group.aborted = True
                group.condition.notify_all()
                raise LocalCommAborted(
                    f"rank {self._rank} timed out waiting for a message from rank {source}"
                )
            return group.mailboxes[key].popleft()


def _buffer(spec):

    if isinstance(spec, (list, tuple)):
        return spec[0]
    return spec


def run_ranks(comms, function) -> list:


    if len(comms) == 1:
        return [function(comms[0])]
    results: list = [None] * len(comms)
    errors: list = [None] * len(comms)

    def body(rank: int) -> None:


        try:
            results[rank] = function(comms[rank])
        except BaseException as error:  # noqa: BLE001
            errors[rank] = error
            comms[rank].abort()

    threads = [
        threading.Thread(target=body, args=(rank,), name=f"local-rank-{rank}")
        for rank in range(len(comms))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    first = next(
        (e for e in errors if e is not None and not isinstance(e, LocalCommAborted)),
        None,
    )
    if first is None:
        first = next((e for e in errors if e is not None), None)
    if first is not None:
        raise first
    return results
