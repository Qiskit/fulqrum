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
from fulqrum.gpu.distributed.local_comm import LocalComm
from fulqrum.gpu.distributed.rank_agreement import raise_on_every_rank

from .distribute import exchange_rows, strictly_ascending

COPY_STREAM_PRIORITY = -1000
WORD_BITS = 64


def exchange_domains(comm, domain_size=None):


    rank = comm.Get_rank()
    error = (
        None
        if domain_size is None or int(domain_size) > 0
        else f"domain_size must be positive, got {domain_size}"
    )
    raise_on_every_rank(comm, error, what="two-level domains")
    if domain_size is None:
        domain_comm = _collective.backend_for(comm)[0].comm_split_type_node(comm)
    else:
        domain_comm = comm.Split(rank // int(domain_size), rank)
    leaders = comm.allgather(min(domain_comm.allgather(rank)))
    grouped = {}
    for world, leader in enumerate(leaders):
        grouped.setdefault(int(leader), []).append(world)
    return domain_comm, tuple(tuple(grouped[leader]) for leader in sorted(grouped))


def obstacle(backend, domains):

    if backend.needs_pinned_staging:
        return "the two-level exchange needs a device backend (NCCL), not a host-staged one"
    if len(domains) < 2:
        return "the two-level exchange needs at least two shared-memory domains"
    return None


def _or_reduce(domain_comm, bits):

    if isinstance(domain_comm, LocalComm):
        for other in domain_comm.allgather(bits):
            np.bitwise_or(bits, other, out=bits)
        return
    from mpi4py import MPI  # noqa: PLC0415

    domain_comm.Allreduce(MPI.IN_PLACE, bits, op=MPI.BOR)


def _copy_bit_range(target, source, low, high):

    if high <= low:
        return
    first, last = low // WORD_BITS, (high - 1) // WORD_BITS
    low_mask = np.uint64((~0 << (low % WORD_BITS)) & ((1 << WORD_BITS) - 1))
    high_mask = np.uint64(
        (1 << ((high - 1) % WORD_BITS + 1)) - 1
        if (high - 1) % WORD_BITS != WORD_BITS - 1
        else (1 << WORD_BITS) - 1
    )
    if first == last:
        mask = low_mask & high_mask
        target[first] = (target[first] & ~mask) | (source[first] & mask)
        return
    target[first] = (target[first] & ~low_mask) | (source[first] & low_mask)
    target[first + 1 : last] = source[first + 1 : last]
    target[last] = (target[last] & ~high_mask) | (source[last] & high_mask)


def extend_to_domain_union(comm, domain_comm, domains, block, receiver):


    from fulqrum.gpu.distributed.rank_agreement import run_on_every_rank  # noqa: PLC0415

    rank = comm.Get_rank()
    own_domain = next(domain for domain in domains if rank in domain)
    bits = run_on_every_rank(
        comm, "two-level halo", lambda: receiver.halo_bitmap(block.num_rows)
    )
    union = bits.copy()
    _or_reduce(domain_comm, union)
    for owner in own_domain:
        _copy_bit_range(
            union,
            bits,
            int(block.block_starts[owner]),
            int(block.block_starts[owner + 1]),
        )
    del bits
    run_on_every_rank(comm, "two-level halo", lambda: receiver.extend_halo(union))


class TwoLevelExchange:


    def __init__(self, comm, block, halo_rows, domains):
        self.mode = "two_level"
        self.comm = comm
        self.backend = _collective.backend_for(comm)[0]
        size, rank = comm.Get_size(), comm.Get_rank()
        raise_on_every_rank(
            comm, obstacle(self.backend, domains), what="two-level exchange"
        )
        domain_of = np.empty(size, dtype=np.int64)
        position = np.empty(size, dtype=np.int64)
        for index, domain in enumerate(domains):
            for place, member in enumerate(domain):
                domain_of[member], position[member] = index, place
        mine = int(domain_of[rank])
        own_domain = domains[mine]
        self.domains = domains

        def delegate(owner, domain):

            return domain[int(position[owner]) % len(domain)]

        halo = np.ascontiguousarray(halo_rows, dtype=np.int64)
        halo_starts = np.searchsorted(halo, block.block_starts).astype(np.int64)
        error = None
        if halo.size and (
            not strictly_ascending(halo) or halo[0] < 0 or halo[-1] >= block.num_rows
        ):
            error = "halo rows must be ascending, unique global rows"
        elif halo_starts[rank + 1] != halo_starts[rank]:
            error = "a halo row is an own row"
        raise_on_every_rank(comm, error, what="two-level exchange")
        lengths = np.diff(halo_starts)

        ask = np.array(
            [
                lengths[q]
                if q != rank
                and (domain_of[q] == mine or delegate(q, own_domain) == rank)
                else 0
                for q in range(size)
            ],
            dtype=np.int64,
        )
        starts = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        request = (
            np.concatenate([halo[starts[q] : starts[q] + ask[q]] for q in range(size)])
            if ask.sum()
            else np.zeros(0, dtype=np.int64)
        )
        requests, counts = exchange_rows(comm, request.reshape(-1, 1), ask)
        local = requests[:, 0] - block.row_start
        error = None
        if local.size and (local.min() < 0 or local.max() >= block.n_local):
            error = "a peer asks for a row that this rank does not own"
        raise_on_every_rank(comm, error, what="two-level exchange")
        from_peer = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

        def wanted(peer):

            return local[from_peer[peer] : from_peer[peer + 1]]

        self.num_halo = int(halo_starts[-1])
        num_domains = len(domains)

        order = [p for p in own_domain if p != rank]
        order += [
            delegate(rank, domains[(mine + k) % num_domains])
            for k in range(1, num_domains)
        ]
        pieces, send_at = [], {}
        offset = 0
        for peer in order:
            rows = wanted(peer)
            send_at[peer] = (offset, int(rows.size))
            pieces.append(rows)
            offset += int(rows.size)
        self.num_send = offset
        index = (
            np.concatenate(pieces).astype(np.int32)
            if pieces
            else np.zeros(0, dtype=np.int32)
        )
        self.send_index = cupy.asarray(index)

        def segment(owner):

            return int(halo_starts[owner]), int(lengths[owner])

        def received(k):

            source = domains[(mine - k) % num_domains]
            return [
                q for q in source if delegate(q, own_domain) == rank and lengths[q] > 0
            ]

        def forwards(k, sends, receives):

            source = domains[(mine - k) % num_domains]
            for q in source:
                d = delegate(q, own_domain)
                displ, count = segment(q)
                if count == 0:
                    continue
                if d == rank:
                    sends.extend((p, displ, count) for p in own_domain if p != rank)
                else:
                    receives.append((d, displ, count))


        steps = []
        sends, receives = [], []
        for p in own_domain:
            if p != rank:
                if send_at[p][1]:
                    sends.append((p, self.num_halo + send_at[p][0], send_at[p][1]))
                displ, count = segment(p)
                if count:
                    receives.append((p, displ, count))
        steps.append((sends, receives))
        for k in range(1, num_domains + 1):
            sends, receives = [], []
            if k < num_domains:
                target = delegate(rank, domains[(mine + k) % num_domains])
                at, count = send_at[target]
                if count:
                    sends.append((target, self.num_halo + at, count))
                receives.extend((q,) + segment(q) for q in received(k))
            if k > 1:
                forwards(k - 1, sends, receives)
            steps.append((sends, receives))
        self.steps = [self._arrays(sends, receives) for sends, receives in steps]
        self._check_pairing(comm, steps)


        self.send_segments = [send_at[p] for p in order if send_at[p][1]]
        temporary = self.num_halo + self.num_send
        reverse, self.reverse_adds = [], []
        self.temp_elements = 0
        for sends, receives in reversed(steps):
            back_sends = list(receives)
            back_receives, adds, offset = [], [], 0
            for peer, displ, count in sends:
                if displ >= self.num_halo:
                    back_receives.append((peer, displ, count))
                else:
                    back_receives.append((peer, temporary + offset, count))
                    adds.append((temporary + offset, displ, count))
                    offset += count
            self.temp_elements = max(self.temp_elements, offset)
            reverse.append((back_sends, back_receives))
            self.reverse_adds.append(adds)
        self.reverse_steps = [
            self._arrays(sends, receives) for sends, receives in reverse
        ]
        self._check_pairing(comm, reverse)
        inter = set(range(size)) - set(own_domain)
        self.traffic = {
            "inter_sent": 8 * sum(c for s, _ in steps for p, _, c in s if p in inter),
            "inter_received": 8
            * sum(c for _, r in steps for p, _, c in r if p in inter),
            "intra_sent": 8
            * sum(c for s, _ in steps for p, _, c in s if p not in inter),
            "intra_received": 8
            * sum(c for _, r in steps for p, _, c in r if p not in inter),
            "peers": len({p for s, r in steps for p, _, _ in s + r}),
        }
        self.copy_stream = cupy.cuda.Stream(
            non_blocking=True, priority=COPY_STREAM_PRIORITY
        )

    @property
    def tail_elements(self):

        return self.num_send

    @property
    def reverse_tail_elements(self):

        return self.num_send + self.temp_elements

    def reverse(self, y_own, y_halo, stream):


        import cupyx  # noqa: PLC0415

        if y_halo.size < self.num_halo + self.reverse_tail_elements:
            raise ValueError(
                f"y_halo needs {self.num_halo + self.reverse_tail_elements} values, got {y_halo.size}"
            )
        self.copy_stream.wait_event(stream.record())
        for (send_counts, send_displs, dst, recv_counts, recv_displs, src), adds in zip(
            self.reverse_steps, self.reverse_adds
        ):
            with self.copy_stream:
                self.backend.grouped_sendrecv_device(
                    y_halo,
                    send_counts,
                    send_displs,
                    dst,
                    y_halo,
                    recv_counts,
                    recv_displs,
                    src,
                    self.comm,
                    stream=self.copy_stream,
                )
                for temporary, displ, count in adds:
                    y_halo[displ : displ + count] += y_halo[
                        temporary : temporary + count
                    ]
        received = y_halo[self.num_halo : self.num_halo + self.num_send]
        with self.copy_stream:
            for at, count in self.send_segments:
                cupyx.scatter_add(
                    y_own, self.send_index[at : at + count], received[at : at + count]
                )
        stream.wait_event(self.copy_stream.record())

    @staticmethod
    def _arrays(sends, receives):


        def pack(entries):
            return (
                np.array([e[2] for e in entries], dtype=np.int64),
                np.array([e[1] for e in entries], dtype=np.int64),
                np.array([e[0] for e in entries], dtype=np.int32),
            )

        return pack(sends) + pack(receives)

    @staticmethod
    def _check_pairing(comm, steps):

        rank = comm.Get_rank()
        mine = (
            [(i, rank, p, c) for i, (s, _) in enumerate(steps) for p, _, c in s],
            [(i, p, rank, c) for i, (_, r) in enumerate(steps) for p, _, c in r],
        )
        everything = comm.allgather(mine)
        sends = sorted(entry for s, _ in everything for entry in s)
        receives = sorted(entry for _, r in everything for entry in r)
        raise_on_every_rank(
            comm,
            None
            if sends == receives
            else "the two-level messages do not pair across ranks",
            what="two-level exchange",
        )

    def run(self, x_own, x_halo, stream):


        if x_halo.size < self.num_halo + self.num_send:
            raise ValueError(
                f"x_halo needs {self.num_halo + self.num_send} values (halo and send region), "
                f"got {x_halo.size}"
            )
        if self.num_send:
            with stream:
                cupy.take(
                    x_own,
                    self.send_index,
                    out=x_halo[self.num_halo : self.num_halo + self.num_send],
                )
        self.copy_stream.wait_event(stream.record())
        for send_counts, send_displs, dst, recv_counts, recv_displs, src in self.steps:
            with self.copy_stream:
                self.backend.grouped_sendrecv_device(
                    x_halo,
                    send_counts,
                    send_displs,
                    dst,
                    x_halo,
                    recv_counts,
                    recv_displs,
                    src,
                    self.comm,
                    stream=self.copy_stream,
                )
        stream.wait_event(self.copy_stream.record())
