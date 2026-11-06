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

import hashlib
import json
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .rank_agreement import raise_on_every_rank

FORWARD = "forward"
SEND_LOCAL = "send_local"


@dataclass(frozen=True)
class Step:


    channel: str
    transfer: str
    send_to: int
    receive_from: int
    source: int
    slot: int


def ring_steps(channel, members, index, double_buffer):


    size = len(members)
    steps = []
    for k in range(1, size):
        source = int(members[(index + k) % size])
        if double_buffer:
            steps.append(
                Step(
                    channel,
                    FORWARD,
                    int(members[(index - 1) % size]),
                    int(members[(index + 1) % size]),
                    source,
                    k & 1,
                )
            )
        else:
            steps.append(
                Step(
                    channel,
                    SEND_LOCAL,
                    int(members[(index - k) % size]),
                    source,
                    source,
                    0,
                )
            )
    return steps


def window_steps(hops, double_buffer):


    return [
        Step(
            "mixed",
            SEND_LOCAL,
            int(dst),
            int(src),
            int(src),
            (index & 1) if double_buffer else 0,
        )
        for index, (dst, src) in enumerate(hops)
    ]


EXCHANGE_AUTO = "auto"
EXCHANGE_RING = "ring"
EXCHANGE_TWO_LEVEL = "two_level"
EXCHANGE_MODES = (EXCHANGE_AUTO, EXCHANGE_RING, EXCHANGE_TWO_LEVEL)


def exchange_mode(environ):


    mode = environ.get("FQ_EXCHANGE", EXCHANGE_AUTO).strip().lower()
    if mode not in EXCHANGE_MODES:
        raise ValueError(f"FQ_EXCHANGE must be one of {EXCHANGE_MODES}; got {mode!r}")
    return mode


def two_level_obstacle(
    nccl_on_every_rank, domains, host_mpi_on_every_rank=False, requested=False
):


    if not (nccl_on_every_rank or (host_mpi_on_every_rank and requested)):
        if host_mpi_on_every_rank:
            return "the two-level exchange on host_mpi needs FQ_EXCHANGE=two_level"
        return "the two-level exchange needs the NCCL backend, or host_mpi with FQ_EXCHANGE=two_level, on every rank"
    if domains is None or len(domains) < 2:
        return "the two-level exchange needs at least two shared-memory domains"
    if max(len(domain) for domain in domains) < 2:
        return (
            "the two-level exchange needs a shared-memory domain with two or more ranks"
        )
    return None


def choose_exchange(mode, nccl_on_every_rank, domains, host_mpi_on_every_rank=False):


    if mode == EXCHANGE_RING:
        return EXCHANGE_RING, "FQ_EXCHANGE=ring"
    obstacle = two_level_obstacle(
        nccl_on_every_rank, domains, host_mpi_on_every_rank, mode == EXCHANGE_TWO_LEVEL
    )
    if mode == EXCHANGE_TWO_LEVEL and obstacle is not None:
        raise ValueError(f"FQ_EXCHANGE=two_level cannot run: {obstacle}")
    if obstacle is not None:
        return EXCHANGE_RING, obstacle
    return (
        EXCHANGE_TWO_LEVEL,
        f"{len(domains)} shared-memory domains with {'NCCL' if nccl_on_every_rank else 'host_mpi'}",
    )


def settings_digest(settings):


    text = json.dumps(settings, sort_keys=True, default=list)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def pairing_errors(steps_by_rank):


    size = len(steps_by_rank)
    sends, receives = [], []
    for rank, steps in enumerate(steps_by_rank):
        for dst, src in steps:
            if dst >= 0:
                sends.append((rank, int(dst)))
            if src >= 0:
                receives.append((int(src), rank))
    errors = [
        f"rank {a} or {b} is outside world size {size}"
        for a, b in sends + receives
        if not (0 <= a < size and 0 <= b < size)
    ]
    send_count, receive_count = Counter(sends), Counter(receives)
    unmatched_sends = sorted((send_count - receive_count).elements())
    unmatched_receives = sorted((receive_count - send_count).elements())
    errors += [
        f"rank {a} sends to rank {b}, which does not receive from it"
        for a, b in unmatched_sends
    ]
    errors += [
        f"rank {b} receives from rank {a}, which does not send to it"
        for a, b in unmatched_receives
    ]
    return errors


def check_same_settings(comm, settings, what="exchange settings"):


    digest = settings_digest(settings)
    digests = comm.allgather(digest)
    error = None
    if len(set(digests)) > 1:
        error = (
            f"{what} differ across ranks (digests {digests}); this rank uses {settings}"
        )
    raise_on_every_rank(comm, error)
    return digest


def check_pairing(comm, steps, name):


    steps_by_rank = comm.allgather([(int(dst), int(src)) for dst, src in steps])
    errors = pairing_errors(steps_by_rank)
    error = f"{name}: " + "; ".join(errors[:4]) if errors else None
    raise_on_every_rank(comm, error)


def _mixed_window_schedule(
    p_alpha: int,
    p_beta: int,
    coord_alpha: int,
    coord_beta: int,
    dst_ranks,
    src_ranks,
):


    dst_set = set(int(r) for r in dst_ranks)
    src_set = set(int(r) for r in src_ranks)
    sched: list[tuple[int, int, int]] = []
    for da in range(1, p_alpha):
        for db in range(1, p_beta):
            dst = ((coord_alpha + da) % p_alpha) * p_beta + ((coord_beta + db) % p_beta)
            src = ((coord_alpha - da) % p_alpha) * p_beta + ((coord_beta - db) % p_beta)
            dst = dst if dst in dst_set else -1
            src = src if src in src_set else -1
            if dst >= 0 or src >= 0:
                sched.append((dst, src))
    covered_src = {s for _d, s in sched if s >= 0}
    covered_dst = {d for d, _s in sched if d >= 0}
    if covered_src != src_set:
        raise RuntimeError(
            f"mixed window schedule missed sources: {sorted(src_set - covered_src)}"
        )
    if covered_dst != dst_set:
        raise RuntimeError(
            f"mixed window schedule missed destinations: {sorted(dst_set - covered_dst)}"
        )
    return sched


@dataclass(frozen=True)
class _HierarchicalExchangeRound:


    send_dst: int
    recv_src: int
    gathered_sources: tuple[int, ...]


@dataclass(frozen=True)
class _HierarchicalExchangePlan:


    domain_index: int
    domain_members: tuple[int, ...]
    domain_rank: int
    local_sources: tuple[int, ...]
    rounds: tuple[_HierarchicalExchangeRound, ...]


def _hierarchical_exchange_plan(
    rank: int,
    domains,
    sources_by_rank,
) -> _HierarchicalExchangePlan:


    domains = tuple(tuple(int(r) for r in domain) for domain in domains)
    sources = tuple(frozenset(int(s) for s in srcs) for srcs in sources_by_rank)
    size = len(sources)
    flat = tuple(r for domain in domains for r in domain)
    if sorted(flat) != list(range(size)) or len(set(flat)) != size:
        raise ValueError("domains must partition world ranks exactly once")
    if any(not domain for domain in domains):
        raise ValueError("domains must be non-empty")
    if not 0 <= int(rank) < size:
        raise ValueError(f"rank {rank} is outside world size {size}")
    if any(any(src < 0 or src >= size for src in srcs) for srcs in sources):
        raise ValueError("sources_by_rank contains an invalid world rank")

    domain_of = {}
    for domain_index, members in enumerate(domains):
        for domain_rank, member in enumerate(members):
            domain_of[member] = (domain_index, domain_rank)
    domain_index, domain_rank = domain_of[int(rank)]
    local_members = domains[domain_index]

    needed_by_domain = tuple(
        frozenset().union(*(sources[consumer] for consumer in members))
        for members in domains
    )
    local_sources = tuple(
        src if src in needed_by_domain[domain_index] else -1 for src in local_members
    )

    rounds: list[_HierarchicalExchangeRound] = []
    domain_count = len(domains)
    for shift in range(1, domain_count):
        subrounds = max(
            (
                len(domains[src_domain])
                + len(domains[(src_domain + shift) % domain_count])
                - 1
            )
            // len(domains[(src_domain + shift) % domain_count])
            for src_domain in range(domain_count)
        )
        outgoing_domain = (domain_index + shift) % domain_count
        incoming_domain = (domain_index - shift) % domain_count
        outgoing_members = domains[outgoing_domain]
        incoming_members = domains[incoming_domain]

        for subround in range(subrounds):
            send_dst = -1
            send_subround, send_owner = divmod(domain_rank, len(outgoing_members))
            if (
                subround == send_subround
                and int(rank) in needed_by_domain[outgoing_domain]
            ):
                send_dst = outgoing_members[send_owner]

            gathered = []
            for owner in range(len(local_members)):
                source_index = subround * len(local_members) + owner
                if source_index < len(incoming_members):
                    source_rank = incoming_members[source_index]
                    if source_rank not in needed_by_domain[domain_index]:
                        source_rank = -1
                else:
                    source_rank = -1
                gathered.append(source_rank)

            rounds.append(
                _HierarchicalExchangeRound(
                    send_dst=int(send_dst),
                    recv_src=int(gathered[domain_rank]),
                    gathered_sources=tuple(gathered),
                )
            )

    return _HierarchicalExchangePlan(
        domain_index=domain_index,
        domain_members=local_members,
        domain_rank=domain_rank,
        local_sources=local_sources,
        rounds=tuple(rounds),
    )


def _hierarchical_local_ring_sources(
    plan: _HierarchicalExchangePlan,
    gathered_sources,
) -> tuple[tuple[int, int], ...]:


    gathered = tuple(int(src) for src in gathered_sources)
    domain_size = len(plan.domain_members)
    if len(gathered) != domain_size:
        raise ValueError("gathered_sources must have one entry per domain member")
    domain_rank = int(plan.domain_rank)
    return tuple(
        (
            gathered[(domain_rank - step + 1) % domain_size],
            gathered[(domain_rank - step) % domain_size],
        )
        for step in range(1, domain_size)
    )


def _transfer_counts(
    send_peer: int, send_elems: int, recv_peer: int, recv_elems: int
) -> dict:


    empty64 = np.zeros(0, dtype=np.int64)
    empty32 = np.zeros(0, dtype=np.int32)
    send, recv = send_peer >= 0, recv_peer >= 0
    return {
        "send_counts": np.asarray([send_elems], dtype=np.int64) if send else empty64,
        "send_displs": np.zeros(1, dtype=np.int64) if send else empty64,
        "dst_ranks": np.asarray([send_peer], dtype=np.int32) if send else empty32,
        "recv_counts": np.asarray([recv_elems], dtype=np.int64) if recv else empty64,
        "recv_displs": np.zeros(1, dtype=np.int64) if recv else empty64,
        "src_ranks": np.asarray([recv_peer], dtype=np.int32) if recv else empty32,
    }


def _two_level_local_ring_args(plan, gathered_sources, tile_elems) -> tuple:


    domain_size = len(plan.domain_members)
    right_peer = (plan.domain_rank + 1) % domain_size
    left_peer = (plan.domain_rank - 1) % domain_size
    steps = []
    for send_source, recv_source in _hierarchical_local_ring_sources(
        plan, gathered_sources
    ):
        send_source, recv_source = int(send_source), int(recv_source)
        steps.append(
            {
                "send_source": send_source,
                "recv_source": recv_source,
                **_transfer_counts(
                    right_peer if send_source >= 0 else -1,
                    tile_elems[send_source] if send_source >= 0 else 0,
                    left_peer if recv_source >= 0 else -1,
                    tile_elems[recv_source] if recv_source >= 0 else 0,
                ),
            }
        )
    return tuple(steps)


def _two_level_round_args(plan, rank: int, tile_elems) -> list:


    round_args = []
    for exchange_round in plan.rounds:
        send_dst = int(exchange_round.send_dst)
        recv_source = int(exchange_round.recv_src)
        round_args.append(
            {
                "send_dst": send_dst,
                "recv_source": recv_source,
                **_transfer_counts(
                    send_dst,
                    tile_elems[rank],
                    recv_source,
                    tile_elems[recv_source] if recv_source >= 0 else 0,
                ),
                "gathered_sources": exchange_round.gathered_sources,
                "local_ring_args": _two_level_local_ring_args(
                    plan, exchange_round.gathered_sources, tile_elems
                ),
            }
        )
    return round_args


def _two_level_remote_bytes(plan, domains, sources, round_args, tile_elems) -> tuple:


    domain_of = {
        member: index for index, members in enumerate(domains) for member in members
    }
    direct = sum(
        tile_elems[src] * 8 for src in sources if domain_of[src] != plan.domain_index
    )
    two_level = sum(
        tile_elems[step["recv_source"]] * 8
        for step in round_args
        if step["recv_source"] >= 0
    )
    return direct, two_level
