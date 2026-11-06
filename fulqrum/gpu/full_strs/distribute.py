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

from dataclasses import dataclass

import numpy as np

from fulqrum.gpu.distributed.local_comm import LocalComm
from fulqrum.gpu.distributed.rank_agreement import (
    raise_on_every_rank,
    run_on_every_rank,
)

WORD_BITS = 64
MPI_CHUNK = 1 << 28
SAMPLE_FLOOR = 128
EXCHANGE_TAG = 7301
KIND_NAMES = ("intersection", "union")
GROUP_LAYOUTS = ("by_rows", "by_key")

PROBE_SLICES = 4
PROBE_MIN_ROWS = 2_000_000


@dataclass(frozen=True)
class KeyBlock:


    keys: np.ndarray
    row_start: int
    num_rows: int
    block_starts: np.ndarray
    num_orbitals: int

    @property
    def n_local(self):

        return int(self.keys.shape[0])

    def owner_of(self, rows):


        return (
            np.searchsorted(
                self.block_starts, np.asarray(rows, dtype=np.int64), side="right"
            ).astype(np.int64)
            - 1
        )


def exchange_rows(comm, send, send_counts):


    size, rank = comm.Get_size(), comm.Get_rank()
    send = np.ascontiguousarray(send)
    counts = np.asarray(send_counts, dtype=np.int64)
    error = None
    if send.ndim != 2:
        error = f"the send array must be 2-D, got shape {send.shape}"
    elif (
        counts.shape != (size,)
        or np.any(counts < 0)
        or int(counts.sum()) != send.shape[0]
    ):
        error = f"send counts {counts.tolist()} do not match {size} ranks and {send.shape[0]} rows"
    raise_on_every_rank(comm, error, what="row exchange")
    matrix = np.array(comm.allgather([int(c) for c in counts]), dtype=np.int64).reshape(
        size, size
    )
    recv_counts = matrix[:, rank].copy()
    send_starts = np.concatenate(([0], np.cumsum(counts)))
    recv_starts = np.concatenate(([0], np.cumsum(recv_counts)))
    recv = np.empty((int(recv_starts[-1]), send.shape[1]), dtype=send.dtype)
    recv[recv_starts[rank] : recv_starts[rank + 1]] = send[
        send_starts[rank] : send_starts[rank + 1]
    ]
    peers = [(rank + step) % size for step in range(1, size)]
    if isinstance(comm, LocalComm):
        for peer in peers:
            comm.post(peer, send[send_starts[peer] : send_starts[peer + 1]].copy())
        for peer in peers:
            recv[recv_starts[peer] : recv_starts[peer + 1]] = comm.take(peer)
    else:
        _exchange_mpi(comm, send, send_starts, recv, recv_starts, peers)
    return recv, recv_counts


def _exchange_mpi(comm, send, send_starts, recv, recv_starts, peers):


    from mpi4py import MPI  # noqa: PLC0415
    from mpi4py.util.dtlib import from_numpy_dtype  # noqa: PLC0415

    datatype = from_numpy_dtype(send.dtype)
    width = max(1, send.shape[1])
    piece_rows = max(1, MPI_CHUNK // width)
    send_flat = send.reshape(-1)
    recv_flat = recv.reshape(-1)
    requests = []
    for peer in peers:
        for start, stop in _pieces(
            recv_starts[peer], recv_starts[peer + 1], piece_rows
        ):
            requests.append(
                comm.Irecv(
                    [recv_flat[start * width : stop * width], datatype],
                    source=peer,
                    tag=EXCHANGE_TAG,
                )
            )
    for peer in peers:
        for start, stop in _pieces(
            send_starts[peer], send_starts[peer + 1], piece_rows
        ):
            requests.append(
                comm.Isend(
                    [send_flat[start * width : stop * width], datatype],
                    dest=peer,
                    tag=EXCHANGE_TAG,
                )
            )
    MPI.Request.Waitall(requests)


def _pieces(start, stop, piece_rows):

    for first in range(int(start), int(stop), piece_rows):
        yield first, min(first + piece_rows, int(stop))


def sort_columns(keys):


    return keys[:, ::-1].T


def sort_unique(keys):


    if keys.shape[0] == 0:
        return np.ascontiguousarray(keys)
    columns = sort_columns(keys)
    order = np.lexsort(columns[::-1])
    ordered = keys[order]
    keep = np.ones(ordered.shape[0], dtype=bool)
    keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    return np.ascontiguousarray(ordered[keep])


def lower_bound(columns, value):


    low, high = 0, columns.shape[1]
    for column, target in zip(columns, value):
        part = column[low:high]
        first = low + int(np.searchsorted(part, target, side="left"))
        last = low + int(np.searchsorted(part, target, side="right"))
        if first == last:
            return first
        low, high = first, last
    return low


def _check_local(keys, num_orbitals):


    num_words = keys.shape[1] // 2
    if keys.shape[1] < 2 or keys.shape[1] % 2:
        return (
            f"keys must have an even, positive number of words per row, got {keys.shape[1]}",
            None,
        )
    if not 1 <= num_orbitals <= WORD_BITS * num_words:
        return (
            f"{num_orbitals} orbitals do not fit {num_words} word(s) per sector",
            None,
        )
    if keys.shape[0] == 0:
        return None, None
    word_index = np.arange(num_words)
    free_bits = np.clip(num_orbitals - WORD_BITS * word_index, 0, WORD_BITS)
    masks = np.array(
        [
            ~np.uint64(0) if bits == WORD_BITS else np.uint64((1 << int(bits)) - 1)
            for bits in free_bits
        ],
        dtype=np.uint64,
    )
    masks = np.concatenate((masks, masks))
    outside = np.any(keys & ~masks, axis=1)
    if outside.any():
        return (
            f"row {int(np.argmax(outside))} has a bit at or above orbital {num_orbitals}",
            None,
        )
    counts = np.bitwise_count(keys).astype(np.int64)
    alpha = counts[:, :num_words].sum(axis=1)
    beta = counts[:, num_words:].sum(axis=1)
    bad = (alpha != alpha[0]) | (beta != beta[0])
    if bad.any():
        row = int(np.argmax(bad))
        return (
            f"row {row} has {int(alpha[row])} alpha and {int(beta[row])} beta electrons, row 0 has "
            f"{int(alpha[0])} and {int(beta[0])}"
        ), None
    return None, (int(alpha[0]), int(beta[0]))


def _check_agreement(reports, size):


    widths = {report[0] for report in reports}
    if len(widths) != 1:
        return (
            f"ranks give keys of different widths: {[report[0] for report in reports]}"
        )
    orbitals = {report[1] for report in reports}
    if len(orbitals) != 1:
        return (
            f"ranks give different orbital counts: {[report[1] for report in reports]}"
        )
    electrons = {report[2] for report in reports if report[2] is not None}
    if len(electrons) > 1:
        found = [
            f"rank {rank}: {report[2]}"
            for rank, report in enumerate(reports)
            if report[2] is not None
        ]
        return "ranks give different (alpha, beta) electron counts: " + ", ".join(found)
    if not electrons:
        return f"no rows on any of the {size} ranks"
    return None


def distribute_keys(comm, keys, num_orbitals):


    size = comm.Get_size()
    keys = np.asarray(keys)
    error = None
    if keys.ndim != 2:
        error = f"keys must have shape (n, 2W), got shape {keys.shape}"
    elif keys.shape[0] and not np.issubdtype(keys.dtype, np.integer):
        error = f"keys must be integers, got dtype {keys.dtype}"
    raise_on_every_rank(comm, error, what="key distribution")
    keys = np.ascontiguousarray(keys, dtype=np.uint64)
    error, electrons = _check_local(keys, int(num_orbitals))
    raise_on_every_rank(comm, error, what="key distribution")
    reports = comm.allgather((int(keys.shape[1]), int(num_orbitals), electrons))
    error = _check_agreement(reports, size)
    if error is not None:
        raise ValueError(f"key distribution failed: {error}")
    local = sort_unique(keys)
    splitters = _splitters(comm, local)
    columns = np.ascontiguousarray(sort_columns(local))
    bounds = [lower_bound(columns, splitter) for splitter in splitters]
    send_counts = np.diff(np.concatenate(([0], bounds, [local.shape[0]]))).astype(
        np.int64
    )
    received, _ = exchange_rows(comm, local, send_counts)
    local = sort_unique(received)
    counts = np.array(comm.allgather(int(local.shape[0])), dtype=np.int64)
    total = int(counts.sum())
    if total < size:
        raise ValueError(
            f"key distribution failed: {total} unique rows cannot fill {size} ranks; "
            "use fewer ranks"
        )
    starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    current = KeyBlock(
        keys=local,
        row_start=int(starts[comm.Get_rank()]),
        num_rows=total,
        block_starts=starts,
        num_orbitals=int(num_orbitals),
    )
    return redistribute(comm, current, row_cuts(total, size))


def row_cuts(num_rows, size):

    return np.array([num_rows * r // size for r in range(size + 1)], dtype=np.int64)


def redistribute(comm, block, block_starts):


    size, rank = comm.Get_size(), comm.Get_rank()
    targets = np.asarray(block_starts, dtype=np.int64)
    error = None
    if (
        targets.shape != (size + 1,)
        or targets[0] != 0
        or targets[-1] != block.num_rows
        or np.any(np.diff(targets) < 1)
    ):
        error = (
            f"block starts {targets.tolist()} are not a cut of {block.num_rows} rows over {size} ranks "
            "with at least one row per rank"
        )
    raise_on_every_rank(comm, error, what="row redistribution")
    if len({tuple(t) for t in comm.allgather(targets.tolist())}) != 1:
        raise ValueError("row redistribution failed: ranks give different block starts")
    first = block.row_start
    last = first + block.n_local
    overlap = np.clip(
        np.minimum(targets[1:], last) - np.maximum(targets[:-1], first), 0, None
    ).astype(np.int64)
    keys, _ = exchange_rows(comm, block.keys, overlap)
    return KeyBlock(
        keys=keys,
        row_start=int(targets[rank]),
        num_rows=block.num_rows,
        block_starts=targets,
        num_orbitals=block.num_orbitals,
    )


def _splitters(comm, local):


    size = comm.Get_size()
    count = min(local.shape[0], max(2 * size, SAMPLE_FLOOR))
    positions = (np.arange(count, dtype=np.int64) * local.shape[0]) // max(count, 1)
    samples = comm.allgather(local[positions])
    merged = sort_unique(np.concatenate(samples))
    picks = (np.arange(1, size, dtype=np.int64) * merged.shape[0]) // size
    return [tuple(row) for row in sort_columns(merged[picks]).T]


def group_kinds(block, alpha_kind=None, beta_kind=None):


    num_words = block.keys.shape[1] // 2
    counts = np.bitwise_count(block.keys[0]).astype(np.int64)
    kinds = []
    for name, electrons in (
        (alpha_kind, int(counts[:num_words].sum())),
        (beta_kind, int(counts[num_words:].sum())),
    ):
        if name is None:
            kinds.append(1 if block.num_orbitals - electrons < electrons else 0)
        else:
            kinds.append(KIND_NAMES.index(name))
    return tuple(kinds)


def join_groups(
    comm,
    block,
    num_slices,
    alpha_kind=None,
    beta_kind=None,
    slices=None,
    fetch_keys=True,
    groups="by_rows",
    vectors="halo",
):


    error = None
    if groups not in GROUP_LAYOUTS:
        error = f"groups must be one of {GROUP_LAYOUTS}, got {groups!r}"
    elif vectors not in ("halo", "full") or (vectors == "full" and groups != "by_key"):
        error = f"vectors must be 'halo', or 'full' with groups='by_key'; got {vectors!r} with {groups!r}"
    elif block.keys.shape[1] != 2:
        error = f"the group join supports one word per sector, got {block.keys.shape[1] // 2}"
    elif int(num_slices) < 1:
        error = f"the slice count must be positive, got {num_slices}"
    raise_on_every_rank(comm, error, what="group join")
    kinds = group_kinds(block, alpha_kind, beta_kind)
    settings = comm.allgather((int(num_slices), kinds))
    if len(set(settings)) != 1:
        raise ValueError(
            f"group join failed: ranks give different (slices, kinds): {settings}"
        )
    return _join(
        comm,
        block,
        block.keys,
        block.row_start,
        block.block_starts,
        np.arange(comm.Get_size(), dtype=np.int64),
        None,
        int(num_slices),
        kinds,
        slices,
        fetch_keys,
        keep_local=groups == "by_key",
        global_stride=int(np.diff(block.block_starts).max())
        if vectors == "full"
        else None,
    )


def _join(
    comm,
    emitter,
    receive_keys,
    receive_start,
    run_starts,
    run_ranks,
    copy_comm,
    num_slices,
    kinds,
    slices=None,
    fetch_keys=True,
    keep_local=False,
    global_stride=None,
):


    from .host import GroupReceiver, emit_records, join_runs  # noqa: PLC0415

    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    size = comm.Get_size()
    indices = list(range(num_slices) if slices is None else slices)
    receiver = run_on_every_rank(
        comm,
        "group join",
        lambda: GroupReceiver(
            receive_keys, receive_start, num_slices, int(run_starts[-1])
        ),
    )

    def emit(index):
        return emit_records(
            emitter.keys,
            emitter.row_start,
            emitter.num_orbitals,
            kinds[0],
            kinds[1],
            num_slices,
            index,
            size,
        )


    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(emit, indices[0]) if indices else None
        for position, index in enumerate(indices):
            records, counts = run_on_every_rank(comm, "group join", pending.result)
            pending = (
                pool.submit(emit, indices[position + 1])
                if position + 1 < len(indices)
                else None
            )
            received, _ = exchange_rows(comm, records, counts)
            del records
            if keep_local:
                everything = np.array([0, int(run_starts[-1])], dtype=np.int64)
                headers, _, members, _ = run_on_every_rank(
                    comm,
                    "group join",
                    lambda received=received: join_runs(received, everything),
                )
                del received
                run_on_every_rank(
                    comm,
                    "group join",
                    lambda headers=headers, members=members: receiver.add_slice(
                        index, headers, members
                    ),
                )
                del headers, members
                continue
            headers, header_counts, members, member_counts = run_on_every_rank(
                comm,
                "group join",
                lambda received=received: join_runs(received, run_starts),
            )
            del received
            received_headers, _ = exchange_rows(
                comm, headers, _to_ranks(header_counts, run_ranks, size)
            )
            del headers
            received_members, _ = exchange_rows(
                comm, members, _to_ranks(member_counts, run_ranks, size)
            )
            del members
            if copy_comm is not None:
                received_headers = _forward(copy_comm, received_headers)
                received_members = _forward(copy_comm, received_members)
            run_on_every_rank(
                comm,
                "group join",
                lambda received_headers=received_headers, received_members=received_members: (
                    receiver.add_slice(index, received_headers, received_members)
                ),
            )

            del received_headers, received_members
    if global_stride is not None:
        run_on_every_rank(
            comm,
            "group join",
            lambda: receiver.finish_global(run_starts, int(global_stride)),
        )
        return receiver
    run_on_every_rank(comm, "group join", receiver.finish)
    if fetch_keys:
        fetch_halo_keys(comm, emitter, receiver)
    return receiver


def _to_ranks(counts, run_ranks, size):

    out = np.zeros(size, dtype=np.int64)
    out[np.asarray(run_ranks, dtype=np.int64)] = counts
    return out


def _forward(copy_comm, rows):

    copies, copy = copy_comm.Get_size(), copy_comm.Get_rank()
    if copy == 0:
        send = (
            np.ascontiguousarray(np.concatenate([rows] * (copies - 1)))
            if copies > 1
            else rows[:0]
        )
        counts = np.array([0] + [rows.shape[0]] * (copies - 1), dtype=np.int64)
    else:
        send = rows[:0]
        counts = np.zeros(copies, dtype=np.int64)
    received, _ = exchange_rows(copy_comm, send, counts)
    return rows if copy == 0 else received


def fetch_halo_keys(comm, block, receiver):


    size, rank = comm.Get_size(), comm.Get_rank()
    rows = receiver.halo_row_view()
    keys = run_on_every_rank(comm, "halo keys", receiver.halo_key_buffer)
    starts = np.searchsorted(rows, block.block_starts).astype(np.int64)
    for step in range(1, size):
        target, source = (rank + step) % size, (rank - step) % size
        ask = np.zeros(size, dtype=np.int64)
        ask[target] = starts[target + 1] - starts[target]
        asked, _ = exchange_rows(
            comm,
            np.ascontiguousarray(rows[starts[target] : starts[target + 1]]).reshape(
                -1, 1
            ),
            ask,
        )
        local = asked[:, 0] - block.row_start
        error = None
        if local.size and (local.min() < 0 or local.max() >= block.n_local):
            error = "a rank asks for the key of a row that this rank does not own"
        raise_on_every_rank(comm, error, what="halo keys")
        reply = np.zeros(size, dtype=np.int64)
        reply[source] = local.size
        received, _ = exchange_rows(
            comm, np.ascontiguousarray(block.keys[local]), reply
        )
        keys[starts[target] : starts[target + 1]] = received
    del keys
    run_on_every_rank(comm, "halo keys", receiver.verify_halo_keys)


def output_memberships(receiver):

    return receiver.output_memberships()


def work_cuts(comm, block, work):


    size, rank = comm.Get_size(), comm.Get_rank()
    work = np.asarray(work, dtype=np.int64)
    totals = np.array(comm.allgather(int(work.sum())), dtype=np.int64)
    total = int(totals.sum())
    if total == 0:
        return row_cuts(block.num_rows, size)
    offset = int(totals[:rank].sum())
    before = offset + np.concatenate(([0], np.cumsum(work)))
    mine = {}
    for r in range(1, size):
        threshold = total * r // size
        if offset <= threshold < offset + int(totals[rank]):
            mine[r] = block.row_start + int(
                np.searchsorted(before, threshold, side="left")
            )
    found = {}
    for part in comm.allgather(mine):
        found.update(part)
    cuts = np.zeros(size + 1, dtype=np.int64)
    cuts[size] = block.num_rows
    for r in range(1, size):
        cuts[r] = max(found.get(r, cuts[r - 1]), cuts[r - 1] + 1)
    for r in range(size - 1, 0, -1):
        cuts[r] = min(cuts[r], cuts[r + 1] - 1)
    return cuts


def work_imbalance(comm, work):

    totals = np.array(comm.allgather(int(np.sum(work))), dtype=np.float64)
    return float(totals.max() / totals.mean()) if totals.sum() > 0 else 1.0


def build_distributed_groups(
    comm,
    keys,
    num_orbitals,
    num_slices,
    alpha_kind=None,
    beta_kind=None,
    recut=True,
    share_starts=None,
    coefficients=None,
    entry_cost=0.0,
    probe_slices=None,
    fetch_keys=True,
    groups="by_rows",
    vectors="halo",
):


    block = distribute_keys(comm, keys, num_orbitals)
    if share_starts is not None:
        block = redistribute(comm, block, share_starts)
        recut = False
    if groups == "by_key":
        recut = False
    default_probe = (
        max(PROBE_SLICES, int(num_slices)) if block.num_rows >= PROBE_MIN_ROWS else 1
    )
    probe = 1 if not recut else int(probe_slices or default_probe)
    receiver = join_groups(
        comm,
        block,
        num_slices if probe == 1 else probe,
        alpha_kind,
        beta_kind,
        slices=None if probe == 1 else [0],
        fetch_keys=fetch_keys and not recut,
        groups=groups,
        vectors=vectors,
    )
    if (
        groups == "by_key"
    ):
        work = np.array(
            [
                receiver.chunk_views(c, cls)[0].size
                for c in range(receiver.num_chunks())
                for cls in range(5)
            ]
        )
    else:
        work = output_memberships(receiver)
    before = work_imbalance(comm, work)
    info = {
        "imbalance_before": before,
        "imbalance_after": before,
        "block_starts_before": block.block_starts,
    }
    if not recut:
        return block, receiver, info
    if coefficients is None:
        cuts = work_cuts(comm, block, work)
    else:
        size = comm.Get_size()
        halo_block = (
            np.searchsorted(block.block_starts, receiver.halo_row_view(), side="right")
            - 1
        ).astype(np.int32)
        profile = receiver.work_profile(
            np.asarray(coefficients, dtype=np.float64).reshape(35),
            halo_block,
            size,
            comm.Get_rank(),
        )
        weighted = profile[0] + float(entry_cost) * profile[2]
        cuts = cut_by_work(comm, block.row_start, weighted, block.num_rows, size)
    del receiver
    block = redistribute(comm, block, cuts)
    receiver = join_groups(
        comm, block, num_slices, alpha_kind, beta_kind, fetch_keys=fetch_keys
    )
    info["imbalance_after"] = work_imbalance(comm, output_memberships(receiver))
    return block, receiver, info


def strictly_ascending(values, piece=1 << 26):

    for first in range(0, max(int(values.size) - 1, 0), piece):
        part = values[first : first + piece + 1]
        if np.any(part[1:] <= part[:-1]):
            return False
    return True


def packing_lists(comm, block, halo_rows):


    size, rank = comm.Get_size(), comm.Get_rank()
    halo = np.ascontiguousarray(halo_rows, dtype=np.int64)
    halo_starts = np.searchsorted(halo, block.block_starts).astype(np.int64)
    error = None
    if halo.size and (
        not strictly_ascending(halo) or halo[0] < 0 or halo[-1] >= block.num_rows
    ):
        error = "halo rows must be ascending, unique global rows"
    elif halo_starts[rank + 1] != halo_starts[rank]:
        error = "a halo row is an own row"
    raise_on_every_rank(comm, error, what="halo packing")
    requests, counts = exchange_rows(comm, halo.reshape(-1, 1), np.diff(halo_starts))
    local = requests[:, 0] - block.row_start
    error = None
    if local.size and (local.min() < 0 or local.max() >= block.n_local):
        error = "a peer asks for a row that this rank does not own"
    raise_on_every_rank(comm, error, what="halo packing")
    starts = np.concatenate(([0], np.cumsum(counts)))
    send_rows = [
        np.ascontiguousarray(local[starts[p] : starts[p + 1]]) for p in range(size)
    ]
    return send_rows, halo_starts


@dataclass
class CopyLayout:


    share: KeyBlock
    block_keys: np.ndarray
    block_start: int
    block_starts: np.ndarray
    block: int
    copy: int
    copy_comm: object
    pieces: object
    plan: dict
    profile: object = None


def block_work(receiver, block_starts, block):


    from .host import CLASSES  # noqa: PLC0415

    num_blocks = len(block_starts) - 1
    own = receiver.num_own
    halo_owner = (
        np.searchsorted(block_starts, receiver.halo_rows(), side="right").astype(
            np.int64
        )
        - 1
    )
    work = np.zeros(num_blocks, dtype=np.int64)
    for chunk in range(receiver.num_chunks()):
        for cls in range(len(CLASSES)):
            members, sizes, _ = receiver.chunk_views(chunk, cls)
            if not sizes.size:
                continue
            group = np.repeat(
                np.arange(sizes.size, dtype=np.int64), sizes.astype(np.int64)
            )
            in_block = members < own
            block_members = np.bincount(group[in_block], minlength=sizes.size).astype(
                np.int64
            )
            work[block] += int((block_members * (block_members - 1)).sum())
            halo = ~in_block
            np.add.at(work, halo_owner[members[halo] - own], block_members[group[halo]])
    return work


def assign_sources(work, block, copies):


    loads = [float(work[block])] * copies
    assignment = np.full(len(work), -1, dtype=np.int64)
    order = sorted(
        (q for q in range(len(work)) if q != block and work[q] > 0),
        key=lambda q: (-float(work[q]), q),
    )
    for q in order:
        copy = min(range(copies), key=lambda c: (loads[c], c))
        assignment[q] = copy
        loads[copy] += float(work[q]) * copies
    return assignment


def build_copy_groups(
    comm,
    keys,
    num_orbitals,
    num_slices,
    copies,
    alpha_kind=None,
    beta_kind=None,
    share_starts=None,
    assignment=None,
    coefficients=None,
    entry_cost=0.0,
):


    size, rank = comm.Get_size(), comm.Get_rank()
    copies = int(copies)
    error = (
        None
        if copies >= 1 and size % copies == 0
        else f"{copies} copies do not divide {size} ranks"
    )
    raise_on_every_rank(comm, error, what="copy layout")
    share = distribute_keys(comm, keys, num_orbitals)
    if share_starts is not None:
        share = redistribute(comm, share, share_starts)
    block, copy = rank // copies, rank % copies
    copy_comm = comm.Split(block, copy)
    block_starts = np.ascontiguousarray(share.block_starts[::copies])
    block_start = int(block_starts[block])
    block_keys, _ = exchange_rows(
        copy_comm,
        np.concatenate([share.keys] * copies),
        np.full(copies, share.n_local, dtype=np.int64),
    )
    kinds = group_kinds(share, alpha_kind, beta_kind)
    settings = comm.allgather((int(num_slices), kinds, copies))
    if len(set(settings)) != 1:
        raise ValueError(
            f"copy layout failed: ranks give different (slices, kinds, copies): {settings}"
        )
    receiver = _join(
        comm,
        share,
        block_keys,
        block_start,
        block_starts,
        np.arange(len(block_starts) - 1, dtype=np.int64) * copies,
        copy_comm,
        int(num_slices),
        kinds,
    )
    num_blocks = len(block_starts) - 1
    halo_block = (
        np.searchsorted(block_starts, receiver.halo_rows(), side="right") - 1
    ).astype(np.int32)
    coef = (
        np.ones(35)
        if coefficients is None
        else np.asarray(coefficients, dtype=np.float64).reshape(35)
    )
    profile = run_on_every_rank(
        comm,
        "work profile",
        lambda: (
            receiver.work_profile(coef, halo_block, num_blocks, block)
            if copy == 0
            else None
        ),
    )
    work = copy_comm.bcast(
        profile[3] + float(entry_cost) * profile[4] if copy == 0 else None, root=0
    )
    if assignment is None:
        assignment = assign_sources(work, block, copies)
    else:
        fixed = np.asarray(assignment, dtype=np.int64)
        error = (
            None
            if fixed.shape == (num_blocks, num_blocks)
            else f"the assignment must be {num_blocks} x {num_blocks}"
        )
        raise_on_every_rank(comm, error, what="copy layout")
        assignment = fixed[block]
    rows = comm.allgather(
        (block, work.tolist(), assignment.tolist()) if copy == 0 else None
    )
    plan = {
        "copies": copies,
        "work": np.zeros((num_blocks, num_blocks), dtype=np.float64),
        "assignment": np.full((num_blocks, num_blocks), -1, dtype=np.int64),
    }
    for row in rows:
        if row is not None:
            plan["work"][row[0]] = row[1]
            plan["assignment"][row[0]] = row[2]
    halo_owner = np.searchsorted(block_starts, receiver.halo_rows(), side="right") - 1
    assigned = (
        (assignment[halo_owner] == copy).astype(np.uint8)
        if halo_owner.size
        else np.zeros(0, dtype=np.uint8)
    )
    share_begin = share.row_start - block_start
    pieces = run_on_every_rank(
        comm,
        "copy pieces",
        lambda: receiver.pieces(share_begin, share_begin + share.n_local, assigned),
    )
    return CopyLayout(
        share=share,
        block_keys=block_keys,
        block_start=block_start,
        block_starts=block_starts,
        block=block,
        copy=copy,
        copy_comm=copy_comm,
        pieces=pieces,
        plan=plan,
        profile=profile,
    )


def cut_by_work(comm, row_start, work, num_rows, parts):


    work = np.asarray(work, dtype=np.float64)
    totals = np.array(
        comm.allgather(float(work.sum()) if work.size else 0.0), dtype=np.float64
    )
    total = float(totals.sum())
    if total <= 0.0:
        return row_cuts(num_rows, parts)
    offset = float(totals[: comm.Get_rank()].sum())
    before = offset + np.concatenate(([0.0], np.cumsum(work)))
    mine = {}
    for p in range(1, parts):
        threshold = total * p / parts
        if work.size and offset <= threshold < offset + float(totals[comm.Get_rank()]):
            mine[p] = int(row_start) + int(
                np.searchsorted(before, threshold, side="left")
            )
    found = {}
    for part in comm.allgather(mine):
        found.update(part)
    cuts = np.zeros(parts + 1, dtype=np.int64)
    cuts[parts] = num_rows
    for p in range(1, parts):
        cuts[p] = max(found.get(p, cuts[p - 1]), cuts[p - 1] + 1)
    for p in range(parts - 1, 0, -1):
        cuts[p] = min(cuts[p], cuts[p + 1] - 1)
    return cuts


def equal_share_starts(block_starts, copies):

    starts = []
    for low, high in zip(block_starts[:-1], block_starts[1:]):
        starts += [
            int(low) + (int(high) - int(low)) * c // copies for c in range(copies)
        ]
    return np.array(starts + [int(block_starts[-1])], dtype=np.int64)


def balanced_bounds(row_own, remote_loads):


    copies = len(remote_loads)
    rows = len(row_own)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.asarray(row_own, dtype=np.float64)))
    )
    target = (cumulative[-1] + float(sum(remote_loads))) / copies
    bounds = [0]
    for c in range(copies - 1):
        wanted = max(0.0, target - float(remote_loads[c]))
        bound = int(
            np.searchsorted(cumulative, cumulative[bounds[-1]] + wanted, side="left")
        )
        bounds.append(min(max(bound, bounds[-1] + 1), rows - (copies - 1 - c)))
    return bounds + [rows]


def copy_loads(comm, layout, entry_cost=0.0):


    bounds = layout.copy_comm.allgather(layout.share.row_start - layout.block_start) + [
        len(layout.block_keys)
    ]
    loads = None
    if layout.copy == 0:
        copies = layout.copy_comm.Get_size()
        assigned = layout.plan["assignment"][layout.block]
        cumulative = np.concatenate(
            ([0.0], np.cumsum(own_row_work(layout.profile, entry_cost)))
        )
        loads = [
            float(cumulative[bounds[c + 1]] - cumulative[bounds[c]])
            + float(
                sum(
                    remote_work(layout.profile, entry_cost)[q]
                    for q in range(len(assigned))
                    if assigned[q] == c
                )
            )
            for c in range(copies)
        ]
    rows = comm.allgather(loads)
    return [v for row in rows if row is not None for v in row]


def plan_copy_groups(comm, keys, num_orbitals, plan, alpha_kind=None, beta_kind=None):


    copies, slices, coefficients, entry = (
        plan.copies,
        plan.num_slices,
        plan.coefficients,
        float(plan.entry_cost),
    )
    if plan.fixed:
        layout = build_copy_groups(
            comm,
            keys,
            num_orbitals,
            slices,
            copies,
            alpha_kind,
            beta_kind,
            plan.share_starts,
            plan.assignment,
            coefficients,
            entry,
        )
        return layout, {"copy_loads": copy_loads(comm, layout, entry)}
    layout = build_copy_groups(
        comm,
        keys,
        num_orbitals,
        slices,
        copies,
        alpha_kind,
        beta_kind,
        coefficients=coefficients,
        entry_cost=entry,
    )
    measured = {"copy_loads_row_cut": copy_loads(comm, layout, entry)}
    if plan.block_cut != "row":
        work, start = np.zeros(0), 0
        if layout.copy == 0:
            work = (
                layout.profile[2]
                if plan.block_cut == "membership"
                else layout.profile[0] + entry * layout.profile[2]
            )
            start = layout.block_start
        blocks = cut_by_work(
            comm, start, work, layout.share.num_rows, len(layout.block_starts) - 1
        )
        layout = build_copy_groups(
            comm,
            keys,
            num_orbitals,
            slices,
            copies,
            alpha_kind,
            beta_kind,
            equal_share_starts(blocks, copies),
            coefficients=coefficients,
            entry_cost=entry,
        )
        measured["copy_loads_block_cut"] = copy_loads(comm, layout, entry)
    if plan.shares in ("balanced", "own") and copies > 1:
        bounds = None
        if layout.copy == 0:
            assigned = layout.plan["assignment"][layout.block]
            remote = [
                float(
                    sum(
                        remote_work(layout.profile, entry)[q]
                        for q in range(len(assigned))
                        if assigned[q] == c
                    )
                )
                if plan.shares == "balanced"
                else 0.0
                for c in range(copies)
            ]
            bounds = balanced_bounds(own_row_work(layout.profile, entry), remote)
        bounds = layout.copy_comm.bcast(bounds, root=0)
        rows = comm.allgather(
            [layout.block_start + b for b in bounds[:-1]] if layout.copy == 0 else None
        )
        starts = [v for row in rows if row is not None for v in row] + [
            layout.share.num_rows
        ]
        layout = build_copy_groups(
            comm,
            keys,
            num_orbitals,
            slices,
            copies,
            alpha_kind,
            beta_kind,
            np.array(starts, dtype=np.int64),
            layout.plan["assignment"],
            coefficients,
            entry,
        )
    measured["copy_loads"] = copy_loads(comm, layout, entry)
    return layout, measured


def own_row_work(profile, entry_cost):

    return profile[1] + float(entry_cost) * profile[2]


def remote_work(profile, entry_cost):

    return profile[3] + float(entry_cost) * profile[4]


def layout_metrics(
    comm, keys, num_orbitals, num_slices, copies, alpha_kind=None, beta_kind=None
):


    from .plan import LayoutPlan  # noqa: PLC0415

    unit = [1.0] * 35
    if copies == 1:
        block, receiver, _ = build_distributed_groups(
            comm,
            keys,
            num_orbitals,
            num_slices,
            alpha_kind,
            beta_kind,
            recut=True,
            coefficients=unit,
        )
        return (
            float(receiver.halo_rows().size),
            float(sum(receiver.pair_digest(cls)[0] for cls in range(5))),
            int(block.num_rows),
        )
    plan = LayoutPlan(
        copies=copies,
        num_slices=num_slices,
        block_cut="fitted",
        shares="own",
        coefficients=unit,
    )
    layout, _ = plan_copy_groups(comm, keys, num_orbitals, plan, alpha_kind, beta_kind)
    pieces = layout.pieces
    begin = layout.share.row_start - layout.block_start
    end = begin + layout.share.n_local
    other = [
        members[(members < begin) | (members >= end)]
        for chunk in range(pieces.num_chunks())
        for cls in range(5)
        for members in (pieces.chunk_views(chunk, cls, 0)[0],)
    ]
    distinct = np.unique(np.concatenate(other)).size if other else 0
    volume = (
        pieces.halo_rows().size
        + distinct
        + (copies - 1) / copies * len(layout.block_keys)
    )
    work = sum(pieces.pair_digest(kind, cls)[0] for kind in (0, 1) for cls in range(5))
    return float(volume), float(work), int(layout.share.num_rows)


def plan_copies(
    comm,
    keys,
    num_orbitals,
    num_slices,
    options=(1, 2, 4),
    alpha_kind=None,
    beta_kind=None,
):


    from .plan import choose_copies  # noqa: PLC0415

    size = comm.Get_size()
    candidates = {}
    for copies in options:
        if size % copies:
            continue
        volume, work, rows = layout_metrics(
            comm, keys, num_orbitals, num_slices, copies, alpha_kind, beta_kind
        )
        values = comm.allgather((volume, work))
        volumes = np.array([v[0] for v in values])
        works = np.array([v[1] for v in values])
        total = max(rows, 1)
        candidates[copies] = {
            "volume_max": float(volumes.max() / total),
            "volume_mean": float(volumes.mean() / total),
            "work_imbalance": float(works.max() / works.mean())
            if works.mean() > 0
            else 1.0,
        }
    return choose_copies(candidates), candidates
