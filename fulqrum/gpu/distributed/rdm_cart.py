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
import time
import logging

import numpy as np

from .. import config
from .local_comm import mpi_namespace
from .rank_agreement import raise_on_every_rank, run_on_every_rank

logger = logging.getLogger(__name__)
__all__ = ["compute_rdms_cart"]

LANE_BYTES = 100
MAX_ORBITALS = 64
MIN_LANES = 1 << 16
MAX_LANES = 1 << 24
TRANSIENT_SHARE = 0.25
FIT_SHARE = 0.9


CONNECT_SECONDS = 1.0
RELAY_BYTES_PER_SECOND = 50e9

MIN_PIECE = 1 << 20


BUDGET_VECTORS = 2
BUDGET_FLOOR = 64 << 20
_ROUTE_FOR_TESTING = None
_PIECE_FOR_TESTING = (
    None
)


def _fixed_ne(keys):

    keys = np.ascontiguousarray(keys, dtype=np.uint64)
    if keys.size == 0:
        return None
    counts = np.unpackbits(keys.view(np.uint8)).reshape(keys.size, 64).sum(axis=1)
    if counts.min() != counts.max():
        return None
    return int(counts[0])


def _popcount(cp, x):

    if hasattr(cp, "bitwise_count"):
        return cp.bitwise_count(x)
    x = x - ((x >> cp.uint64(1)) & cp.uint64(0x5555555555555555))
    x = (x & cp.uint64(0x3333333333333333)) + (
        (x >> cp.uint64(2)) & cp.uint64(0x3333333333333333)
    )
    x = (x + (x >> cp.uint64(4))) & cp.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * cp.uint64(0x0101010101010101)) >> cp.uint64(56)


def _sorted_index(cp, keys):

    order = cp.argsort(keys).astype(cp.int32)
    return keys[order], order


def _lookup(cp, sorted_keys, order, keys, n):

    pos = cp.searchsorted(sorted_keys, keys)
    pos = cp.clip(pos, 0, n - 1)
    return sorted_keys[pos] == keys, order[pos]


def _occ_vir(cp, keys, m, ne):

    bits = (
        (keys[:, None] >> cp.arange(m, dtype=cp.uint64)[None, :]) & cp.uint64(1)
    ).astype(cp.bool_)
    count = int(keys.size)
    occ = cp.nonzero(bits)[1].reshape(count, ne).astype(cp.int64)
    vir = cp.nonzero(~bits)[1].reshape(count, m - ne).astype(cp.int64)
    return occ, vir


def _excite_sign(cp, source, created, removed):

    one = cp.uint64(1)
    low = cp.minimum(created, removed).astype(cp.uint64)
    high = cp.maximum(created, removed).astype(cp.uint64)
    mask = ((one << high) - one) ^ ((one << (low + one)) - one)
    odd = (_popcount(cp, source & mask) & cp.uint64(1)).astype(cp.bool_)
    return cp.where(odd, -1.0, 1.0)


def _singles_range(cp, keys, sorted_keys, order, lo, hi, m, ne, lanes):


    n = int(keys.size)
    nv = m - ne
    one = cp.uint64(1)
    per = max(1, ne * nv)
    step = max(1, lanes // per)
    src_l, tgt_l, sg_l, lab_l = [], [], [], []
    for c0 in range(lo, hi, step):
        c1 = min(hi, c0 + step)
        rows = cp.arange(c0, c1, dtype=cp.int64)
        block = keys[rows]
        occ, vir = _occ_vir(cp, block, m, ne)
        diag_rows = cp.repeat(rows, ne).astype(cp.int32)
        diag_orb = occ.reshape(-1)
        src_l.append(diag_rows)
        tgt_l.append(diag_rows)
        sg_l.append(cp.ones(diag_rows.size, cp.float64))
        lab_l.append((diag_orb * m + diag_orb).astype(cp.int32))
        removed = cp.repeat(occ, nv, axis=1).reshape(-1)
        created = cp.tile(vir, (1, ne)).reshape(-1)
        bra = cp.repeat(rows, per)
        source = cp.repeat(block, per)
        target = (source & ~(one << removed.astype(cp.uint64))) | (
            one << created.astype(cp.uint64)
        )
        found, ket = _lookup(cp, sorted_keys, order, target, n)
        sign = _excite_sign(cp, source, created, removed)
        src_l.append(bra[found].astype(cp.int32))
        tgt_l.append(ket[found].astype(cp.int32))
        sg_l.append(sign[found])
        lab_l.append((created[found] * m + removed[found]).astype(cp.int32))
    if not src_l:
        return (
            cp.zeros(0, cp.int32),
            cp.zeros(0, cp.int32),
            cp.zeros(0, cp.float64),
            cp.zeros(0, cp.int32),
        )
    return (
        cp.concatenate(src_l),
        cp.concatenate(tgt_l),
        cp.concatenate(sg_l),
        cp.concatenate(lab_l),
    )


def _pair_graph(cp, keys, sorted_keys, order, lo, hi, m, ne, lanes):


    n = int(keys.size)
    nv = m - ne
    one = cp.uint64(1)
    rows_total = hi - lo
    oa, ob = np.triu_indices(ne, k=1)
    va, vb = np.triu_indices(nv, k=1)
    pairs_o, pairs_v = oa.shape[0], va.shape[0]
    oa_d, ob_d = cp.asarray(oa, cp.int64), cp.asarray(ob, cp.int64)
    va_d, vb_d = cp.asarray(va, cp.int64), cp.asarray(vb, cp.int64)
    per = ne * nv + pairs_o * pairs_v
    if per == 0:
        return cp.zeros(rows_total + 1, cp.int64), cp.zeros(0, cp.int32)
    step = max(1, lanes // per)
    counts = cp.empty(rows_total, cp.int64)
    cols = []
    for c0 in range(lo, hi, step):
        c1 = min(hi, c0 + step)
        block = keys[cp.arange(c0, c1, dtype=cp.int64)]
        occ, vir = _occ_vir(cp, block, m, ne)
        column = block[:, None]
        removed = cp.repeat(occ, nv, axis=1)
        created = cp.tile(vir, (1, ne))
        singles = (column & ~(one << removed.astype(cp.uint64))) | (
            one << created.astype(cp.uint64)
        )
        q1 = cp.repeat(occ[:, oa_d], pairs_v, axis=1).astype(cp.uint64)
        q2 = cp.repeat(occ[:, ob_d], pairs_v, axis=1).astype(cp.uint64)
        p1 = cp.tile(vir[:, va_d], (1, pairs_o)).astype(cp.uint64)
        p2 = cp.tile(vir[:, vb_d], (1, pairs_o)).astype(cp.uint64)
        doubles = (((column & ~(one << q1)) & ~(one << q2)) | (one << p1)) | (one << p2)
        candidates = cp.concatenate([singles, doubles], axis=1).reshape(-1)
        found, ket = _lookup(cp, sorted_keys, order, candidates, n)
        counts[c0 - lo : c1 - lo] = found.reshape(c1 - c0, per).sum(axis=1)
        cols.append(ket[found].astype(cp.int32))
        del candidates, found, ket, singles, doubles
    ptr = cp.zeros(rows_total + 1, cp.int64)
    ptr[1:] = cp.cumsum(counts)
    return ptr, (cp.concatenate(cols) if cols else cp.zeros(0, cp.int32))


def _col_sqnorm(cp, slab, lo, hi, width, budget=1 << 24):


    rows = hi - lo
    w = cp.empty(rows, cp.float64)
    step = max(1, budget // max(1, width))
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        block = slab[(lo + r0) * width : (lo + r1) * width].reshape(r1 - r0, width)
        w[r0:r1] = (block * block).sum(axis=1)
    return w


def _device_share(cp, comm):


    free = int(cp.cuda.runtime.memGetInfo()[0]) + int(
        cp.get_default_memory_pool().free_bytes()
    )
    place = (socket.gethostname(), cp.cuda.Device().pci_bus_id)
    places = comm.allgather(place)
    return min(comm.allgather(free // max(1, places.count(place))))


def _device_used(cp, comm):

    cp.cuda.runtime.deviceSynchronize()
    cp.get_default_memory_pool().free_all_blocks()
    free, total = (int(x) for x in cp.cuda.runtime.memGetInfo())
    place = (socket.gethostname(), cp.cuda.Device().pci_bus_id)
    sharing = max(1, comm.allgather(place).count(place))
    return (total - free) // sharing


def _lanes(share):

    return int(min(MAX_LANES, max(MIN_LANES, (TRANSIENT_SHARE * share) // LANE_BYTES)))


def _check_fits(comm, need, share, what):

    error = None
    if need > FIT_SHARE * share:
        error = (
            f"needs {need / 2**30:.2f} GiB of GPU memory, {share / 2**30:.2f} GiB is available; "
            "use more GPUs or call reset_cache() on the operator first"
        )
    raise_on_every_rank(comm, error, MemoryError, what)


def _local(cp, comm, what, function):


    def body():

        result = function()
        cp.cuda.runtime.deviceSynchronize()
        return result

    return run_on_every_rank(comm, what, body)


class _Log:


    def __init__(self, cp, comm):

        self.cp = cp
        self.on = comm.Get_rank() == 0 and bool(config.read("FQ_RDM_VERBOSE"))
        self.logger = logging.LoggerAdapter(logger, {"gpu_rank": comm.Get_rank()})
        self.t0 = time.perf_counter()

    def __call__(self, tag):

        if self.on:
            self.cp.cuda.runtime.deviceSynchronize()
            now = time.perf_counter()
            self.logger.debug(
                "RDM stage completed: %s, stage_elapsed=%.2f s", tag, now - self.t0
            )
            self.t0 = now


def _stream(cp):

    return int(cp.cuda.get_current_stream().ptr)


def _block_of(cp, starts, index):

    inner = cp.asarray(np.asarray(starts[1:-1], np.int64))
    return cp.searchsorted(inner, index.astype(cp.int64), side="right")


def _sector_lists(cp, keys, bra, m, ne, two, lanes, starts):


    lo, hi = bra
    n_bra = hi - lo
    blocks = len(starts) - 1
    mm = m * m
    device = cp.asarray(keys)
    sorted_keys, order = _sorted_index(cp, device)
    src, tgt, sign, label = _singles_range(
        cp, device, sorted_keys, order, lo, hi, m, ne, lanes
    )
    block = _block_of(cp, starts, tgt)
    key = block * (2 * mm) + (src == tgt).astype(cp.int64) * mm + label.astype(cp.int64)
    perm = cp.argsort(key)
    sorted_key = key[perm]
    label_keys = (
        cp.arange(blocks, dtype=cp.int64)[:, None] * (2 * mm)
        + cp.arange(mm + 1, dtype=cp.int64)[None, :]
    )
    single_start = cp.searchsorted(sorted_key, label_keys.reshape(-1)).astype(cp.int64)
    single_bounds = cp.asnumpy(
        cp.searchsorted(sorted_key, cp.arange(2 * blocks + 1, dtype=cp.int64) * mm)
    )
    out = {
        "keys": device,
        "source": cp.ascontiguousarray(src[perm]),
        "target": cp.ascontiguousarray(tgt[perm]),
        "sign": cp.ascontiguousarray(sign[perm]),
        "label": cp.ascontiguousarray(label[perm]),
        "single_start": single_start,
        "single_bounds": single_bounds,
        "edge_ptr": None,
        "cols": None,
    }
    if two:
        ptr, cols = _pair_graph(cp, device, sorted_keys, order, lo, hi, m, ne, lanes)
        rows = cp.repeat(cp.arange(n_bra, dtype=cp.int64), cp.diff(ptr))
        edge_key = _block_of(cp, starts, cols) * n_bra + rows
        perm = cp.argsort(edge_key)
        counts = (
            cp.bincount(edge_key, minlength=blocks * n_bra)
            if n_bra
            else cp.zeros(0, cp.int64)
        )
        edge_ptr = cp.zeros(blocks * n_bra + 1, cp.int64)
        edge_ptr[1:] = cp.cumsum(counts)
        out["edge_ptr"] = edge_ptr
        out["cols"] = cp.ascontiguousarray(cols[perm])
    return out


def _beta_order(cp, lists, m, starts):


    block = _block_of(cp, starts, lists["target"])
    key = block * 2 + (lists["source"] == lists["target"]).astype(cp.int64)
    perm = cp.lexsort(cp.stack([lists["source"].astype(cp.int64), key]))
    label = lists["label"][perm].astype(cp.int64)
    lists["ab_source"] = cp.ascontiguousarray(lists["source"][perm])
    lists["ab_target"] = cp.ascontiguousarray(lists["target"][perm])
    lists["ab_sign"] = cp.ascontiguousarray(lists["sign"][perm])
    lists["ab_offset"] = cp.ascontiguousarray((label // m) * (m * m) + label % m)
    lists["ab_bounds"] = cp.asnumpy(
        cp.searchsorted(key[perm], cp.arange(2 * len(starts) - 1, dtype=cp.int64))
    )


def _build_lists(cp, a_uniq, b_uniq, bra, m, ne, two, lanes, starts):

    alpha = _sector_lists(cp, a_uniq, bra[0], m, ne[0], two, lanes, starts[0])
    beta = _sector_lists(cp, b_uniq, bra[1], m, ne[1], two, lanes, starts[1])
    if two:
        groups = cp.arange(m * m, dtype=cp.int64)
        alpha["label_offset"] = cp.ascontiguousarray(
            (groups // m) * (m * m * m) + (groups % m) * m
        )
        _beta_order(cp, beta, m, starts[1])
    return alpha, beta


def _needed_tiles(ham, me, threads):


    from .tile_discovery import _tile_source_needed  # noqa: PLC0415

    alpha = np.asarray(ham._alpha_uniq, dtype=np.uint64)
    beta = np.asarray(ham._beta_uniq, dtype=np.uint64)
    a0, a1, b0, b1, _ca, _cb = ham._rank_tiles[me]
    needed = np.zeros(len(ham._rank_tiles), dtype=np.int32)
    for rank, (s0, s1, t0, t1, _sa, _sb) in enumerate(ham._rank_tiles):
        if rank == me:
            needed[rank] = 1
        elif s1 > s0 and t1 > t0 and a1 > a0 and b1 > b0:
            needed[rank] = int(
                _tile_source_needed(
                    alpha[a0:a1],
                    beta[b0:b1],
                    alpha[s0:s1],
                    beta[t0:t1],
                    None,
                    threads=threads,
                )
            )
    return needed


def _check_symmetric(comm, needed):

    wanted_by = np.empty_like(needed)
    comm.Alltoall(needed, wanted_by)
    error = None
    if not np.array_equal(needed, wanted_by):
        error = (
            f"source tiles not symmetric: needs {np.flatnonzero(needed).tolist()}, "
            f"needed by {np.flatnonzero(wanted_by).tolist()}"
        )
    raise_on_every_rank(comm, error, RuntimeError, "RDM source tiles")


def _hops(ham, me, needed):


    tiles = ham._rank_tiles
    p_alpha, p_beta = ham.process_grid
    ca, cb = tiles[me][4], tiles[me][5]
    hops = []
    for k in range(1, p_alpha):
        dst, src = (
            ((ca - k) % p_alpha) * p_beta + cb,
            ((ca + k) % p_alpha) * p_beta + cb,
        )
        hops.append((dst if needed[dst] else -1, src if needed[src] else -1))
    for k in range(1, p_beta):
        dst, src = ca * p_beta + (cb - k) % p_beta, ca * p_beta + (cb + k) % p_beta
        hops.append((dst if needed[dst] else -1, src if needed[src] else -1))
    for da in range(1, p_alpha):
        for db in range(1, p_beta):
            dst = ((ca + da) % p_alpha) * p_beta + (cb + db) % p_beta
            src = ((ca - da) % p_alpha) * p_beta + (cb - db) % p_beta
            hops.append((dst if needed[dst] else -1, src if needed[src] else -1))
    return hops


def _exchange(backend, comm, own, slot, hop, count):

    dst, src = hop
    empty64 = np.zeros(0, dtype=np.int64)
    empty32 = np.zeros(0, dtype=np.int32)
    zero = np.zeros(1, dtype=np.int64)
    sends, receives = dst >= 0, src >= 0
    backend.grouped_sendrecv_device(
        own if sends else own[:0],
        np.asarray([own.size], dtype=np.int64) if sends else empty64,
        zero if sends else empty64,
        np.asarray([dst], dtype=np.int32) if sends else empty32,
        slot if receives else slot[:0],
        np.asarray([count], dtype=np.int64) if receives else empty64,
        zero if receives else empty64,
        np.asarray([src], dtype=np.int32) if receives else empty32,
        comm,
    )


def _apart_from(cp, buffers, state):

    if not isinstance(state, cp.ndarray):
        return list(buffers or [])
    lo = int(state.data.ptr)
    hi = lo + int(state.nbytes)
    return [
        b
        for b in (buffers or [])
        if int(b.data.ptr) + int(b.nbytes) <= lo or int(b.data.ptr) >= hi
    ]


def _upload_into(cp, target, state):

    if isinstance(state, cp.ndarray):
        target[...] = state.reshape(target.shape).astype(cp.float64, copy=False)
    else:
        target.set(
            np.ascontiguousarray(
                np.asarray(state, dtype=np.float64).reshape(target.shape)
            )
        )


def _upload_transposed(cp, target, state, rows, cols):


    if isinstance(state, cp.ndarray):
        target[...] = state.reshape(rows, cols).T
    else:
        host = np.asarray(state, dtype=np.float64).reshape(rows, cols)
        target.set(np.ascontiguousarray(host.T))


def _row_sqnorm_of_columns(cp, slab, rows, cols, budget=1 << 24):

    w = cp.zeros(cols, cp.float64)
    step = max(1, budget // max(1, cols))
    for r0 in range(0, rows, step):
        block = slab[r0 : min(rows, r0 + step)]
        w += (block * block).sum(axis=0)
    return w


def _lent_elems(wants, sizes):

    free, lent = sorted(sizes), 0
    for want in sorted(wants, reverse=True):
        fit = next((i for i, size in enumerate(free) if size >= want), None)
        if fit is not None:
            free.pop(fit)
            lent += want
    return lent


def _relay_tiles(cp, comm, backend, acc, tiles, needed, moves, hops, peer):


    from ._collective import BACKEND_NCCL  # noqa: PLC0415


    overlap = acc.moving is not None and getattr(backend, "name", None) == BACKEND_NCCL
    copy_stream = cp.cuda.Stream(non_blocking=True, priority=-1000) if overlap else None
    sizes = [(t[1] - t[0]) * (t[3] - t[2]) for t in tiles]
    me = comm.Get_rank()
    cur, inc = acc.slot, acc.moving
    if acc.moving is None:
        _relay_in_pieces(
            cp, comm, backend, acc, tiles, needed, moves, hops, peer, sizes
        )
        return
    _relay_start(cp, comm, acc, tiles[me], cur)
    zero = np.zeros(1, dtype=np.int64)
    held = (0, 0)
    for index, ((_step, offset), (dst, src)) in enumerate(zip(moves, hops)):
        previous = peer(*held)
        sent = sizes[previous]

        def step():
            """Launch this hop's transfer, then the kernels of the tile that arrived in the last hop."""
            backend.grouped_sendrecv_device(
                cur,
                np.asarray([sent], np.int64),
                zero,
                np.asarray([dst], np.int32),
                inc,
                np.asarray([sizes[peer(*offset)]], np.int64),
                zero,
                np.asarray([src], np.int32),
                comm,
                stream=copy_stream,
            )
            if previous != me and needed[previous]:
                acc.tile(tiles[previous], cur)

        _local(cp, comm, f"RDM hop {index}", step)
        cur, inc = inc, cur
        held = offset
    last = peer(*held)

    _local(
        cp,
        comm,
        "RDM last tile",
        lambda: (
            acc.tile(tiles[last], cur)
            if moves and last != me and needed[last]
            else None
        ),
    )


def _connect_neighbours(cp, comm, backend, tiles, grid):


    p_alpha, p_beta = grid
    me = comm.Get_rank()
    at = {(int(t[4]), int(t[5])): r for r, t in enumerate(tiles)}
    ca, cb = int(tiles[me][4]), int(tiles[me][5])
    peers = sorted(
        {
            at[((ca + da) % p_alpha, (cb + db) % p_beta)]
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))
        }
        - {me}
    )

    def connect():

        if not peers:
            return
        k = len(peers)
        send, recv = cp.zeros(8 * k), cp.zeros(8 * k)
        counts = np.full(k, 8, np.int64)
        displs = np.arange(k, dtype=np.int64) * 8
        ranks = np.asarray(peers, np.int32)
        backend.grouped_sendrecv_device(
            send, counts, displs, ranks, recv, counts, displs, ranks, comm
        )

    _local(cp, comm, "RDM neighbour connections", connect)


def _relay_start(cp, comm, acc, own_tile, cur):


    nal, nbl = acc.a1 - acc.a0, acc.b1 - acc.b0

    def start():

        cur[: nal * nbl].reshape(nbl, nal)[...] = acc.own_alpha.T

    _local(cp, comm, "RDM relay start", start)
    _local(cp, comm, "RDM own tile", lambda: acc.tile(own_tile, None, own_beta=cur))


def _relay_in_pieces(cp, comm, backend, acc, tiles, needed, moves, hops, peer, sizes):


    me = comm.Get_rank()
    cur, piece = acc.slot, acc.piece
    chunk = int(piece.size)

    pieces = -(-max(sizes) // chunk)
    _relay_start(cp, comm, acc, tiles[me], cur)
    held = (0, 0)
    for index, ((_step, offset), (dst, src)) in enumerate(zip(moves, hops)):
        sent, received = sizes[peer(*held)], sizes[peer(*offset)]
        arrived = peer(*offset)

        def step():
            """Send-and-replace the moving tile piece by piece, then launch the kernels of the arrived tile."""
            for lo in range(0, pieces * chunk, chunk):
                n_send = max(0, min(chunk, sent - lo))
                n_recv = max(0, min(chunk, received - lo))
                _exchange(
                    backend,
                    comm,
                    cur[lo : lo + max(n_send, 1)],
                    piece,
                    (dst if n_send else -1, src if n_recv else -1),
                    n_recv,
                )
                if n_recv:
                    cur[lo : lo + n_recv] = piece[:n_recv]
                cp.cuda.runtime.deviceSynchronize()
            if arrived != me and needed[arrived]:
                acc.tile(tiles[arrived], cur)

        _local(cp, comm, f"RDM hop {index}", step)
        held = offset


def _plan(comm, prefer_relay, tile, max_tile, lent_sizes, lists_bytes, edges, strings):


    allowance = max(8 * BUDGET_VECTORS * tile, BUDGET_FLOOR)
    small = edges + strings

    def fresh(relay, piece):

        wants = [max_tile, max_tile, tile] + ([] if relay else [tile])
        if relay:
            wants.append(max_tile if piece == 0 else piece)
        return 8 * (sum(wants) - _lent_elems(wants, lent_sizes) + small)

    def best_piece():

        room = (allowance - lists_bytes - fresh(True, MIN_PIECE)) // 8 + MIN_PIECE
        return int(min(max_tile, max(MIN_PIECE, room)))

    choices = [(prefer_relay, 0), (True, best_piece())]

    mine = next(
        (
            i
            for i, (relay, piece) in enumerate(choices)
            if lists_bytes + fresh(relay, piece) <= allowance
        ),
        len(choices) - 1,
    )
    index = max(comm.allgather(mine))
    relay, piece = choices[index]
    if _PIECE_FOR_TESTING is not None and relay:
        piece = int(_PIECE_FOR_TESTING)
    if relay and piece:
        piece = min(
            comm.allgather(piece)
        )
    need = fresh(relay, piece)
    over = max(comm.allgather(lists_bytes + need - allowance))
    return relay, piece, need, over


def _relay_moves(p_alpha, p_beta):


    moves, d_alpha, d_beta, sign = [], 0, 0, 1
    for row in range(p_alpha):
        for _ in range(p_beta - 1):
            d_beta += sign
            moves.append(((0, sign), (d_alpha, d_beta)))
        if row < p_alpha - 1:
            d_alpha += 1
            moves.append(((1, 0), (d_alpha, d_beta)))
        sign = -sign
    return moves


def _use_relay(comm, tiles, needed, p_alpha, p_beta):


    me = comm.Get_rank()
    sizes = (
        np.asarray(
            [(s1 - s0) * (t1 - t0) for s0, s1, t0, t1, _sa, _sb in tiles], np.float64
        )
        * 8
    )
    ca, cb = tiles[me][4], tiles[me][5]
    neighbours = {
        ((ca + da) % p_alpha) * p_beta + (cb + db) % p_beta
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))
    } - {me}
    needed_bytes = float(sizes[np.flatnonzero(needed)].sum()) - float(sizes[me])
    direct = (
        int(needed.sum()) - 1
    ) * CONNECT_SECONDS + needed_bytes / RELAY_BYTES_PER_SECOND
    relay = (
        len(neighbours) * CONNECT_SECONDS
        + (float(sizes.sum()) - float(sizes[me])) / RELAY_BYTES_PER_SECOND
    )
    costs = comm.allgather((direct, relay))
    if _ROUTE_FOR_TESTING is not None:
        return _ROUTE_FOR_TESTING == "relay"
    return max(c[1] for c in costs) < max(c[0] for c in costs)


class _Accumulator:


    def __init__(
        self,
        cp,
        ham,
        evec_local,
        alpha,
        beta,
        m,
        two,
        max_tile,
        scratch=None,
        relay=False,
        piece=0,
    ):


        self.cp, self.alpha, self.beta, self.m, self.two = cp, alpha, beta, m, two
        self.a0, self.a1, self.b0, self.b1, self.ca, self.cb = ham._rank_tiles[
            ham._comm.Get_rank()
        ]
        nal, nbl = self.a1 - self.a0, self.b1 - self.b0
        scratch = _apart_from(cp, scratch, evec_local)


        self.one_own = bool(relay)
        names = ["slot", "slot_alpha"] + (["moving"] if relay and not piece else [])
        names += ["own_alpha"] if self.one_own else ["own", "own_alpha"]
        names += ["piece"] if relay and piece else []
        wants = {
            name: max(
                1,
                int(
                    max_tile if name in ("slot", "slot_alpha", "moving") else nal * nbl
                ),
            )
            for name in names
        }
        if "piece" in wants:
            wants["piece"] = max(1, int(piece))
        free = sorted(scratch or [], key=lambda b: b.size)
        got = {}
        for name in names:
            fit = next((i for i, b in enumerate(free) if b.size >= wants[name]), None)
            got[name] = (
                free.pop(fit)[: wants[name]]
                if fit is not None
                else cp.empty(wants[name], cp.float64)
            )
        self.borrowed = len(scratch or []) - len(free)
        self.own_alpha = got["own_alpha"].reshape(nal, nbl)
        if self.one_own:
            self.own = None
            _upload_transposed(cp, self.own_alpha, evec_local, nbl, nal)
        else:
            self.own = got["own"].reshape(nbl, nal)
            _upload_into(cp, self.own, evec_local)
            self.own_alpha[...] = self.own.T
        self.slot, self.slot_alpha = got["slot"], got["slot_alpha"]
        self.moving = got.get("moving")
        self.piece = got.get("piece")
        self.gamma = [cp.zeros(m * m, cp.float64), cp.zeros(m * m, cp.float64)]
        self.blocks = [cp.zeros(m**4, cp.float64) for _ in range(3)] if two else None
        self.overlap = (
            [
                cp.empty(max(1, int(alpha["cols"].size)), cp.float64),
                cp.empty(max(1, int(beta["cols"].size)), cp.float64),
            ]
            if two
            else None
        )

    def tile(self, tile, values, own_beta=None):


        s0, s1, t0, t1, sa, sb = tile
        rows, cols = t1 - t0, s1 - s0
        if values is None:
            own_beta = self.own if not self.one_own else own_beta
            if own_beta is None:
                raise RuntimeError(
                    "the own tile's beta-outer values are needed (relay start not run)"
                )
            nal, nbl = self.a1 - self.a0, self.b1 - self.b0
            beta_outer, alpha_major = (
                own_beta[: nal * nbl].reshape(nbl, nal),
                self.own_alpha,
            )
        else:
            beta_outer = values[: rows * cols].reshape(rows, cols)
            alpha_major = self.slot_alpha[: rows * cols].reshape(cols, rows)
            alpha_major[...] = beta_outer.T
        if sb == self.cb:
            self._sector(
                0,
                sa,
                (self.own_alpha, alpha_major, self.b1 - self.b0, self.a0, s0),
                self.a0,
                self.a1 - self.a0,
            )
        if sa == self.ca:
            self._sector(
                1, sb, self._beta_bra(beta_outer, t0), self.b0, self.b1 - self.b0
            )
        if self.two:
            self._alpha_beta(sa, sb, alpha_major, rows, s0, t0)
        if values is None:
            self._diagonal()

    def _beta_bra(self, ket, ket_offset):


        nal, nbl = self.a1 - self.a0, self.b1 - self.b0
        if not self.one_own:
            return (self.own, ket, nal, self.b0, ket_offset)
        return (self.own_alpha, ket, nal, self.b0, ket_offset, 1, nbl)

    def _sector(self, index, block, slabs, lo, n_bra):

        from ..rdm import kernel_launch  # noqa: PLC0415

        lists = self.alpha if index == 0 else self.beta
        other, column_offset = (
            (self.beta, self.b0) if index == 0 else (self.alpha, self.a0)
        )
        m = self.m
        stream = _stream(self.cp)
        pointers = (
            int(slabs[0].data.ptr),
            int(slabs[1].data.ptr),
            *(int(x) for x in slabs[2:]),
        )
        first, last = (
            int(lists["single_bounds"][2 * block]),
            int(lists["single_bounds"][2 * block + 1]),
        )
        if last > first:
            kernel_launch.launch_one_body(
                index == 1,
                pointers,
                lists["source"][first:].data.ptr,
                lists["target"][first:].data.ptr,
                lists["sign"][first:].data.ptr,
                lists["label"][first:].data.ptr,
                last - first,
                self.gamma[index].data.ptr,
                self.blocks[1].data.ptr if self.two else 0,
                other["keys"].data.ptr,
                column_offset,
                m,
                stream,
            )
        if not self.two or n_bra == 0:
            return
        row_ptr = lists["edge_ptr"][block * n_bra :]
        out = self.blocks[0 if index == 0 else 2]
        kernel_launch.launch_pair_overlap(
            pointers,
            row_ptr.data.ptr,
            lists["cols"].data.ptr,
            self.overlap[index].data.ptr,
            lists["keys"].data.ptr,
            lo,
            n_bra,
            stream,
        )
        kernel_launch.launch_same_scatter(
            row_ptr.data.ptr,
            lists["cols"].data.ptr,
            self.overlap[index].data.ptr,
            lists["keys"].data.ptr,
            lo,
            n_bra,
            out.data.ptr,
            m,
            stream,
        )

    def _alpha_beta(self, alpha_block, beta_block, source, source_width, s0, t0):

        from ..rdm import kernel_launch  # noqa: PLC0415

        alpha, beta, mm = self.alpha, self.beta, self.m * self.m
        first, last = (
            int(beta["ab_bounds"][2 * beta_block]),
            int(beta["ab_bounds"][2 * beta_block + 1]),
        )
        if (
            last == first
            or alpha["single_bounds"][2 * alpha_block]
            == alpha["single_bounds"][2 * alpha_block + 1]
        ):
            return
        tiles = {
            "own": int(self.own_alpha.data.ptr),
            "source": int(source.data.ptr),
            "own_width": self.b1 - self.b0,
            "source_width": int(source_width),
            "own_row_offset": self.a0,
            "own_column_offset": self.b0,
            "source_row_offset": int(s0),
            "source_column_offset": int(t0),
        }
        pointers = {
            "source": int(alpha["source"].data.ptr),
            "target": int(alpha["target"].data.ptr),
            "sign": int(alpha["sign"].data.ptr),
            "label_start": int(
                alpha["single_start"][alpha_block * (mm + 1) :].data.ptr
            ),
            "offset": int(alpha["label_offset"].data.ptr),
        }
        kernel_launch.launch_alpha_beta(
            tiles,
            pointers,
            beta["ab_source"][first:].data.ptr,
            beta["ab_target"][first:].data.ptr,
            beta["ab_sign"][first:].data.ptr,
            beta["ab_offset"][first:].data.ptr,
            last - first,
            self.blocks[1].data.ptr,
            self.m,
            _stream(self.cp),
        )

    def _diagonal(self):


        from ..rdm import kernel_launch  # noqa: PLC0415

        cp, m = self.cp, self.m
        stream = _stream(cp)
        nal, nbl = self.a1 - self.a0, self.b1 - self.b0
        block = self.blocks[1].data.ptr if self.two else 0
        kernel_launch.launch_diagonal(
            False,
            (
                int(self.own_alpha.data.ptr),
                int(self.own_alpha.data.ptr),
                nbl,
                self.a0,
                self.a0,
            ),
            self.alpha["keys"].data.ptr,
            self.a0,
            nal,
            self.gamma[0].data.ptr,
            block,
            self.beta["keys"].data.ptr,
            self.b0,
            m,
            stream,
        )
        beta_own = (
            (
                int(self.own_alpha.data.ptr),
                int(self.own_alpha.data.ptr),
                nal,
                self.b0,
                self.b0,
                1,
                nbl,
            )
            if self.one_own
            else (int(self.own.data.ptr), int(self.own.data.ptr), nal, self.b0, self.b0)
        )
        kernel_launch.launch_diagonal(
            True,
            beta_own,
            self.beta["keys"].data.ptr,
            self.b0,
            nbl,
            self.gamma[1].data.ptr,
            0,
            self.alpha["keys"].data.ptr,
            self.a0,
            m,
            stream,
        )
        if not self.two:
            return
        weights = (
            _col_sqnorm(cp, self.own_alpha.reshape(-1), 0, nal, nbl),
            _row_sqnorm_of_columns(cp, self.own_alpha, nal, nbl)
            if self.one_own
            else _col_sqnorm(cp, self.own.reshape(-1), 0, nbl, nal),
        )
        for weight, lists, lo, hi, out in (
            (weights[0], self.alpha, self.a0, self.a1, self.blocks[0]),
            (weights[1], self.beta, self.b0, self.b1, self.blocks[2]),
        ):
            kernel_launch.launch_same_diagonal(
                weight.data.ptr,
                lists["keys"].data.ptr,
                lo,
                hi - lo,
                out.data.ptr,
                m,
                stream,
            )

    def host(self):

        cp, m = self.cp, self.m
        gamma = tuple(
            np.ascontiguousarray(cp.asnumpy(g).reshape(m, m)) for g in self.gamma
        )
        if not self.two:
            return gamma, None
        g_aa = np.ascontiguousarray(cp.asnumpy(self.blocks[0]).reshape(m, m, m, m))
        g_ab = np.ascontiguousarray(
            cp.asnumpy(self.blocks[1]).reshape(m, m, m, m).transpose(0, 2, 1, 3)
        )
        g_bb = np.ascontiguousarray(cp.asnumpy(self.blocks[2]).reshape(m, m, m, m))
        return gamma, (g_aa, g_ab, g_bb)


def compute_rdms_cart(ham, evec_local, *, rank=1, spin_summed=False):
    """Compute reduced density matrices for a distributed tensor-product state.

    This is the implementation used by ``TileGPUResidentHamiltonian.compute_rdms``.
    Every communicator rank supplies its local part of the same real state and
    calls with the same ``rank`` and ``spin_summed`` options. All ranks receive
    the full result as host NumPy ``float64`` arrays after GPU work and collective
    communication complete.

    The subspace must have equal alpha and beta orbital counts, at most 64 per
    sector, and a fixed electron count within each sector. Local amplitudes use
    the Hamiltonian's sorted half-string order, flattened with beta outermost
    and alpha innermost: local index ``b * ham.n_a_local + a``.

    No normalization is applied. Scaling the full state by a real factor scales
    every RDM by its square. Normalize the full state before distributing it if
    you need RDMs for a normalized state. Spin and tensor-index conventions follow
    PySCF ``make_rdm12s(reorder=True)``.

    Args:
        ham: Constructed ``TileGPUResidentHamiltonian`` supplying the product
            subspace, local partition, communicator and GPU execution backend.
        evec_local: NumPy-compatible or CuPy array containing ``ham.n_local`` real
            amplitudes, or shape ``(ham.n_b_local, ham.n_a_local)``. Typically one
            local eigenvector column returned by ``distributed_primme_eigsh``.
            CuPy input must be on this rank's GPU.
        rank: RDM order, 1 or 2; defaults to 1. This is not an MPI rank.
        spin_summed: Combine alpha and beta spin blocks; defaults to False.

    Returns:
        For ``rank=1``, ``(gamma_a, gamma_b)`` or spin-summed ``gamma``.
        For ``rank=2``, ``((gamma_a, gamma_b), (G_aa, G_ab, G_bb))`` or
        spin-summed ``(gamma, G)``. Each one-particle array has shape ``(m, m)``
        and each two-particle array ``(m, m, m, m)``, where ``m`` is the common
        orbital count per sector. The returned arrays are replicated on all ranks.

    Raises:
        ImportError: CuPy is unavailable.
        ValueError: The RDM order, local input shape, orbital counts or sector
            electron counts are invalid.
        MemoryError: RDM storage cannot fit within the available device memory
            or the RDM memory budget.
        RuntimeError: GPU execution or collective communication fails.
    """
    import cupy as cp

    from .exchange_plan import check_pairing  # noqa: PLC0415

    started = time.perf_counter()

    if rank not in (1, 2):
        raise ValueError(f"rank must be 1 or 2, got {rank!r}")
    m = int(ham._num_alpha)
    if int(ham._num_beta) != m:
        raise ValueError(
            f"the RDM needs equal alpha and beta orbital counts, got {m} and {int(ham._num_beta)}"
        )
    if m > MAX_ORBITALS:
        raise ValueError(
            f"the RDM supports at most {MAX_ORBITALS} orbitals per sector, got {m}"
        )
    a_uniq = np.asarray(ham._alpha_uniq, dtype=np.uint64)
    b_uniq = np.asarray(ham._beta_uniq, dtype=np.uint64)
    ne_a, ne_b = _fixed_ne(a_uniq), _fixed_ne(b_uniq)
    if ne_a is None or ne_b is None:
        raise ValueError("the RDM needs one electron count in each spin sector")

    comm, backend = ham._comm, ham._backend
    mpi = mpi_namespace(comm)
    me = comm.Get_rank()
    tiles = ham._rank_tiles
    a0, a1, b0, b1, _ca, _cb = tiles[me]
    starts = (
        np.asarray([lo for lo, _hi in ham._alpha_ranges] + [a_uniq.size], np.int64),
        np.asarray([lo for lo, _hi in ham._beta_ranges] + [b_uniq.size], np.int64),
    )
    two = rank == 2
    log = _Log(cp, comm)


    base = _device_used(cp, comm)
    lanes = _lanes(_device_share(cp, comm))
    alpha, beta = _local(
        cp,
        comm,
        "RDM pair lists",
        lambda: _build_lists(
            cp,
            a_uniq,
            b_uniq,
            ((a0, a1), (b0, b1)),
            m,
            (ne_a, ne_b),
            two,
            lanes,
            starts,
        ),
    )

    cp.get_default_memory_pool().free_all_blocks()


    lists_bytes = max(0, _device_used(cp, comm) - base)
    _connect_neighbours(cp, comm, backend, tiles, ham.process_grid)
    connect_bytes = max(0, _device_used(cp, comm) - base - lists_bytes)
    log(
        f"lists lanes={lanes:,} lists={lists_bytes / 2**30:.2f} GiB connections={connect_bytes / 2**30:.2f} GiB"
    )


    threads = max(1, int(getattr(ham, "_build_threads", 1) or 1))
    needed = _local(
        cp, comm, "RDM source tiles", lambda: _needed_tiles(ham, me, threads)
    )
    _check_symmetric(comm, needed)
    p_alpha, p_beta = ham.process_grid
    max_tile = max((s1 - s0) * (t1 - t0) for s0, s1, t0, t1, _sa, _sb in tiles)
    tile = (a1 - a0) * (b1 - b0)
    edges = (int(alpha["cols"].size) + int(beta["cols"].size)) if two else 0
    tensors = 3 * m**4 + 2 * m * m if two else 2 * m * m


    scratch = _apart_from(
        cp, ham.rdm_scratch() if hasattr(ham, "rdm_scratch") else [], evec_local
    )
    relay, piece, need, over = _plan(
        comm,
        _use_relay(comm, tiles, needed, p_alpha, p_beta),
        tile,
        max_tile,
        [int(b.size) for b in scratch],
        lists_bytes,
        edges,
        (a1 - a0) + (b1 - b0),
    )
    if relay:
        at = {(int(t[4]), int(t[5])): r for r, t in enumerate(tiles)}
        ca, cb = int(tiles[me][4]), int(tiles[me][5])

        def peer(da, db):
            """Return the world rank at grid offset (da, db) from this rank."""
            return at[((ca + da) % p_alpha, (cb + db) % p_beta)]

        moves = _relay_moves(p_alpha, p_beta)
        hops = [(peer(-step[0], -step[1]), peer(*step)) for step, _offset in moves]
    else:
        hops = _hops(ham, me, needed)
    check_pairing(comm, hops, "RDM exchange")
    if log.on and two:
        k_a, k_b = int(alpha["source"].size), int(beta["source"].size)
        off_a = k_a - int((alpha["source"] == alpha["target"]).sum())
        off_b = k_b - int((beta["source"] == beta["target"]).sum())
        log.logger.debug(
            "Alpha-beta RDM work: candidate_products=%d, off_diagonal_products=%d",
            k_a * k_b,
            off_a * off_b,
        )
    log(f"sources {int(needed.sum()) - 1} hops {len(hops)}")

    share = _device_share(cp, comm)
    if log.on:
        plan = (
            "direct"
            if not relay
            else ("relay overlap" if piece == 0 else f"relay pieces of {piece:,}")
        )
        log.logger.debug(
            f"RDM execution plan: rank=0, grid={p_alpha}x{p_beta}, tile_elements={tile:,}, "
            f"exchange={plan}, required_working_memory={need / 2**30:.2f} GiB "
            f"lists={lists_bytes / 2**30:.2f} GiB connections={connect_bytes / 2**30:.2f} GiB "
            f"budget={max(BUDGET_VECTORS * 8 * tile, BUDGET_FLOOR) / 2**30:.2f} GiB "
            f"over_eigsolve_peak={max(0, over) / 2**30:.2f} GiB share={share / 2**30:.2f} GiB"
        )
    error = None
    if over > 0:
        error = (
            f"the RDM needs {over / 2**30:.2f} GiB of GPU memory more than the eigensolver's peak "
            f"(its two {8 * tile / 2**30:.2f} GiB vectors); use more GPUs"
        )
    raise_on_every_rank(comm, error, MemoryError, "RDM memory budget")
    _check_fits(comm, need + 8 * tensors, share, "RDM memory check")

    acc = _local(
        cp,
        comm,
        "RDM buffers",
        lambda: _Accumulator(
            cp,
            ham,
            evec_local,
            alpha,
            beta,
            m,
            two,
            max_tile,
            scratch=scratch,
            relay=relay,
            piece=piece,
        ),
    )
    scratch = None
    if not relay:
        _local(cp, comm, "RDM own tile", lambda: acc.tile(tiles[me], None))
    if relay:
        _relay_tiles(cp, comm, backend, acc, tiles, needed, moves, hops, peer)
    else:
        own = acc.own.reshape(-1)
        for index, hop in enumerate(hops):
            source = hop[1]
            count = (
                (tiles[source][1] - tiles[source][0])
                * (tiles[source][3] - tiles[source][2])
                if source >= 0
                else 0
            )
            _exchange(backend, comm, own, acc.slot, hop, count)
            _local(
                cp,
                comm,
                f"RDM hop {index}",
                lambda: acc.tile(tiles[source], acc.slot) if source >= 0 else None,
            )
    log("tiles")

    gamma, blocks = _local(cp, comm, "RDM host copy", acc.host)
    acc = alpha = beta = None
    cp.get_default_memory_pool().free_all_blocks()
    gamma_a, gamma_b = gamma
    comm.Allreduce(mpi.IN_PLACE, [gamma_a, mpi.DOUBLE])
    comm.Allreduce(mpi.IN_PLACE, [gamma_b, mpi.DOUBLE])
    if not two:
        if me == 0:
            log.logger.info(
                "RDM calculation completed: order=1, elapsed=%.2f s",
                time.perf_counter() - started,
            )
        return (gamma_a + gamma_b) if spin_summed else (gamma_a, gamma_b)
    g_aa, g_ab, g_bb = blocks
    for block in (g_aa, g_ab, g_bb):
        comm.Allreduce(mpi.IN_PLACE, [block, mpi.DOUBLE])
    if me == 0:
        log.logger.info(
            "RDM calculation completed: order=2, elapsed=%.2f s",
            time.perf_counter() - started,
        )
    if spin_summed:
        return gamma_a + gamma_b, g_aa + g_bb + g_ab + g_ab.transpose(2, 3, 0, 1)
    return (gamma_a, gamma_b), (g_aa, g_ab, g_bb)
