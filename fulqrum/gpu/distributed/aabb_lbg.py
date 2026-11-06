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


from dataclasses import dataclass

import numpy as np

SEGMENT_WORDS = 128
UNIFORM_MIN = 16
RSL_ROWS_PER_JOB = 4
MAX_ROWS = 65535


@dataclass
class LbgBucket:


    bucket: int
    first: int
    count: int
    panel_row: int
    out_stride: int
    per_launch: int


@dataclass
class RslGroup:


    members_per_clique: int
    block_first: object
    block_count: object
    members: object
    coef_table: object
    num_blocks: int
    per_launch: int


@dataclass
class LbgPlan:


    edge_words: object
    segment_begin: object
    segment_end: object
    segment_pair: object
    num_segments: int
    panel_columns: int
    buckets: tuple
    rsl: tuple
    rsl_panel_rows: int
    scratch_doubles: int
    stats: dict


def retile(edge_words, group_offsets, group_pairs, tile_group_offsets, tile_rows):


    words = edge_words.astype(np.uint64)
    goff = group_offsets.astype(np.int64)
    gpair = group_pairs.astype(np.int64)
    toff = tile_group_offsets.astype(np.int64)
    gtile = np.repeat(np.arange(len(toff) - 1), np.diff(toff))
    gsize = np.diff(goff)
    etile = np.repeat(gtile, gsize)
    epair = np.repeat(gpair, gsize)
    erow = etile * int(tile_rows) + (
        (words >> np.uint64(32)) & np.uint64(0xFFFF)
    ).astype(np.int64)
    if len(erow) and int(erow.max()) > MAX_ROWS:
        raise ValueError(
            f"LBG needs at most {MAX_ROWS + 1} output rows (16-bit row field)"
        )
    order = np.lexsort((erow, epair))
    out = (words & ~(np.uint64(0xFFFF) << np.uint64(32))) | (
        erow.astype(np.uint64) << np.uint64(32)
    )
    out, epair, erow = out[order], epair[order], erow[order]
    starts = (
        np.flatnonzero(np.r_[True, epair[1:] != epair[:-1]])
        if len(epair)
        else np.zeros(0, np.int64)
    )
    goff_n = np.r_[starts, len(epair)].astype(np.int64)
    gid = np.repeat(np.arange(len(starts)), np.diff(goff_n))
    if len(np.unique(gid * (MAX_ROWS + 2) + erow)) != len(erow):
        raise ValueError("LBG needs distinct rows inside a beta-pair group")
    return out, goff_n, epair[starts].astype(np.int32)


def segments(goff, gpair, seg=SEGMENT_WORDS):

    lengths = np.diff(goff)
    per = (lengths + seg - 1) // seg
    group = np.repeat(np.arange(len(gpair)), per)
    k = np.arange(len(group)) - np.repeat(np.cumsum(per) - per, per)
    begin = goff[group] + k * seg
    end = np.minimum(begin + seg, goff[group + 1])
    return begin.astype(np.int32), end.astype(np.int32), gpair[group].astype(np.int32)


def small_cliques(
    alpha_strings,
    residual_off,
    residual_lengths,
    residual_pack,
    residual_order,
    num_alpha,
):


    from .aabb_parents import _residual_sell, choose_parent_plan, unpack_sell  # noqa: PLC0415

    rows, words = unpack_sell(residual_off, residual_lengths, residual_pack)
    if residual_order is not None:
        rows = residual_order[rows]
    srt = np.argsort(rows, kind="stable")
    off, lengths, pack = _residual_sell(rows[srt], words[srt], len(alpha_strings))
    plan = choose_parent_plan(
        np.asarray(alpha_strings, dtype=np.uint64),
        off,
        lengths,
        pack,
        num_alpha,
        2,
        kind="auto",
    )
    if plan["stats"]["residual_edges"]:
        raise ValueError(
            f"RSL needs the residual to be complete cliques; {plan['stats']['residual_edges']} edges "
            "are left over"
        )
    ptr = plan["parent_ptr"].astype(np.int64)
    mem = plan["members"]
    sizes = np.diff(ptr)
    if len(sizes) and (sizes.min() < 2 or sizes.max() > 5):
        raise ValueError("RSL needs residual cliques of 2 to 5 members")
    groups = {}
    for p in range(len(ptr) - 1):
        ms = mem[ptr[p] : ptr[p + 1]]
        orbs = tuple(((ms >> np.uint64(32)) & np.uint64(0xFF)).astype(int).tolist())
        groups.setdefault((len(ms), orbs), []).append(ms)
    out, pools = {}, {}
    for (m, _orbs), cl in sorted(groups.items()):
        if len(cl) < UNIFORM_MIN:
            pools.setdefault(m, []).extend(cl)
            continue
        firsts, counts, members = out.setdefault((m, False), ([], [], []))
        base = len(members) // m
        for s in range(0, len(cl), 32):
            firsts.append(base + s)
            counts.append(min(32, len(cl) - s))
        for ms in cl:
            members.extend(ms.tolist())
    for m, cl in pools.items():
        firsts, counts, members = out.setdefault((m, True), ([], [], []))
        for s in range(0, len(cl), 32):
            firsts.append(s)
            counts.append(min(32, len(cl) - s))
        for ms in cl:
            members.extend(ms.tolist())
    return {
        k: (np.asarray(f, np.int32), np.asarray(c, np.int32), np.asarray(mm, np.uint64))
        for k, (f, c, mm) in out.items()
    }


def mixed_table(bf, bc, mem, m, coefficients_by_pair, orbitals):

    orb = ((mem >> np.uint64(32)) & np.uint64(0xFF)).astype(np.int64).reshape(-1, m)
    ii, jj = np.triu_indices(m, 1)
    offs = np.zeros((len(bf), m * (m - 1) // 2, 32), np.int64)
    valid = np.zeros((len(bf), 1, 32), bool)
    for bi in range(len(bf)):
        c = int(bc[bi])
        o = orb[bf[bi] : bf[bi] + c]
        offs[bi, :, :c] = (o[:, ii] * orbitals + o[:, jj]).T
        valid[bi, 0, :c] = True
    table = coefficients_by_pair[:, offs] * valid[None]
    return np.ascontiguousarray(table.transpose(1, 0, 2, 3))


def build_plan(
    cross,
    panel_row,
    n_rows,
    n_src_rows,
    n_src_cols,
    budget_bytes,
    alpha_strings,
    num_alpha,
    with_rsl=True,
):


    import cupy  # noqa: PLC0415

    if cross.transposed:
        raise ValueError("LBG is for sources that are not transposed")
    host = {
        k: cupy.asnumpy(getattr(cross, k))
        for k in ("edge_words", "group_offsets", "group_pairs", "tile_group_offsets")
    }
    words, goff, gpair = retile(
        host["edge_words"],
        host["group_offsets"],
        host["group_pairs"],
        host["tile_group_offsets"],
        cross.tile_rows,
    )
    sb, se, sp = segments(goff, gpair)
    columns = int((words & np.uint64(0xFFFFFFFF)).max()) + 1 if len(words) else 0
    start = cross.bucket_start
    per_parent = []
    for b in range(len(start) - 1):
        nmax = 8 + 4 * b
        out_stride = 8 * ((nmax + 7) // 8)
        per_parent.append((columns * panel_row(b) + n_rows * out_stride, out_stride))
    budget = budget_bytes // 8
    buckets = []
    for b in range(len(start) - 1):
        count = int(start[b + 1]) - int(start[b])
        if count == 0:
            continue
        doubles, out_stride = per_parent[b]
        per = count if not budget else max(1, min(count, budget // doubles))
        buckets.append(
            LbgBucket(b, int(start[b]), count, panel_row(b), out_stride, per)
        )
    lbg_doubles = max(
        (bk.per_launch * per_parent[bk.bucket][0] for bk in buckets), default=0
    )
    rsl, rsl_doubles, stats = (
        [],
        0,
        dict(lbg_words=int(len(words)), segments=int(len(sb)), columns=columns),
    )
    groups = {}
    if with_rsl and cross.residual_edges:
        try:
            tasks = cupy.asnumpy(cross.tasks).reshape(-1, 3)
            if not np.array_equal(tasks[:, 0], np.arange(len(tasks))):
                raise ValueError("RSL needs one residual task per output row, in order")
            groups = small_cliques(
                alpha_strings,
                cupy.asnumpy(cross.residual_off),
                cupy.asnumpy(cross.residual_lengths),
                cupy.asnumpy(cross.residual_pack),
                None
                if cross.residual_order is None
                else cupy.asnumpy(cross.residual_order),
                num_alpha,
            )
        except ValueError as error:
            stats["rsl_unavailable"] = str(error)[:120]
    if groups:
        orbitals = int(cross.orbitals)
        gorb = cupy.asnumpy(cross.coefficients).reshape(-1, orbitals * orbitals)
        for (m, mixed), (bf, bc, mem) in sorted(groups.items()):
            per_block = n_src_rows * m * 32
            per = len(bf) if not budget else max(1, min(len(bf), budget // per_block))
            table = (
                cupy.asarray(mixed_table(bf, bc, mem, m, gorb, orbitals))
                if mixed
                else None
            )
            rsl.append(
                RslGroup(
                    m,
                    cupy.asarray(bf),
                    cupy.asarray(bc),
                    cupy.asarray(mem),
                    table,
                    len(bf),
                    per,
                )
            )
            rsl_doubles = max(rsl_doubles, per * per_block)
        stats.update(
            rsl_blocks={
                f"{g.members_per_clique}{'x' if g.coef_table is not None else 'u'}": g.num_blocks
                for g in rsl
            },
            rsl_table_mb=sum(
                g.coef_table.nbytes for g in rsl if g.coef_table is not None
            )
            / 2**20,
        )
    stats.update(
        lbg_launches=sum(-(-bk.count // bk.per_launch) for bk in buckets),
        rsl_launches=sum(-(-g.num_blocks // g.per_launch) for g in rsl),
    )
    return LbgPlan(
        cupy.asarray(words),
        cupy.asarray(sb),
        cupy.asarray(se),
        cupy.asarray(sp),
        int(len(sb)),
        columns,
        tuple(buckets),
        tuple(rsl),
        int(n_src_rows),
        max(lbg_doubles, rsl_doubles),
        stats,
    )
