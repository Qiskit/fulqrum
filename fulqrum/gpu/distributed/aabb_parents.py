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


INTERSECTION_BIT = 47
PARENT_KINDS = ("union", "intersection")
CLIQUE_BUCKET_MAX_MEMBERS = (8, 12, 16, 20, 24, 28, 32)


CLIQUE_MAX_SHARED_BYTES = 227 * 1024


def clique_shared_bytes(nmax, eb, tile_rows):

    return 4 * (tile_rows * nmax + nmax * eb) * 8


def validate_clique_plan(
    *,
    transposed,
    nparents,
    residual_edges,
    n_beta_edges,
    n_rows,
    orbitals,
    n_b_src,
    n_b_out,
    tile_rows,
    eb,
    grouped_edges,
    max_shared_bytes,
):


    errors = []
    if nparents <= 0:
        errors.append(f"no parents ({nparents})")
    if residual_edges < 0:
        errors.append(f"negative residual edge count {residual_edges}")
    if not 2 <= orbitals <= 64:
        errors.append(f"{orbitals} orbitals, outside 2..64")
    if transposed and residual_edges != n_beta_edges:
        errors.append(
            f"residual edges {residual_edges} differ from the beta edges {n_beta_edges}"
        )
    if transposed and n_b_src != n_b_out:
        errors.append(
            f"source beta count {n_b_src} differs from the output beta count {n_b_out}"
        )
    if not transposed and n_beta_edges <= 0:
        errors.append("no beta edges")
    if grouped_edges is not None:
        start = list(grouped_edges["bucket_start"])
        if not 1 <= tile_rows <= 1024:
            errors.append(f"row tile of {tile_rows} rows, outside 1..1024")
        if eb not in (4, 8):
            errors.append(f"{eb} edges per batch, not 4 or 8")
        ntiles = int(grouped_edges["ntiles"])
        if not (ntiles * tile_rows >= n_rows > (ntiles - 1) * tile_rows):
            errors.append(
                f"{ntiles} row tiles of {tile_rows} rows do not cover {n_rows} rows exactly"
            )
        if int(grouped_edges["ngroups"]) <= 0:
            errors.append("no edge groups")
        if transposed and int(grouped_edges["nwords"]) <= 0:
            errors.append("no tile words")
        if not transposed and int(grouped_edges["nwords"]) != n_beta_edges:
            errors.append(
                f"{grouped_edges['nwords']} tile words for {n_beta_edges} beta edges"
            )
        if (
            len(start) != len(CLIQUE_BUCKET_MAX_MEMBERS) + 1
            or start[0] != 0
            or start[-1] != nparents
        ):
            errors.append(f"bucket starts {start} do not span the {nparents} parents")
        for b, nmax in enumerate(CLIQUE_BUCKET_MAX_MEMBERS):
            if b + 1 < len(start) and start[b + 1] < start[b]:
                errors.append(f"bucket starts {start} decrease")
            elif b + 1 < len(start) and start[b + 1] > start[b]:
                needed = clique_shared_bytes(nmax, eb, tile_rows)
                if needed > min(max_shared_bytes, CLIQUE_MAX_SHARED_BYTES):
                    errors.append(
                        f"bucket {b} needs {needed} bytes of shared memory, more than the "
                        f"{min(max_shared_bytes, CLIQUE_MAX_SHARED_BYTES)} bytes of the device"
                    )
    if errors:
        raise ValueError(
            "clique aabb plan does not fit the clique kernels: " + "; ".join(errors)
        )


def clique_tile_options(environ):


    rows = int(environ.get("FQ_AABB_CLIQUE_OUTPUT_ROWS_PER_TILE", "64"))
    eb = int(environ.get("FQ_AABB_CLIQUE_EDGES_PER_BATCH", "4"))
    if not 1 <= rows <= 1024 or eb not in (4, 8):
        raise ValueError(
            "FQ_AABB_CLIQUE_OUTPUT_ROWS_PER_TILE must be in [1,1024] and "
            "FQ_AABB_CLIQUE_EDGES_PER_BATCH 4 or 8"
        )
    if clique_shared_bytes(32, eb, rows) > CLIQUE_MAX_SHARED_BYTES:
        raise ValueError(
            "FQ_AABB_CLIQUE_OUTPUT_ROWS_PER_TILE too large for the 32-member bucket's shared tile"
        )
    return rows, eb


def clique_edge_group_plan(words, rows, n_rows, tile_rows):


    import numpy as np

    words = np.asarray(words, dtype=np.uint64)
    rows = np.asarray(rows, dtype=np.int64)
    if len(words) != len(rows) or (
        len(rows) and (rows.min() < 0 or rows.max() >= n_rows)
    ):
        raise ValueError("beta words and rows disagree")
    bpid = ((words >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.int64)
    tile = rows // tile_rows
    order = np.lexsort((bpid, tile))
    t_s, b_s, w_s = tile[order], bpid[order], words[order]
    row_in = (rows[order] - t_s * tile_rows).astype(np.uint64)
    ew = (
        (w_s & np.uint64(0xFFFFFFFF))
        | (row_in << np.uint64(32))
        | (w_s & np.uint64(1 << 63))
    )
    key = t_s * 65536 + b_s
    starts = (
        np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        if len(key)
        else np.empty(0, np.int64)
    )

    group_of = (
        np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(key)]))
        if len(key)
        else np.empty(0, np.int64)
    )
    by_row = np.lexsort((row_in, group_of))
    gr, rr = group_of[by_row], row_in[by_row]
    distinct_rows = bool(
        len(key) < 2 or not np.any((gr[1:] == gr[:-1]) & (rr[1:] == rr[:-1]))
    )
    ntiles = (int(n_rows) + tile_rows - 1) // tile_rows
    return dict(
        words=np.ascontiguousarray(ew, dtype=np.uint64),
        group_ptr=_int32(np.r_[starts, len(key)], "clique group offsets"),
        group_bpid=b_s[starts].astype(np.int32),
        tile_group_ptr=_int32(
            np.searchsorted(t_s[starts], np.arange(ntiles + 1)),
            "clique tile group offsets",
        ),
        ntiles=int(ntiles),
        ngroups=int(len(starts)),
        distinct_rows=distinct_rows,
    )


def clique_size_buckets(parent_ptr):


    import numpy as np

    sizes = np.diff(np.asarray(parent_ptr, dtype=np.int64))
    if np.any(sizes > 32) or np.any(sizes < 1):
        raise ValueError("clique buckets need groups of 1..32 members")
    bucket = np.searchsorted(np.asarray(CLIQUE_BUCKET_MAX_MEMBERS), sizes)
    ids = np.argsort(bucket, kind="stable").astype(np.int32)
    start = np.searchsorted(
        bucket[ids], np.arange(len(CLIQUE_BUCKET_MAX_MEMBERS) + 1)
    ).astype(np.int32)
    return ids, start


def remote_alpha_mode(environ):


    parent_on = environ.get("FQ_AABB_CLIQUE_ENABLE", "1") == "1"
    mode = environ.get(
        "FQ_AABB_REMOTE_ALPHA_METHOD", "transposed_clique" if parent_on else "baseline"
    )
    if mode not in ("baseline", "transposed_clique"):
        raise ValueError(
            "FQ_AABB_REMOTE_ALPHA_METHOD must be 'baseline' or 'transposed_clique'"
        )
    if mode == "transposed_clique" and not parent_on:
        raise ValueError(
            "FQ_AABB_REMOTE_ALPHA_METHOD=transposed_clique requires the clique arm (FQ_AABB_CLIQUE_ENABLE=1)"
        )
    return mode


def alpha_task_words(off, lengths, pack, n_out):


    import numpy as np

    off = np.asarray(off, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    pack = np.asarray(pack, dtype=np.uint64)
    if len(lengths) < n_out:
        raise ValueError("SELL lengths shorter than the output alpha count")
    rows, cols = [], []
    for a in range(int(n_out)):
        n = int(lengths[a])
        if n:
            idx = off[a >> 5] + (np.arange(n, dtype=np.int64) << 5) + (a & 31)
            rows.append(np.full(n, a, dtype=np.int64))
            cols.append(idx)
    if not rows:
        return np.empty(0, dtype=np.uint64), np.stack(
            [np.arange(n_out), np.zeros(n_out), np.zeros(n_out)], axis=1
        ).astype(np.int32)
    rows = np.concatenate(rows)
    words = pack[np.concatenate(cols)]
    ja = (words & np.uint64(0xFFFFFFFF)).astype(np.int64)
    keep = ja < 2**31
    rows, words = rows[keep], words[keep]
    counts = np.bincount(rows, minlength=int(n_out))
    ends = np.cumsum(counts)
    tasks = _int32(
        np.stack([np.arange(n_out), ends - counts, ends], axis=1),
        "cross direct task ranges",
    )
    return np.ascontiguousarray(words, dtype=np.uint64), tasks


def parent_options(environ):


    flag = environ.get("FQ_AABB_CLIQUE_ENABLE", "1")
    timing = environ.get("FQ_AABB_TIME_CLIQUE_KERNELS", "0")
    threshold = int(environ.get("FQ_AABB_CLIQUE_MIN_MEMBERS", "6"))
    if flag not in ("0", "1") or timing not in ("0", "1") or not 2 <= threshold <= 32:
        raise ValueError(
            "parent flags must be 0/1; FQ_AABB_CLIQUE_MIN_MEMBERS must be in [2,32]"
        )
    return flag == "1", threshold, timing == "1"


def remote_beta_options(environ):


    parent_on = environ.get("FQ_AABB_CLIQUE_ENABLE", "1") == "1"
    mode = environ.get(
        "FQ_AABB_REMOTE_BETA_METHOD", "clique" if parent_on else "baseline"
    )
    timing = environ.get("FQ_AABB_TIME_EACH_SOURCE", "0")
    if mode not in ("baseline", "clique") or timing not in ("0", "1"):
        raise ValueError(
            "FQ_AABB_REMOTE_BETA_METHOD must be baseline or clique; FQ_AABB_TIME_EACH_SOURCE must be 0 or 1"
        )
    if mode != "baseline":
        parent_options(dict(environ, FQ_AABB_CLIQUE_ENABLE="1"))
    if mode == "clique" and not parent_on:
        raise ValueError(
            "remote-beta clique requires FQ_AABB_CLIQUE_ENABLE=1 to share its alpha plan"
        )
    return mode, timing == "1"


def parent_kind(environ):


    kind = environ.get("FQ_AABB_CLIQUE_GROUPING_KIND", "auto").strip().lower()
    if kind not in ("auto",) + PARENT_KINDS:
        raise ValueError(
            "FQ_AABB_CLIQUE_GROUPING_KIND must be auto, union or intersection"
        )
    return kind


def parent_plan_score(plan):

    st = plan["stats"]
    return (
        st["selected_edges"],
        st.get("edge_weighted_size", 0.0),
        st["kind"] == "union",
    )


def choose_parent_plan(alpha, off, lengths, pack, num_alpha, min_parent=8, kind="auto"):

    if kind != "auto":
        return parent_plan(alpha, off, lengths, pack, num_alpha, min_parent, kind=kind)
    plans = [
        parent_plan(alpha, off, lengths, pack, num_alpha, min_parent, kind=k)
        for k in PARENT_KINDS
    ]
    best = max(plans, key=parent_plan_score)
    best["stats"]["alternative_selected_edges"] = min(
        p["stats"]["selected_edges"] for p in plans
    )
    return best


def parent_masks(plan):

    import numpy as np

    ptr, members = plan["parent_ptr"], plan["members"]
    sizes = np.diff(ptr.astype(np.int64))
    if (
        ptr.ndim != 1
        or ptr.dtype != np.dtype(np.int32)
        or not len(ptr)
        or members.ndim != 1
        or members.dtype != np.dtype(np.uint64)
        or ptr[0] != 0
        or ptr[-1] != len(members)
        or np.any(sizes < 2)
        or np.any(sizes > 32)
    ):
        raise ValueError("invalid parent member ranges")
    ids = np.arange(len(sizes), dtype=np.int32)
    if not len(sizes):
        return np.empty(0, dtype=np.uint64), ids
    orbitals = ((members >> np.uint64(32)) & np.uint64(0xFF)).astype(np.int32)
    phases = (members >> np.uint64(48)).astype(np.uint16).view(np.int16)
    boundary = np.zeros(len(members), dtype=bool)
    boundary[ptr[:-1]] = True
    if (
        np.any(orbitals >= 64)
        or np.any((phases != 1) & (phases != -1))
        or np.any((np.diff(orbitals) <= 0) & ~boundary[1:])
    ):
        raise ValueError("parent members must have ascending orbitals and +/-1 phases")
    bits = np.uint64(1) << orbitals.astype(np.uint64)
    return np.bitwise_or.reduceat(bits, ptr[:-1]), ids


def _parity(words):


    import numpy as np

    x = np.array(words, dtype=np.uint64, copy=True)
    for shift in (32, 16, 8, 4, 2, 1):
        x ^= x >> np.uint64(shift)
    return (x & np.uint64(1)).astype(np.int16)


def _int32(values, name):

    import numpy as np

    x = np.asarray(values, dtype=np.int64)
    info = np.iinfo(np.int32)
    if x.size and (int(x.min()) < info.min or int(x.max()) > info.max):
        raise OverflowError(f"{name} do not fit int32 (largest {int(x.max())})")
    return x.astype(np.int32)


def _integers(values, name):


    import numpy as np

    x = np.asarray(values)
    if (
        x.ndim != 1
        or (x.size and x.dtype.kind not in "iu")
        or np.any(x < 0)
        or np.any(x > np.iinfo(np.int32).max)
    ):
        raise ValueError(f"{name} must fit nonnegative int32")
    return x.astype(np.int64)


def unpack_sell(off, lengths, pack):

    import numpy as np

    off, lengths = _integers(off, "offsets"), _integers(lengths, "lengths")
    pack = np.asarray(pack)
    if pack.ndim != 1 or pack.dtype != np.dtype(np.uint64):
        raise ValueError("packed edges must be a uint64 vector")
    n = len(lengths)
    padded = np.zeros(((n + 31) // 32) * 32, dtype=np.int64)
    padded[:n] = lengths
    widths = padded.reshape(-1, 32).max(axis=1)
    if (
        len(off) != len(widths) + 1
        or off[0] != 0
        or off[-1] != len(pack)
        or not np.array_equal(np.diff(off), widths * 32)
    ):
        raise ValueError("offsets do not describe the supplied C=32 SELL payload")
    rows = np.repeat(np.arange(n, dtype=np.int64), lengths)
    ptr = np.r_[0, np.cumsum(lengths)]
    slot = off[rows // 32] + (np.arange(len(rows)) - ptr[rows]) * 32 + rows % 32
    return rows, pack[slot]


def _residual_sell(rows, words, n):


    import numpy as np

    lengths = np.bincount(rows, minlength=n).astype(np.int64)
    ptr = np.r_[0, np.cumsum(lengths)]
    padded = np.zeros(((n + 31) // 32) * 32, dtype=np.int64)
    padded[:n] = lengths
    off = np.r_[0, np.cumsum(padded.reshape(-1, 32).max(axis=1) * 32)].astype(np.int64)
    pack = np.zeros(int(off[-1]), dtype=np.uint64)
    slot = off[rows // 32] + (np.arange(len(rows)) - ptr[rows]) * 32 + rows % 32
    pack[slot] = words
    return off, lengths.astype(np.int32), pack


def degree_sorted_sell(off, lengths, pack):


    import numpy as np

    rows, words = unpack_sell(off, lengths, pack)
    lengths = np.asarray(lengths).astype(np.int64)
    active = np.flatnonzero(lengths > 0)
    order = active[np.argsort(-lengths[active], kind="stable")]
    rank = np.empty(len(lengths), dtype=np.int64)
    rank[order] = np.arange(len(order), dtype=np.int64)
    new_rows = rank[rows]
    by_row = np.argsort(new_rows, kind="stable")
    new_off, new_len, new_pack = _residual_sell(
        new_rows[by_row], words[by_row], len(order)
    )
    return new_off, new_len, new_pack, order.astype(np.int32)


def parent_plan(alpha, off, lengths, pack, num_alpha, min_parent=8, kind="union"):


    import numpy as np

    if not isinstance(num_alpha, (int, np.integer)) or not 2 <= num_alpha <= 64:
        raise ValueError("num_alpha must be in [2, 64]")
    if not isinstance(min_parent, (int, np.integer)) or not 2 <= min_parent <= 32:
        raise ValueError("min_parent must be in [2, 32]")
    if kind not in PARENT_KINDS:
        raise ValueError("kind must be union or intersection")
    alpha = np.asarray(alpha)
    if (
        alpha.ndim != 1
        or (alpha.size and alpha.dtype.kind not in "iu")
        or np.any(alpha < 0)
    ):
        raise ValueError("alpha must be a nonnegative integer vector")
    alpha = alpha.astype(np.uint64)
    if (num_alpha < 64 and np.any(alpha >> np.uint64(num_alpha))) or len(
        np.unique(alpha)
    ) != len(alpha):
        raise ValueError("alpha strings must be unique and fit num_alpha")
    populations = {int(a).bit_count() for a in alpha}
    if len(populations) > 1:
        raise ValueError("parent replay requires fixed alpha electron count")
    rows, words = unpack_sell(off, lengths, pack)
    if len(alpha) != len(lengths):
        raise ValueError("alpha strings must match SELL output rows (self-alpha only)")
    cols = (words & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32)
    pid = ((words >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.int64)
    if np.any(pid >= num_alpha * (num_alpha - 1) // 2):
        raise ValueError("packed alpha pair ID outside G")
    live = cols >= 0
    r, j, w = rows[live], cols[live].astype(np.int64), words[live]
    if np.any(j >= len(alpha)) or np.any(r == j):
        raise ValueError("non-self or diagonal alpha edge in self-alpha graph")
    edge_ids = (r.astype(np.uint64) << np.uint64(32)) | j.astype(np.uint64)
    if len(np.unique(edge_ids)) != len(edge_ids):
        raise ValueError("duplicate directed alpha edges are not supported")
    unions = alpha[r] | alpha[j]
    removed_r, removed_j = unions ^ alpha[r], unions ^ alpha[j]
    if (
        np.any(removed_r == 0)
        or np.any(removed_j == 0)
        or np.any(removed_r & (removed_r - np.uint64(1)))
        or np.any(removed_j & (removed_j - np.uint64(1)))
    ):
        raise ValueError("alpha edge is not a single excitation")
    powers = np.uint64(1) << np.arange(num_alpha, dtype=np.uint64)
    p, q = np.searchsorted(powers, removed_r), np.searchsorted(powers, removed_j)
    lo, hi = np.minimum(p, q), np.maximum(p, q)
    expected_pid = lo * (2 * num_alpha - lo - 1) // 2 + hi - lo - 1
    sign = (w >> np.uint64(48)).astype(np.uint16).view(np.int16)
    if kind == "union":
        parents_of_edges = unions
        tr = 1 - 2 * _parity(unions & (removed_r - np.uint64(1)))
        tj = 1 - 2 * _parity(unions & (removed_j - np.uint64(1)))
        expected_sign = -tr * tj
    else:
        parents_of_edges = alpha[r] & alpha[j]
        tr = 1 - 2 * _parity(parents_of_edges & (removed_j - np.uint64(1)))
        tj = 1 - 2 * _parity(parents_of_edges & (removed_r - np.uint64(1)))
        expected_sign = tr * tj
    if not np.array_equal(expected_pid, pid[live]) or not np.array_equal(
        sign, expected_sign
    ):
        raise ValueError("alpha strings disagree with packed orbital-pair IDs or signs")

    keys, group, edge_counts = np.unique(
        parents_of_edges, return_inverse=True, return_counts=True
    )
    tagged = group.astype(np.uint64) << np.uint64(32)
    membership = np.unique(
        np.r_[tagged | r.astype(np.uint64), tagged | j.astype(np.uint64)]
    )
    mg = (membership >> np.uint64(32)).astype(np.int64)
    mr = (membership & np.uint64(0xFFFFFFFF)).astype(np.int64)
    counts = np.bincount(mg, minlength=len(keys))
    chosen = (
        (counts >= min_parent) & (counts <= 32) & (edge_counts == counts * (counts - 1))
    )
    selected = np.zeros(len(words), dtype=bool)
    selected[live] = chosen[group]
    selected_edges = int(selected.sum())
    mr, mg = mr[chosen[mg]], mg[chosen[mg]]
    removed = keys[mg] ^ alpha[mr]
    orbital = np.searchsorted(powers, removed)
    phase = 1 - 2 * _parity(keys[mg] & (removed - np.uint64(1)))
    flag = np.uint64(1 << INTERSECTION_BIT) if kind == "intersection" else np.uint64(0)
    order = np.lexsort((orbital, mg))
    members = (
        mr.astype(np.uint64)
        | (orbital.astype(np.uint64) << np.uint64(32))
        | flag
        | (
            (phase.astype(np.int64).astype(np.uint64) & np.uint64(0xFFFF))
            << np.uint64(48)
        )
    )[order]
    parent_ptr = np.r_[0, np.cumsum(counts[chosen])]
    if parent_ptr[-1] > np.iinfo(np.int32).max:
        raise OverflowError("parent memberships exceed int32")
    roff, rlen, rpack = _residual_sell(rows[~selected], words[~selected], len(alpha))
    sizes = counts[chosen]
    work = sizes.astype(np.float64) * (sizes - 1)
    ew_size = float((work * sizes).sum() / work.sum()) if work.sum() else 0.0
    return dict(
        parent_ptr=parent_ptr.astype(np.int32),
        members=members,
        residual_off=roff,
        residual_lengths=rlen,
        residual_pack=rpack,
        stats=dict(
            parents=int(chosen.sum()),
            members=len(members),
            original_edges=len(words),
            selected_edges=selected_edges,
            residual_edges=int((~selected).sum()),
            min_size=int(sizes.min()) if len(sizes) else 0,
            max_size=int(sizes.max()) if len(sizes) else 0,
            min_parent=int(min_parent),
            electron_count=next(iter(populations), 0),
            kind=kind,
            edge_weighted_size=ew_size,
        ),
    )


def orbital_g(g, num_alpha):


    import numpy as np

    g = np.asarray(g)
    if (
        not 2 <= num_alpha <= 64
        or g.dtype != np.dtype(np.float64)
        or g.ndim != 2
        or g.shape[1] != num_alpha * (num_alpha - 1) // 2
    ):
        raise ValueError("G must be FP64 [beta pairs, alpha pairs]")
    p, q = np.triu_indices(num_alpha, 1)
    result = np.zeros((len(g), num_alpha, num_alpha), dtype=np.float64)
    result[:, p, q] = g
    result[:, q, p] = g
    return result
