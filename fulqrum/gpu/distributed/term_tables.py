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


import numpy as np

from ..common.hash_utils import _EMPTY_KEY, _splitmix64


_EMPTY_KEY_U32 = np.uint32(0xFFFFFFFF)


def _splitmix32_host(key):


    k = int(key) & 0xFFFFFFFF
    k = ((k >> 16) ^ k) * 0x45D9F3B & 0xFFFFFFFF
    k = ((k >> 16) ^ k) * 0x45D9F3B & 0xFFFFFFFF
    return (k ^ (k >> 16)) & 0xFFFFFFFF


def _build_ht_u32_kv(keys_int, vals_int):


    n = len(keys_int)
    cap = 1
    while cap < 3 * max(n, 1):
        cap <<= 1
    mask = int(cap - 1)
    ht_keys = np.full(cap, _EMPTY_KEY_U32, dtype=np.uint32)
    ht_vals = np.full(cap, np.int32(-1), dtype=np.int32)
    bw = (cap + 31) // 32
    ht_bitset = np.full(bw, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    for key, val in zip(keys_int, vals_int):
        slot = _splitmix32_host(key) & mask
        while ht_keys[slot] != _EMPTY_KEY_U32:
            slot = (slot + 1) & mask
        ht_keys[slot] = np.uint32(key)
        ht_vals[slot] = np.int32(val)
        word, bit = slot >> 5, slot & 31
        ht_bitset[word] = np.uint32(int(ht_bitset[word]) & ~(1 << bit))
    return ht_keys, ht_vals, np.uint32(mask), ht_bitset


def _build_ht_u64_kv(keys_int, vals_int):


    n = len(keys_int)
    cap = 1
    while cap < 3 * max(n, 1):
        cap <<= 1
    mask = int(cap - 1)
    ht_keys = np.full(cap, _EMPTY_KEY, dtype=np.uint64)
    ht_vals = np.full(cap, np.int32(-1), dtype=np.int32)
    bw = (cap + 63) // 64
    ht_bitset = np.full(bw, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    for key, val in zip(keys_int, vals_int):
        k = int(key) & 0xFFFFFFFFFFFFFFFF
        slot = _splitmix64(k) & mask
        while ht_keys[slot] != _EMPTY_KEY:
            slot = (slot + 1) & mask
        ht_keys[slot] = np.uint64(k)
        ht_vals[slot] = np.int32(val)
        word, bit = slot >> 6, slot & 63
        ht_bitset[word] = np.uint64(int(ht_bitset[word]) & ~(1 << bit))
    return ht_keys, ht_vals, np.uint64(mask), ht_bitset


def _group_terms(gpu_data, grp_offdiag_sizes, num_groups, size):


    ladder_ptrs = np.asarray(gpu_data["grp_ladder_ptrs"]).flatten().astype(np.int64)
    lo = int(gpu_data["ladder_offset"])
    groups = np.flatnonzero(np.asarray(grp_offdiag_sizes)[:num_groups] == size)
    if len(groups) == 0 or lo < 2:
        return groups, np.empty(0, np.int64), np.empty(0, np.int64)
    starts = ladder_ptrs[groups * lo]
    ends = ladder_ptrs[groups * lo + lo - 1]
    inner = ladder_ptrs[(groups * lo)[:, None] + np.arange(lo)[None, :]]
    if np.any(np.diff(inner, axis=1) < 0):
        raise AssertionError("ladder pointers decrease inside a group")
    counts = np.maximum(ends - starts, 0)
    owner = np.repeat(groups, counts)
    terms = np.repeat(starts - np.r_[0, np.cumsum(counts)[:-1]], counts) + np.arange(
        int(counts.sum())
    )
    return groups, owner, terms


def _assert_aaaa_bbbb_proj_invariance(
    gpu_data, grp_offdiag_inds, grp_offdiag_sizes, num_groups, num_alpha
):


    term_num_proj = np.asarray(gpu_data["term_num_proj"]).astype(np.int32)
    term_proj_inds = np.asarray(gpu_data["term_proj_inds"]).astype(np.int32)
    if term_proj_inds.size == 0:
        return
    proj_view = (
        term_proj_inds if term_proj_inds.ndim >= 2 else term_proj_inds.reshape(-1, 1)
    )
    num_alpha = int(num_alpha)
    oi = np.asarray(grp_offdiag_inds)
    groups, owner, terms = _group_terms(gpu_data, grp_offdiag_sizes, num_groups, 4)
    if len(terms) == 0:
        return
    n_a = (
        (oi[:, :4] < num_alpha).sum(axis=1)
        if oi.ndim == 2 and oi.shape[1] >= 4
        else np.zeros(len(oi), int)
    )
    kind = n_a[owner]
    keep = (kind == 4) | (kind == 0)
    owner, terms, kind = owner[keep], terms[keep], kind[keep]
    npj = term_num_proj[terms]
    k = np.arange(proj_view.shape[1])[None, :]
    pos = proj_view[terms]
    live = k < npj[:, None]
    low = np.where(kind == 4, num_alpha, 0)[:, None]
    high = np.where(kind == 4, 1 << 30, num_alpha)[:, None]
    bad = live & (pos >= low) & (pos < high)
    if bad.any():
        row, col = np.argwhere(bad)[0]
        g, t, p_ = int(owner[row]), int(terms[row]), int(pos[row, col])
        arm_name = "aaaa" if kind[row] == 4 else "bbbb"
        raise AssertionError(
            f"base cache invariance violated: {arm_name} group "
            f"g={g} term t={t} has proj_ind={p_} in the "
            f"forbidden range [{int(low[row, 0])}, {int(high[row, 0])}). The aaaa and bbbb caches "
            "cannot hold this operator; use the CPU operator."
        )


def _assert_aaaa_bbbb_proj_invariance_reference(
    gpu_data, grp_offdiag_inds, grp_offdiag_sizes, num_groups, num_alpha
):


    ladder_ptrs = np.asarray(gpu_data["grp_ladder_ptrs"]).flatten().astype(np.int64)
    lo = int(gpu_data["ladder_offset"])
    term_num_proj = np.asarray(gpu_data["term_num_proj"]).astype(np.int32)
    term_proj_inds = np.asarray(gpu_data["term_proj_inds"]).astype(np.int32)
    if term_proj_inds.size == 0:
        return
    if term_proj_inds.ndim >= 2:
        proj_view = term_proj_inds
    else:
        proj_view = term_proj_inds.reshape(-1, 1)
    oi = np.asarray(grp_offdiag_inds)
    os_ = np.asarray(grp_offdiag_sizes)
    num_alpha = int(num_alpha)

    for g in range(num_groups):
        sz = int(os_[g])
        if sz != 4:
            continue
        inds = oi[g, :4]
        n_a = sum(1 for idx in inds if int(idx) < num_alpha)
        if n_a == 4:
            forbid_low, forbid_high = num_alpha, 1 << 30
            arm_name = "aaaa"
        elif n_a == 0:
            forbid_low, forbid_high = 0, num_alpha
            arm_name = "bbbb"
        else:
            continue
        for ri in range(lo - 1):
            t0 = int(ladder_ptrs[g * lo + ri])
            t1 = int(ladder_ptrs[g * lo + ri + 1])
            for t in range(t0, t1):
                npj = int(term_num_proj[t])
                if npj == 0:
                    continue
                for k in range(npj):
                    pos = int(proj_view[t, k])
                    if forbid_low <= pos < forbid_high:
                        raise AssertionError(
                            f"base cache invariance violated: {arm_name} group "
                            f"g={g} term t={t} has proj_ind={pos} in the "
                            f"forbidden range [{forbid_low}, {forbid_high}). The aaaa and bbbb caches "
                            "cannot hold this operator; use the CPU operator."
                        )


def _assert_aa_bb_factor_cache_invariance(
    gpu_data, grp_offdiag_inds, grp_offdiag_sizes, num_groups, num_alpha
):


    term_num_proj = np.asarray(gpu_data["term_num_proj"]).astype(np.int32)
    term_proj_inds = np.asarray(gpu_data["term_proj_inds"]).astype(np.int32)
    term_num_ops = np.asarray(gpu_data["term_num_ops"]).astype(np.int32)
    term_inds = np.asarray(gpu_data["term_inds"]).astype(np.int32)
    proj_view = (
        term_proj_inds.reshape(-1, 1) if term_proj_inds.ndim == 1 else term_proj_inds
    )
    inds_view = term_inds.reshape(-1, 1) if term_inds.ndim == 1 else term_inds
    num_alpha = int(num_alpha)
    oi = np.asarray(grp_offdiag_inds)
    groups, owner, terms = _group_terms(gpu_data, grp_offdiag_sizes, num_groups, 2)
    if len(terms) == 0:
        return
    n_a = (
        (oi[:, :2] < num_alpha).sum(axis=1)
        if oi.ndim == 2 and oi.shape[1] >= 2
        else np.zeros(len(oi), int)
    )
    kind = n_a[owner]
    keep = (kind == 2) | (kind == 0)
    owner, terms, kind = owner[keep], terms[keep], kind[keep]
    if len(terms) == 0:
        return
    low = np.where(kind == 2, num_alpha, 0)[:, None]
    high = np.where(kind == 2, 1 << 30, num_alpha)[:, None]
    parts = []
    if proj_view.size:
        pos = (
            proj_view[terms]
            if len(proj_view) > int(terms.max())
            else np.zeros((len(terms), 0), np.int32)
        )
        live = np.arange(pos.shape[1])[None, :] < term_num_proj[terms][:, None]
        parts.append(np.where(live & (pos >= low) & (pos < high), pos, -1))
    if inds_view.size:
        pos = inds_view[terms]
        live = np.arange(pos.shape[1])[None, :] < term_num_ops[terms][:, None]
        parts.append(np.where(live & (pos >= low) & (pos < high), pos, -1))
    if not parts:
        return
    seen = np.sort(np.concatenate(parts, axis=1), axis=1)


    nonneg = seen >= 0
    first = nonneg.any(axis=1).astype(int)
    distinct = first + ((seen[:, 1:] != seen[:, :-1]) & nonneg[:, :-1]).sum(axis=1)
    bad = np.flatnonzero(distinct > 1)
    if len(bad):
        row = int(bad[0])
        bits = sorted(set(int(x) for x in seen[row] if x >= 0))
        arm_name = "aa" if kind[row] == 2 else "bb"
        raise AssertionError(
            f"factor cache invariance violated: {arm_name} group g={int(owner[row])} "
            f"term t={int(terms[row])} touches {len(bits)} distinct cross-spin "
            f"bits {bits} (max allowed: 1)."
        )


def _assert_aa_bb_factor_cache_invariance_reference(
    gpu_data, grp_offdiag_inds, grp_offdiag_sizes, num_groups, num_alpha
):


    ladder_ptrs = np.asarray(gpu_data["grp_ladder_ptrs"]).flatten().astype(np.int64)
    lo = int(gpu_data["ladder_offset"])
    term_num_proj = np.asarray(gpu_data["term_num_proj"]).astype(np.int32)
    term_proj_inds = np.asarray(gpu_data["term_proj_inds"]).astype(np.int32)
    term_num_ops = np.asarray(gpu_data["term_num_ops"]).astype(np.int32)
    term_inds = np.asarray(gpu_data["term_inds"]).astype(np.int32)
    if term_proj_inds.ndim == 1:
        proj_view = term_proj_inds.reshape(-1, 1)
    else:
        proj_view = term_proj_inds
    if term_inds.ndim == 1:
        inds_view = term_inds.reshape(-1, 1)
    else:
        inds_view = term_inds
    oi = np.asarray(grp_offdiag_inds)
    os_ = np.asarray(grp_offdiag_sizes)
    num_alpha = int(num_alpha)

    for g in range(num_groups):
        sz = int(os_[g])
        if sz != 2:
            continue
        inds = oi[g, :2]
        n_a = sum(1 for idx in inds if int(idx) < num_alpha)
        if n_a == 2:
            cross_low, cross_high = num_alpha, 1 << 30
            arm_name = "aa"
        elif n_a == 0:
            cross_low, cross_high = 0, num_alpha
            arm_name = "bb"
        else:
            continue
        for ri in range(lo - 1):
            t0 = int(ladder_ptrs[g * lo + ri])
            t1 = int(ladder_ptrs[g * lo + ri + 1])
            for t in range(t0, t1):
                bits_seen = set()
                for k in range(int(term_num_proj[t])):
                    pos = int(proj_view[t, k])
                    if cross_low <= pos < cross_high:
                        bits_seen.add(pos)
                for k in range(int(term_num_ops[t])):
                    pos = int(inds_view[t, k])
                    if cross_low <= pos < cross_high:
                        bits_seen.add(pos)
                if len(bits_seen) > 1:
                    raise AssertionError(
                        f"factor cache invariance violated: {arm_name} group g={g} "
                        f"term t={t} touches {len(bits_seen)} distinct cross-spin "
                        f"bits {sorted(bits_seen)} (max allowed: 1)."
                    )


def _build_direct_integral_tables(
    gpu_data, grp_offdiag_inds, grp_offdiag_sizes, num_groups, num_alpha
):


    ladder_ptrs = np.asarray(gpu_data["grp_ladder_ptrs"]).flatten().astype(np.int64)
    lo = int(gpu_data["ladder_offset"])
    coeffs_re = np.asarray(gpu_data["term_coeffs_re"], dtype=np.float64)
    real_phase = np.asarray(gpu_data["term_real_phase"], dtype=np.int8)
    oi = np.asarray(grp_offdiag_inds)
    os_ = np.asarray(grp_offdiag_sizes)
    num_alpha = int(num_alpha)

    grp2_direct = np.zeros(num_groups, dtype=np.float64)
    grp4_direct = np.zeros(num_groups, dtype=np.float64)

    for g in range(num_groups):
        sz = int(os_[g])
        if sz != 4:
            continue
        inds = oi[g, :4]
        n_a = sum(1 for idx in inds if int(idx) < num_alpha)
        if n_a != 2:
            continue
        for ri in range(lo - 1):
            t0 = int(ladder_ptrs[g * lo + ri])
            t1 = int(ladder_ptrs[g * lo + ri + 1])
            if t0 < t1:
                grp4_direct[g] = float(coeffs_re[t0]) * float(real_phase[t0])
                break

    return grp2_direct, grp4_direct


def _build_grp_pos_tables(
    grp_offdiag_inds, grp_offdiag_sizes, num_alpha=None, grp_rowint_length=None
):


    keys2, vals2, keys4, vals4 = [], [], [], []
    pack_meta = grp_rowint_length is not None
    for g, sz in enumerate(grp_offdiag_sizes):
        inds = grp_offdiag_inds[g, :sz]
        if pack_meta:
            if g >= (1 << 24):
                raise ValueError(
                    f"group/length pack: group index g={g} exceeds 24 bits "
                    f"(max {(1 << 24) - 1}). Disable the pack by passing "
                    "grp_rowint_length=None, or extend the layout."
                )
            nl_g = int(grp_rowint_length[g])
            if not (0 <= nl_g < 256):
                raise ValueError(
                    f"group/length pack: grp_rowint_length[{g}]={nl_g} doesn't fit in 8 bits."
                )
            val = (g << 8) | nl_g
        else:
            val = g
        if sz == 2:
            key = int(inds[0]) | (int(inds[1]) << 16)
            keys2.append(key)
            vals2.append(val)
        elif sz == 4:
            if num_alpha is not None:
                n_alpha = sum(1 for k in range(4) if int(inds[k]) < num_alpha)
                if n_alpha == 2:
                    continue
            key = (
                int(inds[0])
                | (int(inds[1]) << 16)
                | (int(inds[2]) << 32)
                | (int(inds[3]) << 48)
            )
            keys4.append(key)
            vals4.append(val)
    g2 = _build_ht_u32_kv(keys2, vals2)
    g4 = _build_ht_u64_kv(keys4, vals4)
    return g2 + g4


def _build_aabb_g_dense(
    grp_offdiag_inds, grp_offdiag_sizes, grp4_direct, num_alpha, num_beta
):


    oi = np.asarray(grp_offdiag_inds)
    os_ = np.asarray(grp_offdiag_sizes)
    grp4 = np.asarray(grp4_direct, dtype=np.float64)
    num_alpha = int(num_alpha)
    num_beta = int(num_beta)
    n_apairs = num_alpha * (num_alpha - 1) // 2
    n_bpairs = num_beta * (num_beta - 1) // 2

    g = np.zeros(n_bpairs * n_apairs, dtype=np.float64)
    g4 = np.where(os_ == 4)[0]
    if len(g4) == 0:
        return g
    inds = oi[g4, :4].astype(np.int64)
    is_a = inds < num_alpha
    keep = is_a.sum(axis=1) == 2
    g4, inds, is_a = g4[keep], inds[keep], is_a[keep]
    gval = grp4[g4]
    aorb = np.sort(inds[is_a].reshape(-1, 2), axis=1)
    borb = np.sort((inds[~is_a] - num_alpha).reshape(-1, 2), axis=1)
    a0, a1 = aorb[:, 0], aorb[:, 1]
    b0, b1 = borb[:, 0], borb[:, 1]
    apid = a0 * (2 * num_alpha - a0 - 1) // 2 + (a1 - a0 - 1)
    bpid = b0 * (2 * num_beta - b0 - 1) // 2 + (b1 - b0 - 1)
    g[bpid * n_apairs + apid] = gval
    return g
