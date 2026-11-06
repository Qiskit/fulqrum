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


@dataclass
class ArmPairTablesPayload:


    aa_sell_off: int = 0
    aa_sell_len: int = 0
    aa_sell_pack: int = 0
    aa_c0_sell: int = 0
    aa_t: int = 0
    bb_word: int = 0
    bb_t: int = 0
    aa_enable: int = 0
    aa_pairs: int = 0
    aa_t_rows: int = 0
    bb_enable: int = 0
    bb_pairs: int = 0
    bb_t_cols: int = 0
    aaaa_min_blocks: int = 0
    aaaa_identity_slot: int = 0


def pair_payload_error(payload: ArmPairTablesPayload, n_b_rows: int, n_a_columns: int):


    errors = []
    if payload.aa_enable:
        if not all(
            (
                payload.aa_sell_off,
                payload.aa_sell_len,
                payload.aa_sell_pack,
                payload.aa_c0_sell,
                payload.aa_t,
            )
        ):
            errors.append("aa pair table without its device tables")
        if not 0 < payload.aa_pairs <= 65536:
            errors.append(f"aa pair count {payload.aa_pairs} outside 1..65536")
        if payload.aa_t_rows < n_b_rows:
            errors.append(
                f"aa pair table has {payload.aa_t_rows} rows for {n_b_rows} beta rows"
            )
    if payload.bb_enable:
        if not (payload.bb_word and payload.bb_t):
            errors.append("bb pair table without its device tables")
        if payload.bb_pairs <= 0:
            errors.append(f"bb pair count {payload.bb_pairs}")
        if payload.bb_t_cols < n_a_columns:
            errors.append(
                f"bb pair table has {payload.bb_t_cols} columns for {n_a_columns} alpha columns"
            )
    if payload.aaaa_min_blocks not in (0, 5, 6):
        errors.append(f"aaaa minimum blocks {payload.aaaa_min_blocks} not 0, 5 or 6")
    return "; ".join(errors) or None


def pair_table_options(environ):


    aa = environ.get("FQ_AA_PAIR_TABLE", "1")
    bb = environ.get("FQ_BB_PAIR_TABLE", "1")
    if aa not in ("0", "1") or bb not in ("0", "1"):
        raise ValueError("FQ_AA_PAIR_TABLE and FQ_BB_PAIR_TABLE must be 0 or 1")
    return aa == "1", bb == "1"


def pair_id(p, q, n):

    import numpy as np

    p = np.asarray(p, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64)
    lo, hi = np.minimum(p, q), np.maximum(p, q)
    return lo * (2 * int(n) - lo - 1) // 2 + (hi - lo - 1)


def between_parity_sign(bra, p, q):

    import numpy as np

    lo = np.minimum(p, q).astype(np.uint64)
    hi = np.maximum(p, q).astype(np.uint64)
    mask = ((np.uint64(1) << hi) - np.uint64(1)) ^ (
        (np.uint64(1) << (lo + np.uint64(1))) - np.uint64(1)
    )
    x = np.asarray(bra, dtype=np.uint64) & mask
    for s in (32, 16, 8, 4, 2, 1):
        x ^= x >> np.uint64(s)
    return np.where((x & np.uint64(1)) != 0, -1, 1).astype(np.int8)


def derive_pair_table(cpb, sign, pairs, n_pairs, n_other):


    import numpy as np

    pairs = np.asarray(pairs, dtype=np.int64)
    e = len(pairs)
    cpb = np.asarray(cpb, dtype=np.float64)
    if cpb.size < e * int(n_other):
        return None
    d = np.zeros((n_pairs, n_other), dtype=np.float64)
    present = np.zeros(n_pairs, dtype=bool)
    if e == 0:
        return d, present
    if pairs.min() < 0 or pairs.max() >= n_pairs:
        return None
    vals = (
        cpb[: e * int(n_other)].reshape(e, n_other)
        * np.asarray(sign, dtype=np.float64)[:, None]
    )
    first = np.full(n_pairs, -1, dtype=np.int64)
    first[pairs[::-1]] = np.arange(e - 1, -1, -1)
    present = first >= 0
    d[present] = vals[first[present]]
    if not np.array_equal(vals, d[pairs]):
        return None
    return d, present


def merge_pair_tables(d_base, present_base, d_new, present_new):


    if not tables_agree(d_base, present_base, d_new, present_new):
        return None
    d = d_base.copy()
    add = present_new & ~present_base
    d[add] = d_new[add]
    return d, present_base | present_new


def tables_agree(d1, present1, d2, present2):

    import numpy as np

    both = present1 & present2
    return bool(np.array_equal(d1[both], d2[both]))


def occupation_bits(strings, n_orb):


    import numpy as np

    s = np.asarray(strings, dtype=np.uint64)
    shifts = np.arange(n_orb, dtype=np.uint64)
    return ((s[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.float64)


def aa_table(d, bra_beta, n_beta):

    import numpy as np

    return np.ascontiguousarray(occupation_bits(bra_beta, n_beta) @ d.T)


def bb_table(d, bra_alpha, n_alpha):

    import numpy as np

    return np.ascontiguousarray(d @ occupation_bits(bra_alpha, n_alpha).T)


def csr_to_sell_values(values, ptr, off, lengths):

    import numpy as np

    ptr = np.asarray(ptr, dtype=np.int64)
    off = np.asarray(off, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    n = len(lengths)
    if len(ptr) != n + 1 or not np.array_equal(np.diff(ptr), lengths):
        raise ValueError("CSR row lengths disagree with SELL lengths")
    rows = np.repeat(np.arange(n, dtype=np.int64), lengths)
    k = np.arange(int(ptr[-1]), dtype=np.int64) - ptr[rows]
    from .tile_tables import SELL_SLICE_ROWS

    slot = off[rows // SELL_SLICE_ROWS] + k * SELL_SLICE_ROWS + rows % SELL_SLICE_ROWS
    out = np.zeros(int(off[-1]), dtype=np.float64)
    out[slot] = np.asarray(values, dtype=np.float64)[: int(ptr[-1])]
    return out


def bb_words(ptr, p0, p1, bra_beta, n_beta):

    import numpy as np

    ptr = np.asarray(ptr, dtype=np.int64)
    nnz = int(ptr[-1])
    rows = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))
    p = np.asarray(p0[:nnz], dtype=np.int64)
    q = np.asarray(p1[:nnz], dtype=np.int64)
    pid = pair_id(p, q, n_beta)
    neg = between_parity_sign(np.asarray(bra_beta, dtype=np.uint64)[rows], p, q) < 0
    from .tile_tables import SIGNED_PAIR_NEGATIVE

    negative = np.where(neg, np.uint32(SIGNED_PAIR_NEGATIVE), np.uint32(0))
    return (pid.astype(np.uint32) | negative), pid, np.where(neg, -1, 1)
