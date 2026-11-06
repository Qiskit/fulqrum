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

from typing import TYPE_CHECKING

import numpy as np

from ..common.hash_utils import _popcount_u64
from .hash_utils import _build_hash_table_simple, _ht_lookup_batch

if TYPE_CHECKING:
    from .dist_matvec import TileClass

try:
    from .tile_build import (
        build_tile_flip_csr_dense as _native_dense,
        build_tile_flip_csr_masks as _native_masks,
        tile_has_popcount as _native_has_popcount,
    )

    _NATIVE_TILE_BUILD = True
except ImportError:  # pragma: no cover
    _NATIVE_TILE_BUILD = False


def _native_tile_build_enabled() -> bool:

    return _NATIVE_TILE_BUILD


def _tile_has_popcount(
    alpha_left, alpha_right, wanted: tuple[int, ...], threads: int = 1
) -> bool:


    if len(alpha_left) == 0 or len(alpha_right) == 0:
        return False
    if _native_tile_build_enabled():
        bra = np.ascontiguousarray(alpha_left, dtype=np.uint64)
        src = np.ascontiguousarray(alpha_right, dtype=np.uint64)
        return any(_native_has_popcount(bra, src, int(w), int(threads)) for w in wanted)
    wanted_arr = np.asarray(wanted, dtype=np.int32)
    batch = 256
    right = alpha_right.astype(np.uint64)
    for start in range(0, len(alpha_left), batch):
        left = alpha_left[start : start + batch].astype(np.uint64)
        pc = _popcount_u64(right[:, None] ^ left[None, :], dtype=np.int32)
        if np.isin(pc, wanted_arr).any():
            return True
    return False


def _tile_source_needed(
    alpha_bra: np.ndarray,
    beta_bra: np.ndarray,
    alpha_src: np.ndarray,
    beta_src: np.ndarray,
    flip_masks: dict | None = None,
    threads: int = 1,
) -> bool:


    a0 = np.intersect1d(alpha_bra, alpha_src, assume_unique=True).size > 0
    b0 = np.intersect1d(beta_bra, beta_src, assume_unique=True).size > 0
    if a0 and b0:
        return True

    if flip_masks is None:
        a2 = _tile_has_popcount(alpha_bra, alpha_src, (2,), threads=threads)
        a4 = _tile_has_popcount(alpha_bra, alpha_src, (4,), threads=threads)
        b2 = _tile_has_popcount(beta_bra, beta_src, (2,), threads=threads)
        b4 = _tile_has_popcount(beta_bra, beta_src, (4,), threads=threads)
    else:
        a2 = _tile_has_connection_hybrid(
            alpha_bra, alpha_src, 2, flip_masks["alpha2"], threads=threads
        )
        a4 = _tile_has_connection_hybrid(
            alpha_bra, alpha_src, 4, flip_masks["alpha4"], threads=threads
        )
        b2 = _tile_has_connection_hybrid(
            beta_bra, beta_src, 2, flip_masks["beta2"], threads=threads
        )
        b4 = _tile_has_connection_hybrid(
            beta_bra, beta_src, 4, flip_masks["beta4"], threads=threads
        )
    return (b0 and (a2 or a4)) or (a0 and (b2 or b4)) or (a2 and b2)


def _group_flip_masks(grp_offdiag_inds, grp_offdiag_sizes, num_alpha: int):


    sizes = np.asarray(grp_offdiag_sizes, dtype=np.int32)
    inds_all = np.asarray(grp_offdiag_inds, dtype=np.int64)
    num_alpha = int(num_alpha)
    out = {key: {} for key in ("alpha2", "alpha4", "beta2", "beta4")}
    if inds_all.ndim != 2 or len(sizes) == 0:
        return {key: [] for key in out}
    width = inds_all.shape[1]
    for size in (2, 4):
        if width < size:
            continue
        sel = np.flatnonzero(sizes == size)
        if len(sel) == 0:
            continue
        inds = inds_all[sel, :size]
        is_alpha = inds < num_alpha
        n_alpha = is_alpha.sum(axis=1)
        alpha_pos = np.where(is_alpha, inds, -1)
        beta_pos = np.where(is_alpha, -1, inds - num_alpha)

        def masks_of(pos, rows):
            """Return the unique (mask, sorted positions) of the given rows (positions >= 0)."""
            if len(rows) == 0:
                return []
            p = np.sort(
                np.where(pos[rows] >= 0, pos[rows], np.iinfo(np.int64).max), axis=1
            )
            count = int((pos[rows[0]] >= 0).sum())
            p = p[:, :count]
            mask = np.bitwise_or.reduce(
                np.left_shift(np.uint64(1), p.astype(np.uint64)), axis=1
            )
            _, first = np.unique(mask, return_index=True)
            return [(int(mask[i]), tuple(int(x) for x in p[i])) for i in first]

        if size == 2:
            for name, pos, want in (("alpha2", alpha_pos, 2), ("beta2", beta_pos, 0)):
                out[name].update(masks_of(pos, np.flatnonzero(n_alpha == want)))
        else:
            for name, pos, want in (("alpha4", alpha_pos, 4), ("beta4", beta_pos, 0)):
                out[name].update(masks_of(pos, np.flatnonzero(n_alpha == want)))
            mixed = np.flatnonzero(n_alpha == 2)
            out["alpha2"].update(masks_of(alpha_pos, mixed))
            out["beta2"].update(masks_of(beta_pos, mixed))
    return {
        key: [
            (np.uint64(mask), tuple(np.uint8(q) for q in positions))
            for mask, positions in sorted(value.items())
        ]
        for key, value in out.items()
    }


def _group_flip_masks_reference(grp_offdiag_inds, grp_offdiag_sizes, num_alpha: int):

    masks = {
        "alpha2": {},
        "alpha4": {},
        "beta2": {},
        "beta4": {},
    }
    sizes = np.asarray(grp_offdiag_sizes, dtype=np.int32)
    inds_all = np.asarray(grp_offdiag_inds, dtype=np.int32)
    for g, size in enumerate(sizes):
        if size not in (2, 4):
            continue
        inds = inds_all[g, :size]
        alpha_pos = sorted(int(x) for x in inds if int(x) < num_alpha)
        beta_pos = sorted(int(x) - num_alpha for x in inds if int(x) >= num_alpha)

        if size == 2:
            if len(alpha_pos) == 2:
                mask = sum(1 << p for p in alpha_pos)
                masks["alpha2"][mask] = tuple(alpha_pos)
            elif len(beta_pos) == 2:
                mask = sum(1 << p for p in beta_pos)
                masks["beta2"][mask] = tuple(beta_pos)
        else:
            if len(alpha_pos) == 4:
                mask = sum(1 << p for p in alpha_pos)
                masks["alpha4"][mask] = tuple(alpha_pos)
            elif len(beta_pos) == 4:
                mask = sum(1 << p for p in beta_pos)
                masks["beta4"][mask] = tuple(beta_pos)
            elif len(alpha_pos) == 2 and len(beta_pos) == 2:
                amask = sum(1 << p for p in alpha_pos)
                bmask = sum(1 << p for p in beta_pos)
                masks["alpha2"][amask] = tuple(alpha_pos)
                masks["beta2"][bmask] = tuple(beta_pos)

    return {
        key: [
            (np.uint64(mask), tuple(np.uint8(p) for p in positions))
            for mask, positions in sorted(value.items())
        ]
        for key, value in masks.items()
    }


def _tile_has_mask_connection(
    bra_vals: np.ndarray,
    src_ht_keys: np.ndarray,
    src_ht_vals: np.ndarray,
    src_ht_mask: int,
    masks: list[tuple[np.uint64, tuple[np.uint8, ...]]],
) -> bool:


    if len(bra_vals) == 0 or not masks:
        return False
    bra = bra_vals.astype(np.uint64)
    for mask, _positions in masks:
        candidates = bra ^ mask
        found = _ht_lookup_batch(src_ht_keys, src_ht_vals, src_ht_mask, candidates)
        if (found >= 0).any():
            return True
    return False


def _tile_has_connection_hybrid(
    bra_vals: np.ndarray,
    src_vals: np.ndarray,
    wanted_popcount: int,
    masks: list[tuple[np.uint64, tuple[np.uint8, ...]]],
    threads: int = 1,
) -> bool:


    dense_work = int(len(bra_vals)) * int(len(src_vals))
    mask_work = int(len(bra_vals)) * int(len(masks))
    if masks and mask_work * 8 < dense_work:
        src_ht = _build_hash_table_simple(src_vals.astype(np.uint64))
        return _tile_has_mask_connection(bra_vals, *src_ht, masks)
    return _tile_has_popcount(bra_vals, src_vals, (wanted_popcount,), threads=threads)


def _bits_from_xor_values(xors: np.ndarray, nbits: int) -> tuple[np.ndarray, ...]:


    bits = []
    vals_i = xors.astype(np.uint64).view(np.int64).copy()
    for _ in range(nbits):
        lsb_i = vals_i & -vals_i
        lsb_u = lsb_i.view(np.uint64)

        pos = np.where(
            lsb_u != 0,
            np.log2(np.where(lsb_u != 0, lsb_u.astype(np.float64), 1.0)),
            0.0,
        ).astype(np.uint8)
        bits.append(pos)
        vals_i ^= lsb_i
    return tuple(bits)


def _build_tile_flip_csr_dense(
    bra_vals: np.ndarray,
    src_vals: np.ndarray,
    popcount: int,
    threads: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if _native_tile_build_enabled():
        return _native_dense(
            np.ascontiguousarray(bra_vals, dtype=np.uint64),
            np.ascontiguousarray(src_vals, dtype=np.uint64),
            int(popcount),
            int(threads),
        )

    n_bra = len(bra_vals)
    ptr = np.zeros(n_bra + 1, dtype=np.int64)
    if n_bra == 0 or len(src_vals) == 0:
        z_i = np.zeros(1, dtype=np.int32)
        z_u = np.zeros(1, dtype=np.uint8)
        return ptr, z_i, z_u, z_u, z_u, z_u

    src = src_vals.astype(np.uint64)
    batch = 256

    def _scan_block(row_range):


        r0, r1 = row_range
        r_ch: list = []
        j_ch: list = []
        p_ch: list = [[], [], [], []]
        for start in range(r0, r1, batch):
            stop = min(start + batch, r1)
            bra = bra_vals[start:stop].astype(np.uint64)
            xors = src[:, None] ^ bra[None, :]
            pc = _popcount_u64(xors, dtype=np.int32)
            src_j, bidx = np.where(pc == int(popcount))
            if len(src_j) == 0:
                continue
            xr = xors[src_j, bidx]
            bits = _bits_from_xor_values(xr, int(popcount))
            r_ch.append((bidx + start).astype(np.int32))
            j_ch.append(src_j.astype(np.int32))
            for k in range(int(popcount)):
                p_ch[k].append(bits[k])
        return r_ch, j_ch, p_ch

    n_batches = (n_bra + batch - 1) // batch
    nthreads = max(1, min(int(threads), n_batches))
    if nthreads <= 1:
        blocks = [(0, n_bra)]
    else:
        per = ((n_batches + nthreads - 1) // nthreads) * batch
        blocks = [(i, min(i + per, n_bra)) for i in range(0, n_bra, per)]

    if len(blocks) <= 1:
        results = [_scan_block(blocks[0])]
    else:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        with ThreadPoolExecutor(max_workers=len(blocks)) as ex:
            results = list(ex.map(_scan_block, blocks))

    rows_chunks: list = []
    j_chunks: list = []
    p_chunks: list = [[], [], [], []]
    for r_ch, j_ch, p_ch in results:
        rows_chunks.extend(r_ch)
        j_chunks.extend(j_ch)
        for k in range(4):
            p_chunks[k].extend(p_ch[k])

    if not rows_chunks:
        z_i = np.zeros(1, dtype=np.int32)
        z_u = np.zeros(1, dtype=np.uint8)
        return ptr, z_i, z_u, z_u, z_u, z_u

    rows = np.concatenate(rows_chunks)
    j_rel = np.concatenate(j_chunks)
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    j_rel = j_rel[order]

    out_bits = []
    for k in range(4):
        if k < int(popcount):
            out_bits.append(np.concatenate(p_chunks[k])[order].astype(np.uint8))
        else:
            out_bits.append(np.zeros(len(j_rel), dtype=np.uint8))

    counts = np.bincount(rows, minlength=n_bra).astype(np.int32)
    ptr[1:] = np.cumsum(counts)
    return ptr, j_rel, out_bits[0], out_bits[1], out_bits[2], out_bits[3]


def _build_tile_flip_csr_from_masks(
    bra_vals: np.ndarray,
    src_vals: np.ndarray,
    masks: list[tuple[np.uint64, tuple[np.uint8, ...]]],
    src_ht: tuple | None = None,
    threads: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


    if _native_tile_build_enabled():
        n_m = len(masks)
        mask_vals = np.fromiter(
            (int(_m) for _m, _ in masks), dtype=np.uint64, count=n_m
        )
        mask_pos = np.zeros((n_m, 4), dtype=np.uint8)
        for _i, (_m, _positions) in enumerate(masks):
            for _k in range(len(_positions)):
                mask_pos[_i, _k] = np.uint8(int(_positions[_k]))
        return _native_masks(
            np.ascontiguousarray(bra_vals, dtype=np.uint64),
            np.ascontiguousarray(src_vals, dtype=np.uint64),
            mask_vals,
            mask_pos,
            int(threads),
        )

    n_bra = len(bra_vals)
    ptr = np.zeros(n_bra + 1, dtype=np.int64)
    if n_bra == 0 or len(src_vals) == 0 or not masks:
        z_i = np.zeros(1, dtype=np.int32)
        z_u = np.zeros(1, dtype=np.uint8)
        return ptr, z_i, z_u, z_u, z_u, z_u

    if src_ht is None:
        src_ht = _build_hash_table_simple(src_vals.astype(np.uint64))
    src_ht_keys, src_ht_vals, src_ht_mask = src_ht
    bra = bra_vals.astype(np.uint64)

    def _scan_masks(mask_group):


        r_ch: list = []
        j_ch: list = []
        p_ch: list = [[], [], [], []]
        for mask, positions in mask_group:
            candidates = bra ^ mask
            src_j = _ht_lookup_batch(
                src_ht_keys, src_ht_vals, int(src_ht_mask), candidates
            )
            rows = np.where(src_j >= 0)[0]
            if len(rows) == 0:
                continue
            r_ch.append(rows.astype(np.int32))
            j_ch.append(src_j[rows].astype(np.int32))
            for k in range(4):
                if k < len(positions):
                    p_ch[k].append(np.full(len(rows), positions[k], dtype=np.uint8))
                else:
                    p_ch[k].append(np.zeros(len(rows), dtype=np.uint8))
        return r_ch, j_ch, p_ch

    nthreads = max(1, min(int(threads), len(masks)))
    if nthreads <= 1:
        groups = [masks]
    else:
        step = (len(masks) + nthreads - 1) // nthreads
        groups = [masks[i : i + step] for i in range(0, len(masks), step)]

    if len(groups) <= 1:
        results = [_scan_masks(groups[0])]
    else:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        with ThreadPoolExecutor(max_workers=len(groups)) as ex:
            results = list(ex.map(_scan_masks, groups))

    rows_chunks: list = []
    j_chunks: list = []
    p_chunks: list = [[], [], [], []]
    for r_ch, j_ch, p_ch in results:
        rows_chunks.extend(r_ch)
        j_chunks.extend(j_ch)
        for k in range(4):
            p_chunks[k].extend(p_ch[k])

    if not rows_chunks:
        z_i = np.zeros(1, dtype=np.int32)
        z_u = np.zeros(1, dtype=np.uint8)
        return ptr, z_i, z_u, z_u, z_u, z_u

    rows = np.concatenate(rows_chunks)
    j_rel = np.concatenate(j_chunks)
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    j_rel = j_rel[order]

    out_bits = []
    for k in range(4):
        out_bits.append(np.concatenate(p_chunks[k])[order].astype(np.uint8))

    counts = np.bincount(rows, minlength=n_bra).astype(np.int32)
    ptr[1:] = np.cumsum(counts)
    return ptr, j_rel, out_bits[0], out_bits[1], out_bits[2], out_bits[3]


def _build_tile_flip_csr_hybrid(
    bra_vals: np.ndarray,
    src_vals: np.ndarray,
    masks: list[tuple[np.uint64, tuple[np.uint8, ...]]],
    popcount: int,
    src_ht: tuple | None = None,
    threads: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


    dense_work = int(len(bra_vals)) * int(len(src_vals))
    mask_work = int(len(bra_vals)) * int(len(masks))
    if masks and mask_work * 8 < dense_work:
        return _build_tile_flip_csr_from_masks(
            bra_vals, src_vals, masks, src_ht=src_ht, threads=threads
        )
    return _build_tile_flip_csr_dense(bra_vals, src_vals, popcount, threads=threads)


def _empty_csr_table(n_rows: int):

    ptr = np.zeros(n_rows + 1, dtype=np.int64)
    empty = np.zeros(0, dtype=np.int32)
    return (ptr, empty, empty, empty, empty, empty)


def _build_tile_connection_tables(
    alpha_bra,
    beta_bra,
    alpha_src,
    beta_src,
    flip_masks,
    tile_cls: "TileClass | None" = None,
    threads: int = 1,
):

    alpha_ht = _build_hash_table_simple(alpha_src.astype(np.uint64))
    beta_ht = _build_hash_table_simple(beta_src.astype(np.uint64))

    return {
        "alpha2": _build_tile_flip_csr_hybrid(
            alpha_bra, alpha_src, flip_masks["alpha2"], 2, alpha_ht, threads=threads
        ),
        "alpha4": _build_tile_flip_csr_hybrid(
            alpha_bra, alpha_src, flip_masks["alpha4"], 4, alpha_ht, threads=threads
        ),
        "beta2": _build_tile_flip_csr_hybrid(
            beta_bra, beta_src, flip_masks["beta2"], 2, beta_ht, threads=threads
        ),
        "beta4": _build_tile_flip_csr_hybrid(
            beta_bra, beta_src, flip_masks["beta4"], 4, beta_ht, threads=threads
        ),
    }


def _csr_referenced_columns(
    ptr: np.ndarray, cols: np.ndarray, row0: int, row1: int
) -> np.ndarray:


    start = int(ptr[int(row0)])
    stop = int(ptr[int(row1)])
    if start >= stop:
        return np.zeros(0, dtype=np.int32)
    return np.unique(cols[start:stop].astype(np.int32, copy=False))


def _build_compact_source_descriptor(
    tables: dict,
    *,
    n_a_src: int,
    n_b_src: int,
    n_b_bra: int,
    a_gpu_start: int,
    a_gpu_stop: int,
    same_alpha: bool,
    same_beta: bool,
    add_diag: bool,
) -> dict:


    a_rels = []
    b_rels = []

    if add_diag:
        a_rels.append(np.arange(a_gpu_start, a_gpu_stop, dtype=np.int32))
        b_rels.append(np.arange(n_b_bra, dtype=np.int32))

    alpha2_ptr, alpha2_j = tables["alpha2"][0], tables["alpha2"][1]
    alpha4_ptr, alpha4_j = tables["alpha4"][0], tables["alpha4"][1]
    beta2_ptr, beta2_j = tables["beta2"][0], tables["beta2"][1]
    beta4_ptr, beta4_j = tables["beta4"][0], tables["beta4"][1]

    alpha2_cols = _csr_referenced_columns(alpha2_ptr, alpha2_j, a_gpu_start, a_gpu_stop)
    if len(alpha2_cols) > 0:
        a_rels.append(alpha2_cols)

    if same_beta:
        alpha4_cols = _csr_referenced_columns(
            alpha4_ptr, alpha4_j, a_gpu_start, a_gpu_stop
        )
        if len(alpha4_cols) > 0:
            a_rels.append(alpha4_cols)
        if len(alpha2_cols) > 0 or len(alpha4_cols) > 0:
            b_rels.append(np.arange(n_b_bra, dtype=np.int32))

    beta2_cols = _csr_referenced_columns(beta2_ptr, beta2_j, 0, n_b_bra)
    if len(beta2_cols) > 0:
        b_rels.append(beta2_cols)

    if same_alpha:
        beta4_cols = _csr_referenced_columns(beta4_ptr, beta4_j, 0, n_b_bra)
        if len(beta4_cols) > 0:
            b_rels.append(beta4_cols)
        if len(beta2_cols) > 0 or len(beta4_cols) > 0:
            a_rels.append(np.arange(a_gpu_start, a_gpu_stop, dtype=np.int32))

    alpha_rels = (
        np.unique(np.concatenate(a_rels).astype(np.int32, copy=False))
        if a_rels
        else np.zeros(0, dtype=np.int32)
    )
    beta_rels = (
        np.unique(np.concatenate(b_rels).astype(np.int32, copy=False))
        if b_rels
        else np.zeros(0, dtype=np.int32)
    )

    alpha_slot = np.full(max(int(n_a_src), 1), -1, dtype=np.int32)
    beta_slot = np.full(max(int(n_b_src), 1), -1, dtype=np.int32)
    if len(alpha_rels) > 0:
        alpha_slot[alpha_rels] = np.arange(len(alpha_rels), dtype=np.int32)
    if len(beta_rels) > 0:
        beta_slot[beta_rels] = np.arange(len(beta_rels), dtype=np.int32)

    if len(alpha_rels) == 0 or len(beta_rels) == 0:
        dense_offsets = np.zeros(0, dtype=np.int64)
    else:
        dense_offsets = (
            beta_rels[:, None].astype(np.int64) * np.int64(n_a_src)
            + alpha_rels[None, :].astype(np.int64)
        ).ravel()

    return {
        "alpha_slot": alpha_slot,
        "beta_slot": beta_slot,
        "dense_offsets": dense_offsets,
        "n_alpha_compact": int(len(alpha_rels)),
    }
