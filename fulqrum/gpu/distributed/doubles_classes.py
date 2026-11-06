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

ELEMENT_SCREEN = (
    1e-14
)
TILE_ROWS = 64
CHUNK = 16
CHUNK_DOUBLES = CHUNK // 4 * 8 * 32


@dataclass
class DoubleClassTables:


    blocks: np.ndarray
    block_offsets: np.ndarray
    tile_info: np.ndarray
    row_members: np.ndarray
    col_members: np.ndarray
    residual_offsets: np.ndarray
    residual_columns: np.ndarray
    residual_values: np.ndarray
    stats: dict
    in_class: object = None


def _members(xp, classes, items, span):


    key = classes * span + items
    unique, inverse = xp.unique(key, return_inverse=True)
    return xp.stack((unique // span, unique % span), axis=1), inverse.reshape(-1)


def _cumsum0(xp, counts):

    out = xp.zeros(len(counts) + 1, dtype=xp.int64)
    out[1:] = xp.cumsum(counts)
    return out


def double_class_tables(
    offsets, columns, values, out_strings, src_strings, min_size, xp=np
) -> DoubleClassTables:


    offsets = xp.asarray(offsets, dtype=xp.int64)
    n_rows = len(offsets) - 1
    nnz = int(offsets[-1])
    columns = xp.asarray(columns[:nnz], dtype=xp.int64)
    values = xp.asarray(values[:nnz], dtype=xp.float64)

    starts = xp.zeros(nnz + 1, dtype=xp.int64)
    xp.add.at(starts, offsets[1:-1], 1)
    rows = xp.cumsum(starts[:nnz])
    applied = xp.abs(values) > ELEMENT_SCREEN
    keys = (
        xp.asarray(out_strings, dtype=xp.uint64)[rows]
        | xp.asarray(src_strings, dtype=xp.uint64)[columns]
    )
    ukeys, cls = xp.unique(keys, return_inverse=True)
    cls = cls.reshape(-1).astype(xp.int64)
    take = applied & xp.isfinite(values)
    take_idx = xp.flatnonzero(take)

    span = max(n_rows, int(columns.max()) + 1 if nnz else 1, 1)
    n_cls = len(ukeys)
    if n_cls * span >= 2**62:
        raise OverflowError("class member keys exceed int64")
    t_cls = cls[take_idx]
    row_pairs, row_pos = _members(xp, t_cls, rows[take_idx], span)
    col_pairs, col_pos = _members(xp, t_cls, columns[take_idx], span)
    n_row = xp.bincount(row_pairs[:, 0], minlength=n_cls)
    n_col = xp.bincount(col_pairs[:, 0], minlength=n_cls)
    big = (n_row >= min_size) & (n_col >= min_size)
    in_class = xp.zeros(nnz, dtype=bool)
    in_class[take_idx] = big[t_cls]
    row_start = _cumsum0(xp, n_row)
    col_start = _cumsum0(xp, n_col)

    r_local = row_pos - row_start[t_cls]
    c_local = col_pos - col_start[t_cls]
    keep_t = big[t_cls]

    kept = xp.flatnonzero(big)
    new_id = -xp.ones(n_cls, dtype=xp.int64)
    new_id[kept] = xp.arange(len(kept))
    k_rows, k_cols = n_row[kept], n_col[kept]
    row_members = row_pairs[big[row_pairs[:, 0]], 1].astype(xp.int32)
    col_members = col_pairs[big[col_pairs[:, 0]], 1].astype(xp.int32)
    k_row_start = _cumsum0(xp, k_rows)[:-1]
    k_col_start = _cumsum0(xp, k_cols)[:-1]
    tiles_per = (k_rows + TILE_ROWS - 1) // TILE_ROWS
    chunks_per = (k_cols + CHUNK - 1) // CHUNK
    first_tile = _cumsum0(xp, tiles_per)
    n_tiles = int(first_tile[-1])
    first_tile = first_tile[:-1]

    tile_class = xp.searchsorted(first_tile, xp.arange(n_tiles), side="right") - 1
    tile_index = xp.arange(n_tiles) - first_tile[tile_class]
    tile_doubles = chunks_per[tile_class] * CHUNK_DOUBLES
    block_offsets = _cumsum0(xp, tile_doubles)
    total = int(block_offsets[-1])
    block_offsets = block_offsets[:-1]
    tile_info = xp.stack(
        (
            k_row_start[tile_class] + TILE_ROWS * tile_index,
            xp.minimum(TILE_ROWS, k_rows[tile_class] - TILE_ROWS * tile_index),
            k_col_start[tile_class],
            k_cols[tile_class],
        ),
        axis=1,
    ).astype(xp.int32)


    e_cls = new_id[t_cls[keep_t]]
    r = r_local[keep_t]
    k = c_local[keep_t]
    tile = first_tile[e_cls] + r // TILE_ROWS
    rr = r % TILE_ROWS
    pos = (
        block_offsets[tile]
        + (k // CHUNK) * CHUNK_DOUBLES
        + ((k % CHUNK) // 4) * 256
        + (rr // 8) * 32
        + (rr % 8) * 4
        + k % 4
    )
    blocks = xp.zeros(max(total, 2), dtype=xp.float64)

    xp.add.at(blocks, pos, values[take_idx][keep_t])

    resid = xp.flatnonzero(applied & ~in_class)

    residual_counts = (
        xp.bincount(rows[resid], minlength=n_rows)
        if len(resid)
        else xp.zeros(n_rows, xp.int64)
    )
    residual_offsets = _cumsum0(xp, residual_counts)
    class_edges = int(keep_t.sum())
    issued = int((tiles_per * chunks_per).sum()) * TILE_ROWS * CHUNK
    stats = dict(
        classes=int(len(kept)),
        tiles=n_tiles,
        class_edges=class_edges,
        residual_edges=int(len(resid)),
        block_mib=total * 8 / 2**20,
        fill=class_edges / max(issued, 1),
    )
    if n_tiles > np.iinfo(np.int32).max or total >= 2**62:
        raise OverflowError("class tables exceed the kernel index range")
    return DoubleClassTables(
        blocks=blocks,
        block_offsets=block_offsets,
        tile_info=xp.ascontiguousarray(tile_info),
        row_members=row_members,
        col_members=col_members,
        residual_offsets=residual_offsets,
        residual_columns=columns[resid].astype(xp.int32),
        residual_values=values[resid],
        stats=stats,
        in_class=in_class,
    )


def sell_from_csr(offsets, columns, values, xp=np, slice_rows=32):


    offsets = xp.asarray(offsets, dtype=xp.int64)
    n_rows = len(offsets) - 1
    lengths = offsets[1:] - offsets[:-1]
    n_slice = (n_rows + slice_rows - 1) // slice_rows
    padded = xp.zeros(n_slice * slice_rows, dtype=xp.int64)
    padded[:n_rows] = lengths
    width = (
        padded.reshape(n_slice, slice_rows).max(axis=1)
        if n_slice
        else xp.zeros(0, dtype=xp.int64)
    )
    off = _cumsum0(xp, width * slice_rows)
    nnz = int(offsets[-1])
    starts = xp.zeros(nnz + 1, dtype=xp.int64)
    xp.add.at(starts, offsets[1:-1], 1)
    rows = xp.cumsum(starts[:nnz])
    k = xp.arange(nnz, dtype=xp.int64) - offsets[rows]
    slot = off[rows // slice_rows] + k * slice_rows + rows % slice_rows
    total = max(int(off[-1]), 1)
    sell_columns = xp.zeros(total, dtype=xp.int32)
    sell_values = xp.zeros(total, dtype=xp.float64)
    sell_columns[slot] = xp.asarray(columns[:nnz], dtype=xp.int32)
    sell_values[slot] = xp.asarray(values[:nnz], dtype=xp.float64)
    return off, lengths.astype(xp.int32), sell_columns, sell_values
