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

import dataclasses
import time
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional

import numpy as np

from .construction import _build_axis_connection_tables
from .tile_tables import (
    CROSS_MAX_PAIR,
    CROSS_PAIR_SHIFT,
    CROSS_SIGN_SHIFT,
    SELL_MAX_SLOTS,
    SELL_SLICE_ROWS,
)


@dataclass(frozen=True)
class Csr:


    ptr: Any
    j: Any
    positions: tuple


@dataclass(frozen=True)
class SinglesSell:


    off: Any
    length: Any
    j: Any
    apid: Any
    asign: Any
    pack: Any


@dataclass(frozen=True)
class DoublesSell:


    off: Any
    length: Any
    j: Any
    positions: tuple


@dataclass(frozen=True)
class AlphaTables:


    singles: Csr
    doubles: Csr
    apid: Any = None
    asign: Any = None
    singles_sell: Optional[SinglesSell] = None
    doubles_sell: Optional[DoublesSell] = None

    def cross_sector_bytes(self) -> int:

        return array_bytes((self.apid, self.asign, self.singles_sell))


@dataclass(frozen=True)
class BetaTables:


    singles: Csr
    doubles: Csr


@dataclass(frozen=True)
class ArmCacheLengths:


    aaaa: int
    bbbb: int
    aa: int
    bb: int


@dataclass(frozen=True)
class SourceTables:


    alpha: AlphaTables
    beta: BetaTables

    @cached_property
    def launch_view(self) -> dict:

        alpha, beta = self.alpha, self.beta
        view = {
            "d_alpha2_ptr": alpha.singles.ptr,
            "d_alpha2_j": alpha.singles.j,
            "d_alpha2_p0": alpha.singles.positions[0],
            "d_alpha2_p1": alpha.singles.positions[1],
            "d_alpha2_apid": alpha.apid,
            "d_alpha2_asign": alpha.asign,
            "d_alpha4_ptr": alpha.doubles.ptr,
            "d_alpha4_j": alpha.doubles.j,
            "d_beta2_ptr": beta.singles.ptr,
            "d_beta2_j": beta.singles.j,
            "d_beta2_p0": beta.singles.positions[0],
            "d_beta2_p1": beta.singles.positions[1],
            "d_beta4_ptr": beta.doubles.ptr,
            "d_beta4_j": beta.doubles.j,
        }
        for index in range(4):
            view[f"d_alpha4_p{index}"] = alpha.doubles.positions[index]
            view[f"d_beta4_p{index}"] = beta.doubles.positions[index]
        sell = alpha.singles_sell
        for name in ("off", "len", "j", "apid", "asign", "pack"):
            field = "length" if name == "len" else name
            view[f"d_alpha2_sell_{name}"] = (
                None if sell is None else getattr(sell, field)
            )
        sell4 = alpha.doubles_sell
        view["d_alpha4_sell_off"] = None if sell4 is None else sell4.off
        view["d_alpha4_sell_len"] = None if sell4 is None else sell4.length
        view["d_alpha4_sell_j"] = None if sell4 is None else sell4.j
        for index in range(4):
            view[f"d_alpha4_sell_p{index}"] = (
                None if sell4 is None else sell4.positions[index]
            )
        return view

    def arm_cache_lengths(
        self, same_alpha: bool, same_beta: bool, doubles_sell: bool, skip_unread: bool
    ) -> ArmCacheLengths:


        doubles = self.alpha.doubles_sell.j if doubles_sell else self.alpha.doubles.j
        aaaa, aa = int(doubles.size), int(self.alpha.singles.j.size)
        bbbb, bb = int(self.beta.doubles.j.size), int(self.beta.singles.j.size)
        if skip_unread and not same_beta:
            aaaa = aa = 0
        if skip_unread and not same_alpha:
            bbbb = bb = 0
        return ArmCacheLengths(aaaa, bbbb, aa, bb)


@dataclass(frozen=True)
class TileTables:


    alpha: dict
    beta: dict
    sources: dict

    def connection_bytes(self) -> int:

        return (
            array_bytes((tuple(self.alpha.values()), tuple(self.beta.values())))
            - self.cross_sector_bytes()
        )

    def cross_sector_bytes(self) -> int:

        return sum(tables.cross_sector_bytes() for tables in self.alpha.values())


@dataclass(frozen=True)
class TableOptions:


    num_alpha: int
    factored: bool


def build_tile_tables(
    self_tables: dict,
    alpha_bra,
    beta_bra,
    coords: tuple,
    alpha_peers: dict,
    beta_peers: dict,
    source_coords: dict,
    flip_masks: dict,
    options: TableOptions,
) -> tuple:


    started = time.perf_counter()
    coord_alpha, coord_beta = coords
    alpha = {
        coord_alpha: alpha_tables(
            self_tables["alpha2"], self_tables["alpha4"], alpha_bra, options
        )
    }
    beta = {coord_beta: beta_tables(self_tables["beta2"], self_tables["beta4"])}
    for coord in sorted(alpha_peers):
        built = _build_axis_connection_tables(
            alpha_bra, alpha_peers[coord], flip_masks["alpha2"], flip_masks["alpha4"]
        )
        alpha[coord] = alpha_tables(built["csr2"], built["csr4"], alpha_bra, options)
    for coord in sorted(beta_peers):
        built = _build_axis_connection_tables(
            beta_bra, beta_peers[coord], flip_masks["beta2"], flip_masks["beta4"]
        )
        beta[coord] = beta_tables(built["csr2"], built["csr4"])
    host_seconds = time.perf_counter() - started
    started = time.perf_counter()
    alpha = {coord: to_device(tables) for coord, tables in alpha.items()}
    beta = {coord: to_device(tables) for coord, tables in beta.items()}
    sources = {
        rank: SourceTables(alpha[ca], beta[cb])
        for rank, (ca, cb) in source_coords.items()
    }
    return TileTables(alpha, beta, sources), host_seconds, time.perf_counter() - started


def alpha_tables(
    singles: tuple, doubles: tuple, row_alpha, options: TableOptions
) -> AlphaTables:


    a2_ptr, a2_j, a2_p0, a2_p1 = singles[:4]
    a4_ptr, a4_j, a4_p0, a4_p1, a4_p2, a4_p3 = doubles
    tables = dict(singles=_csr(singles[:4]), doubles=_csr(doubles))
    sell = _alpha4_sell(a4_ptr, a4_j, a4_p0, a4_p1, a4_p2, a4_p3)
    tables["doubles_sell"] = DoublesSell(sell[0], sell[1], sell[2], tuple(sell[3:]))
    if options.factored:
        apid, asign = _alpha2_apid_asign(
            a2_ptr, a2_p0, a2_p1, row_alpha, options.num_alpha
        )
        tables["apid"], tables["asign"] = _host(apid, np.int32), _host(asign, np.int8)
        tables["singles_sell"] = SinglesSell(*_alpha2_sell(a2_ptr, a2_j, apid, asign))
    return AlphaTables(**tables)


def beta_tables(singles: tuple, doubles: tuple) -> BetaTables:

    return BetaTables(_csr(singles[:4]), _csr(doubles))


def to_device(value):

    import cupy  # noqa: PLC0415

    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value,
            **{
                field.name: to_device(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, tuple):
        return tuple(to_device(item) for item in value)
    if isinstance(value, np.ndarray):
        return cupy.asarray(value)
    return value


def array_bytes(value) -> int:

    if value is None:
        return 0
    if dataclasses.is_dataclass(value):
        return sum(
            array_bytes(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, tuple):
        return sum(array_bytes(item) for item in value)
    return int(value.nbytes)


def arm_cache_sizes(lengths: ArmCacheLengths, num_alpha: int, num_beta: int) -> dict:


    return dict(
        aaaa=lengths.aaaa,
        bbbb=lengths.bbbb,
        aa_c0=lengths.aa,
        aa_cpb=lengths.aa * num_beta,
        bb_c0=lengths.bb,
        bb_cpb=lengths.bb * num_alpha,
    )


def hbm_fit_error(
    predicted: dict, free_bytes: int, total_bytes: int, fraction: float
) -> Optional[str]:


    total = sum(predicted.values())
    if total <= fraction * free_bytes:
        return None
    return (
        f"predicted={total} B exceeds {fraction:.0%} of free_bytes={free_bytes} B "
        f"(total={total_bytes} B). Breakdown: {predicted}."
    )


def values_on_edges(
    gpu_ptr, gpu_columns, host_offsets, host_columns, host_values
) -> np.ndarray:


    width = (
        int(
            max(
                int(np.max(gpu_columns, initial=0)),
                int(np.max(host_columns, initial=0)),
            )
        )
        + 1
    )
    gpu_keys = _row_keys(gpu_ptr, gpu_columns, width)
    host_keys = _row_keys(host_offsets, host_columns, width)
    order = np.argsort(host_keys, kind="stable")
    host_keys, host_values = (
        host_keys[order],
        np.asarray(host_values, dtype=np.float64)[order],
    )
    values = np.zeros(gpu_keys.size)
    if host_keys.size == 0:
        return values
    position = np.minimum(np.searchsorted(host_keys, gpu_keys), host_keys.size - 1)
    matched = host_keys[position] == gpu_keys
    if int(np.count_nonzero(matched)) != host_keys.size:
        raise ValueError("a host connection has no GPU connection")
    values[matched] = host_values[position[matched]]
    return values


def pair_table(factors: np.ndarray, orbitals: int, tolerance: float) -> np.ndarray:


    num_pairs = orbitals * (orbitals - 1) // 2
    table = np.zeros((factors.shape[0], num_pairs))
    if num_pairs == 0 or factors.size == 0:
        return table
    forward, backward = factors[:, 0::2], factors[:, 1::2]
    if np.abs(forward - backward).max() > tolerance * max(np.abs(factors).max(), 1.0):
        raise ValueError(
            "a same-sector cross factor depends on the excitation direction"
        )
    gpu, core = _pair_ids(orbitals)
    table[:, gpu] = forward[:, core]
    return table


def aabb_coefficients(
    cross_sector: np.ndarray, num_alpha: int, num_beta: int
) -> np.ndarray:


    alpha_pairs = num_alpha * (num_alpha - 1) // 2
    beta_pairs = num_beta * (num_beta - 1) // 2
    coefficients = np.zeros((beta_pairs, alpha_pairs))
    if alpha_pairs == 0 or beta_pairs == 0:
        return coefficients.reshape(-1)
    alpha_gpu, alpha_core = _pair_ids(num_alpha)
    beta_gpu, beta_core = _pair_ids(num_beta)
    coefficients[np.ix_(beta_gpu, alpha_gpu)] = cross_sector[
        np.ix_(2 * beta_core, 4 * alpha_core)
    ]
    return coefficients.reshape(-1)


def _pair_ids(orbitals: int) -> tuple:

    high = np.array([q for q in range(orbitals) for _ in range(q)], dtype=np.int64)
    low = np.array([p for q in range(orbitals) for p in range(q)], dtype=np.int64)
    return low * (2 * orbitals - low - 1) // 2 + (high - low - 1), high * (
        high - 1
    ) // 2 + low


def _row_keys(offsets, columns, width: int) -> np.ndarray:

    offsets = np.asarray(offsets, dtype=np.int64)
    rows = np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))
    return rows * width + np.asarray(columns, dtype=np.int64)[: rows.size]


def _csr(arrays: tuple) -> Csr:

    return Csr(
        _host(arrays[0], np.int64),
        _host(arrays[1], np.int32),
        tuple(_host(position, np.uint8) for position in arrays[2:]),
    )


def _host(array, dtype) -> np.ndarray:

    host = np.ascontiguousarray(array)
    if host.dtype != np.dtype(dtype):
        host = host.astype(dtype, copy=False)
    return host


def _popcount_u64(x):

    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def _alpha2_apid_asign(a2_ptr, a2_p0, a2_p1, row_alpha_dets, num_alpha):


    p0 = np.asarray(a2_p0, dtype=np.int64)
    p1 = np.asarray(a2_p1, dtype=np.int64)
    m = np.int64(num_alpha)
    apid = (p0 * (2 * m - p0 - 1) // 2 + (p1 - p0 - 1)).astype(np.int32)
    ptr = np.ascontiguousarray(a2_ptr, dtype=np.int64)
    counts = np.diff(ptr)
    row = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), counts)
    a_val = np.asarray(row_alpha_dets, dtype=np.uint64)[row]
    up0 = p0.astype(np.uint64)
    up1 = p1.astype(np.uint64)
    mask = ((np.uint64(1) << up1) - np.uint64(1)) ^ (
        (np.uint64(1) << (up0 + np.uint64(1))) - np.uint64(1)
    )
    odd = _popcount_u64(a_val & mask) & np.uint8(1)
    asign = np.where(odd, np.int8(-1), np.int8(1)).astype(np.int8)
    return apid, asign


def _sell_layout(ptr, warp=SELL_SLICE_ROWS):


    ptr = np.ascontiguousarray(ptr, dtype=np.int64)
    n_rows = len(ptr) - 1
    row_len = np.diff(ptr)
    n_slice = (n_rows + warp - 1) // warp
    padded = np.zeros(n_slice * warp, dtype=np.int64)
    padded[:n_rows] = row_len
    slice_w = padded.reshape(n_slice, warp).max(axis=1)
    off = np.zeros(n_slice + 1, dtype=np.int64)
    off[1:] = np.cumsum(slice_w * warp)
    if int(off[-1]) > SELL_MAX_SLOTS:
        raise OverflowError(
            f"SELL layout needs {int(off[-1])} padded entries, beyond {SELL_MAX_SLOTS}"
        )
    E = int(ptr[-1])
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), row_len)
    k = np.arange(E, dtype=np.int64) - ptr[rows]
    dst = off[rows // warp] + k * warp + (rows % warp)
    return off, row_len, dst


def _alpha2_sell(a2_ptr, a2_j, apid, asign, warp=SELL_SLICE_ROWS):


    off, row_len, dst = _sell_layout(a2_ptr, warp)
    total = int(off[-1])
    sj = np.zeros(total, dtype=np.int32)
    sapid = np.zeros(total, dtype=np.int32)
    sasign = np.zeros(total, dtype=np.int8)
    sj[dst] = np.ascontiguousarray(a2_j, dtype=np.int32)
    sapid[dst] = np.ascontiguousarray(apid, dtype=np.int32)
    sasign[dst] = np.ascontiguousarray(asign, dtype=np.int8)
    if sapid.size:
        if int(sapid.max()) > CROSS_MAX_PAIR:
            raise ValueError(
                f"alpha2 pair-id {int(sapid.max())} exceeds the 16-bit mode-10 field"
            )
    _a2_asign = np.ascontiguousarray(asign, dtype=np.int8)
    if _a2_asign.size:
        if not set(np.unique(_a2_asign).tolist()) <= {-1, 1}:
            raise ValueError("alpha2 asign not in {-1,+1}")
    spack = (
        sj.astype(np.uint64)
        | (
            (sapid.astype(np.uint64) & np.uint64(CROSS_MAX_PAIR))
            << np.uint64(CROSS_PAIR_SHIFT)
        )
        | (
            ((sasign.astype(np.int64) & 0xFFFF).astype(np.uint64))
            << np.uint64(CROSS_SIGN_SHIFT)
        )
    )
    return (
        off,
        row_len.astype(np.int32),
        sj,
        sapid,
        sasign,
        spack,
    )


def _alpha4_sell(a4_ptr, a4_j, a4_p0, a4_p1, a4_p2, a4_p3, warp=SELL_SLICE_ROWS):


    off, row_len, dst = _sell_layout(a4_ptr, warp)
    total = int(off[-1])
    sj = np.zeros(total, dtype=np.int32)
    sp0 = np.zeros(total, dtype=np.uint8)
    sp1 = np.zeros(total, dtype=np.uint8)
    sp2 = np.zeros(total, dtype=np.uint8)
    sp3 = np.zeros(total, dtype=np.uint8)
    sj[dst] = np.ascontiguousarray(a4_j, dtype=np.int32)
    sp0[dst] = np.ascontiguousarray(a4_p0, dtype=np.uint8)
    sp1[dst] = np.ascontiguousarray(a4_p1, dtype=np.uint8)
    sp2[dst] = np.ascontiguousarray(a4_p2, dtype=np.uint8)
    sp3[dst] = np.ascontiguousarray(a4_p3, dtype=np.uint8)
    return off, row_len.astype(np.int32), sj, sp0, sp1, sp2, sp3
