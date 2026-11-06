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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dist_matvec import TileClass, _split_ranges_even
from .rank_agreement import raise_on_every_rank
from .tile_discovery import _tile_source_needed

Range = tuple[int, int]
Tile = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class TileGrid:


    p_alpha: int
    p_beta: int
    alpha_ranges: tuple[Range, ...]
    beta_ranges: tuple[Range, ...]

    @classmethod
    def build(
        cls, p_alpha: int, p_beta: int, alpha_strings, beta_strings
    ) -> "TileGrid":


        n_a, n_b = len(alpha_strings), len(beta_strings)
        return cls(
            p_alpha,
            p_beta,
            tuple(_split_ranges_even(n_a, p_alpha)),
            tuple(_split_ranges_even(n_b, p_beta)),
        )

    @property
    def size(self) -> int:

        return self.p_alpha * self.p_beta

    def coords(self, rank: int) -> tuple[int, int]:

        return rank // self.p_beta, rank % self.p_beta

    def rank_of(self, coord_alpha: int, coord_beta: int) -> int:

        return coord_alpha * self.p_beta + coord_beta

    def tile(self, rank: int) -> Tile:

        coord_alpha, coord_beta = self.coords(rank)
        a0, a1 = self.alpha_ranges[coord_alpha]
        b0, b1 = self.beta_ranges[coord_beta]
        return a0, a1, b0, b1, coord_alpha, coord_beta

    def tiles(self) -> list:

        return [self.tile(rank) for rank in range(self.size)]

    def tile_class(self, rank: int, source: int) -> TileClass:

        coord_alpha, coord_beta = self.coords(rank)
        source_alpha, source_beta = self.coords(source)
        same_alpha = source_alpha == coord_alpha
        same_beta = source_beta == coord_beta
        if same_alpha and same_beta:
            return TileClass.DIAGONAL
        if same_beta:
            return TileClass.ALPHA_ONLY
        if same_alpha:
            return TileClass.BETA_ONLY
        return TileClass.MIXED


@dataclass(frozen=True)
class GridComms:


    world: object
    alpha: object
    beta: object

    @classmethod
    def split(cls, comm, grid: TileGrid) -> "GridComms":

        coord_alpha, coord_beta = grid.coords(comm.Get_rank())
        rank = comm.Get_rank()
        return cls(
            comm,
            comm.Split(color=coord_alpha, key=rank),
            comm.Split(color=coord_beta, key=rank),
        )


@dataclass(frozen=True)
class SourceSet:


    sources: dict
    mixed_sources: list
    mixed_destinations: list
    reference: Optional[dict] = None

    def class_counts(self) -> dict:

        counts: dict = {}
        for tile_class in self.sources.values():
            counts[tile_class.name] = counts.get(tile_class.name, 0) + 1
        return counts


@dataclass(frozen=True)
class _ScanInput:


    grid: TileGrid
    rank: int
    alpha_strings: np.ndarray
    beta_strings: np.ndarray
    flip_masks: dict
    threads: int

    def bra(self) -> tuple[np.ndarray, np.ndarray]:

        a0, a1, b0, b1, _coord_alpha, _coord_beta = self.grid.tile(self.rank)
        return self.alpha_strings[a0:a1], self.beta_strings[b0:b1]

    def item(self, source: int, tile_class: Optional[TileClass] = None) -> tuple:

        a0, a1, b0, b1, _coord_alpha, _coord_beta = self.grid.tile(source)
        if tile_class is None:
            tile_class = self.grid.tile_class(self.rank, source)
        return source, tile_class, self.alpha_strings[a0:a1], self.beta_strings[b0:b1]


def _scan(scan: _ScanInput, items: list) -> dict:


    if not items:
        return {}
    alpha_bra, beta_bra = scan.bra()
    threads = max(1, int(scan.threads))
    per_check = min(threads, 4)
    workers = max(1, min(len(items), threads // per_check))

    def check(item):

        source, tile_class, alpha_source, beta_source = item
        needed = _tile_source_needed(
            alpha_bra,
            beta_bra,
            alpha_source,
            beta_source,
            scan.flip_masks,
            threads=per_check,
        )
        return (source, tile_class) if needed else None

    if workers <= 1:
        results = [check(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(check, items))
    return {result[0]: result[1] for result in results if result is not None}


def _signatures(scan: _ScanInput) -> tuple[int, int]:

    alpha_bra, beta_bra = scan.bra()
    signature = []
    for bra, strings, keys in (
        (alpha_bra, scan.alpha_strings, ("alpha2", "alpha4")),
        (beta_bra, scan.beta_strings, ("beta2", "beta4")),
    ):
        value = 0
        for key in keys:
            for mask, _positions in scan.flip_masks.get(key, []):
                flipped = bra.astype(np.uint64) ^ np.uint64(mask)
                if np.isin(flipped, strings, assume_unique=False).any():
                    value |= int(mask)
        signature.append(value)
    return signature[0], signature[1]


def _mixed_sources(scan: _ScanInput, comms: GridComms) -> dict:

    grid = scan.grid
    coord_alpha, coord_beta = grid.coords(scan.rank)
    signature_alpha, signature_beta = _signatures(scan)
    column_alpha = comms.beta.allgather(signature_alpha)
    row_beta = comms.alpha.allgather(signature_beta)
    any_masks = any(
        len(scan.flip_masks.get(key, [])) > 0
        for key in ("alpha2", "alpha4", "beta2", "beta4")
    )
    items = []
    for source_alpha in range(grid.p_alpha):
        if source_alpha == coord_alpha:
            continue
        if any_masks and (column_alpha[coord_alpha] & column_alpha[source_alpha]) == 0:
            continue
        for source_beta in range(grid.p_beta):
            if source_beta == coord_beta:
                continue
            if any_masks and (row_beta[coord_beta] & row_beta[source_beta]) == 0:
                continue
            items.append(
                scan.item(grid.rank_of(source_alpha, source_beta), TileClass.MIXED)
            )
    return _scan(scan, items)


def _row_and_column_sources(scan: _ScanInput) -> dict:

    grid = scan.grid
    coord_alpha, coord_beta = grid.coords(scan.rank)
    items = [
        scan.item(grid.rank_of(source_alpha, coord_beta))
        for source_alpha in range(grid.p_alpha)
    ]
    items += [
        scan.item(grid.rank_of(coord_alpha, source_beta))
        for source_beta in range(grid.p_beta)
        if source_beta != coord_beta
    ]
    return _scan(scan, items)


def _cross_check_error(sources: dict, reference: dict) -> Optional[str]:

    if sources == reference:
        return None
    only_reference = sorted(set(reference) - set(sources))
    only_fast = sorted(set(sources) - set(reference))
    class_diff = {
        rank: (reference.get(rank), sources.get(rank))
        for rank in set(reference) | set(sources)
        if reference.get(rank) != sources.get(rank)
    }
    return (
        f"the full-scan reference and the row and column scan disagree: only_reference={only_reference} "
        f"only_scan={only_fast} class_diff={class_diff}"
    )


def discover_sources(
    grid: TileGrid,
    comms: GridComms,
    alpha_strings,
    beta_strings,
    flip_masks: dict,
    threads: int,
    cross_check_max_ranks: int,
) -> SourceSet:


    rank = comms.world.Get_rank()
    scan = _ScanInput(
        grid,
        rank,
        np.asarray(alpha_strings),
        np.asarray(beta_strings),
        flip_masks,
        threads,
    )
    reference = None
    if grid.size <= cross_check_max_ranks:
        reference = _scan(scan, [scan.item(source) for source in range(grid.size)])
    sources = _row_and_column_sources(scan)
    sources.update(_mixed_sources(scan, comms))
    error = None if reference is None else _cross_check_error(sources, reference)
    raise_on_every_rank(comms.world, error, RuntimeError, "source-tile discovery")
    mixed_sources = sorted(
        source
        for source, tile_class in sources.items()
        if tile_class is TileClass.MIXED
    )
    need_receive = np.zeros(grid.size, dtype=np.int32)
    need_receive[mixed_sources] = 1
    need_send = np.empty_like(need_receive)
    comms.world.Alltoall(need_receive, need_send)
    mixed_destinations = sorted(np.flatnonzero(need_send).tolist())
    raise_on_every_rank(
        comms.world,
        None
        if mixed_sources == mixed_destinations
        else f"MIXED edge set asymmetric: sources={mixed_sources} destinations={mixed_destinations}. Either the "
        f"operator is not Hermitian or the discovery misclassified an edge.",
        RuntimeError,
        "mixed source discovery",
    )
    return SourceSet(sources, mixed_sources, mixed_destinations, reference)
