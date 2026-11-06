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
import pytest

from fulqrum.gpu.distributed.dist_matvec import TileClass
from fulqrum.gpu.distributed.grid import TileGrid


def strings(count):
    values = sorted((1 << i) | (1 << j) for i in range(12) for j in range(i + 1, 12))
    return np.asarray(values[:count], dtype=np.uint64)


@pytest.mark.parametrize(
    "p_alpha,p_beta,n_a,n_b",
    [(1, 1, 5, 7), (2, 3, 10, 7), (4, 1, 9, 9), (3, 4, 13, 20)],
)
def test_even_partition_covers_every_string_once(p_alpha, p_beta, n_a, n_b):
    grid = TileGrid.build(p_alpha, p_beta, strings(n_a), strings(n_b))
    for ranges, count in ((grid.alpha_ranges, n_a), (grid.beta_ranges, n_b)):
        assert ranges[0][0] == 0 and ranges[-1][1] == count
        assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
        sizes = [end - begin for begin, end in ranges]
        assert max(sizes) - min(sizes) <= 1


def test_coordinates_and_tiles_are_consistent():
    grid = TileGrid.build(3, 4, strings(13), strings(20))
    assert grid.size == 12
    for rank in range(grid.size):
        coord_alpha, coord_beta = grid.coords(rank)
        assert grid.rank_of(coord_alpha, coord_beta) == rank
        a0, a1, b0, b1, tile_alpha, tile_beta = grid.tile(rank)
        assert (a0, a1) == grid.alpha_ranges[coord_alpha] and (
            b0,
            b1,
        ) == grid.beta_ranges[coord_beta]
        assert (tile_alpha, tile_beta) == (coord_alpha, coord_beta)
    assert grid.tiles() == [grid.tile(rank) for rank in range(grid.size)]


def test_tile_classes_follow_the_shared_coordinate():
    grid = TileGrid.build(2, 2, strings(4), strings(4))
    assert grid.tile_class(0, 0) is TileClass.DIAGONAL
    assert grid.tile_class(0, 2) is TileClass.ALPHA_ONLY
    assert grid.tile_class(0, 1) is TileClass.BETA_ONLY
    assert grid.tile_class(0, 3) is TileClass.MIXED
