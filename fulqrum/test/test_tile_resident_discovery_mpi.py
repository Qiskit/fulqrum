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

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mpi4py")
from mpi4py import MPI

if MPI.COMM_WORLD.Get_size() != 4:
    pytest.skip(
        f"test_tile_resident_discovery_mpi requires `mpirun -n 4` "
        f"(got COMM_WORLD size={MPI.COMM_WORLD.Get_size()})",
        allow_module_level=True,
    )

from fulqrum.gpu.distributed import _collective as col
from fulqrum.gpu.distributed.dist_matvec import TileClass
from fulqrum.gpu.distributed.dist_matvec_tile_resident import TileGPUResidentHamiltonian

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


def _hsub_with_singles_coupling() -> SimpleNamespace:
    flip_masks = {
        "alpha2": [(np.uint64(0b11), (np.uint8(0), np.uint8(1)))],
        "alpha4": [],
        "beta2": [(np.uint64(0b11), (np.uint8(0), np.uint8(1)))],
        "beta4": [],
    }
    return SimpleNamespace(spmv=SimpleNamespace(is_real=True, flip_masks=flip_masks))


def _hsub_with_no_coupling() -> SimpleNamespace:
    flip_masks = {"alpha2": [], "alpha4": [], "beta2": [], "beta4": []}
    return SimpleNamespace(spmv=SimpleNamespace(is_real=True, flip_masks=flip_masks))


def test_2x2_discovery_cross_check_with_hand_rolled_flip_masks():
    col.reset_for_testing()
    hsub = _hsub_with_singles_coupling()
    n_a, n_b = 4, 4
    alpha_strs = np.arange(n_a, dtype=np.uint64)
    beta_strs = np.arange(n_b, dtype=np.uint64)

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=COMM,
        process_grid=(2, 2),
        comm_backend="host_mpi",
    )

    assert H._sources_2d == H._source_set.reference

    sources = H._sources_2d
    assert len(sources) == 4
    assert set(sources.values()) == {
        TileClass.DIAGONAL,
        TileClass.ALPHA_ONLY,
        TileClass.BETA_ONLY,
        TileClass.MIXED,
    }

    diagonal_src = next(r for r, c in sources.items() if c == TileClass.DIAGONAL)
    assert diagonal_src == RANK


def test_2x2_discovery_no_coupling_returns_only_diagonal():
    col.reset_for_testing()
    hsub = _hsub_with_no_coupling()
    alpha_strs = np.array([0b000_000, 0b000_111, 0b111_000, 0b111_111], dtype=np.uint64)
    beta_strs = alpha_strs.copy()

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=COMM,
        process_grid=(2, 2),
        comm_backend="host_mpi",
    )

    assert H._sources_2d == H._source_set.reference
    assert H._sources_2d == {RANK: TileClass.DIAGONAL}
