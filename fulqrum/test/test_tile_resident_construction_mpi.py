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
        f"test_tile_resident_construction_mpi requires `mpirun -n 4` "
        f"(got COMM_WORLD size={MPI.COMM_WORLD.Get_size()})",
        allow_module_level=True,
    )

from fulqrum.gpu.distributed import _collective as col
from fulqrum.gpu.distributed.dist_matvec import TileClass
from fulqrum.gpu.distributed.dist_matvec_tile_resident import TileGPUResidentHamiltonian

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


def _make_tiny_hsub() -> SimpleNamespace:
    return SimpleNamespace(spmv=SimpleNamespace(is_real=True))


def test_alpha_beta_subcomms_partition_world_2x2():
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
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

    assert H.process_grid == (2, 2)
    assert H.coord_alpha == RANK // 2
    assert H.coord_beta == RANK % 2

    assert H._alpha_comm.Get_size() == 2
    assert H._beta_comm.Get_size() == 2

    assert H._alpha_rank == H.coord_beta
    assert H._beta_rank == H.coord_alpha

    alpha_peers = H._alpha_comm.allgather(H.coord_alpha)
    assert all(ca == H.coord_alpha for ca in alpha_peers)

    beta_peers = H._beta_comm.allgather(H.coord_beta)
    assert all(cb == H.coord_beta for cb in beta_peers)


def test_uneven_split_pins_global_peer_max():
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    n_a, n_b = 5, 6
    H = TileGPUResidentHamiltonian(
        hsub,
        np.arange(n_a, dtype=np.uint64),
        np.arange(n_b, dtype=np.uint64),
        comm=COMM,
        process_grid=(2, 2),
        comm_backend="host_mpi",
    )

    assert H._alpha_ranges == [(0, 3), (3, 5)]
    assert H._beta_ranges == [(0, 3), (3, 6)]
    assert H._n_a_peer_max == 3
    assert H._n_b_peer_max == 3
    expected_n_a_local = 3 if H.coord_alpha == 0 else 2
    expected_n_b_local = 3
    assert H.n_a_local == expected_n_a_local
    assert H.n_b_local == expected_n_b_local


def test_mixed_neighbors_at_2x2_with_singles_coupling():
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    n_a, n_b = 4, 4
    H = TileGPUResidentHamiltonian(
        hsub,
        np.arange(n_a, dtype=np.uint64),
        np.arange(n_b, dtype=np.uint64),
        comm=COMM,
        process_grid=(2, 2),
        comm_backend="host_mpi",
    )

    expected_neighbor = 3 - RANK
    assert H._mixed_sources == [expected_neighbor]
    assert H._mixed_destinations == [expected_neighbor]
    assert H._mixed_indegree == 1
    assert H._mixed_outdegree == 1


def test_mixed_neighbors_empty_at_2x2_with_no_coupling():
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
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

    assert H._mixed_sources == []
    assert H._mixed_destinations == []
    assert H._mixed_indegree == 0
    assert H._mixed_outdegree == 0


def test_cross_check_passes_with_empty_flip_masks_at_2x2():
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    n_a, n_b = 4, 4
    H = TileGPUResidentHamiltonian(
        hsub,
        np.arange(n_a, dtype=np.uint64),
        np.arange(n_b, dtype=np.uint64),
        comm=COMM,
        process_grid=(2, 2),
        comm_backend="host_mpi",
    )

    assert H._sources_2d == H._source_set.reference
    classes_seen = set(H._sources_2d.values())
    assert classes_seen == {
        TileClass.DIAGONAL,
        TileClass.ALPHA_ONLY,
        TileClass.BETA_ONLY,
        TileClass.MIXED,
    }


def _construct_2x2():
    """Construct the 2x2 operator on the tiny subspace with the host_mpi backend."""
    col.reset_for_testing()
    return TileGPUResidentHamiltonian(
        _make_tiny_hsub(),
        np.arange(4, dtype=np.uint64),
        np.arange(4, dtype=np.uint64),
        comm=COMM,
        process_grid=(2, 2),
        comm_backend="host_mpi",
    )


def test_explicit_two_level_without_nccl_raises_on_every_rank(monkeypatch):
    """FQ_EXCHANGE=two_level on host_mpi raises ValueError on every rank, with no hang."""
    monkeypatch.setenv("FQ_EXCHANGE", "two_level")
    with pytest.raises(ValueError, match="NCCL"):
        _construct_2x2()
    COMM.Barrier()


def test_exchange_mode_differing_across_ranks_raises_on_every_rank(monkeypatch):
    """FQ_EXCHANGE set on one rank only raises ValueError on every rank, with no hang."""
    if RANK == 1:
        monkeypatch.setenv("FQ_EXCHANGE", "ring")
    else:
        monkeypatch.delenv("FQ_EXCHANGE", raising=False)
    with pytest.raises(ValueError, match="differs across ranks"):
        _construct_2x2()
    COMM.Barrier()


def test_auto_exchange_on_host_mpi_runs_ring_and_records_it(monkeypatch):
    """FQ_EXCHANGE=auto on host_mpi runs ring and records the reason and the settings digest."""
    monkeypatch.delenv("FQ_EXCHANGE", raising=False)
    H = _construct_2x2()
    timing = H.construction_timing
    assert timing["exchange"] == "ring"
    assert "NCCL" in timing["exchange_reason"]
    assert len(set(COMM.allgather(timing["exchange_digest"]))) == 1


def test_verify_hbm_fits_failing_on_one_rank_raises_on_every_rank(monkeypatch):
    """verify_hbm_fits that fails on one rank only raises RuntimeError on every rank."""
    H = _construct_2x2()
    monkeypatch.setattr(H, "_hbm_fit_error", lambda: "too small" if RANK == 2 else None)
    with pytest.raises(RuntimeError, match="rank 2: too small"):
        H.verify_hbm_fits()
    COMM.Barrier()


@pytest.mark.parametrize("double_buffer", ["1", "0"])
def test_exchange_steps_match_the_ring_and_window_plans(monkeypatch, double_buffer):
    """The ring steps visit every other member of the row or column in order, and the window steps follow the hops."""
    from fulqrum.gpu.distributed.exchange_plan import FORWARD, SEND_LOCAL

    monkeypatch.setenv("FQ_EXCHANGE_DOUBLE_BUFFER", double_buffer)
    H = _construct_2x2()
    for steps, members, index in (
        (H._beta_ring_steps, list(H._alpha_comm_world_ranks), H._alpha_rank),
        (H._alpha_ring_steps, list(H._beta_comm_world_ranks), H._beta_rank),
    ):
        size = len(members)
        peers = [members[(index + k) % size] for k in range(1, size)]
        send_peers = [members[(index - k) % size] for k in range(1, size)]
        left, right = members[(index - 1) % size], members[(index + 1) % size]
        assert [step.source for step in steps] == peers
        if double_buffer == "1":
            assert all(
                (s.transfer, s.send_to, s.receive_from) == (FORWARD, left, right)
                for s in steps
            )
        else:
            assert all(s.transfer == SEND_LOCAL for s in steps)
            assert [step.send_to for step in steps] == send_peers
    assert [(step.send_to, step.receive_from) for step in H._mixed_steps] == list(
        H._mixed_window_hops
    )
