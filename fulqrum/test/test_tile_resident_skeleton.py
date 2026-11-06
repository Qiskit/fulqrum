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

from fulqrum.gpu.distributed import _collective as col
from fulqrum.gpu.distributed.dist_matvec import TileClass
from fulqrum.gpu.distributed.dist_matvec_tile_resident import TileGPUResidentHamiltonian


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False
    return True


def _make_tiny_hsub() -> SimpleNamespace:
    return SimpleNamespace(spmv=SimpleNamespace(is_real=True))


def test_skeleton_instantiates_on_comm_self():
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    n_a, n_b = 3, 4
    alpha_strs = np.arange(n_a, dtype=np.uint64)
    beta_strs = np.arange(n_b, dtype=np.uint64)

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )

    n_global = n_a * n_b
    assert H.shape == (n_global, n_global)
    assert H.dtype == np.float64
    assert H.process_grid == (1, 1)
    assert H.coord_alpha == 0
    assert H.coord_beta == 0
    assert H.comm is MPI.COMM_SELF
    assert H.backend_name == "host_mpi"

    assert H._alpha_comm.Get_size() == 1
    assert H._beta_comm.Get_size() == 1
    assert H._alpha_rank == 0
    assert H._beta_rank == 0

    assert H.n_a_local == n_a
    assert H.n_b_local == n_b
    assert H.n_local == n_a * n_b
    assert H._n_a_peer_max == n_a
    assert H._n_b_peer_max == n_b
    assert H._a0 == 0 and H._a1 == n_a
    assert H._b0 == 0 and H._b1 == n_b

    assert H._d_mixed_slots is None

    if not _has_cupy():
        assert H._cupy_available is False
        assert H._d_v_local is None
        assert H._d_out_local is None
        assert H._d_alpha_ring is None
        assert H._d_beta_ring is None

    assert H._source_set.reference == {0: TileClass.DIAGONAL}
    assert H._sources_2d == {0: TileClass.DIAGONAL}
    assert H._source_set.reference is not None


def test_predicted_hbm_bytes_shape():
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    n_a, n_b = 3, 4
    alpha_strs = np.arange(n_a, dtype=np.uint64)
    beta_strs = np.arange(n_b, dtype=np.uint64)

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )

    pred = H.predicted_hbm_bytes
    assert set(pred.keys()) == {
        "v_local",
        "out_local",
        "alpha_ring_total",
        "beta_ring_total",
        "hierarchical_stage",
        "hierarchical_gather_total",
        "exchange_pool",
        "operator",
        "scratch",
        "mixed_recv",
        "arm_caches",
        "aabb_csr",
        "aabb_parent_extra",
        "aabb_remote_beta_metadata",
    }
    assert pred["v_local"] == 12 * 8
    assert pred["out_local"] == 12 * 8
    assert pred["alpha_ring_total"] == 0
    assert pred["beta_ring_total"] == 0
    assert pred["exchange_pool"] == 0
    assert pred["operator"] == 0
    assert pred["aabb_csr"] == 0  # tiny mock hsub has no aabb groups


def test_verify_hbm_fits_returns_none_when_cupy_missing():
    if _has_cupy():
        pytest.skip("Needs proper distributed cluster.")
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    H = TileGPUResidentHamiltonian(
        hsub,
        np.arange(3, dtype=np.uint64),
        np.arange(4, dtype=np.uint64),
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )
    assert H._cupy_available is False
    assert H.verify_hbm_fits() is None


def test_uneven_alpha_split_pins_global_peer_max_at_p_alpha_1():
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    n_a, n_b = 5, 7
    H = TileGPUResidentHamiltonian(
        hsub,
        np.arange(n_a, dtype=np.uint64),
        np.arange(n_b, dtype=np.uint64),
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )
    assert H._alpha_ranges == [(0, 5)]
    assert H._beta_ranges == [(0, 7)]
    assert H._n_a_peer_max == 5
    assert H._n_b_peer_max == 7
    assert H.n_local == 35


@pytest.mark.skipif(not _has_cupy(), reason="raise-on-overrun branch needs cupy")
def test_verify_hbm_fits_raises_on_overrun(monkeypatch):
    import cupy

    monkeypatch.setattr(
        cupy.cuda.runtime,
        "memGetInfo",
        lambda: (1024, 1024),
    )
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    n_a, n_b = 32, 32
    with pytest.raises(RuntimeError, match=r"predicted.*free_bytes"):
        TileGPUResidentHamiltonian(
            hsub,
            np.arange(n_a, dtype=np.uint64),
            np.arange(n_b, dtype=np.uint64),
            comm=MPI.COMM_SELF,
            comm_backend="host_mpi",
        )


def test_mixed_neighbors_empty_at_comm_self():
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    n_a, n_b = 3, 4
    H = TileGPUResidentHamiltonian(
        hsub,
        np.arange(n_a, dtype=np.uint64),
        np.arange(n_b, dtype=np.uint64),
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )

    assert H._mixed_sources == []
    assert H._mixed_destinations == []
    assert H._mixed_indegree == 0
    assert H._mixed_outdegree == 0


def test_skeleton_matvec_raises_without_cupy():
    if _has_cupy():
        pytest.skip("test pins the cupy-missing branch; covered by CCC tests")
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    n_a, n_b = 2, 2
    alpha_strs = np.arange(n_a, dtype=np.uint64)
    beta_strs = np.arange(n_b, dtype=np.uint64)

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )

    v = np.zeros(n_a * n_b, dtype=np.float64)
    with pytest.raises(NotImplementedError, match="cupy and a CUDA GPU"):
        H._matvec(v)


def test_matvec_raises_when_cupy_unavailable():
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    n_a, n_b = 2, 2
    alpha_strs = np.arange(n_a, dtype=np.uint64)
    beta_strs = np.arange(n_b, dtype=np.uint64)

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )

    v = np.zeros(H.n_local, dtype=np.float64)
    if not _has_cupy():
        with pytest.raises(NotImplementedError, match=r"_matvec requires cupy"):
            H._matvec(v)
    else:
        with pytest.raises(NotImplementedError, match=r"operator tables were uploaded"):
            H._matvec(v)


def test_matvec_input_size_mismatch_raises():
    col.reset_for_testing()
    hsub = _make_tiny_hsub()
    n_a, n_b = 3, 4
    alpha_strs = np.arange(n_a, dtype=np.uint64)
    beta_strs = np.arange(n_b, dtype=np.uint64)

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )

    if not _has_cupy():
        pytest.skip("cupy unavailable; the no-cupy branch shadows the size check")

    bad = np.zeros(H.n_local + 1, dtype=np.float64)
    with pytest.raises(ValueError, match=r"expected local vector length"):
        H._matvec(bad)
