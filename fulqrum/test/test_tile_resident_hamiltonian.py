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

from pathlib import Path

import numpy as np
import pytest

try:
    import cupy
except ImportError:
    cupy = None


def _has_cupy() -> bool:
    return cupy is not None


_H2_PATH = Path(__file__).parent / "data/h2.json"


def _make_h2_hsub():
    from fulqrum import FermionicOperator, Subspace, SubspaceHamiltonian

    fop = FermionicOperator.from_json(_H2_PATH)
    qop = fop.extended_jw_transformation()

    alpha_strs = ["01", "10"]
    beta_strs = ["01", "10"]
    sub = Subspace([alpha_strs, beta_strs])
    hsub = SubspaceHamiltonian(qop, sub)
    return hsub, alpha_strs, beta_strs, qop


def test_export_tile_resident_class():
    import fulqrum.gpu as fg

    assert hasattr(fg, "TileGPUResidentHamiltonian")
    assert hasattr(fg, "DistributedMPIGPUHamiltonian")

    cls = fg.TileGPUResidentHamiltonian
    factory = fg.DistributedMPIGPUHamiltonian
    assert callable(cls)
    assert callable(factory)


def test_factory_dispatch_rejects_unknown_layout():
    pytest.importorskip("mpi4py")
    from fulqrum.gpu.distributed.dist_matvec import DistributedMPIGPUHamiltonian

    with pytest.raises(ValueError, match=r"layout="):
        DistributedMPIGPUHamiltonian(hsub=None, layout="tile_residentX")

    with pytest.raises(ValueError, match=r"layout="):
        DistributedMPIGPUHamiltonian(hsub=None, layout="2d")


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_factory_dispatch_routes_to_tile_resident():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed import _collective as col
    from fulqrum.gpu.distributed.dist_matvec import DistributedMPIGPUHamiltonian
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    col.reset_for_testing()
    hsub, _alpha_strs, _beta_strs, _qop = _make_h2_hsub()

    H_explicit = DistributedMPIGPUHamiltonian(
        hsub,
        layout="tile_resident",
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )
    assert isinstance(H_explicit, TileGPUResidentHamiltonian)


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_construction_timing_populated():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed import _collective as col
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    col.reset_for_testing()
    hsub, alpha_strs, beta_strs, _qop = _make_h2_hsub()

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )
    ct = H.construction_timing

    for key in (
        "process_grid",
        "n_a_local",
        "n_b_local",
        "n_local",
        "local_tile_bytes",
        "backend",
        "source_discovery",
        "mixed_neighbor_setup",
        "gpu_upload",
        "hbm_check",
        "source_tiles",
        "tile_class_counts",
        "predicted_hbm_bytes",
        "operator_bytes",
        "mixed_indegree",
        "mixed_outdegree",
    ):
        assert key in ct, f"construction_timing missing key {key!r}"

    assert ct["process_grid"] == (1, 1)
    assert ct["n_local"] == ct["n_a_local"] * ct["n_b_local"]
    assert ct["local_tile_bytes"] == ct["n_local"] * 8
    assert ct["source_tiles"] >= 1
    assert ct["tile_class_counts"]["DIAGONAL"] == 1
    assert ct["source_discovery"] >= 0
    assert ct["gpu_upload"] >= 0
    assert ct["predicted_hbm_bytes"] > 0


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_diag_local_and_source_ranks_exposed():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed import _collective as col
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    col.reset_for_testing()
    hsub, alpha_strs, beta_strs, _qop = _make_h2_hsub()
    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )

    assert H._is_primary is True
    assert H._primme_comm is MPI.COMM_SELF
    assert H._comm is MPI.COMM_SELF

    diag = H.diag_local
    assert diag.shape == (H.n_local,)
    assert diag.dtype == np.float64
    assert np.all(np.isfinite(diag))

    assert 0 in H.source_ranks
    assert H.assigned_source_ranks == H.source_ranks

    assert H.run_workers() is None
    assert H.stop_workers() is None


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_last_timing_populated_after_matvec():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed import _collective as col
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    col.reset_for_testing()
    hsub, alpha_strs, beta_strs, _qop = _make_h2_hsub()
    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )

    rng = np.random.default_rng(seed=2026)
    v = rng.standard_normal(H.n_local).astype(np.float64)
    _ = H @ v

    lt = H.last_timing
    for key in (
        "h2d",
        "exchange",
        "compute",
        "sync",
        "d2h",
        "source_tiles",
        "kernel_launches",
        "wall",
        "h2d_bytes",
        "d2h_bytes",
        "allgatherv_bytes",
    ):
        assert key in lt, f"last_timing missing key {key!r}"

    assert lt["wall"] > 0
    assert lt["source_tiles"] >= 1
    assert lt["kernel_launches"] >= 1
    assert lt["h2d_bytes"] == H.n_local * 8
    assert lt["d2h_bytes"] == H.n_local * 8


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_backend_equivalence_host_mpi_vs_cuda_mpi():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed import _collective as col
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()
    if nranks != 4:
        pytest.skip(f"test requires 4 ranks, got {nranks}")

    col.reset_for_testing()
    if not col._probe_cuda_aware_mpi(comm):
        pytest.skip("cuda-aware MPI not available on this cluster")

    hsub, alpha_strs, beta_strs, _qop = _make_h2_hsub()

    rng = np.random.default_rng(seed=42)
    v_global = rng.standard_normal(16).astype(np.float64)

    out_per_backend = {}
    for backend in ("host_mpi", "cuda_mpi"):
        col.reset_for_testing()
        H = TileGPUResidentHamiltonian(
            hsub,
            alpha_strs,
            beta_strs,
            process_grid=(2, 2),
            comm=comm,
            comm_backend=backend,
        )
        a0, a1 = H._alpha_ranges[H._coord_alpha]
        b0, b1 = H._beta_ranges[H._coord_beta]
        v_2d = v_global.reshape(4, 4)
        v_local = np.ascontiguousarray(v_2d[b0:b1, a0:a1]).reshape(-1)
        out_per_backend[backend] = H._matvec(v_local)

    np.testing.assert_allclose(
        out_per_backend["host_mpi"],
        out_per_backend["cuda_mpi"],
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_construction_memory_smoke_4rank():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed import _collective as col
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()
    if nranks != 4:
        pytest.skip(f"test requires 4 ranks, got {nranks}")

    col.reset_for_testing()
    hsub, alpha_strs, beta_strs, _qop = _make_h2_hsub()
    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        process_grid=(2, 2),
        comm=comm,
        comm_backend="host_mpi",
    )

    hbm = H.predicted_hbm_bytes
    for key in (
        "v_local",
        "out_local",
        "alpha_ring_total",
        "beta_ring_total",
        "operator",
        "scratch",
        "mixed_recv",
    ):
        assert key in hbm, f"predicted_hbm_bytes missing key {key!r}"
        assert isinstance(hbm[key], int)
        assert hbm[key] >= 0

    total = sum(hbm.values())
    assert total > 0
    assert total < (2 << 30)

    ct = H.construction_timing
    for key in ("source_discovery", "mixed_neighbor_setup", "gpu_upload", "hbm_check"):
        v = ct[key]
        assert isinstance(v, (int, float))
        assert 0 <= v < 60
