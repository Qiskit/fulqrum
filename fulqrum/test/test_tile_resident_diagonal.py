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

from pathlib import Path

import numpy as np
import pytest

try:
    import cupy
except ImportError:
    cupy = None

from fulqrum.gpu.distributed import _collective as col


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


def _subspace_dense(hsub):
    n = hsub.shape[0]
    return np.column_stack([hsub @ unit for unit in np.eye(n)])


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
@pytest.mark.parametrize("source", ["auto", "kernel"])
@pytest.mark.parametrize("recompute", ["0", "auto"])
def test_diagonal_construction_uploads_operator_tables(monkeypatch, source, recompute):
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    monkeypatch.setenv("FQ_ARM_TABLE_SOURCE", source)
    monkeypatch.setenv("FQ_DIAG_RECOMPUTE", recompute)
    col.reset_for_testing()
    hsub, alpha_strs, beta_strs, _qop = _make_h2_hsub()

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )

    stored = recompute == "0"
    assert H._diag_recompute_enabled is not stored
    must_be_set = [
        "_d_alpha_uniq",
        "_d_beta_uniq",
        "_d_alpha_self_slot",
        "_d_beta_self_slot",
    ] + (
        ["_d_diag_local"] if stored else ["_d_diag_alpha", "_d_diag_beta", "_d_diag_Mb"]
    )
    for name in must_be_set:
        val = getattr(H, name)
        assert val is not None, f"{name} was not populated by the table upload"
    if not stored:
        assert H._d_diag_local is None
    term_tables = [
        "_d_term_header",
        "_d_term_inds_vals",
        "_d_term_proj_inds",
        "_d_term_proj_bits",
        "_d_grp_ladder_ptrs",
        "_d_oper_elems",
        "_d_grp2_ht_keys",
        "_d_grp2_ht_vals",
        "_d_grp2_ht_bitset",
        "_d_grp4_ht_keys",
        "_d_grp4_ht_vals",
        "_d_grp4_ht_bitset",
    ]
    assert H._host_arms == (source == "auto")
    for name in term_tables:
        assert (getattr(H, name) is None) == H._host_arms, (
            f"{name}: host_arms={H._host_arms}"
        )
    bra_tables = (
        "d_alpha2_ptr",
        "d_alpha2_j",
        "d_alpha2_p0",
        "d_alpha2_p1",
        "d_alpha4_ptr",
        "d_alpha4_j",
        "d_alpha4_p0",
        "d_alpha4_p1",
        "d_alpha4_p2",
        "d_alpha4_p3",
        "d_beta2_ptr",
        "d_beta2_j",
        "d_beta2_p0",
        "d_beta2_p1",
        "d_beta4_ptr",
        "d_beta4_j",
        "d_beta4_p0",
        "d_beta4_p1",
        "d_beta4_p2",
        "d_beta4_p3",
    )
    tables = H._tile_tables.sources[H._rank].launch_view
    for name in bra_tables:
        assert tables[name] is not None, f"{name} was not populated by the table upload"

    assert H._num_alpha == 2
    assert H._num_beta == 2
    assert (H._term_inds_stride > 0) == (not H._host_arms)

    np.testing.assert_array_equal(
        cupy.asnumpy(H._d_alpha_self_slot),
        np.arange(H.n_a_local, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        cupy.asnumpy(H._d_beta_self_slot),
        np.arange(H.n_b_local, dtype=np.int32),
    )


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_diagonal_kernel_zero_v_leaves_out_zero():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

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

    v_zero = cupy.zeros(H.n_local, dtype=cupy.float64)
    out = cupy.zeros(H.n_local, dtype=cupy.float64)
    H._launch_diagonal(d_v_local=v_zero, d_out_local=out)
    cupy.cuda.Stream.null.synchronize()

    np.testing.assert_array_equal(
        cupy.asnumpy(out),
        np.zeros(H.n_local, dtype=np.float64),
    )


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_diagonal_kernel_full_v_matches_dense_Hv():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    col.reset_for_testing()
    hsub, alpha_strs, beta_strs, qop = _make_h2_hsub()

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )

    rng = np.random.default_rng(seed=1234)
    n = H.shape[0]
    v_host = rng.standard_normal(n).astype(np.float64)

    n_a = H.n_a_local
    n_b = H.n_b_local
    n_local = H.n_local

    v_kernel_layout = cupy.asarray(v_host)
    out_kernel_layout = cupy.zeros(n_local, dtype=cupy.float64)
    H._launch_diagonal(d_v_local=v_kernel_layout, d_out_local=out_kernel_layout)
    cupy.cuda.Stream.null.synchronize()

    out_host = cupy.asnumpy(out_kernel_layout).ravel()
    assert n_a * n_b == n_local

    M_dense = _subspace_dense(hsub)
    assert M_dense.shape == (n, n)
    np.testing.assert_allclose(
        out_host, np.diag(M_dense) * v_host, atol=1e-10, rtol=1e-10
    )
    full = np.asarray(H._matvec(v_host)).ravel()
    np.testing.assert_allclose(full, (M_dense @ v_host).ravel(), atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_diagonal_construction_4rank_resolves_self_source():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed.dist_matvec import TileClass
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
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

    assert H._sources_2d.get(rank) is TileClass.DIAGONAL, (
        f"rank {rank}: expected self-source DIAGONAL, got {H._sources_2d.get(rank)}"
    )

    assert H._d_alpha_uniq is not None, f"rank {rank}: strings not uploaded"
    tables = H._tile_tables.sources[rank].launch_view
    assert tables["d_alpha2_ptr"] is not None, f"rank {rank}: alpha2 CSR not built"
    assert tables["d_beta2_ptr"] is not None, f"rank {rank}: beta2 CSR not built"
    assert H._d_diag_local is not None, f"rank {rank}: diag_local not uploaded"

    v_zero = cupy.zeros(H.n_local, dtype=cupy.float64)
    out = cupy.zeros(H.n_local, dtype=cupy.float64)
    H._launch_diagonal(d_v_local=v_zero, d_out_local=out)
    cupy.cuda.Stream.null.synchronize()

    np.testing.assert_array_equal(
        cupy.asnumpy(out),
        np.zeros(H.n_local, dtype=np.float64),
        err_msg=f"rank {rank}: zero-v should yield zero out_local",
    )
