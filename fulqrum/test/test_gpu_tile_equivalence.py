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


import gc

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
MPI = pytest.importorskip("mpi4py.MPI")

from .conftest import (  # noqa: E402
    FE4S4_CLUSTER_PREFIXES,
    fe4s4_alpha_strings,
    fe4s4_fcidump,
)

RECIPE = {
    "FQ_AABB_CLIQUE_ENABLE": "1",
    "FQ_AABB_REMOTE_BETA_METHOD": "clique",
    "FQ_AABB_REMOTE_ALPHA_METHOD": "transposed_clique",
    "FQ_AA_PAIR_TABLE": "1",
    "FQ_BB_PAIR_TABLE": "1",
}
COMM = MPI.COMM_WORLD
MAX_RANKS = 4

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        COMM.Get_size() > MAX_RANKS,
        reason="each rank holds the full Fe4S4 operator (several GB); run with at most 4 ranks",
    ),
]


@pytest.fixture(scope="module")
def fe4s4_operator(tmp_path_factory):
    from fulqrum.convert import fcidump_to_fq_fermionic_op

    op = fcidump_to_fq_fermionic_op(
        str(fe4s4_fcidump(tmp_path_factory.mktemp("fe4s4")))
    ).extended_jw_transformation()
    return op, fe4s4_alpha_strings()


def release_gpu_memory():
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    COMM.Barrier()


@pytest.fixture(autouse=True)
def released_gpu_memory():
    release_gpu_memory()
    yield


def pick_strings(strings, rule, count):
    if rule == "clusters":
        prefix_a, prefix_b = FE4S4_CLUSTER_PREFIXES
        group_a = [x for x in strings if x.startswith(prefix_a)]
        rest_a = {x[8:] for x in group_a}
        group_b = [
            x for x in strings if x.startswith(prefix_b) and x[8:] not in rest_a
        ]
        return sorted(group_a[: count // 2] + group_b[: count - count // 2])
    return sorted(strings[:count])


def grids_for(size):
    if size == 1:
        return [(1, 1)]
    out = []
    for p_alpha in range(1, size + 1):
        if size % p_alpha == 0:
            out.append((p_alpha, size // p_alpha))
    return out


def gpu_matvec(hsub, half, grid, x):
    from fulqrum.gpu.distributed import _collective
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    _collective.reset_for_testing()
    op = TileGPUResidentHamiltonian(
        hsub,
        list(half),
        list(half),
        comm=COMM,
        process_grid=grid,
        comm_backend="host_mpi",
    )
    tile = x.reshape(op._n_b, op._n_a)[op._b0 : op._b1, op._a0 : op._a1]
    local = op._matvec(np.ascontiguousarray(tile).ravel())
    local = op._matvec(np.ascontiguousarray(tile).ravel())
    pieces = COMM.allgather((op._a0, op._a1, op._b0, op._b1, np.asarray(local)))
    out = np.zeros((op._n_b, op._n_a))
    for a0, a1, b0, b1, data in pieces:
        out[b0:b1, a0:a1] = data.reshape(b1 - b0, a1 - a0)
    sources = COMM.allgather(len(op.source_ranks))
    del op
    release_gpu_memory()
    return out.ravel(), sources


@pytest.mark.parametrize("source", ["host", "kernel"])
@pytest.mark.parametrize("flags", ["default", "recipe"])
@pytest.mark.parametrize("rule,count", [("first", 60), ("clusters", 120)])
def test_tile_matvec_matches_cpu(
    fe4s4_operator, monkeypatch, flags, rule, count, source
):
    import fulqrum as fq

    monkeypatch.setenv("FQ_ARM_TABLE_SOURCE", source)
    if flags == "recipe":
        for key, value in RECIPE.items():
            monkeypatch.setenv(key, value)
    op, strings = fe4s4_operator
    half = pick_strings(strings, rule, count)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([list(half), list(half)]))
    x = np.random.RandomState(17).uniform(-1.0, 1.0, hsub.shape[0])
    expected = hsub @ x
    for grid in grids_for(COMM.Get_size()):
        result, sources = gpu_matvec(hsub, half, grid, x)
        error = np.linalg.norm(result - expected) / np.linalg.norm(expected)
        assert error < 1e-12, (grid, flags, rule, source, error)
        if rule == "clusters" and grid[0] >= 4:
            assert min(sources) < COMM.Get_size(), (grid, sources)
    del hsub
    gc.collect()


@pytest.mark.parametrize(
    "source,record",
    [("auto", "on"), ("host", "on"), ("kernel", "FQ_ARM_TABLE_SOURCE=kernel")],
)
def test_arm_table_source_is_recorded(fe4s4_operator, monkeypatch, source, record):
    import fulqrum as fq
    from fulqrum.gpu.distributed import _collective
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    monkeypatch.setenv("FQ_ARM_TABLE_SOURCE", source)
    op, strings = fe4s4_operator
    half = pick_strings(strings, "first", 60)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([list(half), list(half)]))
    _collective.reset_for_testing()
    gpu = TileGPUResidentHamiltonian(
        hsub,
        list(half),
        list(half),
        comm=COMM,
        process_grid=(COMM.Get_size(), 1),
        comm_backend="host_mpi",
    )
    assert gpu.construction_timing["host_arm_values"] == record
    del gpu
    release_gpu_memory()


def test_explicit_host_source_with_an_unmet_prerequisite_raises(
    fe4s4_operator, monkeypatch
):
    import fulqrum as fq
    from fulqrum.gpu.distributed import _collective
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    monkeypatch.setenv("FQ_ARM_TABLE_SOURCE", "host")
    monkeypatch.setenv("FQ_AA_PAIR_TABLE", "0")
    op, strings = fe4s4_operator
    half = pick_strings(strings, "first", 60)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([list(half), list(half)]))
    _collective.reset_for_testing()
    with pytest.raises(ValueError, match="FQ_ARM_TABLE_SOURCE=host cannot run"):
        TileGPUResidentHamiltonian(
            hsub,
            list(half),
            list(half),
            comm=COMM,
            process_grid=(COMM.Get_size(), 1),
            comm_backend="host_mpi",
        )


def test_arm_plans_follow_the_source_class(fe4s4_operator):
    import fulqrum as fq
    from fulqrum.gpu.distributed import _collective
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    op, strings = fe4s4_operator
    half = pick_strings(strings, "first", 120)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([list(half), list(half)]))
    expected = {
        (True, True): "clique",
        (True, False): "clique",
        (False, True): "transposed_clique",
        (False, False): "baseline",
    }
    for grid in grids_for(COMM.Get_size()):
        _collective.reset_for_testing()
        gpu = TileGPUResidentHamiltonian(
            hsub,
            list(half),
            list(half),
            comm=COMM,
            process_grid=grid,
            comm_backend="host_mpi",
        )
        wrong = {
            source: plan.aabb_path
            for source, plan in gpu._arm_plans.items()
            if plan.aabb_path != expected[gpu._same_coordinates(source)]
        }
        assert not any(COMM.allgather(wrong)), (grid, COMM.allgather(wrong))
        del gpu
        release_gpu_memory()


def cross_formula(hsub, bra, source, x):
    import scipy.sparse as sp

    from fulqrum.gpu.distributed.tile_tables import cross_sector_table, sector_tiles

    a0, a1, b0, b1 = bra
    sa0, sa1, sb0, sb1 = source
    alpha = sector_tiles(hsub.spmv, 0, a0, a1, [sa0, sa1])[0]
    beta = sector_tiles(hsub.spmv, 1, b0, b1, [sb0, sb1])[0]
    table = cross_sector_table(hsub.spmv)
    alpha_rows = np.repeat(
        np.arange(a1 - a0), np.diff(np.asarray(alpha["single_offsets"], dtype=np.int64))
    )
    beta_rows = np.repeat(
        np.arange(b1 - b0), np.diff(np.asarray(beta["single_offsets"], dtype=np.int64))
    )
    alpha_index = 2 * np.asarray(alpha["directed_pairs"], dtype=np.int64) + (
        np.asarray(alpha["jw_signs"]) < 0
    )
    beta_pairs = np.asarray(beta["directed_pairs"], dtype=np.int64)
    out = np.zeros((b1 - b0, a1 - a0))
    for pair in np.unique(beta_pairs):
        chosen = beta_pairs == pair
        rows = sp.csr_matrix(
            (
                np.asarray(beta["jw_signs"], dtype=float)[chosen],
                (
                    beta_rows[chosen],
                    np.asarray(beta["columns"], dtype=np.int64)[chosen],
                ),
            ),
            shape=(b1 - b0, sb1 - sb0),
        )
        columns = sp.csr_matrix(
            (
                table[pair, alpha_index],
                (alpha_rows, np.asarray(alpha["columns"], dtype=np.int64)),
            ),
            shape=(a1 - a0, sa1 - sa0),
        )
        out += rows @ (columns @ x.T).T
    return out


@pytest.mark.parametrize("source", ["host", "kernel"])
@pytest.mark.parametrize("degree_order", ["1", "0"])
def test_clique_arm_matches_the_cross_formula(
    fe4s4_operator, monkeypatch, source, degree_order
):
    import fulqrum as fq
    from fulqrum.gpu.distributed import _collective
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    monkeypatch.setenv("FQ_ARM_TABLE_SOURCE", source)
    monkeypatch.setenv("FQ_AABB_CLIQUE_RESIDUAL_DEGREE_SORT", degree_order)
    op, strings = fe4s4_operator
    half = pick_strings(strings, "first", 120)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([list(half), list(half)]))
    rng = np.random.default_rng(17 + COMM.Get_rank())
    seen = set()
    for grid in grids_for(COMM.Get_size()):
        _collective.reset_for_testing()
        gpu = TileGPUResidentHamiltonian(
            hsub,
            list(half),
            list(half),
            comm=COMM,
            process_grid=grid,
            comm_backend="host_mpi",
        )
        bra = gpu._rank_tiles[gpu._rank][:4]
        errors = []
        for src_rank, plan in sorted(gpu._arm_plans.items()):
            if plan.aabb_path not in ("clique", "transposed_clique"):
                continue
            seen.add(plan.aabb_path + ("_remote_beta" if plan.remote_beta else ""))
            if plan.aabb_path == "transposed_clique":
                residual = gpu._aabb_remote_alpha_plans[src_rank]["device"]
                sell = gpu._tile_tables.sources[src_rank].launch_view[
                    "d_alpha2_sell_pack"
                ]
                assert (residual.get("residual_order") is not None) == (
                    degree_order == "1"
                )
                assert (residual["residual_pack"] is sell) == (degree_order == "0")
            tile = gpu._rank_tiles[src_rank][:4]
            x = rng.standard_normal((tile[3] - tile[2], tile[1] - tile[0]))
            out = cupy.zeros((bra[3] - bra[2]) * (bra[1] - bra[0]))
            gpu._launch_cross_clique_arm(
                src_rank, cupy.asarray(x.ravel()), out, tile[1] - tile[0]
            )
            expected = cross_formula(hsub, bra, tile, x)
            got = cupy.asnumpy(out).reshape(expected.shape)
            error = np.abs(got - expected).max() / np.abs(expected).max()
            if not (np.abs(expected).max() > 0 and error < 1e-13):
                errors.append((grid, src_rank, plan.aabb_path, float(error)))
        assert not any(COMM.allgather(errors)), COMM.allgather(errors)
        del gpu
        release_gpu_memory()
    seen = set().union(*COMM.allgather(seen))
    assert "clique" in seen
    if COMM.Get_size() == 4:
        assert {"clique", "clique_remote_beta", "transposed_clique"} <= seen, seen


def test_ring_skips_peers_that_are_not_sources(fe4s4_operator):
    import fulqrum as fq
    from fulqrum.gpu.distributed import _collective
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    if COMM.Get_size() < 4:
        pytest.skip("needs at least 4 ranks so that some ring peers have no coupling")
    op, strings = fe4s4_operator
    half = pick_strings(strings, "clusters", 120)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([list(half), list(half)]))
    _collective.reset_for_testing()
    gpu = TileGPUResidentHamiltonian(
        hsub,
        list(half),
        list(half),
        comm=COMM,
        process_grid=(COMM.Get_size(), 1),
        comm_backend="host_mpi",
    )
    launches = []
    original = gpu._launch_source

    def counting_launch(src_rank, **kwargs):
        launches.append(src_rank)
        return original(src_rank, **kwargs)

    gpu._launch_source = counting_launch
    x = np.random.RandomState(17).uniform(-1.0, 1.0, hsub.shape[0])
    tile = x.reshape(gpu._n_b, gpu._n_a)[gpu._b0 : gpu._b1, gpu._a0 : gpu._a1]
    gpu._matvec(np.ascontiguousarray(tile).ravel())
    outside = sorted(set(launches) - set(gpu.source_ranks))
    assert all(COMM.allgather(not outside)), outside
    tables_outside = sorted(
        set(gpu._tile_tables.sources) - set(gpu.source_ranks) - {COMM.Get_rank()}
    )
    assert all(COMM.allgather(not tables_outside)), tables_outside
    ring_peers = {
        step.source for step in (*gpu._alpha_ring_steps, *gpu._beta_ring_steps)
    }
    assert any(COMM.allgather(bool(ring_peers - set(gpu.source_ranks))))
    remote_sources = set(gpu.source_ranks) - {COMM.Get_rank()}
    assert all(COMM.allgather(remote_sources <= set(launches)))


@pytest.mark.parametrize("domain_size", [2, 3])
def test_two_level_exchange_matches_cpu(fe4s4_operator, monkeypatch, domain_size):
    import fulqrum as fq
    from fulqrum.gpu.distributed import _collective
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    if COMM.Get_size() != 4:
        pytest.skip("needs 4 ranks: two shared-memory domains")
    monkeypatch.setenv("FQ_EXCHANGE", "two_level")
    monkeypatch.setenv("_FULQRUM_TWO_LEVEL_ANY_BACKEND", "1")
    op, strings = fe4s4_operator
    half = pick_strings(strings, "clusters", 120)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([list(half), list(half)]))
    x = np.random.RandomState(17).uniform(-1.0, 1.0, hsub.shape[0])
    expected = hsub @ x
    for grid in grids_for(COMM.Get_size()):
        _collective.reset_for_testing()
        gpu = TileGPUResidentHamiltonian(
            hsub,
            list(half),
            list(half),
            comm=COMM,
            process_grid=grid,
            comm_backend="host_mpi",
            intra_node_size=domain_size,
        )
        assert gpu.construction_timing["exchange"] == "two_level"
        tile = x.reshape(gpu._n_b, gpu._n_a)[gpu._b0 : gpu._b1, gpu._a0 : gpu._a1]
        gpu._matvec(np.ascontiguousarray(tile).ravel())
        local = gpu._matvec(np.ascontiguousarray(tile).ravel())
        pieces = COMM.allgather((gpu._a0, gpu._a1, gpu._b0, gpu._b1, np.asarray(local)))
        result = np.zeros((gpu._n_b, gpu._n_a))
        for a0, a1, b0, b1, data in pieces:
            result[b0:b1, a0:a1] = data.reshape(b1 - b0, a1 - a0)
        error = np.linalg.norm(result.ravel() - expected) / np.linalg.norm(expected)
        assert error < 1e-12, (grid, error)
        del gpu
        release_gpu_memory()
