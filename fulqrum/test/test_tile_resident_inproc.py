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


import sys

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from fulqrum.gpu.distributed import _collective as col  # noqa: E402
from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: E402

from .conftest import make_lih_hsub  # noqa: E402

GRIDS = [(1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (1, 4)]


def inproc_matvecs(hsub, alpha, beta, grid, vectors, reset_before=None, **options):
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    col.reset_for_testing()

    def rank(comm):
        cupy.cuda.Device(0).use()
        op = TileGPUResidentHamiltonian(
            hsub,
            alpha,
            beta,
            gpu_ids=(0,),
            comm=comm,
            process_grid=grid,
            comm_backend="inproc",
            build_threads=1,
            **options,
        )
        outputs = []
        for index, vector in enumerate(vectors):
            if index == reset_before:
                op.reset_cache()
            tile = vector.reshape(op._n_b, op._n_a)[op._b0 : op._b1, op._a0 : op._a1]
            outputs.append(np.asarray(op._matvec(np.ascontiguousarray(tile).ravel())))
        return (
            (op._a0, op._a1, op._b0, op._b1),
            outputs,
            (op.backend_name, op.last_timing["allgatherv_bytes"]),
        )

    results = run_ranks(LocalComm.create(grid[0] * grid[1]), rank)
    n = hsub.shape[0]
    n_a, n_b = len(alpha), len(beta)
    assembled = []
    for index in range(len(vectors)):
        out = np.zeros((n_b, n_a))
        for (a0, a1, b0, b1), outputs, _ in results:
            out[b0:b1, a0:a1] = outputs[index].reshape(b1 - b0, a1 - a0)
        assembled.append(out.reshape(n))
    return (
        assembled,
        {name for _, _, (name, _) in results},
        sum(moved for _, _, (_, moved) in results),
    )


def assert_matches_cpu(hsub, vectors, outputs):
    for vector, output in zip(vectors, outputs):
        expected = hsub @ vector
        scale = float(np.max(np.abs(expected)))
        np.testing.assert_allclose(output, expected, atol=1e-12 * scale, rtol=1e-12)


@pytest.mark.parametrize("source", ["host", "kernel"])
@pytest.mark.parametrize("grid", GRIDS, ids=[f"{p}x{q}" for p, q in GRIDS])
def test_inproc_matvec_matches_cpu(monkeypatch, grid, source):
    monkeypatch.setenv("FQ_ARM_TABLE_SOURCE", source)
    hsub, alpha, beta = make_lih_hsub()
    rng = np.random.default_rng(seed=11)
    vectors = [rng.standard_normal(hsub.shape[0]) for _ in range(2)]
    outputs, backends, moved = inproc_matvecs(hsub, alpha, beta, grid, vectors)
    assert backends == {"inproc"}
    assert (moved > 0) == (grid != (1, 1))
    assert_matches_cpu(hsub, vectors, outputs)


@pytest.mark.parametrize(
    "flags,domain",
    [
        ({"FQ_EXCHANGE": "two_level", "_FULQRUM_TWO_LEVEL_ANY_BACKEND": "1"}, 2),
        ({"FQ_EXCHANGE_DOUBLE_BUFFER": "0"}, None),
    ],
    ids=["two_level", "single_slot"],
)
def test_inproc_exchange_variants_match_cpu(monkeypatch, flags, domain):
    for name, value in flags.items():
        monkeypatch.setenv(name, value)
    hsub, alpha, beta = make_lih_hsub()
    rng = np.random.default_rng(seed=5)
    vectors = [rng.standard_normal(hsub.shape[0]) for _ in range(2)]
    outputs, _, moved = inproc_matvecs(
        hsub, alpha, beta, (2, 2), vectors, intra_node_size=domain
    )
    assert moved > 0
    assert_matches_cpu(hsub, vectors, outputs)


def test_inproc_runs_without_mpi4py(monkeypatch):
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", None)
    monkeypatch.setenv("FQ_ARM_TABLE_SOURCE", "kernel")
    hsub, alpha, beta = make_lih_hsub()
    rng = np.random.default_rng(seed=3)
    x = rng.standard_normal(hsub.shape[0])
    outputs, _, _ = inproc_matvecs(hsub, alpha, beta, (2, 1), [x, x, x], reset_before=2)
    assert_matches_cpu(hsub, [x, x, x], outputs)
    np.testing.assert_array_equal(outputs[2], outputs[1])


def test_inproc_rejects_another_backend():
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    hsub, alpha, beta = make_lih_hsub()
    col.reset_for_testing()

    def rank(comm):
        cupy.cuda.Device(0).use()
        TileGPUResidentHamiltonian(
            hsub,
            alpha,
            beta,
            gpu_ids=(0,),
            comm=comm,
            process_grid=(2, 1),
            comm_backend="host_mpi",
            build_threads=1,
        )

    with pytest.raises(ValueError, match="inproc"):
        run_ranks(LocalComm.create(2, timeout=60.0), rank)
