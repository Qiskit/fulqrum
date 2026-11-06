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
import scipy.sparse.linalg as spla

from .conftest import gpu_available, make_lih_hsub

pytestmark = [
    pytest.mark.skipif(
        not gpu_available(), reason="GPU build / CUDA device unavailable"
    ),
    pytest.mark.gpu,
]

LIH_GROUND_ENERGY = -7.875652564927877


@pytest.mark.parametrize(
    "gpu_ids", [None, (0,), (0, 0), (0, 0, 0, 0)], ids=["default", "1", "2", "4"]
)
def test_matvec_matches_cpu(gpu_ids):
    from fulqrum.gpu import GPUHamiltonian

    hsub, _, _ = make_lih_hsub()
    op = GPUHamiltonian(hsub, gpu_ids=gpu_ids)
    assert op.gpu_ids == ((0,) if gpu_ids is None else gpu_ids)
    rng = np.random.default_rng(seed=17)
    for _ in range(2):
        x = rng.standard_normal(op.shape[0])
        expected = hsub @ x
        scale = float(np.max(np.abs(expected)))
        np.testing.assert_allclose(
            op.matvec(x), expected, atol=1e-12 * scale, rtol=1e-12
        )
        column = op.matvec(x.reshape(-1, 1))
        assert column.shape == (op.shape[0], 1)
        np.testing.assert_allclose(
            column[:, 0], expected, atol=1e-12 * scale, rtol=1e-12
        )


def test_eigsh_finds_the_ground_state_on_two_ranks():
    from fulqrum.gpu import GPUHamiltonian

    hsub, _, _ = make_lih_hsub()
    op = GPUHamiltonian(hsub, gpu_ids=(0, 0))
    evals, _ = spla.eigsh(op, k=1, which="SA", v0=np.ones(op.shape[0]))
    assert np.isclose(evals[0], LIH_GROUND_ENERGY, atol=1e-8)


def test_runs_without_mpi4py_and_rebuilds_caches(monkeypatch):
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", None)
    monkeypatch.setenv("FQ_ARM_TABLE_SOURCE", "kernel")
    from fulqrum.gpu import GPUHamiltonian

    hsub, _, _ = make_lih_hsub()
    op = GPUHamiltonian(hsub, gpu_ids=(0, 0))
    x = np.random.default_rng(seed=2).standard_normal(op.shape[0])
    first = op.matvec(x)
    steady = op.matvec(x)
    op.reset_cache()
    np.testing.assert_array_equal(op.matvec(x), steady)
    expected = hsub @ x
    np.testing.assert_allclose(
        first, expected, atol=1e-12 * float(np.max(np.abs(expected))), rtol=1e-12
    )
    assert all(timing["caches_built_this_call"] for timing in op.last_timing)
