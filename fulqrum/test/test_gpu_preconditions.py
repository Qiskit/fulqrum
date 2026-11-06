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


import itertools

import numpy as np
import pytest

import fulqrum as fq  # noqa: E402
from fulqrum.test.test_type2_differential import (
    NORB,
    ORDERINGS,
    bitstring,
    make_operator,
)  # noqa: E402

cupy = pytest.importorskip("cupy")
MPI = pytest.importorskip("mpi4py.MPI")


pytestmark = pytest.mark.gpu

ALPHA, BETA = ORDERINGS["block"]
EQUAL_WEIGHTS = [
    bitstring(occupied, NORB) for occupied in itertools.combinations(range(NORB), 2)
]
MIXED_WEIGHTS = [
    bitstring(occupied, NORB)
    for occupied in [(0,), (1,), (0, 1), (0, 2), (1, 3), (0, 1, 2)]
]
SINGLES = [
    ("+-", [ALPHA[1], ALPHA[0]], 0.2),
    ("+-", [ALPHA[0], ALPHA[1]], 0.2),
    ("+-", [BETA[1], BETA[0]], 0.2),
    ("+-", [BETA[0], BETA[1]], 0.2),
]
SPIN_FLIPS = [("+-", [ALPHA[i], BETA[i]], 0.3) for i in range(NORB)] + [
    ("+-", [BETA[i], ALPHA[i]], 0.3) for i in range(NORB)
]


def tile_operator(fermionic_op, halves):
    from fulqrum.gpu.distributed import _collective
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    qubit_op = fermionic_op.extended_jw_transformation()
    hamiltonian = fq.SubspaceHamiltonian(
        qubit_op, fq.Subspace([list(halves), list(halves)])
    )
    _collective.reset_for_testing()
    gpu = TileGPUResidentHamiltonian(
        hamiltonian,
        sorted(halves),
        sorted(halves),
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )
    return hamiltonian, gpu


@pytest.mark.parametrize(
    "name,fermionic_op,halves",
    [
        (
            "aabb_uniform_singles",
            make_operator("aabb_uniform_singles", ALPHA, BETA),
            EQUAL_WEIGHTS,
        ),
        (
            "same_sector_doubles",
            make_operator("same_sector_doubles", ALPHA, BETA),
            EQUAL_WEIGHTS,
        ),
    ],
)
def test_supported_operator_matches_cpu(name, fermionic_op, halves):
    hamiltonian, gpu = tile_operator(fermionic_op, halves)
    x = np.random.RandomState(3).uniform(-1.0, 1.0, hamiltonian.shape[0])
    expected = hamiltonian @ x
    result = gpu._matvec(x)
    result = gpu._matvec(x)
    assert np.linalg.norm(result - expected) <= 1e-12 * np.linalg.norm(expected), name


@pytest.mark.parametrize(
    "name,fermionic_op,halves",
    [
        (
            "mixed_electron_counts",
            make_operator("aabb_uniform_singles", ALPHA, BETA),
            MIXED_WEIGHTS,
        ),
        (
            "spin_flip_groups",
            fq.FermionicOperator(2 * NORB, SPIN_FLIPS + SINGLES),
            MIXED_WEIGHTS,
        ),
        (
            "spin_flip_groups_equal_counts",
            fq.FermionicOperator(2 * NORB, SPIN_FLIPS + SINGLES),
            EQUAL_WEIGHTS,
        ),
        (
            "aabb_direction_dependent",
            make_operator("aabb_random", ALPHA, BETA),
            EQUAL_WEIGHTS,
        ),
        (
            "aabb_not_number_conserving_per_sector",
            make_operator("sector_transfer", ALPHA, BETA),
            EQUAL_WEIGHTS,
        ),
        (
            "aabb_partial_directions",
            make_operator("aabb_partial", ALPHA, BETA),
            EQUAL_WEIGHTS,
        ),
    ],
)
def test_unsupported_input_raises(name, fermionic_op, halves):
    with pytest.raises(ValueError):
        tile_operator(fermionic_op, halves)


@pytest.mark.parametrize("explicit", [False, True])
def test_mixed_electron_counts_raise_for_every_arm(monkeypatch, explicit):
    if explicit:
        monkeypatch.setenv("FQ_AABB_CLIQUE_ENABLE", "1")
    with pytest.raises(ValueError, match="one electron count"):
        tile_operator(make_operator("aabb_uniform_singles", ALPHA, BETA), MIXED_WEIGHTS)
