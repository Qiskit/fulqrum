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

import pytest

import fulqrum as fq
from fulqrum.gpu.distributed.operator_check import (
    block_tier_counts,
    cross_group_values,
    unsupported_groups,
)
from fulqrum.test.test_type2_differential import (
    NORB,
    ORDERINGS,
    bitstring,
    make_operator,
)

ALPHA, BETA = ORDERINGS["block"]
HALVES = [
    bitstring(occupied, NORB) for occupied in itertools.combinations(range(NORB), 2)
]
TIER_NAMES = {
    0: "generic",
    1: "same_sector_single",
    2: "same_sector_double",
    3: "cross_sector_double",
}


def operator(kind):
    qubit_op = make_operator(kind, ALPHA, BETA).extended_jw_transformation()
    return fq.SubspaceHamiltonian(qubit_op, fq.Subspace([HALVES, HALVES])).spmv


@pytest.mark.parametrize(
    "kind", ["aabb_uniform_singles", "same_sector_doubles", "aabb_random"]
)
def test_block_tiers_match_the_cpu_tiers(kind):
    spmv = operator(kind)
    counts = {
        TIER_NAMES[tier]: count for tier, count in block_tier_counts(spmv, NORB).items()
    }
    diagnostics = spmv._cpu_diagnostics()
    assert (
        sum(counts.values())
        == diagnostics["tp_groups"] + diagnostics["residual_groups"]
    )
    assert counts.get("generic", 0) == diagnostics["residual_groups"]


@pytest.mark.parametrize("kind", ["aabb_uniform_singles", "same_sector_doubles"])
def test_supported_operators_have_no_unsupported_groups(kind):
    assert unsupported_groups(operator(kind), NORB) == []


@pytest.mark.parametrize("kind", ["aabb_random", "aabb_partial"])
def test_direction_dependent_cross_values_are_found(kind):
    reasons = {
        reason for _group, _flips, reason in unsupported_groups(operator(kind), NORB)
    }
    assert reasons == {"direction_dependent_cross_value"}


def test_gpu_coefficient_must_match_the_cross_value():
    spmv = operator("aabb_uniform_singles")
    values = cross_group_values(spmv, NORB)
    coefficients = [0.0] * (max(values) + 1)
    for group, group_values in values.items():
        coefficients[group] = abs(group_values[0])
    assert unsupported_groups(spmv, NORB, coefficients) == []
    first = min(values)
    coefficients[first] *= 2.0
    found = unsupported_groups(spmv, NORB, coefficients)
    assert [(group, reason) for group, _flips, reason in found] == [
        (first, "gpu_coefficient_mismatch")
    ]


def test_terms_that_move_electrons_between_halves_have_zero_cross_values():
    values = cross_group_values(operator("sector_transfer"), NORB)
    assert values
    assert all(value == 0 for group_values in values.values() for value in group_values)
