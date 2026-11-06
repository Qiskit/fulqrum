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

import fulqrum as fq
from fulqrum.gpu.distributed.hash_utils import _pack_half_strs_to_uint64
from fulqrum.gpu.distributed.tile_tables import (
    cross_sector_table,
    cross_table,
    sector_pairs,
    sector_tiles,
    tile_input_error,
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
KINDS = ["same_sector_doubles", "aabb_uniform_singles", "aabb_random", "mixed"]


def hamiltonian(kind):
    qubit_op = make_operator(kind, ALPHA, BETA).extended_jw_transformation()
    return fq.SubspaceHamiltonian(qubit_op, fq.Subspace([HALVES, HALVES]))


def gpu_strings():
    alpha, beta = _pack_half_strs_to_uint64(HALVES, HALVES)
    return np.sort(alpha), np.sort(beta)


def ranges(count, parts):
    return [count * part // parts for part in range(parts + 1)]


def sector_elements(spmv, sector, num_strings, cross, parts):
    elements = {}
    bounds = ranges(num_strings[sector], parts)
    for bra_begin, bra_end in zip(bounds[:-1], bounds[1:]):
        tiles = sector_tiles(spmv, sector, bra_begin, bra_end, bounds)
        for block, tile in enumerate(tiles):
            singles, doubles = tile["single_offsets"], tile["double_offsets"]
            for row in range(bra_end - bra_begin):
                for edge in range(singles[row], singles[row + 1]):
                    column = bounds[block] + int(tile["columns"][edge])
                    pair = int(tile["directed_pairs"][edge])
                    for other in range(num_strings[1 - sector]):
                        value = (
                            tile["single_values"][edge]
                            + tile["own_signs"][edge] * cross[other, pair]
                        )
                        elements[(bra_begin + row, column, other)] = value
                for edge in range(doubles[row], doubles[row + 1]):
                    column = bounds[block] + int(tile["double_columns"][edge])
                    for other in range(num_strings[1 - sector]):
                        key = (bra_begin + row, column, other)
                        elements[key] = (
                            elements.get(key, 0.0) + tile["double_values"][edge]
                        )
    return elements


def assembled_offdiagonal(spmv, num_strings, parts):
    n_a, n_b = num_strings
    matrix = np.zeros((n_a * n_b, n_a * n_b))
    for sector in range(2):
        cross = cross_table(spmv, sector, 0, num_strings[1 - sector])
        for (row, column, other), value in sector_elements(
            spmv, sector, num_strings, cross, parts
        ).items():
            if sector == 0:
                matrix[other * n_a + row, other * n_a + column] += value
            else:
                matrix[row * n_a + other, column * n_a + other] += value
    table = cross_sector_table(spmv)
    alpha = sector_tiles(spmv, 0, 0, n_a, [0, n_a])[0]
    beta = sector_tiles(spmv, 1, 0, n_b, [0, n_b])[0]
    for b in range(n_b):
        for kb in range(beta["single_offsets"][b], beta["single_offsets"][b + 1]):
            b2, beta_pair = int(beta["columns"][kb]), int(beta["directed_pairs"][kb])
            for a in range(n_a):
                for ka in range(
                    alpha["single_offsets"][a], alpha["single_offsets"][a + 1]
                ):
                    a2, alpha_pair = (
                        int(alpha["columns"][ka]),
                        int(alpha["directed_pairs"][ka]),
                    )
                    negative = int(alpha["jw_signs"][ka] < 0)
                    value = (
                        beta["jw_signs"][kb]
                        * table[beta_pair, 2 * alpha_pair + negative]
                    )
                    matrix[b * n_a + a, b2 * n_a + a2] += value
    return matrix


@pytest.mark.parametrize("kind", KINDS)
def test_tiles_give_the_off_diagonal_of_the_operator(kind):
    hsub = hamiltonian(kind)
    alpha, beta = gpu_strings()
    assert tile_input_error(hsub.spmv, NORB, alpha, beta) is None
    dense = hsub.spmv.to_csr_array().toarray()
    expected = dense - np.diag(np.diag(dense))
    assembled = assembled_offdiagonal(hsub.spmv, (alpha.size, beta.size), 2)
    assert np.abs(expected).max() > 0
    np.testing.assert_allclose(assembled, expected, atol=1e-13)


def test_ranged_cross_table_is_a_slice_of_the_full_table():
    spmv = hamiltonian("mixed").spmv
    num_pairs = sector_pairs(spmv)
    for sector in range(2):
        full = cross_table(spmv, sector, 0, len(HALVES))
        assert full.shape == (len(HALVES), 2 * num_pairs[sector])
        np.testing.assert_array_equal(
            cross_table(spmv, sector, 2, 5, threads=1), full[2:5]
        )


def test_columns_outside_the_blocks_are_dropped():
    spmv = hamiltonian("mixed").spmv
    full = sector_tiles(spmv, 0, 0, len(HALVES), [0, len(HALVES)])[0]
    part = sector_tiles(spmv, 0, 0, len(HALVES), [1, 4])[0]
    kept = sum(1 <= column < 4 for column in full["columns"])
    assert part["columns"].size == kept
    assert part["columns"].max() < 3


@pytest.mark.parametrize(
    "change,reason",
    [
        ("split", "splits at qubit"),
        ("strings", "differ from the GPU strings"),
        ("count", "strings in the subspace"),
    ],
)
def test_tile_inputs_are_checked(change, reason):
    spmv = hamiltonian("mixed").spmv
    alpha, beta = gpu_strings()
    num_alpha = NORB
    if change == "split":
        num_alpha = NORB - 1
    elif change == "strings":
        alpha = alpha[::-1].copy()
    else:
        beta = beta[:-1]
    assert reason in tile_input_error(spmv, num_alpha, alpha, beta)


def test_complex_operators_are_refused():
    qubit_op = make_operator("complex", ALPHA, BETA).extended_jw_transformation()
    spmv = fq.SubspaceHamiltonian(qubit_op, fq.Subspace([HALVES, HALVES])).spmv
    alpha, beta = gpu_strings()
    assert "complex" in tile_input_error(spmv, NORB, alpha, beta)


@pytest.mark.parametrize(
    "kind", ["aabb_uniform_singles", "aabb_uniform", "same_sector_doubles"]
)
def test_aabb_coefficients_match_the_term_table_builder(kind):
    pytest.importorskip("fulqrum.gpu.common.operator_arrays")
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import _operator_tables
    from fulqrum.gpu.distributed.tables import aabb_coefficients
    from fulqrum.gpu.distributed.term_tables import (
        _build_aabb_g_dense,
        _build_direct_integral_tables,
    )

    spmv = hamiltonian(kind).spmv
    data = _operator_tables(spmv)
    inds = np.ascontiguousarray(data["grp_offdiag_inds"])
    sizes = np.ascontiguousarray(data["grp_offdiag_sizes"])
    _, direct = _build_direct_integral_tables(
        data, inds, sizes, int(data["num_groups"]), NORB
    )
    expected = _build_aabb_g_dense(inds, sizes, direct, NORB, NORB)
    assert kind == "same_sector_doubles" or np.abs(expected).max() > 0
    np.testing.assert_array_equal(
        aabb_coefficients(cross_sector_table(spmv), NORB, NORB), expected
    )
