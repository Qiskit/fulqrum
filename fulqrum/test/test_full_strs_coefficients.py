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

"""Coefficient tables of the full-string GPU path against the operator terms."""

import lzma
from pathlib import Path

import numpy as np
import pytest

import fulqrum as fq
from fulqrum.convert.integrals import (
    fcidump_to_fq_fermionic_op,
    integrals_to_fq_fermionic_op,
)
from fulqrum.gpu.full_strs.host import HostOperator

FE4S4 = Path(__file__).parent / "data" / "fe4s4"
MOVES = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]


def random_operator(norb, seed, active=None):
    """Return the qubit operator of random 8-fold symmetric integrals in block layout."""
    rng = np.random.default_rng(seed)
    orbitals = np.arange(norb) if active is None else np.asarray(active)
    m = orbitals.size
    h_active = rng.normal(size=(m, m))
    h_active = 0.5 * (h_active + h_active.T)
    v_active = 0.1 * rng.normal(size=(m, m, m, m))
    v_active = v_active + v_active.transpose(1, 0, 2, 3)
    v_active = v_active + v_active.transpose(0, 1, 3, 2)
    v_active = v_active + v_active.transpose(2, 3, 0, 1)
    h = np.zeros((norb, norb))
    v = np.zeros((norb, norb, norb, norb))
    h[np.ix_(orbitals, orbitals)] = h_active
    v[np.ix_(orbitals, orbitals, orbitals, orbitals)] = v_active
    return integrals_to_fq_fermionic_op(h, v, constant=0.5).extended_jw_transformation()


def random_strings(rng, allowed, count, rows):
    """Return rows random strings with count electrons on the allowed orbitals.

    Returns a uint64 array.
    """
    allowed = np.asarray(allowed)
    keys = np.argsort(rng.random((rows, allowed.size)), axis=1)[:, :count]
    out = np.zeros(rows, dtype=np.uint64)
    for k in range(count):
        out |= np.left_shift(np.uint64(1), allowed[keys[:, k]].astype(np.uint64))
    return out


def excite(rng, strings, allowed, level):
    """Return each string with level electrons moved to empty orbitals among the allowed ones."""
    out = strings.copy()
    if level == 0:
        return out
    allowed = np.asarray(allowed, dtype=np.uint64)
    for row in range(strings.size):
        bits = (strings[row] >> allowed) & np.uint64(1)
        occupied = allowed[bits == 1]
        empty = allowed[bits == 0]
        if occupied.size < level or empty.size < level:
            continue
        flips = np.concatenate(
            [
                rng.choice(occupied, level, replace=False),
                rng.choice(empty, level, replace=False),
            ]
        )
        out[row] = strings[row] ^ np.bitwise_or.reduce(
            np.left_shift(np.uint64(1), flips)
        )
    return out


def compare_moves(host, keys, allowed, rng, rows):
    """Return the largest relative difference of table and term elements over random moves."""
    picks = keys[
        rng.choice(keys.shape[0], size=min(rows, keys.shape[0]), replace=False)
    ]
    worst = 0.0
    nonzero = []
    for alpha_level, beta_level in MOVES:
        sources = np.stack(
            [
                excite(rng, picks[:, 0], allowed, alpha_level),
                excite(rng, picks[:, 1], allowed, beta_level),
            ],
            axis=1,
        )
        from_tables = host.table_elements(picks, sources)
        from_terms = host.term_elements(picks, sources)
        scale = np.maximum(1.0, np.abs(from_terms))
        worst = max(worst, float(np.max(np.abs(from_tables - from_terms) / scale)))
        nonzero.append(int(np.count_nonzero(from_terms)))
    return worst, nonzero


def test_tables_equal_terms_on_small_operators():
    rng = np.random.default_rng(3)
    for norb, num_alpha, num_beta, seed in [(10, 5, 4, 1), (12, 3, 6, 2)]:
        op = random_operator(norb, seed)
        keys = np.stack(
            [
                random_strings(rng, np.arange(norb), num_alpha, 400),
                random_strings(rng, np.arange(norb), num_beta, 400),
            ],
            axis=1,
        )
        host = HostOperator(op, keys)
        assert host.electrons == (num_alpha, num_beta)
        worst, nonzero = compare_moves(
            host, host.sorted_keys(), np.arange(norb), rng, 300
        )
        assert worst <= 1e-14
        assert min(nonzero) > 100


def test_diagonal_equals_the_subspace_hamiltonian():
    rng = np.random.default_rng(4)
    norb = 8
    op = random_operator(norb, 5)
    keys = np.stack(
        [
            random_strings(rng, np.arange(norb), 4, 200),
            random_strings(rng, np.arange(norb), 3, 200),
        ],
        axis=1,
    )
    host = HostOperator(op, keys)
    rows = host.sorted_keys()
    labels = []
    for alpha, beta in rows:
        bits = int(alpha) | (int(beta) << norb)
        labels.append(format(bits, f"0{2 * norb}b"))
    csr = fq.SubspaceHamiltonian(op, fq.Subspace([labels])).to_csr_linearoperator()
    matrix = csr.matrix if hasattr(csr, "matrix") else csr
    expected_diagonal = np.asarray(matrix.diagonal()).real
    np.testing.assert_allclose(
        np.sort(host.diagonal()), np.sort(expected_diagonal), rtol=0, atol=1e-12
    )


def test_word_boundary_orbitals():
    rng = np.random.default_rng(6)
    norb = 64
    active = [0, 1, 17, 30, 31, 32, 33, 47, 60, 61, 62, 63]
    rest = [k for k in range(norb) if k not in active]
    op = random_operator(norb, 7, active)
    alpha = random_strings(rng, active, 4, 300) | random_strings(rng, rest, 3, 300)
    beta = random_strings(rng, active, 5, 300) | random_strings(rng, rest, 2, 300)
    host = HostOperator(op, np.stack([alpha, beta], axis=1))
    worst, nonzero = compare_moves(host, host.sorted_keys(), active, rng, 300)
    assert worst <= 1e-14
    assert min(nonzero) > 50


def test_projector_on_a_spectator_raises_with_the_group():
    op = random_operator(6, 8)
    extra = fq.QubitOperator(
        12, [("++--1", [0, 1, 2, 3, 5], 0.3), ("--++1", [0, 1, 2, 3, 5], 0.3)]
    )
    extra.set_type(2)
    op += extra
    rng = np.random.default_rng(9)
    keys = np.stack(
        [
            random_strings(rng, np.arange(6), 3, 20),
            random_strings(rng, np.arange(6), 3, 20),
        ],
        axis=1,
    )
    with pytest.raises(
        ValueError, match=r"Operator group \d+ \(flips alpha \{0,1,2,3\}"
    ):
        HostOperator(op, keys)


@pytest.mark.long
@pytest.mark.parametrize("norb, num_alpha, num_beta", [(36, 27, 27), (40, 12, 10)])
def test_tables_equal_terms_on_random_dense_operators(norb, num_alpha, num_beta):
    rng = np.random.default_rng(norb)
    op = random_operator(norb, norb)
    keys = np.stack(
        [
            random_strings(rng, np.arange(norb), num_alpha, 2000),
            random_strings(rng, np.arange(norb), num_beta, 2000),
        ],
        axis=1,
    )
    host = HostOperator(op, keys)
    worst, nonzero = compare_moves(host, host.sorted_keys(), np.arange(norb), rng, 1000)
    assert worst <= 1e-14
    assert min(nonzero) > 500


@pytest.fixture(scope="module")
def fe4s4_operator(tmp_path_factory):
    """The Fe4S4 (36 orbitals, 27 + 27 electrons) qubit operator from the bundled compressed FCIDUMP."""
    path = tmp_path_factory.mktemp("fe4s4") / "fcidump_Fe4S4_MO.txt"
    path.write_bytes(lzma.decompress((FE4S4 / "fcidump_Fe4S4_MO.txt.xz").read_bytes()))
    return fcidump_to_fq_fermionic_op(str(path)).extended_jw_transformation()


@pytest.mark.parametrize("sample", ["head", "uniform", "random"])
def test_tables_equal_terms_on_fe4s4(fe4s4_operator, sample):
    """1000 determinants of the Fe4S4 1e9 Cartesian subspace (most important block, evenly spaced, random):
    the coefficient tables equal the operator terms for every excitation level."""
    with lzma.open(FE4S4 / f"fe4s4_1000_{sample}.txt.xz", "rt") as handle:
        text = np.array(handle.read().split(), dtype="U72")
    beta = np.array([int(s[:36], 2) for s in text], dtype=np.uint64)
    alpha = np.array([int(s[36:], 2) for s in text], dtype=np.uint64)
    host = HostOperator(fe4s4_operator, np.stack([alpha, beta], axis=1))
    assert host.electrons == (27, 27)
    rng = np.random.default_rng(10)
    worst, nonzero = compare_moves(host, host.sorted_keys(), np.arange(36), rng, 1000)
    assert worst <= 1e-14
    assert min(nonzero) > 500
