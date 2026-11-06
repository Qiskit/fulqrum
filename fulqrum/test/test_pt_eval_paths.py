# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2024.
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


@pytest.mark.parametrize("width", [65, 140])
@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("complex_values", [False, True])
def test_cached_masks_match_reference_after_subspace_update(
    width, kind, complex_values
):
    positions = [0, 2, 63, 64, width - 1]
    positions = sorted(set(positions))
    p, q = positions[:2]
    r, s = positions[-2:]
    coefficient = 0.2 + 0.3j if complex_values else 0.2
    terms = [
        ("I", [0], -1.2),
        ("0Z", [63, width - 1], 0.125),
        ("11", [2, 64], -0.75),
        ("+-", [p, q], coefficient),
        ("-+", [p, q], np.conj(coefficient)),
        ("+-1", [r, s, 2], coefficient),
        ("-+1", [r, s, 2], np.conj(coefficient)),
        ("--++", [p, q, r, s], 0.5),
        ("++--", [p, q, r, s], 0.5),
    ]
    if kind == 1:
        terms.extend([("YY", [63, 64], 0.3), ("XX", [63, 64], -0.3)])
        if complex_values:
            terms.append(("Y", [width - 1], 0.4))
    op = fq.QubitOperator(width, terms)
    op.set_type(kind)
    strings = []
    for pattern in itertools.product("01", repeat=len(positions)):
        value = ["0"] * width
        for pos, bit in zip(positions, pattern):
            value[width - 1 - pos] = bit
        strings.append("".join(value))
    subspace = fq.Subspace([strings[::-1]])
    h = fq.SubspaceHamiltonian(op, subspace)
    snapshots = []
    for selected in [strings[::-1], strings[::2]]:
        subspace = fq.Subspace([selected])
        h.update_subspace(subspace)
        reference = np.array(
            [[op.matrix_element(a, b) for b in selected] for a in selected]
        )
        np.testing.assert_allclose(
            reference, reference.conj().T, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            h.diagonal_vector(), reference.diagonal(), atol=1e-12, rtol=1e-12
        )
        vector = (
            np.random.default_rng(62)
            .normal(size=2 * len(selected))
            .astype(h.dtype)[::2]
        )
        if complex_values:
            vector += 0.4j
        expected = reference @ vector
        np.testing.assert_allclose(h.matvec(vector), expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            h.matvec(vector[:, None]), expected[:, None], atol=1e-12, rtol=1e-12
        )
        csr = h.to_csr_linearoperator().matrix
        np.testing.assert_allclose(csr.toarray(), reference, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            h.to_csr_linearoperator_fast().matrix.toarray(),
            reference,
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            h._to_linearoperator() @ vector, expected, atol=1e-12, rtol=1e-12
        )
        snapshots.append((csr, reference.copy()))
    for csr, reference in snapshots:
        np.testing.assert_allclose(csr.toarray(), reference, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("constant", [0.0, -2.5])
def test_empty_term_masks_preserve_constant_and_zero_outputs(kind, constant):
    op = fq.QubitOperator.from_label("II", constant)
    op.set_type(kind)
    h = fq.SubspaceHamiltonian(op, fq.Subspace([["11", "00", "10", "01"]]))
    expected = np.eye(4) * constant
    np.testing.assert_allclose(h @ np.arange(4.0), expected @ np.arange(4.0))
    np.testing.assert_allclose(h.to_csr_linearoperator().matrix.toarray(), expected)
    np.testing.assert_allclose(
        h.to_csr_linearoperator_fast().matrix.toarray(), expected
    )


def test_diagonal_only_masks_follow_fast_term_sort():
    width = 12
    rng = np.random.default_rng(51)
    terms = []
    for p in range(width):
        for q in range(p, width):
            terms.append(
                ("1" if p == q else "11", [p] if p == q else [p, q], rng.normal())
            )
    rng.shuffle(terms)
    terms.append(("I", [0], -3.25))
    op = fq.QubitOperator(width, terms)
    op.set_type(2)
    strings = [
        format(int(i), f"0{width}b") for i in rng.choice(1 << width, 40, replace=False)
    ]
    subspace = fq.Subspace([strings])
    expected = np.array([op.matrix_element(s, s) for s in strings])
    fast = fq.SubspaceHamiltonian(op, subspace).diagonal_vector()
    general = fq.SubspaceHamiltonian(op, subspace).diagonal_vector(
        disable_fast_mode=True
    )
    np.testing.assert_allclose(fast, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(general, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("tol,expected_size", [(0.1, 2), (0.12, 1)])
def test_ramps_masks_preserve_general_diagonal_candidate_thresholds(
    restricted, tol, expected_size
):
    from fulqrum.ramps import ramps_open, ramps_restricted_simple

    op = fq.QubitOperator(
        2, [("1", [0], 0.5), ("1", [1], 1.2), ("+-", [0, 1], 0.2), ("-+", [0, 1], 0.2)]
    )
    op.set_type(2)
    space = fq.Subspace([["01", "10"]])
    h = fq.SubspaceHamiltonian(op, space)
    assert not h.spmv.fast_diag_compatible()
    target = fq.Subspace([["01"]])
    energy = op.matrix_element("01", "01").real
    if restricted:
        result = ramps_restricted_simple(
            h, target, energy, space, max_recursion=1, tol=tol
        )
    else:
        result = ramps_open(h, target, energy, max_recursion=1, tol=tol)
    assert result.size() == expected_size
