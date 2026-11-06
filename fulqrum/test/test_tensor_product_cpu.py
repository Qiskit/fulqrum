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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fulqrum as fq


def half_strings(width, active, electrons):
    values = []
    for occupied in itertools.combinations(active, electrons):
        value = sum(1 << bit for bit in occupied)
        values.append(format(value, f"0{width}b"))
    return sorted(values)


def matrix_reference(op, strings):
    return np.array([[op.matrix_element(a, b) for b in strings] for a in strings])


def check_routes(op, halves, expected_residual=None):
    strings = sorted(b + a for b in halves[1] for a in halves[0])
    reference = matrix_reference(op, strings)
    for route in ("auto", "retained"):
        space = fq.Subspace(
            [halves[0].copy(), halves[1].copy()]
            if route == "auto"
            else [strings.copy()]
        )
        h = fq.SubspaceHamiltonian(op, space)
        diagnostic = h.spmv._cpu_diagnostics()
        if route != "retained":
            assert (
                diagnostic["tp_groups"] + diagnostic["residual_groups"] == h.num_groups
            )
        if route == "auto" and expected_residual is not None:
            assert diagnostic["residual_groups"] == expected_residual
            if expected_residual == 0 and h.num_groups:
                assert diagnostic["tp_ready"]
                assert diagnostic["tp_groups"] == h.num_groups
        x = np.random.default_rng(53).normal(size=2 * len(space)).astype(h.dtype)[::2]
        if h.dtype == np.dtype(complex):
            x += 0.3j
        expected = reference @ x
        np.testing.assert_allclose(h.matvec(x), expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            h.matvec(x[:, None]), expected[:, None], atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            h.diagonal_vector(), reference.diagonal(), atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            h.to_csr_linearoperator().matrix.toarray(),
            reference,
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            h.to_csr_linearoperator_fast().matrix.toarray(),
            reference,
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            h._to_linearoperator() @ x, expected, atol=1e-12, rtol=1e-12
        )
        state = space._storage_state()
        assert state["map_materialized"] == (route == "retained")
        assert state["map_build_count"] == (1 if route == "retained" else 0)


@pytest.mark.parametrize("name", ["h2", "lih", "h2o"])
def test_molecular_product_routes(name):
    op = fq.FermionicOperator.from_json(
        Path(__file__).parent / "data" / f"{name}.json"
    ).extended_jw_transformation()
    width = op.width // 2
    halves = [
        half_strings(width, range(width), 1),
        half_strings(width, range(width), 1),
    ]
    check_routes(op, halves, expected_residual=0)


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("residual_count", [0, 1, 3])
def test_mixed_tp_and_residual_groups(complex_values, residual_count):
    c = 0.2 + 0.15j if complex_values else 0.2
    terms = [("I", [0], -1.0), ("11", [0, 4], 0.5)]
    for ops, inds in [("+-", [0, 1]), ("+-", [4, 5])]:
        terms.extend([(ops, inds, c), ("-+", inds, np.conj(c))])
    extra = [
        ("--++1", [0, 1, 2, 3, 6]),
        ("--++1", [4, 5, 6, 7, 2]),
        ("+-+-1", [0, 2, 4, 6, 1]),
    ]
    for ops, inds in extra[:residual_count]:
        terms.extend(
            [
                (ops, inds, c),
                (
                    ops.replace("+", "p").replace("-", "+").replace("p", "-"),
                    inds,
                    np.conj(c),
                ),
            ]
        )
    op = fq.QubitOperator(8, terms)
    op.set_type(2)
    halves = [half_strings(4, range(4), 2), half_strings(4, range(4), 2)]
    check_routes(op, halves, residual_count)


def test_variable_particle_counts_and_packing_limit_use_residuals():
    op = fq.QubitOperator(6, [("+-", [0, 1], 0.25), ("-+", [0, 1], 0.25)])
    op.set_type(2)
    halves = [["000", "001", "010", "011"], ["000", "001"]]
    check_routes(op, halves, 1)
    width = 300
    op = fq.QubitOperator(2 * width, [("+-", [0, 1], 0.25), ("-+", [0, 1], 0.25)])
    op.set_type(2)
    halves = [half_strings(width, [0, 1], 1), half_strings(width, [0, 1], 1)]
    check_routes(op, halves, 1)


@pytest.mark.parametrize("complex_values", [False, True])
def test_unequal_multiword_product_routes(complex_values):
    c = 0.3 + 0.2j if complex_values else 0.3
    f = fq.FermionicOperator(
        135,
        [
            ("+-", [63, 64], c),
            ("+-", [64, 63], np.conj(c)),
            ("+-", [71, 134], c),
            ("+-", [134, 71], np.conj(c)),
            ("+-+-", [0, 0, 71, 71], 0.6),
        ],
    )
    op = f.extended_jw_transformation()
    halves = [half_strings(70, [0, 63, 64, 69], 2), half_strings(65, [1, 63, 64], 1)]
    check_routes(op, halves, 0)


def test_product_updates_invalidate_engines_and_preserve_sparse_ownership():
    op = fq.FermionicOperator.from_json(
        Path(__file__).parent / "data/h2.json"
    ).extended_jw_transformation()
    half = ["01", "10"]
    h = fq.SubspaceHamiltonian(op, fq.Subspace([half, half]))
    previous = h.to_csr_linearoperator().matrix
    previous_copy = previous.copy()
    h.update_subspace(fq.Subspace([["01"], half]))
    strings = ["0101", "1001"]
    expected = matrix_reference(op, strings)
    np.testing.assert_allclose(
        h.to_csr_linearoperator_fast().matrix.toarray(),
        expected,
        atol=1e-12,
        rtol=1e-12,
    )
    assert h.shape == (2, 2)
    np.testing.assert_array_equal(previous.toarray(), previous_copy.toarray())
    full = fq.Subspace([strings])
    h.update_subspace(full)
    np.testing.assert_allclose(
        h @ np.ones(2), expected @ np.ones(2), atol=1e-12, rtol=1e-12
    )
    assert h.spmv._cpu_diagnostics()["route"] == "retained"
    h.update_subspace(fq.Subspace([half, half]))
    np.testing.assert_allclose(
        h.to_csr_linearoperator().matrix.toarray(),
        previous_copy.toarray(),
        atol=1e-12,
        rtol=1e-12,
    )


def test_memory_failure_does_not_materialize_product(monkeypatch):
    op = fq.QubitOperator(4, [("+-", [0, 1], 0.2), ("-+", [0, 1], 0.2)])
    op.set_type(2)
    space = fq.Subspace([["01", "10"], ["01", "10"]])
    h = fq.SubspaceHamiltonian(op, space)
    with monkeypatch.context() as resource:

        def allocation_failure(*args, **kwargs):
            raise MemoryError("injected allocation failure")

        resource.setattr(np, "empty", allocation_failure)
        with pytest.raises(MemoryError):
            h @ np.ones(4)
    assert not space._storage_state()["map_materialized"]
    h @ np.ones(4)
    assert not space._storage_state()["map_materialized"]


def test_mixed_parallel_rows_against_independent_connected_elements():
    terms = [
        ("I", [0], -0.5),
        ("+-", [0, 1], 0.2),
        ("-+", [0, 1], 0.2),
        ("+-", [8, 9], 0.3),
        ("-+", [8, 9], 0.3),
        ("--++1", [2, 3, 4, 5, 8], 0.4),
        ("++--1", [2, 3, 4, 5, 8], 0.4),
    ]
    op = fq.QubitOperator(16, terms)
    op.set_type(2)
    half = half_strings(8, range(8), 4)
    strings = sorted(b + a for b in half for a in half)
    positions = {int(s, 2): i for i, s in enumerate(strings)}
    x = np.random.default_rng(82).normal(size=len(strings))
    expected = np.zeros(len(strings))
    for row, string in enumerate(strings):
        key = int(string, 2)
        expected[row] = op.matrix_element(string, string).real * x[row]
        for flips in [(0, 1), (8, 9), (2, 3, 4, 5)]:
            target = key ^ sum(1 << bit for bit in flips)
            if target in positions:
                col = positions[target]
                expected[row] += op.matrix_element(string, strings[col]).real * x[col]
    for route in ("auto", "retained"):
        space = fq.Subspace([half, half] if route == "auto" else [strings.copy()])
        h = fq.SubspaceHamiltonian(op, space)
        if route == "auto":
            assert h.spmv._cpu_diagnostics()["residual_groups"] == 1
        np.testing.assert_allclose(h @ x, expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            h.to_csr_linearoperator().matrix @ x, expected, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            h.to_csr_linearoperator_fast().matrix @ x, expected, atol=1e-12, rtol=1e-12
        )


@pytest.mark.parametrize("constant", [0.0, -2.5])
def test_constant_product_outputs_do_not_build_a_map(constant):
    op = fq.QubitOperator.from_label("IIII", constant)
    op.set_type(2)
    h = fq.SubspaceHamiltonian(op, fq.Subspace([["01", "10"], ["01", "10"]]))
    expected = np.eye(4) * constant
    np.testing.assert_allclose(h @ np.arange(4.0), expected @ np.arange(4.0))
    np.testing.assert_allclose(h.to_csr_linearoperator().matrix.toarray(), expected)
    np.testing.assert_allclose(
        h.to_csr_linearoperator_fast().matrix.toarray(), expected
    )


def test_nonhermitian_fast_path_check_uses_residuals():
    op = fq.QubitOperator(4, [("+-", [0, 1], 0.3)])
    op.set_type(2)
    space = fq.Subspace([["01", "10"], ["01", "10"]])
    h = fq.SubspaceHamiltonian(op, space)
    assert h.spmv._cpu_diagnostics()["residual_groups"] == 1


@pytest.mark.parametrize("coupling", [0.0, 0.5e-14, 2.5e-14, 1e-12])
def test_product_cancellation_and_pruning_threshold(coupling):
    op = fq.QubitOperator(
        4,
        [
            ("+-", [0, 1], 1.0),
            ("+-", [0, 1], -1.0 + coupling),
            ("-+", [0, 1], 1.0),
            ("-+", [0, 1], -1.0 + coupling),
        ],
    )
    op.set_type(2)
    halves = [["01", "10"], ["01", "10"]]
    strings = [b + a for b in halves[1] for a in halves[0]]
    expected = matrix_reference(op, strings)
    for route in ("auto", "retained"):
        space = fq.Subspace(halves if route == "auto" else [strings.copy()])
        h = fq.SubspaceHamiltonian(op, space)
        np.testing.assert_allclose(
            h @ np.ones(4), expected @ np.ones(4), atol=3e-14, rtol=1e-12
        )
        csr = h.to_csr_linearoperator().matrix
        np.testing.assert_allclose(csr.toarray(), expected, atol=3e-14, rtol=1e-12)
        if coupling == 0:
            assert csr.nnz == 0
        elif coupling >= 2.5e-14:
            assert csr.nnz == 4


@pytest.mark.parametrize("is_real", [0, 1])
def test_sparse_row_container_overflow_fails_before_allocation(is_real):
    from fulqrum.core.csrlike import CSRLike

    maximum = np.iinfo(np.uintp).max
    with pytest.raises(OverflowError):
        CSRLike(int(maximum), is_real)


def test_native_solver_rejects_basis_size_overflow_before_allocation(monkeypatch):
    pytest.importorskip("fulqrum.gpu.eigensolver._primme_mpi")
    MPI = pytest.importorskip("mpi4py.MPI")
    from fulqrum.gpu.eigensolver.primme_distributed import distributed_primme_eigsh

    maximum = int(np.iinfo(np.int32).max)
    operator = SimpleNamespace(
        shape=(maximum, maximum),
        n_local=maximum,
        _comm=MPI.COMM_SELF,
        _matvec=lambda vector: vector,
    )

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("overflow must be rejected before array allocation")

    monkeypatch.setattr(np, "zeros", unexpected_allocation)
    with pytest.raises(OverflowError):
        distributed_primme_eigsh(operator, k=1, ncv=maximum, gpu=False)
