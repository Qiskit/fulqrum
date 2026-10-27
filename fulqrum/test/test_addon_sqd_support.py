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

"""Tests for the SQD addon support module."""

import numpy as np
import pyscf
import pyscf.ao2mo
import pyscf.mcscf
import pytest
import scipy.sparse.linalg

import fulqrum as fq
from fulqrum.addon_sqd_support import sci_solver_from_integrals
from fulqrum.convert.integrals import integrals_to_fq_fermionic_op

qiskit_addon_sqd = pytest.importorskip("qiskit_addon_sqd")


@pytest.fixture(scope="module")
def h4():
    """One- and two-body integrals of a stretched H4 chain."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[
            ["H", (0, 0, 0)],
            ["H", (0.9, 0, 0)],
            ["H", (1.8, 0, 0)],
            ["H", (2.7, 0, 0)],
        ],
        basis="sto-6g",
        verbose=0,
    )
    scf = pyscf.scf.RHF(mol).run()
    norb = mol.nao_nr()
    nelec = (2, 2)
    cas = pyscf.mcscf.CASCI(scf, norb, nelec)
    mo = cas.sort_mo(range(norb), base=0)
    hcore, _ = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)
    return hcore, eri, norb, nelec


# Deliberately asymmetric, so that a transposed result cannot pass by coincidence.
STRINGS_A = np.array([0b0011, 0b0101, 0b1001])
STRINGS_B = np.array([0b0011, 0b0110, 0b1010, 0b1100])


def test_matches_the_addon_default_solver(h4):
    """The energy, amplitudes and occupancies agree with the addon's PySCF solver."""
    from qiskit_addon_sqd.fermion import solve_sci

    hcore, eri, norb, nelec = h4
    expected = solve_sci((STRINGS_A, STRINGS_B), hcore, eri, norb=norb, nelec=nelec)

    solver = sci_solver_from_integrals(hcore, eri, tol=1e-12)
    (result,) = solver([(STRINGS_A, STRINGS_B)], hcore, eri, norb, nelec)

    assert result.energy == pytest.approx(expected.energy, abs=1e-9)
    assert result.sci_state.amplitudes.shape == (len(STRINGS_A), len(STRINGS_B))
    np.testing.assert_array_equal(result.sci_state.ci_strs_a, np.sort(STRINGS_A))
    np.testing.assert_array_equal(result.sci_state.ci_strs_b, np.sort(STRINGS_B))
    # The overall sign of an eigenvector is arbitrary, so compare magnitudes. PySCF's
    # own eigenvector is converged only to about 1e-7 here, which sets the tolerance.
    np.testing.assert_allclose(
        np.abs(result.sci_state.amplitudes),
        np.abs(expected.sci_state.amplitudes),
        atol=1e-6,
    )
    for spin in (0, 1):
        np.testing.assert_allclose(
            result.orbital_occupancies[spin],
            expected.orbital_occupancies[spin],
            atol=1e-6,
        )


def test_amplitudes_are_indexed_by_alpha_then_beta(h4):
    """Amplitude ``(i, j)`` holds the configuration ``(ci_strs_a[i], ci_strs_b[j])``.

    Fulqrum orders a subspace with the beta half string major, which is the transpose
    of what the addon expects. Comparing the whole matrix against another solver
    cannot catch a transposed unpacking, because a wrong index map can still produce
    an amplitude of the right magnitude somewhere. So read each amplitude back using
    the subspace's own bitstring lookup, which is independent of how the eigenvector
    was reshaped.
    """
    hcore, eri, norb, nelec = h4
    solver = sci_solver_from_integrals(hcore, eri, tol=1e-12)
    (result,) = solver([(STRINGS_A, STRINGS_B)], hcore, eri, norb, nelec)

    strings_a = np.sort(STRINGS_A)
    strings_b = np.sort(STRINGS_B)
    subspace = fq.Subspace(
        [
            [format(string, f"0{norb}b") for string in strings_a],
            [format(string, f"0{norb}b") for string in strings_b],
        ]
    )
    # Recover the flat eigenvector in the subspace's own order, then check that every
    # amplitude sits at the index its pair of CI strings names.
    flat = np.empty(len(subspace))
    for i, string_a in enumerate(strings_a):
        for j, string_b in enumerate(strings_b):
            full = format(string_b, f"0{norb}b") + format(string_a, f"0{norb}b")
            flat[subspace.get_bitstring_index(full)] = result.sci_state.amplitudes[i, j]

    # A correct unpacking reproduces a normalized eigenvector of the projected
    # Hamiltonian; a transposed one scrambles it.
    assert np.linalg.norm(flat) == pytest.approx(1.0, abs=1e-8)
    hamiltonian = fq.SubspaceHamiltonian(
        integrals_to_fq_fermionic_op(
            one_body_integrals=hcore, two_body_integrals=eri
        ).extended_jw_transformation(),
        subspace,
    )
    np.testing.assert_allclose(hamiltonian @ flat, result.energy * flat, atol=1e-7)


def test_matrix_free_agrees_with_the_sparse_matrix(h4):
    """Applying the Hamiltonian without building its matrix gives the same state."""
    hcore, eri, norb, nelec = h4
    args = ([(STRINGS_A, STRINGS_B)], hcore, eri, norb, nelec)
    (sparse,) = sci_solver_from_integrals(hcore, eri, tol=1e-12)(*args)
    (free,) = sci_solver_from_integrals(hcore, eri, matrix_free=True, tol=1e-12)(*args)

    assert free.energy == pytest.approx(sparse.energy, abs=1e-10)
    np.testing.assert_allclose(
        np.abs(free.sci_state.amplitudes),
        np.abs(sparse.sci_state.amplitudes),
        atol=1e-10,
    )


def test_unsorted_ci_strings_are_rejected(h4):
    """Unsorted CI strings raise rather than being silently reordered.

    The SQD addon passes them sorted, so this is a tripwire on that guarantee. It
    raises instead of sorting because sorting would reorder the arrays that the
    result reports back as its CI strings.
    """
    hcore, eri, norb, nelec = h4
    solver = sci_solver_from_integrals(hcore, eri, tol=1e-12)

    for strings_a, strings_b in [
        (STRINGS_A[::-1], np.sort(STRINGS_B)),
        (np.sort(STRINGS_A), STRINGS_B[::-1]),
    ]:
        with pytest.raises(ValueError, match="not sorted"):
            solver([(strings_a, strings_b)], hcore, eri, norb, nelec)


def test_mismatched_integrals_are_rejected(h4):
    """Integrals that differ from the ones the solver was built from raise.

    The operator is built once, from the integrals given to the factory, so the ones
    the addon passes on each call are ignored. Diagonalizing the wrong Hamiltonian
    would return a plausible energy, so the mismatch has to be caught.
    """
    hcore, eri, norb, nelec = h4
    solver = sci_solver_from_integrals(hcore, eri, tol=1e-12)
    subspaces = [(np.sort(STRINGS_A), np.sort(STRINGS_B))]

    with pytest.raises(ValueError, match="one-body integrals.*differ"):
        solver(subspaces, hcore * 1.5, eri, norb, nelec)

    with pytest.raises(ValueError, match="two-body integrals.*differ"):
        solver(subspaces, hcore, eri * 1.5, norb, nelec)

    with pytest.raises(ValueError, match="one-body integrals.*differ"):
        solver(subspaces, hcore[:-1, :-1], eri, norb, nelec)


def test_equal_copies_of_the_integrals_are_accepted(h4):
    """A distinct but equal array is accepted, since the check compares values.

    Under MPI the loop broadcasts its arguments, so a rank can legitimately hold a
    copy rather than the object the factory was given.
    """
    hcore, eri, norb, nelec = h4
    solver = sci_solver_from_integrals(hcore, eri, tol=1e-12)
    (result,) = solver(
        [(np.sort(STRINGS_A), np.sort(STRINGS_B))],
        hcore.copy(),
        eri.copy(),
        norb,
        nelec,
    )
    assert np.isfinite(result.energy)


def test_a_batch_yields_one_result_per_subspace(h4):
    """Every subspace in a call gets its own result, in order."""
    hcore, eri, norb, nelec = h4
    subspaces = [
        (STRINGS_A, STRINGS_B),
        (np.array([0b0011, 0b0101]), np.array([0b0011, 0b1100])),
    ]
    results = sci_solver_from_integrals(hcore, eri, tol=1e-12)(
        subspaces, hcore, eri, norb, nelec
    )

    assert len(results) == len(subspaces)
    for (strings_a, strings_b), result in zip(subspaces, results):
        assert result.sci_state.amplitudes.shape == (len(strings_a), len(strings_b))


def test_custom_eigensolver_is_used(h4):
    """The eigensolver argument replaces the default, and is given the subspace size."""
    hcore, eri, norb, nelec = h4
    calls = []

    def spy(linear_operator, **kwargs):
        calls.append(linear_operator.shape)
        # A custom solver receives the column-vector v0, so flatten it for SciPy.
        return scipy.sparse.linalg.eigsh(
            linear_operator, **{**kwargs, "v0": kwargs["v0"][:, 0]}
        )

    solver = sci_solver_from_integrals(hcore, eri, eigensolver=spy, tol=1e-12)
    solver([(STRINGS_A, STRINGS_B)], hcore, eri, norb, nelec)

    dim = len(STRINGS_A) * len(STRINGS_B)
    assert calls == [(dim, dim)]


def test_v0_is_passed_as_a_column_vector(h4):
    """``v0`` holds one starting vector per column, the general convention."""
    hcore, eri, norb, nelec = h4
    shapes = []

    def spy(linear_operator, **kwargs):
        shapes.append(kwargs["v0"].shape)
        # Flatten, because this spy stands in for a block-taking solver but
        # delegates to one that does not.
        return scipy.sparse.linalg.eigsh(
            linear_operator, **{**kwargs, "v0": kwargs["v0"][:, 0]}
        )

    solver = sci_solver_from_integrals(hcore, eri, eigensolver=spy, tol=1e-12)
    solver([(STRINGS_A, STRINGS_B)], hcore, eri, norb, nelec)

    assert shapes == [(len(STRINGS_A) * len(STRINGS_B), 1)]


def test_scipy_eigsh_is_adapted_automatically(h4):
    """Passing scipy's eigsh explicitly works, and matches the default.

    SciPy raises on a two-dimensional ``v0`` once the starting vector is close to
    the answer, so without the automatic adaptation this would fail.
    """
    hcore, eri, norb, nelec = h4
    args = ([(STRINGS_A, STRINGS_B)], hcore, eri, norb, nelec)
    (default,) = sci_solver_from_integrals(hcore, eri, tol=1e-12)(*args)
    (explicit,) = sci_solver_from_integrals(
        hcore, eri, eigensolver=scipy.sparse.linalg.eigsh, tol=1e-12
    )(*args)

    assert explicit.energy == pytest.approx(default.energy, abs=1e-12)


def test_primme_needs_no_adapter(h4):
    """primme.eigsh takes the column-vector ``v0`` directly."""
    primme = pytest.importorskip("primme")

    hcore, eri, norb, nelec = h4
    args = ([(STRINGS_A, STRINGS_B)], hcore, eri, norb, nelec)
    (default,) = sci_solver_from_integrals(hcore, eri, tol=1e-12)(*args)
    (with_primme,) = sci_solver_from_integrals(
        hcore, eri, eigensolver=primme.eigsh, tol=1e-12
    )(*args)

    assert with_primme.energy == pytest.approx(default.energy, abs=1e-10)
    np.testing.assert_allclose(
        np.abs(with_primme.sci_state.amplitudes),
        np.abs(default.sci_state.amplitudes),
        atol=1e-8,
    )


def test_drives_the_addon_configuration_recovery_loop():
    """The solver works as the sci_solver argument of the addon's loop."""
    from qiskit_addon_sqd.counts import generate_bit_array_uniform
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="sto-6g",
        symmetry="Dooh",
        verbose=0,
    )
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    norb = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    nelec = ((n_electrons + mol.spin) // 2, (n_electrons - mol.spin) // 2)
    cas = pyscf.mcscf.CASCI(scf, norb, nelec)
    mo = cas.sort_mo(active_space, base=0)
    hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)
    exact_energy = cas.run().e_tot

    rng = np.random.default_rng(24)
    bit_array = generate_bit_array_uniform(5_000, 2 * norb, rand_seed=rng)
    result = diagonalize_fermionic_hamiltonian(
        hcore,
        eri,
        bit_array,
        samples_per_batch=100,
        norb=norb,
        nelec=nelec,
        num_batches=3,
        max_iterations=3,
        sci_solver=sci_solver_from_integrals(hcore, eri, tol=1e-12),
        symmetrize_spin=True,
        max_dim=50,
        seed=rng,
    )

    energy = result.energy + nuclear_repulsion_energy
    assert energy > exact_energy - 1e-9, "a subspace energy cannot beat the exact one"
    assert energy == pytest.approx(exact_energy, abs=0.05)
