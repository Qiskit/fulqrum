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

"""Support for using Fulqrum as a solver for the Qiskit SQD addon.

This module builds the ``sci_solver`` that
:func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian` expects, so that
it is straightforward for a user to leverage Fulqrum with the addon's configuration
recovery loop.

``qiskit-addon-sqd`` is imported lazily in this module, and the rest of Fulqrum is
usable without it. To install the two together, use the addon's own extra::

    pip install "qiskit-addon-sqd[fulqrum]"
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import scipy.sparse.linalg

import fulqrum as fq
from fulqrum.convert.integrals import integrals_to_fq_fermionic_op

__all__ = [
    "sci_solver_from_integrals",
]


def sci_solver_from_integrals(
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
    **kwargs: Any,
) -> Callable[..., list]:
    """Build an SQD addon ``sci_solver`` that diagonalizes with Fulqrum.

    The returned function matches the ``sci_solver`` interface of
    `qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian
    <https://qiskit.github.io/qiskit-addon-sqd/stubs/qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian.html>`__,
    so it can be passed straight to that function's ``sci_solver`` argument::

        solver = sci_solver_from_integrals(hcore, eri)
        result = diagonalize_fermionic_hamiltonian(
            hcore, eri, bit_array, ..., sci_solver=solver
        )

    The Hamiltonian is converted to a Fulqrum :class:`~fulqrum.QubitOperator` once,
    here, rather than on every diagonalization. The integrals that the addon passes to
    the solver on each call are therefore ignored.

    Args:
        one_body_integrals: The one-body integrals, as given to the SQD addon.
        two_body_integrals: The two-body integrals, as given to the SQD addon.
        **kwargs: Forwarded to :func:`_sci_solver_from_qubit_operator`.

    Returns:
        A function suitable for the SQD addon's ``sci_solver`` argument.
    """
    fermionic_op = integrals_to_fq_fermionic_op(
        one_body_integrals=one_body_integrals,
        two_body_integrals=two_body_integrals,
    )
    solver = _sci_solver_from_qubit_operator(
        fermionic_op.extended_jw_transformation(), **kwargs
    )
    expected = (
        _fingerprint(one_body_integrals),
        _fingerprint(two_body_integrals),
    )

    def checked_solver(ci_strings, one_body, two_body, norb, nelec):
        # The operator was built from the integrals given here, so integrals that
        # arrive later are ignored. Diagonalizing the wrong Hamiltonian would
        # otherwise return a plausible energy, so check that they still match.
        _check_integrals_match(expected, one_body, two_body)
        return solver(ci_strings, one_body, two_body, norb, nelec)

    return checked_solver


def _sci_solver_from_qubit_operator(
    operator,
    *,
    eigensolver: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    matrix_free: bool = False,
    **eigensolver_kwargs: Any,
) -> Callable[..., list]:
    """Build an SQD addon ``sci_solver`` from a Fulqrum qubit operator.

    Private, because the addon requires the one- and two-body integrals as positional
    arguments of ``diagonalize_fermionic_hamiltonian``. A caller holding only an
    operator would have to pass dummy tensors for the loop to ignore, which is a
    question about the addon's interface rather than something to work around here.
    See https://github.com/Qiskit/qiskit-addon-sqd/issues/312.

    Args:
        operator: The Hamiltonian, as a Fulqrum :class:`~fulqrum.QubitOperator`.
        eigensolver: The eigensolver that diagonalizes each projected Hamiltonian. It
            is called as ``eigensolver(linear_operator, k=1, which="SA", v0=v0,
            **eigensolver_kwargs)`` and must return a pair ``(eigenvalues,
            eigenvectors)`` whose eigenvectors are columns, following the convention
            of `scipy.sparse.linalg.eigsh
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`__.
            Defaults to that function.

            ``v0`` is passed as a two-dimensional array of column vectors, the
            general convention: it holds one starting vector per column, which
            ``primme.eigsh`` and ``scipy.sparse.linalg.lobpcg`` both expect.

            SciPy's solvers are the exception, taking a single one-dimensional
            starting vector instead, because ARPACK's ``v0`` is the one residual
            vector it overwrites as it iterates. They are detected and adapted
            automatically, so ``scipy.sparse.linalg.eigsh`` can be passed directly.
            A solver of that shape from elsewhere needs its own wrapper to flatten
            ``v0``.
        matrix_free: Whether to apply the projected Hamiltonian without building a
            sparse matrix for it. Building the matrix is considerably faster, so it is
            the default, but it costs memory proportional to the number of nonzeros.
            Set this to ``True`` for a subspace whose sparse matrix does not fit in
            memory.
        **eigensolver_kwargs: Additional keyword arguments for ``eigensolver``, for
            example ``tol``.

    Returns:
        A function suitable for the SQD addon's ``sci_solver`` argument.
    """
    if eigensolver is None:
        eigensolver = scipy.sparse.linalg.eigsh
    if eigensolver in _SINGLE_VECTOR_EIGENSOLVERS:
        eigensolver = _single_vector_eigensolver(eigensolver)

    # One SubspaceHamiltonian is reused across every call, because splitting and
    # grouping the operator's terms is independent of the subspace and is the
    # expensive part of constructing it.
    subspace_hamiltonian = fq.SubspaceHamiltonian(operator)

    def solve_sci_batch(
        ci_strings: Sequence[tuple[np.ndarray, np.ndarray]],
        _one_body_integrals: np.ndarray,
        _two_body_integrals: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
    ) -> list:
        return [
            _solve_one(
                strings_a,
                strings_b,
                norb=norb,
                nelec=nelec,
                subspace_hamiltonian=subspace_hamiltonian,
                eigensolver=eigensolver,
                matrix_free=matrix_free,
                eigensolver_kwargs=eigensolver_kwargs,
            )
            for strings_a, strings_b in ci_strings
        ]

    return solve_sci_batch


_SINGLE_VECTOR_EIGENSOLVERS = frozenset(
    {scipy.sparse.linalg.eigsh, scipy.sparse.linalg.eigs}
)


def _single_vector_eigensolver(
    eigensolver: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Adapt a solver that takes ``v0`` as a single one-dimensional vector.

    ARPACK's ``v0`` is the residual vector it overwrites while iterating, so SciPy
    copies the argument straight into a length-n work array. A two-dimensional ``v0``
    then raises as soon as ARPACK writes back, which happens once the starting vector
    is close enough to the answer to skip the restart loop -- the usual case here.
    """

    def wrapper(linear_operator, *args, v0=None, **kwargs):
        if v0 is not None:
            v0 = np.asarray(v0)
            if v0.ndim == 2:
                if v0.shape[1] != 1:
                    raise ValueError(
                        f"{getattr(eigensolver, '__name__', eigensolver)} accepts one "
                        f"starting vector, but v0 holds {v0.shape[1]}. Use an "
                        "eigensolver that takes a block, such as primme.eigsh."
                    )
                v0 = v0[:, 0]
        return eigensolver(linear_operator, *args, v0=v0, **kwargs)

    return wrapper


_FINGERPRINT_SAMPLES = 100


def _fingerprint(tensor: np.ndarray, num_samples: int = _FINGERPRINT_SAMPLES):
    """Shape and a fixed sample of values, for comparing two tensors cheaply."""
    tensor = np.asarray(tensor)
    flat = tensor.reshape(-1)
    # Evenly spaced indices over the flattened tensor vary every axis, where a
    # contiguous slice would vary only the last one. The gather costs the same
    # regardless of the tensor's size, so a two-body tensor with 1e8 elements is
    # as cheap to check as a one-body tensor with 1e2.
    count = min(flat.size, num_samples)
    if not count:
        return tensor.shape, flat[:0]
    indices = np.linspace(0, flat.size - 1, count, dtype=np.intp)
    return tensor.shape, flat[indices]


def _check_integrals_match(
    expected, one_body: np.ndarray, two_body: np.ndarray
) -> None:
    """Raise if the integrals differ from the ones the solver was built from.

    Raising suits a single process. Under SPMD, one rank raising while the others
    wait in a collective deadlocks, so this will need to abort collectively once
    Fulqrum supports MPI.
    """
    names = ("one-body", "two-body")
    for name, want, got in zip(names, expected, (one_body, two_body)):
        want_shape, want_values = want
        got_shape, got_values = _fingerprint(got)
        if want_shape != got_shape or not np.array_equal(want_values, got_values):
            raise ValueError(
                f"The {name} integrals passed to this solver differ from the ones it "
                "was built from, so it would diagonalize the wrong Hamiltonian. Build "
                "the solver with the same integrals that are passed to "
                "diagonalize_fermionic_hamiltonian."
            )


def _check_sorted(strings, which: str) -> np.ndarray:
    """Return the CI strings as an array, checking that they are sorted.

    Raising suits a single process; see :func:`_check_integrals_match` for what
    SPMD will require.
    """
    strings = np.asarray(strings)
    if strings.size and not np.all(strings[:-1] <= strings[1:]):
        raise ValueError(
            f"The {which} CI strings are not sorted in ascending order. "
            "diagonalize_fermionic_hamiltonian passes them sorted, so this solver "
            "relies on that ordering to match amplitudes to determinants."
        )
    return strings


def _solve_one(
    strings_a: np.ndarray,
    strings_b: np.ndarray,
    *,
    norb: int,
    nelec: tuple[int, int],
    subspace_hamiltonian,
    eigensolver: Callable[..., tuple[np.ndarray, np.ndarray]],
    matrix_free: bool,
    eigensolver_kwargs: dict,
):
    # Imported here so that the module can be imported without qiskit-addon-sqd
    # installed, which keeps it usable for introspection and documentation builds.
    from qiskit_addon_sqd.fermion import SCIResult, SCIState

    # A Fulqrum Subspace sorts each list of half strings, and the amplitudes are
    # unpacked below on the assumption that the sorted order matches the order of
    # these arrays. The SQD addon guarantees that it passes them sorted, as of
    # https://github.com/Qiskit/qiskit-addon-sqd/pull/423; this checks that
    # guarantee rather than re-establishing it, because sorting here would silently
    # reorder the arrays that the result reports back as its CI strings.
    strings_a = _check_sorted(strings_a, "spin-alpha")
    strings_b = _check_sorted(strings_b, "spin-beta")

    subspace = fq.Subspace(
        [
            [format(string, f"0{norb}b") for string in strings_a],
            [format(string, f"0{norb}b") for string in strings_b],
        ]
    )
    subspace_hamiltonian.update_subspace(subspace)

    if matrix_free:
        linear_operator = subspace_hamiltonian
    else:
        linear_operator = subspace_hamiltonian.to_csr_linearoperator_fast()

    # Start the eigensolver from the lowest-energy determinant in the subspace.
    diagonal = subspace_hamiltonian.diagonal_vector()
    v0 = np.zeros((len(subspace), 1), dtype=subspace_hamiltonian.dtype)
    v0[np.argmin(diagonal), 0] = 1.0

    eigenvalues, eigenvectors = eigensolver(
        linear_operator, k=1, which="SA", v0=v0, **eigensolver_kwargs
    )
    eigenvector = eigenvectors[:, 0]

    occupancies = subspace.get_orbital_occupancies(
        probs=np.abs(eigenvector) ** 2, norb=norb
    )

    # Fulqrum orders a subspace with the beta half string major and the alpha half
    # string minor, which is the transpose of the addon's convention. PySCF requires
    # the C-contiguous result.
    amplitudes = np.ascontiguousarray(
        eigenvector.reshape(len(strings_b), len(strings_a)).T
    )
    sci_state = SCIState(
        amplitudes=amplitudes,
        ci_strs_a=strings_a,
        ci_strs_b=strings_b,
        norb=norb,
        nelec=nelec,
    )
    return SCIResult(
        energy=eigenvalues[0],
        sci_state=sci_state,
        orbital_occupancies=occupancies,
    )
