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
# pylint: disable=no-name-in-module
import numpy as np
import scipy.sparse as sp
import pytest
import psutil
from types import SimpleNamespace
import fulqrum as fq
from fulqrum.utils import qubitoperator_to_matrix

"""All test names ending with `a` uses `use_all_bitset_blocks=False` which
    using first (LSB) block of a bitset for hashing.
"""


def matrix_subspace(A, rows):
    B = A[rows, :]
    B = B[:, rows]
    return B


def test_csrlike_csr1():
    """Test building CSR array from subspace Hamiltonian"""
    num_qubits = 5
    strings = ["XIXII", "0101I", "II0II", "XYIYX", "ZZZZZ"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    M = sp.csr_array(A)

    counts = {}
    for kk in range(2**5):
        counts[bin(kk)[2:].zfill(5)] = None

    S = fq.Subspace([list(counts.keys())])
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


def test_csrlike_csr1a():
    """Test building CSR array from subspace Hamiltonian"""
    num_qubits = 5
    strings = ["XIXII", "0101I", "II0II", "XYIYX", "ZZZZZ"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    M = sp.csr_array(A)

    counts = {}
    for kk in range(2**5):
        counts[bin(kk)[2:].zfill(5)] = None

    S = fq.Subspace([list(counts.keys())], use_all_bitset_blocks=False)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


def test_csrlike_csr2():
    """Test building CSR array from subspace Hamiltonian"""
    num_qubits = 5
    strings = ["01011", "IIIII", "ZZIZZ"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    M = sp.csr_array(A)

    counts = {}
    for kk in range(2**5):
        counts[bin(kk)[2:].zfill(5)] = None

    S = fq.Subspace([list(counts.keys())])
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


def test_csrlike_csr2a():
    """Test building CSR array from subspace Hamiltonian"""
    num_qubits = 5
    strings = ["01011", "IIIII", "ZZIZZ"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    M = sp.csr_array(A)

    counts = {}
    for kk in range(2**5):
        counts[bin(kk)[2:].zfill(5)] = None

    S = fq.Subspace([list(counts.keys())], use_all_bitset_blocks=False)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


def test_csrlike_csr3():
    """Test building CSR array from subspace Hamiltonian"""
    num_qubits = 5
    strings = ["XXIXX", "YYIYY"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    M = sp.csr_array(A)

    counts = {}
    for kk in range(2**5):
        counts[bin(kk)[2:].zfill(5)] = None

    S = fq.Subspace([list(counts.keys())])
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


def test_csrlike_csr3a():
    """Test building CSR array from subspace Hamiltonian"""
    num_qubits = 5
    strings = ["XXIXX", "YYIYY"]
    values = np.array([(-1) ** kk * 3.14159 / (kk + 1) for kk in range(len(strings))])

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    M = sp.csr_array(A)

    counts = {}
    for kk in range(2**5):
        counts[bin(kk)[2:].zfill(5)] = None

    S = fq.Subspace([list(counts.keys())], use_all_bitset_blocks=False)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


def test_csrlike_csr4():
    """Test building CSR array from subspace Hamiltonian"""
    num_qubits = 5
    strings = ["XXIXX", "YYIYY", "-+I+-", "XYZXY", "+-I-+"]
    values = np.array([(-1) ** kk for kk in range(len(strings))])

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    M = sp.csr_array(A)

    counts = {}
    for kk in range(2**5):
        counts[bin(kk)[2:].zfill(5)] = None

    S = fq.Subspace([list(counts.keys())])
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


def test_csrlike_csr4a():
    """Test building CSR array from subspace Hamiltonian, single block hashing"""
    num_qubits = 5
    strings = ["XXIXX", "YYIYY", "-+I+-", "XYZXY", "+-I-+"]
    values = np.array([(-1) ** kk for kk in range(len(strings))])

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    M = sp.csr_array(A)

    counts = {}
    for kk in range(2**5):
        counts[bin(kk)[2:].zfill(5)] = None

    S = fq.Subspace([list(counts.keys())], use_all_bitset_blocks=False)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


def test_csrlike_csr5():
    """Test building CSR array from subspace Hamiltonian, empty result"""
    num_qubits = 5
    strings = ["+-X-+", "XZIXX", "0Y+YZ", "X0+1Y", "-+X+-", "X0-1Y", "0Y-YZ"]
    values = np.array([1 for kk in range(len(strings))])
    rows = [0, 1, 2, 26, 27, 28]

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    B = matrix_subspace(A, rows)
    M = sp.csr_array(B)

    counts = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = fq.Subspace([list(counts.keys())])
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


def test_csrlike_csr5a():
    """Test building CSR array from subspace Hamiltonian with single block hashing
    - empty result"""
    num_qubits = 5
    strings = ["+-X-+", "XZIXX", "0Y+YZ", "X0+1Y", "-+X+-", "X0-1Y", "0Y-YZ"]
    values = np.array([1 for kk in range(len(strings))])
    rows = [0, 1, 2, 26, 27, 28]

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    B = matrix_subspace(A, rows)
    M = sp.csr_array(B)

    counts = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = fq.Subspace([list(counts.keys())], use_all_bitset_blocks=False)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


def test_csrlike_csr6():
    """Test building CSR array from subspace Hamiltonian"""
    num_qubits = 5
    strings = ["+-X-+", "XZIXX", "0Y+YZ", "X0+1Y", "-+X+-", "X0-1Y", "0Y-YZ"]
    values = np.array([1 for kk in range(len(strings))])
    rows = [0, 1, 2, 3, 5, 7, 8, 9, 11, 13, 31]

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    B = matrix_subspace(A, rows)
    M = sp.csr_array(B)

    counts = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = fq.Subspace([list(counts.keys())])
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


def test_csrlike_csr6a():
    """Test building CSR array from subspace Hamiltonian with single bitset block hashing"""
    num_qubits = 5
    strings = ["+-X-+", "XZIXX", "0Y+YZ", "X0+1Y", "-+X+-", "X0-1Y", "0Y-YZ"]
    values = np.array([1 for kk in range(len(strings))])
    rows = [0, 1, 2, 3, 5, 7, 8, 9, 11, 13, 31]

    H = fq.QubitOperator(num_qubits)
    for idx, string in enumerate(strings):
        H += fq.QubitOperator.from_label(string, values[idx])

    A = qubitoperator_to_matrix(H)
    B = matrix_subspace(A, rows)
    M = sp.csr_array(B)

    counts = {bin(rr)[2:].zfill(H.width): 1 for rr in rows}

    S = fq.Subspace([list(counts.keys())], use_all_bitset_blocks=False)
    Hsub = fq.SubspaceHamiltonian(H, S)
    P = Hsub._to_linearoperator().to_csr_array()

    assert np.allclose(P.indptr, M.indptr)
    assert np.allclose(P.indices, M.indices)
    assert np.allclose(P.data, M.data)
    assert P.indptr.dtype == np.int32


@pytest.mark.parametrize("route", ["spmv", "csrlike"])
@pytest.mark.parametrize("label", ["X", "Y"])
def test_csr_memory_preflight_and_recovery(monkeypatch, route, label):
    from fulqrum.exceptions import FulqrumError

    operator = fq.QubitOperator.from_label(label, 1.0)
    hsub = fq.SubspaceHamiltonian(operator, fq.Subspace([["0", "1"]]))
    builder = hsub.spmv if route == "spmv" else hsub.spmv.to_csrlike()
    available_memory = psutil.virtual_memory
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=0))
    with pytest.raises(FulqrumError, match="does not fit within available memory"):
        builder.to_csr_array()
    monkeypatch.setattr(psutil, "virtual_memory", available_memory)
    assert np.allclose(
        builder.to_csr_array().toarray(), qubitoperator_to_matrix(operator)
    )
