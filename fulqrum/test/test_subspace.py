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
"""Test string to vector conversion"""

import pytest
import numpy as np
from fulqrum import Subspace, Bitset

dic = {
    "01010": 1,
    "10101": 1,
    "11111": 1,
    "11100": 1,
}  # sorted keys: "01010","10101","11100","11111"
dic = [list(dic.keys())]
dic2 = {
    "01010": 1,
    "10101": 1,
    "11111": 1,
    "11100": 1,
    "01111": 1,
    "00100": 1,
    "10111": 1,
}  # sorted keys: "00100","01010","01111","10111","11100","11111"
dic2 = [list(dic2.keys())]

copy_sub = {
    "000000000000111111111111111111111111000000000000111111111111111111111111": 1,
    "000000000000111111111111111111111111000000000011110011111111111111111111": 1,
    "000000000000111111111111111111111111000000000101010111111111111111111111": 1,
    "000000000000111111111111111111111111000000000101100111111111111111111111": 1,
    "000000000000111111111111111111111111000000001001010111111111111111111111": 1,
    "000000000000111111111111111111111111000000001001100111111111111111111111": 1,
    "000000000001110111111111111111111111000000000001110111111111111111111111": 1,
    "000000000001110111111111111111111111000000000010111011111111111111111111": 1,
    "000000000001110111111111111111111111000000000100011111111111111111111111": 1,
    "000000000001110111111111111111111111000000000100101111111111111111111111": 1,
    "000000000001110111111111111111111111000000001000011111111111111111111111": 1,
    "000000000001110111111111111111111111000000001000101111111111111111111111": 1,
    "000000000010111011111111111111111111000000000001110111111111111111111111": 1,
    "000000000011110011111111111111111111000000000000111111111111111111111111": 1,
    "000000000100011111111111111111111111000000000001110111111111111111111111": 1,
    "000000000100011111111111111111111111000000001000011111111111111111111111": 1,
    "000000000100011111111111111111111111000000001000101111111111111111111111": 1,
    "000000000100101111111111111111111111000000000001110111111111111111111111": 1,
    "000000000100101111111111111111111111000000001000101111111111111111111111": 1,
    "000000000101010111111111111111111111000000000000111111111111111111111111": 1,
    "000000000101100111111111111111111111000000000000111111111111111111111111": 1,
    "000000001000011111111111111111111111000000000001110111111111111111111111": 1,
    "000000001000011111111111111111111111000000000100011111111111111111111111": 1,
    "000000001000101111111111111111111111000000000001110111111111111111111111": 1,
    "000000001000101111111111111111111111000000000100011111111111111111111111": 1,
    "000000001000101111111111111111111111000000000100101111111111111111111111": 1,
    "000000001001010111111111111111111111000000000000111111111111111111111111": 1,
    "000000001001100111111111111111111111000000000000111111111111111111111111": 1,
}


def test_subspace_vector_order0():
    V = Subspace(dic)
    # Order is only guaranteed up to first bit
    assert V[0].to_string()[-1] == "0"
    assert V[1].to_string()[-1] == "1"
    assert V[2].to_string()[-1] == "0"
    assert V[3].to_string()[-1] == "1"


def test_subspace_vector_order1():
    V = Subspace(dic2)
    # Order is only guaranteed up to first bit
    assert V[0].to_string()[-1] == "0"
    assert V[1].to_string()[-1] == "0"
    assert V[2].to_string()[-1] == "1"
    assert V[3].to_string()[-1] == "1"
    assert V[4].to_string()[-1] == "1"
    assert V[5].to_string()[-1] == "0"
    assert V[6].to_string()[-1] == "1"


def test_subspace_vector_order2():
    V = Subspace(dic)
    assert list(V.to_dict(key_type="str").keys()) == [
        "01010",
        "10101",
        "11100",
        "11111",
    ]


def test_subspace_vector_order3():
    V = Subspace(dic2)
    assert list(V.to_dict(key_type="str").keys()) == [
        "00100",
        "01010",
        "01111",
        "10101",
        "10111",
        "11100",
        "11111",
    ]


def test_get_n_th_bitstring():
    """Test get_n_th_bitstring() method returns the correct bitstring
    Both Python and Fulqrum's emhash8::HashMap dictionaries retain
    the insertion order, and this test checks in that order.
    """
    V = Subspace(dic)
    assert V.get_n_th_bitstring(0) == "01010"
    assert V.get_n_th_bitstring(1) == "10101"
    assert V.get_n_th_bitstring(2) == "11100"
    assert V.get_n_th_bitstring(3) == "11111"


def test_max_num_qubits():
    bitstrings = [["0" * (2**16), "1" * (2**16)]]
    S = Subspace(bitstrings)

    assert S

    bitstrings = [["0" * (2**16 + 1), "1" * (2**16 + 1)]]

    with pytest.raises(ValueError):
        S = Subspace(bitstrings)


def test_get_orbital_occupancies():
    # sorted bitstrings as Subspace input as Subspace sorts them
    # if this is not sorted here, ``probs`` array may have mismatched order.
    bitstrings = [["011101", "101011", "110110"]]  # bit order: bN ... b0, aN ... a0
    probs = np.array([0.50, 0.30, 0.20])
    n_spatial_orb = len(bitstrings[0][0]) // 2
    S = Subspace(bitstrings)

    # occupancies orrder: ([a0, ..., aN], [b0, ..., bN])
    occs_a, occs_b = S.get_orbital_occupancies(probs=probs, norb=n_spatial_orb)

    expected_occs_a = np.array([0.80, 0.50, 0.70])
    expected_occs_b = np.array([0.80, 0.70, 0.50])

    np.testing.assert_allclose(occs_a, expected_occs_a)
    np.testing.assert_allclose(occs_b, expected_occs_b)

    bitstrings = [
        ["00010001", "00101000", "10000100"]
    ]  # bit order: bN ... b0, aN ... a0
    probs = np.array([0.40, 0.50, 0.10])
    n_spatial_orb = len(bitstrings[0][0]) // 2
    S = Subspace(bitstrings)

    occs_a, occs_b = S.get_orbital_occupancies(probs=probs, norb=n_spatial_orb)

    expected_occs_a = np.array([0.40, 0.0, 0.10, 0.50])
    expected_occs_b = np.array([0.40, 0.50, 0.0, 0.10])

    np.testing.assert_allclose(occs_a, expected_occs_a)
    np.testing.assert_allclose(occs_b, expected_occs_b)


def test_subspace_copy():
    """Test subspace copy"""
    S = Subspace([["0100", "1000", "1111", "0000"]])
    B = S.copy()
    del S
    sorted_keys = list(B.to_dict(key_type="str").keys())
    assert sorted_keys == ["0000", "0100", "1000", "1111"]


def test_bitset_dict():
    """Test subspace to_dict as Bitsets"""
    S = Subspace([["0100", "1000", "1111", "0000"]])
    sorted_keys = list(S.to_dict().keys())
    assert sorted_keys == [
        Bitset("0000"),
        Bitset("0100"),
        Bitset("1000"),
        Bitset("1111"),
    ]


def test_half_strs_mode_globally_ascending():
    """Test that half-string mode produces globally ascending insertion order.

    Regression test for https://github.com/Qiskit/fulqrum/issues/19.
    Uses alpha and beta lists of different lengths (3 vs 2) so that any
    incorrect product/concat ordering shows up. Earlier tests with identical
    halves accidentally masked the bug.
    """
    alpha_half_strs = ["001", "010", "100"]
    beta_half_strs = ["01", "10"]
    S = Subspace([alpha_half_strs, beta_half_strs])

    # Full bitstring is `beta + alpha`; iteration order must be ascending.
    expected = ["01001", "01010", "01100", "10001", "10010", "10100"]
    actual = [S.get_n_th_bitstring(i) for i in range(len(S))]
    assert actual == expected
    assert actual == sorted(actual)


def test_bitset_init():
    """Test subspace init using bitsets"""
    A = {Bitset("01010"): 0, Bitset("01111"): 1, Bitset("11100"): 2, Bitset("00000"): 3}
    S = Subspace([list(A.keys())])
    assert list(S.to_dict()) == [
        Bitset("00000"),
        Bitset("01010"),
        Bitset("01111"),
        Bitset("11100"),
    ]


def test_bitset_copy():
    """Test subspace copy works on previous bad example"""
    S = Subspace([list(copy_sub)])
    C = S.copy()
    for kk, bits in enumerate(copy_sub):
        assert bits == C[kk].to_string()
        assert bits == S[kk].to_string()
