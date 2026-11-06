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
import numpy as np
import pytest

import fulqrum as fq
from fulqrum.exceptions import FulqrumError


@pytest.mark.parametrize("product", [False, True])
@pytest.mark.parametrize("all_blocks", [False, True])
@pytest.mark.parametrize("reserve_multiplier", [1, 2])
def test_storage_transition_and_copy(product, all_blocks, reserve_multiplier):
    alpha = [format(i, "065b") for i in [1, 3, 1 << 64]]
    beta = [format(i, "070b") for i in [2, 1 << 69]]
    strings = sorted(b + a for b in beta for a in alpha)
    space = fq.Subspace(
        [alpha, beta] if product else [strings.copy()],
        use_all_bitset_blocks=all_blocks,
        reserve_multiplier=reserve_multiplier,
    )
    assert space._storage_state()["map_materialized"] is False
    before = space._storage_state()["full_buffer_words"]
    assert list(space.to_dict(key_type="str")) == strings
    assert space[-1].to_string() == strings[-1]
    np.testing.assert_array_equal(
        list(space.interpret_vector(np.ones(6), renormalize=False)), strings
    )
    copied = space.copy()
    assert list(copied.to_dict(key_type="str")) == strings
    assert copied._storage_state()["map_materialized"] is False
    if product:
        assert space.get_bitstring_index(strings[4]) == 4
        assert space.alpha_strs == alpha
        assert space.beta_strs == beta
        exposed = space.alpha_strs
        exposed[0] = "bad"
        assert space.alpha_strs == alpha
    h = fq.SubspaceHamiltonian(fq.QubitOperator.from_label("Z" + "I" * 134), space)
    h @ np.ones(len(space))
    assert space._storage_state()["map_build_count"] == 1
    assert space._storage_state()["full_buffer_words"] == 0
    assert before == (0 if product else 18)
    assert list(space.to_dict(key_type="str")) == strings
    h @ np.ones(len(space))
    assert space._storage_state()["map_build_count"] == 1
    after_copy = space.copy()
    assert after_copy._storage_state()["map_materialized"] is False
    del space, h
    assert list(after_copy.to_dict(key_type="str")) == strings


@pytest.mark.parametrize(
    "bad", [[["01", "01"]], [["01", "1"]], [["02"]], [[], ["1"]], [["1"], []]]
)
def test_malformed_storage_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        fq.Subspace(bad)


def test_empty_singleton_and_export_limits():
    empty = fq.Subspace()
    assert len(empty) == 0
    assert len(empty.copy()) == 0
    assert empty.to_dict() == {}
    assert empty.get_bitstring_index("0") is None
    with pytest.raises(IndexError):
        empty[0]
    space = fq.Subspace([["1"], ["10"]])
    assert space[0].to_string() == "101"
    assert space.copy()[0].to_string() == "101"
    a, b = space.to_alpha_beta_arrays()
    np.testing.assert_array_equal(a, [1])
    np.testing.assert_array_equal(b, [2])
    a[0] = 0
    np.testing.assert_array_equal(space.to_alpha_beta_arrays()[0], [1])
    wide = fq.Subspace([["1" * 65], ["0"]])
    with pytest.raises(FulqrumError):
        wide.to_alpha_beta_arrays()


def test_occupancy_and_sqd_key_export_remain_map_free():
    space = fq.Subspace([["01", "10"], ["01", "10"]])
    a, b = space.get_orbital_occupancies(np.full(4, 0.25), 2)
    np.testing.assert_allclose(a, [0.5, 0.5])
    np.testing.assert_allclose(b, [0.5, 0.5])
    assert space._storage_state()["map_materialized"] is False


@pytest.mark.parametrize("product", [False, True])
def test_singleton_materialization_with_minimum_reserve(product):
    space = fq.Subspace([["1"], ["10"]] if product else [["101"]], reserve_multiplier=1)
    h = fq.SubspaceHamiltonian(fq.QubitOperator.from_label("ZZZ"), space)
    np.testing.assert_allclose(h @ np.ones(1), [1.0])
    assert space._storage_state()["map_build_count"] == 1
