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
import gc
import weakref
from itertools import combinations
import pytest
from pathlib import Path

import numpy as np
import fulqrum as fq
from fulqrum.convert.integrals import integrals_to_fq_fermionic_op
from fulqrum.exceptions import FulqrumError

_path = Path(__file__).parent / "data/lih.json"
FOP = fq.FermionicOperator.from_json(_path)
OP = FOP.extended_jw_transformation()


def test_subspacehamiltonian_different_widths():
    """Hamiltonian and subspace with different widths raises"""
    S = fq.Subspace()
    Hsub = fq.SubspaceHamiltonian(OP, S)
    with pytest.raises(FulqrumError) as _:
        Hsub.diagonal_vector()


def test_subspacehamiltonian_update_subspace():
    """Test that updating a subspace works"""
    S = fq.Subspace()
    S2 = fq.Subspace([["1" * 12]])
    Hsub = fq.SubspaceHamiltonian(OP, S)
    Hsub.update_subspace(S2)
    diag = Hsub.diagonal_vector()
    assert np.allclose(diag, np.array([1.77959097]))


def test_subspacehamiltonian_init_no_subspace():
    """Can initialize a SubspaceHamiltonian with no subspace"""
    Hsub = fq.SubspaceHamiltonian(OP)
    assert Hsub.spmv.subspace_dim == 0


def test_csr_linops_free_by_ref():
    """Verify linops freed by reference counting - Issue #141"""
    norb, nelec = 6, 3
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal((norb, norb))
    h1 = h1 + h1.T
    h2 = rng.standard_normal((norb,) * 4)
    h2 = h2 + h2.transpose(1, 0, 2, 3)
    h2 = h2 + h2.transpose(0, 1, 3, 2)
    h2 = h2 + h2.transpose(2, 3, 0, 1)
    hsub = fq.SubspaceHamiltonian(
        integrals_to_fq_fermionic_op(h1, h2).extended_jw_transformation()
    )
    strs = [
        format(sum(1 << i for i in c), f"0{norb}b")
        for c in combinations(range(norb), nelec)
    ]
    hsub.update_subspace(fq.Subspace([strs, strs]))

    gc.disable()
    for method in ("to_csr_linearoperator", "to_csr_linearoperator_fast"):
        op = getattr(hsub, method)()
        ref = weakref.ref(op)
        del op
        assert ref() is None
    gc.enable()
