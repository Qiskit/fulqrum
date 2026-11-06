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
"""Test diagonal vector computation"""

from pathlib import Path
import numpy as np

import fulqrum as fq
from fulqrum.utils.io import json_to_dict


def test_diag_fast_mode():
    """Verify fast mode and default give same diagonal"""
    op_path = Path(__file__).parent / "data/ch4_dimer_jw.json.xz"
    op = fq.QubitOperator.from_json(op_path)
    op.set_type(2)  # can be removed once the type is saved in the json
    dist_path = Path(__file__).parent / "data/dimer_subspace.json.xz"
    dist = json_to_dict(dist_path)
    S = fq.Subspace([list(dist.keys())])
    Hsub = fq.SubspaceHamiltonian(op, S)
    diag1 = Hsub.diagonal_vector()  # This is fast mode
    diag2 = fq.SubspaceHamiltonian(op, S).diagonal_vector(
        disable_fast_mode=True
    )  # This is regular mode
    np.testing.assert_allclose(diag1, diag2, atol=1e-12, rtol=1e-12)


def test_diag_fast_mode_compatibility_check():
    """Verify that diag fast mode compatibility check works"""
    op_path = Path(__file__).parent / "data/lih.json"
    fop = fq.FermionicOperator.from_json(op_path)
    op = fop.extended_jw_transformation()

    # because full operator is not diagonal
    assert not op.fast_diag_compatible()

    diag, off = op.split_diagonal()
    # because diag still has a constant term in it
    assert not diag.fast_diag_compatible()

    new_diag, _ = diag.remove_constant_terms()
    assert new_diag.fast_diag_compatible()


def test_diag_fast_mode_sorting():
    """Verify that fast diag sorting gives the correct projector indices per term"""
    op_path = Path(__file__).parent / "data/lih.json"
    fop = fq.FermionicOperator.from_json(op_path)
    op = fop.extended_jw_transformation()
    diag, off = op.split_diagonal()
    new_diag, _ = diag.remove_constant_terms()
    new_diag.fast_diag_term_sort()

    counter = 0
    for kk in range(new_diag.width):
        for ll in range(kk, new_diag.width):
            if kk == ll:
                assert new_diag[counter].proj_indices.shape[0] == 1
                assert new_diag[counter].proj_indices[0] == kk
            else:
                assert new_diag[counter].proj_indices[0] == kk
                assert new_diag[counter].proj_indices[1] == ll
            counter += 1
