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


import itertools

import numpy as np
import pytest

pytest.importorskip("fulqrum.gpu.common.operator_arrays")

from fulqrum.gpu.distributed import term_tables as tt  # noqa: E402
from fulqrum.gpu.distributed.tile_discovery import (
    _group_flip_masks,
    _group_flip_masks_reference,
)  # noqa: E402



def operator_case(name, directory):
    import fulqrum as fq
    from fulqrum.gpu.common.operator_arrays import to_gpu_arrays

    if name == "fe4s4":
        from fulqrum.convert import fcidump_to_fq_fermionic_op
        from fulqrum.test.conftest import fe4s4_alpha_strings, fe4s4_fcidump

        op = fcidump_to_fq_fermionic_op(
            str(fe4s4_fcidump(directory))
        ).extended_jw_transformation()
        op, _ = op.remove_constant_terms()
        op.set_type(2)
        half = sorted(fe4s4_alpha_strings())[:200]
    else:
        from fulqrum.test.test_type2_differential import (
            NORB,
            ORDERINGS,
            bitstring,
            make_operator,
        )

        alpha, beta = ORDERINGS["block"]
        half = [
            bitstring(occupied, NORB)
            for occupied in itertools.combinations(range(NORB), 2)
        ]
        op = make_operator(name, alpha, beta).extended_jw_transformation()
        op, _ = op.remove_constant_terms()
        op.set_type(2)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([list(half), list(half)]))
    spmv = hsub.spmv
    return to_gpu_arrays(spmv, skip_diag=True), int(spmv.subspace.num_alpha_qubits)


CASES = ["fe4s4", "aabb_uniform_singles", "mixed"]


@pytest.mark.parametrize("name", CASES)
def test_flip_masks_equal_the_reference(name, tmp_path):
    gpu_data, num_alpha = operator_case(name, tmp_path)
    oi, os_ = gpu_data["grp_offdiag_inds"], gpu_data["grp_offdiag_sizes"]
    fast, ref = (
        _group_flip_masks(oi, os_, num_alpha),
        _group_flip_masks_reference(oi, os_, num_alpha),
    )
    assert fast.keys() == ref.keys()
    for key in ref:
        assert [(int(m), tuple(int(x) for x in p)) for m, p in fast[key]] == [
            (int(m), tuple(int(x) for x in p)) for m, p in ref[key]
        ], key
        assert all(isinstance(m, np.uint64) for m, _ in fast[key])


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("check", ["proj", "factor"])
def test_invariance_checks_agree_with_the_reference(name, check, tmp_path):
    gpu_data, num_alpha = operator_case(name, tmp_path)
    fast, ref = (
        (
            tt._assert_aaaa_bbbb_proj_invariance,
            tt._assert_aaaa_bbbb_proj_invariance_reference,
        )
        if check == "proj"
        else (
            tt._assert_aa_bb_factor_cache_invariance,
            tt._assert_aa_bb_factor_cache_invariance_reference,
        )
    )
    oi, os_ = (
        np.asarray(gpu_data["grp_offdiag_inds"]),
        np.asarray(gpu_data["grp_offdiag_sizes"]),
    )
    num_groups = int(gpu_data["num_groups"])
    fast(gpu_data, oi, os_, num_groups, num_alpha)
    ref(gpu_data, oi, os_, num_groups, num_alpha)
    # plant a violation in the first term of the first same-spin group of the checked size
    size = 4 if check == "proj" else 2
    _, owner, terms = tt._group_terms(gpu_data, os_, num_groups, size)
    n_a = (oi[owner, :size] < num_alpha).sum(axis=1)
    same = np.flatnonzero((n_a == size) | (n_a == 0))
    if len(same) == 0:
        pytest.skip(f"no same-spin group of size {size}")
    row = int(same[0])
    t = int(terms[row])
    alpha_group = n_a[row] == size
    lo_bit = num_alpha if alpha_group else 0
    bad = dict(gpu_data)

    def widened(name, count_name, slots):
        table = np.asarray(gpu_data[name])
        table = (
            table.reshape(len(np.asarray(gpu_data[count_name])), -1)
            if table.ndim == 1
            else table
        )
        if table.shape[1] < slots:
            table = np.concatenate(
                [
                    table,
                    np.zeros((table.shape[0], slots - table.shape[1]), table.dtype),
                ],
                axis=1,
            )
        return np.array(table, copy=True), np.array(
            np.asarray(gpu_data[count_name]), copy=True
        )

    if check == "proj":
        # one projector index in the forbidden (opposite-spin) range
        proj, num_proj = widened("term_proj_inds", "term_num_proj", 1)
        proj[t, 0] = lo_bit
        num_proj[t] = max(int(num_proj[t]), 1)
        bad["term_proj_inds"], bad["term_num_proj"] = proj, num_proj
    else:
        # two distinct cross-spin bits in the projectors of one term
        proj, num_proj = widened("term_proj_inds", "term_num_proj", 2)
        proj[t, 0], proj[t, 1] = lo_bit, lo_bit + 1
        num_proj[t] = max(int(num_proj[t]), 2)
        bad["term_proj_inds"], bad["term_num_proj"] = proj, num_proj
    with pytest.raises(AssertionError) as fast_error:
        fast(bad, oi, os_, num_groups, num_alpha)
    with pytest.raises(AssertionError) as ref_error:
        ref(bad, oi, os_, num_groups, num_alpha)
    assert str(fast_error.value) == str(ref_error.value)
