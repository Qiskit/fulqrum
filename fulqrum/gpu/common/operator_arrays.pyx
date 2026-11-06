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
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cimport cython
import numpy as np
cimport numpy as np

from ...core.spmv cimport FulqrumSpMV


@cython.boundscheck(False)
def to_gpu_arrays(FulqrumSpMV spmv, bint skip_diag=False):


    if spmv.oper.type == 2 and spmv.num_groups > 0:
        spmv.check_ladder_bins()

    cdef size_t T = spmv.num_terms
    cdef size_t G = <size_t>spmv.num_groups
    cdef size_t kk, j, sz
    cdef size_t max_inds = 0, max_proj = 0, max_offdiag = 0
    cdef double complex c
    cdef double[::1] mv_term_coeffs_re
    cdef np.int8_t[::1] mv_term_real_phase
    cdef np.uint8_t[::1] mv_term_num_ops
    cdef np.uint8_t[:, ::1] mv_term_inds
    cdef np.uint8_t[:, ::1] mv_term_vals
    cdef np.uint8_t[::1] mv_term_num_proj
    cdef np.uint8_t[:, ::1] mv_term_proj_inds
    cdef np.uint8_t[:, ::1] mv_term_proj_bits
    cdef np.uint8_t[::1] mv_grp_offdiag_sizes
    cdef np.uint8_t[:, ::1] mv_grp_offdiag_inds

    if spmv.width > 256:
        raise ValueError(
            f"The GPU term arrays hold qubit indices in uint8; width {spmv.width} is more than 256."
        )
    if (spmv.num_terms > (np.iinfo(np.uint32).max)):
        raise ValueError(
            f"num_terms={spmv.num_terms} does not fit in uint32_t."
        )

    if not skip_diag:
        spmv.compute_diag_vector()

    if skip_diag:
        diag = None
    else:
        diag = (
            np.asarray(spmv.real_diag_vec)
            if spmv.is_real else np.asarray(spmv.complex_diag_vec)
        )

    if T == 0:
        return {
            "diag_vec": diag,
            "has_nonzero_diag": int(spmv.has_nonzero_diag),
            "num_groups": 0,
            "ladder_offset": int(spmv.ladder_offset),
        }

    for kk in range(T):
        if spmv.oper.terms[kk].indices.size() > max_inds:
            max_inds = spmv.oper.terms[kk].indices.size()
        if spmv.oper.terms[kk].proj_indices.size() > max_proj:
            max_proj = spmv.oper.terms[kk].proj_indices.size()
    for kk in range(G):
        if spmv.group_offdiag_inds[kk].size() > max_offdiag:
            max_offdiag = spmv.group_offdiag_inds[kk].size()

    if max_inds > 255 or max_proj > 255 or max_offdiag > 255:
        raise ValueError("GPU term and group lengths must fit in uint8")
    if any(length > 255 for length in spmv.group_rowint_length):
        raise ValueError("GPU row integer lengths must fit in uint8")
    if any(pointer > np.iinfo(np.uint32).max for pointer in spmv.group_ladder_ptrs):
        raise ValueError("GPU ladder pointers must fit in uint32")

    max_inds = max(max_inds,    1)
    max_proj = max(max_proj,    1)
    max_offdiag = max(max_offdiag, 1)

    term_coeffs_re = np.empty(T,              dtype=np.float64)
    term_real_phase = np.empty(T,              dtype=np.int8)
    term_num_ops = np.empty(T,              dtype=np.uint8)
    term_inds = np.zeros((T, max_inds),  dtype=np.uint8)
    term_vals = np.zeros((T, max_inds),  dtype=np.uint8)
    term_num_proj = np.empty(T,              dtype=np.uint8)
    term_proj_inds = np.zeros((T, max_proj),  dtype=np.uint8)
    term_proj_bits = np.zeros((T, max_proj),  dtype=np.uint8)

    mv_term_coeffs_re = term_coeffs_re
    mv_term_real_phase = term_real_phase
    mv_term_num_ops = term_num_ops
    mv_term_inds = term_inds
    mv_term_vals = term_vals
    mv_term_num_proj = term_num_proj
    mv_term_proj_inds = term_proj_inds
    mv_term_proj_bits = term_proj_bits

    for kk in range(T):
        c = spmv.oper.terms[kk].coeff
        mv_term_coeffs_re[kk]  = c.real
        mv_term_real_phase[kk] = <np.int8_t>spmv.oper.terms[kk].real_phase
        sz = spmv.oper.terms[kk].indices.size()
        mv_term_num_ops[kk] = <np.uint8_t>sz
        for j in range(sz):
            mv_term_inds[kk, j] = <np.uint8_t>spmv.oper.terms[kk].indices[j]
            mv_term_vals[kk, j] = spmv.oper.terms[kk].values[j]
        sz = spmv.oper.terms[kk].proj_indices.size()
        mv_term_num_proj[kk] = <np.uint8_t>sz
        for j in range(sz):
            mv_term_proj_inds[kk, j] = <np.uint8_t>spmv.oper.terms[kk].proj_indices[j]
            mv_term_proj_bits[kk, j] = <np.uint8_t>spmv.oper.terms[kk].proj_bits[j]

    grp_offdiag_inds = np.zeros((G, max_offdiag), dtype=np.uint8)
    grp_offdiag_sizes = np.empty(G, dtype=np.uint8)

    mv_grp_offdiag_sizes = grp_offdiag_sizes
    mv_grp_offdiag_inds = grp_offdiag_inds

    for kk in range(G):
        sz = spmv.group_offdiag_inds[kk].size()
        mv_grp_offdiag_sizes[kk] = <np.uint8_t>sz
        for j in range(sz):
            mv_grp_offdiag_inds[kk, j] = <np.uint8_t>spmv.group_offdiag_inds[kk][j]

    return {
        "term_coeffs_re": term_coeffs_re,
        "term_real_phase": term_real_phase,
        "term_num_ops": term_num_ops,
        "term_inds": term_inds,
        "term_vals": term_vals,
        "term_num_proj": term_num_proj,
        "term_proj_inds": term_proj_inds,
        "term_proj_bits": term_proj_bits,
        "grp_offdiag_inds": grp_offdiag_inds,
        "grp_offdiag_sizes": grp_offdiag_sizes,
        "grp_rowint_length": np.asarray(spmv.group_rowint_length, dtype=np.uint8),
        "grp_ladder_ptrs": np.asarray(spmv.group_ladder_ptrs, dtype=np.uint32),
        "ladder_offset": int(spmv.ladder_offset),
        "num_groups": int(spmv.num_groups),
        "diag_vec": diag,
        "has_nonzero_diag": int(spmv.has_nonzero_diag),
    }
