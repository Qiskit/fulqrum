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
# cython: c_string_type=unicode, c_string_encoding=UTF-8

from ..core.qubit_operator cimport QubitOperator
from ..core.subspace cimport Subspace
from ..core.bitset cimport Bitset
from ..core.bitset cimport bitset_t
from ..core.linear_operator import SubspaceHamiltonian
from ..core.spmv cimport FulqrumSpMV
from ..exceptions import FulqrumError

include "includes/open_header.pxi"

import numpy as np



def ramps_open(object Hsub, Subspace target_subspace, double target_energy,
               unsigned int max_recursion=1, double tol=1e-12):
    """Unrestricted RAMPS around the input target subspace.

    The resultant subspace will be the target subspace with addtional bit-strings
    added that perturbatively affect the energy more than the tolerance value

    Parameters:
        Hsub (SubspaceHamiltonian): Hamiltonian
        target_subspace (Subspace): Target subspace to expand around
        target_energy (double): Target energy from target subspace
        max_recursion (int): Optional, maximum number of recursions to perform, default=1
        tol (double): Optional, tolerance value for truncation, default=1e-12

    Returns:
        Subspace: RAMPS constructed subspace
    """
    if not isinstance(Hsub, SubspaceHamiltonian):
        raise FulqrumError("Input Hamiltonian must be a SubspaceHamiltonian object")
    cdef Subspace out = target_subspace.copy()
    cdef FulqrumSpMV spmv = Hsub.spmv
    out.subspace.prepare_mutation()
    cdef double energy = open_ramps(spmv.oper,
                                    out.subspace.bitstrings,
                                    spmv.diag_oper,
                                    spmv.width,
                                    spmv.has_nonzero_diag,
                                    &spmv.group_ptrs[0],
                                    &spmv.group_ladder_ptrs[0],
                                    &spmv.group_rowint_length[0],
                                    spmv.group_offdiag_inds,
                                    spmv.num_groups,
                                    spmv.ladder_offset,
                                    target_energy,
                                    max_recursion,
                                    tol,
                                    &spmv.term_masks,
                                    &spmv.diag_masks)


    out.subspace.size = out.subspace.bitstrings.size()

    # This is a temp workaround for issues with iteratively expanded
    # subspaces.  This should be removed once those are resolved
    cdef list temp_out = []
    cdef size_t n
    cdef str bs
    for n in range(<size_t>out.size()):
        bs = out.get_n_th_bitstring(n)
        temp_out.append(bs)

    cdef Subspace final_out = Subspace([temp_out])

    return final_out
