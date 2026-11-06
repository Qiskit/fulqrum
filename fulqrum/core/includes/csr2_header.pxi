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
from libcpp.vector cimport vector
from ..core.element_evaluator cimport TermMasks
from ..core.bitset cimport bitset_t
from ..core.bitset_hashmap cimport BitsetHashMapWrapper
from .constants cimport width_t
include "base_header.pxi"


cdef extern from "../src/csr2.hpp":

    void csr_matrix_builder2[T, U](const vector[OperatorTerm_t]& terms,
                              const BitsetHashMapWrapper& subspace,
                              const U * diag_vec,
                              width_t width,
                              size_t subspace_dim,
                              int has_nonzero_diag,
                              const size_t * group_ptrs,
                              const size_t * group_ladder_ptrs,
                              width_t * group_rowint_length,
                              const vector[vector[width_t]]& group_offdiag_inds,
                              size_t num_groups,
                              unsigned int ladder_offset,
                              T * indptr,
                              T * indices,
                              U * data,
                              int compute_values,
                              const TermMasks* prepared_masks) except + nogil
