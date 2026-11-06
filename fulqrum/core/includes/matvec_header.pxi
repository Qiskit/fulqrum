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

cdef extern from "../src/matvec.hpp":
    void omp_matvec[T](vector[OperatorTerm_t]& terms,
                const BitsetHashMapWrapper& subspace,
                T * diag_vec,
                width_t width,
                size_t subspace_dim,
                int has_nonzero_diag,
                size_t * group_ptrs,
                const vector[vector[width_t]]& group_offdiag_inds,
                size_t num_groups,
                const T * in_vec,
                T * out_vec,
                const TermMasks* prepared_masks) except + nogil
