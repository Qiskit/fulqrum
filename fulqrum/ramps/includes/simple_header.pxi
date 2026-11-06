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
from ..core.constants cimport width_t

include "../../core/includes/base_header.pxi"


cdef extern from "../src/simple.hpp":

    double simple_restricted(const QubitOperator_t& oper,
                             const BitsetHashMapWrapper &restricted_subspace,
                             BitsetHashMapWrapper& out_subspace,
                             QubitOperator_t& diag_oper,
                             const width_t width,
                             const size_t subspace_dim,
                             const int has_nonzero_diag,
                             const size_t * group_ptrs,
                             const size_t * group_ladder_ptrs,
                             width_t * group_rowint_length,
                             vector[vector[width_t]]& group_offdiag_inds,
                             const size_t num_groups,
                             const unsigned int ladder_offset,
                             const double target_energy,
                             const unsigned int max_recursion,
                             const double tol,
                             const TermMasks* prepared_masks,
                             const TermMasks* prepared_diag_masks) except + nogil
