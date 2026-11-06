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
from libcpp.pair cimport pair
from ..core.qubit_operator cimport QubitOperator_t
from ..core.bitset cimport bitset_t
from ..core.bitset_hashmap cimport BitsetHashMapWrapper

include "base_header.pxi"

cdef extern from "../src/diag.hpp":
    void compute_diag_vector[T](const BitsetHashMapWrapper& data,
                                T * diag_vec,
                                const QubitOperator_t& diag_oper,
                                const size_t subspace_dim,
                                const TermMasks* prepared_masks) except + nogil


    bool fast_diag_compatible(QubitOperator_t&) except + nogil

    void fast_diag_term_sort(QubitOperator_t& oper) except + nogil

    void compute_diag_vector_fast[T](const BitsetHashMapWrapper& data,
                                     T * diag_vec,
                                     const QubitOperator_t& diag_oper,
                                     const size_t subspace_dim,
                                const TermMasks* prepared_masks) except + nogil

    size_t count_diag_cross_terms(const QubitOperator_t& diag_oper,
                                  const unsigned int num_alpha) nogil

    void compute_diag_factors[T](const unsigned long long* alpha_uint,
                                 const unsigned long long* beta_uint,
                                 const QubitOperator_t& diag_oper,
                                 const unsigned int num_alpha,
                                 const size_t a0,
                                 const size_t a1,
                                 const size_t n_b,
                                 T* d_alpha_out,
                                 T* d_beta_out,
                                 T* Fc_out,
                                 T* G_out) except + nogil

    bool compute_diag_cross_factors[T](const QubitOperator_t& diag_oper,
                                       const unsigned int num_alpha,
                                       const unsigned int num_beta,
                                       T* V_out,
                                       T* da_lin_out,
                                       T* db_lin_out,
                                       T* const_out) except + nogil
