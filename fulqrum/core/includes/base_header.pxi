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

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp cimport bool
from ..core.bitset cimport bitset_t
from ..core.bitset_hashmap cimport BitsetHashMapWrapper
from ..core.constants cimport width_t
import numpy as np
cimport numpy as np 


cdef extern from "../src/base.hpp":
    ctypedef struct OperatorTerm_t:
        double complex coeff
        vector[width_t] indices
        vector[unsigned char] values
        vector[width_t] proj_indices
        vector[width_t] proj_bits
        width_t offdiag_weight
        np.int8_t real_phase
        int group
        unsigned int offdiag_structure
        unsigned int proj_structure
        void sort_term_data()
        void set_proj_indices()
        OperatorTerm_t()
        OperatorTerm_t(complex) except +


    ctypedef struct QubitOperator_t:
        width_t width
        vector[OperatorTerm_t] terms
        int sorted
        int type
        width_t ladder_width
        int weight_sorted
        int off_weight_sorted
        int ladder_sorted
        int structure_sorted
        QubitOperator_t()
        QubitOperator_t(width_t)
        size_t size()
        bool is_real()
        bool is_diagonal()
        QubitOperator_t copy()
        QubitOperator_t& weight_sort()
        QubitOperator_t& offdiag_weight_sort()
        QubitOperator_t& group_sort()
        vector[int] groups()
        vector[size_t] group_ptrs()
        QubitOperator_t& group_term_sort_by_ladder_int(width_t)
        QubitOperator_t combine_repeat_terms(double)
        vector[size_t] offdiag_weight_ptrs()
        QubitOperator_t& from_label(string)
        double constant_energy()
        QubitOperator_t remove_constant_terms()
        pair[QubitOperator_t, QubitOperator_t] split_diagonal()
        QubitOperator_t terms_by_group(int)
        vector[np.int8_t] real_phases()
        vector[complex] coefficients()
        vector[int] extended_terms()
        vector[width_t] ladder_integers()
        vector[width_t] group_ladder_int_bit_lengths()
        vector[size_t] group_ladder_int_ptrs()
        void to_json(string, bool) except +
        QubitOperator_t from_json(string) except +


    ctypedef struct Subspace_t:
        BitsetHashMapWrapper bitstrings
        width_t num_qubits
        size_t size
        width_t alpha_width
        bool product
        bool map_materialized
        size_t map_build_count
        vector[size_t] full_keys
        void configure(int, bool) except +
        void initialize_full(const vector[string]&) except +
        void initialize_product(const vector[string]&, const vector[string]&) except +
        bitset_t key_at(size_t) except +
        void load_key(size_t, bitset_t&) except +
        void ensure_hashmap() except +
        void prepare_mutation() except +
        void copy_to(Subspace_t&) except +
        string half_string(int, size_t) except +
        size_t half_word(int, size_t) except +
        size_t half_size(int) except +


    ctypedef struct FermionicTerm_t:
        double complex coeff
        vector[width_t] indices
        vector[unsigned char] values
        vector[width_t] proj_indices
        vector[width_t] proj_bits
        width_t offdiag_weight
        unsigned int offdiag_structure
        unsigned int proj_structure
        void insertion_sort()


    ctypedef struct FermionicOperator_t:
        width_t width
        vector[FermionicTerm_t] terms
        size_t size()
        int weight_sorted
        void to_json(string, bool) except +
        FermionicOperator_t copy()
        FermionicOperator_t from_json(string) except +
        FermionicOperator_t& combine_repeat_indices() nogil
        FermionicOperator_t weight_sort() nogil
        FermionicOperator_t offdiag_structure_sort() nogil
        FermionicOperator_t& combine_repeat_terms(double)
        QubitOperator_t& extended_jw_transformation() nogil
        vector[size_t] offdiag_structure_ptrs() nogil


    size_t max_offdiag_ptr_size(vector[size_t]&)

    void set_group_offdiag_indices(vector[OperatorTerm_t]& terms,
                                   vector[vector[width_t]]& group_indices,
                                   size_t* group_ptrs,
                                   size_t num_groups)

    OperatorTerm_t& set_proj_indices(OperatorTerm_t&)
    void set_offdiag_weight_phase_struct(OperatorTerm_t& term) nogil
    void set_extended_flag(OperatorTerm_t& term) nogil


cdef extern from "../src/qubit_oper.hpp":
    void check_ladder_bins(const vector[OperatorTerm_t]& terms, const size_t* group_ptrs, size_t num_groups,
                           const size_t* ladder_ptrs, size_t num_ladder_ptrs, size_t ladder_offset) except +
