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
from libcpp.vector cimport vector
from libcpp.complex cimport complex as cpp_complex
from .qubit_operator cimport OperatorTerm_t

cdef extern from "src/element_evaluator.hpp":
    cdef cppclass TermMasks:
        size_t num_words
        vector[size_t] words
        vector[double] real_values
        vector[cpp_complex[double]] complex_values

    TermMasks build_term_masks(const vector[OperatorTerm_t]& terms, size_t num_words) except + nogil
