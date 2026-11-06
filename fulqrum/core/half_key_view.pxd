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
from libcpp cimport bool
from .subspace cimport Subspace_t
from .bitset cimport bitset_t
from .constants cimport width_t

cdef extern from "src/subspace.hpp":
    cdef cppclass HalfKeyView:
        HalfKeyView(const Subspace_t&) except +
        width_t width()
        width_t split_bit()
        bool is_tensor_product()
        size_t size()
        size_t num_words()
        size_t num_half_keys(int) except +
        size_t half_words(int)
        const size_t* half_key(int, size_t) except +
        size_t find_half_key(int, const size_t*) except +
        bitset_t key(size_t) except +
        void load_key(size_t, bitset_t&) except +
