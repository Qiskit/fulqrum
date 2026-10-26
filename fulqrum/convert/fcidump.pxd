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

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "./src/fcidump.hpp":
    
    ctypedef struct FCIDumpData_t:
        vector[double] H1
        vector[double] H2
        vector[int] ORBSYM
        double ECORE
        int NORB
        int NELEC
        int MS2
        int ISYM
        bool UHF
        void two_body_integrals_to_ptr(double * out, int norb, bool permute)


    FCIDumpData_t parse_fcidump(string& filename) except +

cdef class FCIDumpData():
    cdef FCIDumpData_t data

