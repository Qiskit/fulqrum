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

"""PySCF conversion utilities"""
from libcpp.vector cimport vector
from libc.string cimport memcpy
from libcpp cimport bool
from pathlib import Path
import time
import numpy as np
cimport numpy as np
import logging
logger = logging.getLogger(__name__)


cdef class FCIDumpData():
    def __cinit__(self, string filename):
        self.data = parse_fcidump(filename)

    def __repr__(self):

        cdef str data_str = ''
        data_str += f'NORB={self.data.NORB}'
        data_str += f', NELEC={self.data.NELEC}'
        data_str += f', ISYM={self.data.ISYM}'
        data_str += f', MS2={self.data.MS2}'
        data_str += f', UHF={self.data.UHF}'
        data_str += f', ECORE={self.data.ECORE}'
        return f"<FCIDumpData[{data_str}]>"

    @property
    def NORB(self):
        return self.data.NORB

    @property
    def NELEC(self):
        return self.data.NELEC
    
    @property
    def ISYM(self):
        return self.data.ISYM
    
    @property
    def MS2(self):
        return self.data.MS2

    @property
    def UHF(self):
        return self.data.UHF

    @property
    def ECORE(self):
        return self.data.ECORE

    @property
    def H1(self):
        cdef double[::1] arr = <double [:self.data.H1.size()]>self.data.H1.data()
        return np.asarray(arr)

    @property
    def H2(self):
        cdef double[::1] arr = <double [:self.data.H2.size()]>self.data.H2.data()
        return np.asarray(arr)

    @property
    def ORBSYM(self):
        cdef int[::1] arr = <int [:self.data.ORBSYM.size()]>self.data.ORBSYM.data()
        return np.asarray(arr)

    
    def one_body_integrals(self):
        """One body integrals:

        Returns:
            ndarray: NumPy array of integral values
        """
        cdef int norb = self.data.NORB
        cdef int norb2 = norb * norb
        cdef double[::1] out = np.zeros(norb2, dtype=float)
        memcpy(&out[0], self.data.H1.data(), norb2*sizeof(double))
        return np.asarray(out)
    
    
    def two_body_integrals(self, bool permute=0):
        """Two body integrals:

        Parameters:
            permute (bool): Permute indices to Fulqrum convention

        Returns:
            ndarray: NumPy array of integral values
        """
        cdef int norb = self.data.NORB
        cdef double[::1] out = np.zeros(norb * norb * norb * norb, dtype=float)
        self.data.two_body_integrals_to_ptr(&out[0], norb, permute)
        return np.asarray(out)


def read_fcidump(filename: str | Path):
    """Read an fcidump file

    Parameters:
        filename : Path to input fcidump file

    Returns:
        FCIDumpData : Data file
    """
    cdef string string_name = str(filename)
    cdef FCIDumpData out = FCIDumpData(string_name)
    return out
