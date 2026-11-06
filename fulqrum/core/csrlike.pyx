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
cimport cython
from libcpp.vector cimport vector
include "includes/csrlike_header.pxi"
include "includes/types.pxi"
import time
import numpy as np
import scipy.sparse as sp
import psutil
from ..exceptions import FulqrumError

from .constants cimport MAX_SIZE_T


from .tensor_product_cpu cimport checked_multiply, checked_add, checked_index

cdef class CSRLike():
    def __cinit__(self, size_t num_rows, unsigned int is_real=1):
        self.num_rows = num_rows
        self.is_real = is_real
        self._nnz = MAX_SIZE_T
        self.is_int64 = num_rows > np.iinfo(np.int32).max
        self.data_type = 0
        cdef size_t kk
        checked_multiply(num_rows, 2 * sizeof(vector[int]))
        if self.is_real:
            if self.is_int64:
                self.data_d64.cols.resize(self.num_rows)
                self.data_d64.data.resize(self.num_rows)
                self.data_type = 2
            else:
                self.data_d32.cols.resize(self.num_rows)
                self.data_d32.data.resize(self.num_rows)
                self.data_type = 1
        else:
            if self.is_int64:
                self.data_z64.cols.resize(self.num_rows)
                self.data_z64.data.resize(self.num_rows)
                self.data_type = 4
            else:
                self.data_z32.cols.resize(self.num_rows)
                self.data_z32.data.resize(self.num_rows)
                self.data_type = 3

    def __dealloc__(self):
        # Clear cols and data vectors upon deallocation of class
        clear_csrlike_data(self.data_d32.cols,
                           self.data_d32.data,
                           self.data_d64.cols,
                           self.data_d64.data,
                           self.data_z32.cols,
                           self.data_z32.data,
                           self.data_z64.cols,
                           self.data_z64.data)
    @property
    def shape(self):
        return (self.num_rows, self.num_rows)

    @property
    def num_rows(self):
        cdef size_t num_rows = 0
        if self.data_type == 1:
            num_rows = self.data_d32.cols.size()
        elif self.data_type == 2:
            num_rows = self.data_d64.cols.size()
        elif self.data_type == 3:
            num_rows = self.data_z32.cols.size()
        elif self.data_type == 4:
            num_rows = self.data_z64.cols.size()
        return num_rows

    @property
    def type_string(self):
        if self.data_type == 1:
            return 'd32'
        elif self.data_type == 2:
            return 'd64'
        elif self.data_type == 3:
            return 'z32'
        elif self.data_type == 4:
            return 'z64'
        else:
            raise FulqrumError('Invalid data type')

    @property
    def nnz(self):
        # If nnz is already computed do not calculate again
        if self._nnz != MAX_SIZE_T:
            return self._nnz
        cdef size_t nnz = 0
        cdef size_t kk
        if self.data_type == 1:
            for kk in range(self.num_rows):
                nnz = checked_add(nnz, self.data_d32.data[kk].size())
        elif self.data_type == 2:
            for kk in range(self.num_rows):
                nnz = checked_add(nnz, self.data_d64.data[kk].size())
        elif self.data_type == 3:
            for kk in range(self.num_rows):
                nnz = checked_add(nnz, self.data_z32.data[kk].size())
        elif self.data_type == 4:
            for kk in range(self.num_rows):
                nnz = checked_add(nnz, self.data_z64.data[kk].size())
        self._nnz = nnz
        return nnz

    def to_csr_array(self, int verbose=0):
        cdef int[::1] ptr32
        cdef int[::1] inds32
        cdef long long[::1] ptr64
        cdef long long[::1] inds64
        cdef double[::1] real_data
        cdef complex[::1] complex_data

        checked_index[int64](checked_add(self.num_rows, 1))
        cdef size_t nnz = self.nnz
        cdef object mat
        cdef double stop, start = time.perf_counter()
        # break early if no elements
        if nnz == 0:
            _dtype=np.int64 if self.is_int64 else np.int32
            stop = time.perf_counter()
            if verbose:
                print(f'CSR Conversion time: {round(stop-start, 3)}')
            return  sp.csr_array((np.empty(0, dtype=_dtype), np.empty(0, dtype=_dtype), np.zeros(checked_add(self.num_rows, 1), dtype=_dtype)),
                                shape=(self.num_rows,)*2, dtype=float if self.is_real else complex)

        cdef int data_size
        cdef size_t total_bytes
        # does output CSR matrix require int64 indices?
        cdef size_t max_int = np.iinfo(np.int32).max
        cdef int needs_int64 = (checked_add(self.num_rows, 1)) > max_int or nnz > max_int
        if self.is_real:
            data_size = 8 # size of double
        else:
            data_size = 16 # size of double complex
        # Compute the CSR array sizes
        if needs_int64:
            # indptr + indices + data sizes
            total_bytes = checked_add(checked_multiply(checked_add(self.num_rows, 1), 8), checked_multiply(nnz, 8 + data_size))
        else:
            total_bytes = checked_add(checked_multiply(checked_add(self.num_rows, 1), 4), checked_multiply(nnz, 4 + data_size))

        if psutil.virtual_memory().available < total_bytes:
            raise FulqrumError(f"Sparse matrix copy of size {round(total_bytes/(1024**2), 3)} Mb does not fit within available memory.")

        if '32' in self.type_string and not needs_int64:
            ptr32 = np.zeros(checked_add(self.num_rows, 1), dtype=np.int32)
            inds32 = np.empty(nnz, dtype=np.int32)
        else:
            ptr64 = np.zeros(checked_add(self.num_rows, 1), dtype=np.int64)
            inds64 = np.empty(nnz, dtype=np.int64)

        if 'd' in self.type_string:
            real_data = np.empty(nnz, dtype=float)
        else:
            complex_data = np.empty(nnz, dtype=complex)

        if self.type_string == 'd32':
            if not needs_int64:
                set_csr_ptr(self.data_d32.cols, &ptr32[0])
                set_csr_data(self.data_d32.data, self.data_d32.cols, &ptr32[0], &inds32[0], &real_data[0])

                mat = sp.csr_array((real_data, inds32, ptr32),
                                    shape=(self.num_rows,)*2, dtype=float)
            else:
                set_csr_ptr(self.data_d32.cols, &ptr64[0])
                set_csr_data(self.data_d32.data, self.data_d32.cols, &ptr64[0], &inds64[0], &real_data[0])
                mat = sp.csr_array((real_data, inds64, ptr64),
                                    shape=(self.num_rows,)*2, dtype=float)

        elif self.type_string == 'd64':
            set_csr_ptr(self.data_d64.cols, &ptr64[0])
            set_csr_data(self.data_d64.data, self.data_d64.cols, &ptr64[0], &inds64[0], &real_data[0])

            mat = sp.csr_array((real_data, inds64, ptr64),
                                shape=(self.num_rows,)*2, dtype=float)

        elif self.type_string == 'z32':
            if not needs_int64:
                set_csr_ptr(self.data_z32.cols, &ptr32[0])
                set_csr_data(self.data_z32.data, self.data_z32.cols, &ptr32[0], &inds32[0], &complex_data[0])

                mat = sp.csr_array((complex_data, inds32, ptr32),
                                   shape=(self.num_rows,)*2, dtype=complex)
            else:
                set_csr_ptr(self.data_z32.cols, &ptr64[0])
                set_csr_data(self.data_z32.data, self.data_z32.cols, &ptr64[0], &inds64[0], &complex_data[0])

                mat = sp.csr_array((complex_data, inds64, ptr64),
                                   shape=(self.num_rows,)*2, dtype=complex)

        elif self.type_string == 'z64':
            set_csr_ptr(self.data_z64.cols, &ptr64[0])
            set_csr_data(self.data_z64.data, self.data_z64.cols, &ptr64[0], &inds64[0], &complex_data[0])

            mat = sp.csr_array((complex_data, inds64, ptr64),
                                shape=(self.num_rows,)*2, dtype=complex)

        stop = time.perf_counter()
        if verbose:
            print(f'CSR Conversion time: {round(stop-start, 3)}')
        return mat

    def matvec(self,  double_or_complex[::1] x):
        if <size_t>x.shape[0] != self.num_rows:
            raise FulqrumError('Incorrect length of input vector.')

        cdef double_or_complex[::1] out
        if self.is_real:
            out = np.zeros(x.shape[0], dtype=float)
        else:
            out = np.zeros(x.shape[0], dtype=complex)

        if self.type_string == 'd32':
            if double_or_complex is double: #This is here to allow for us to specialize type
                csrlike_spmv(self.data_d32.data, self.data_d32.cols, &x[0], &out[0], <int>self.num_rows)

        elif self.type_string == 'd64':
            if double_or_complex is double:
                csrlike_spmv(self.data_d64.data, self.data_d64.cols, &x[0], &out[0], <long long>self.num_rows)

        if self.type_string == 'z32':
            if double_or_complex is complex:
                csrlike_spmv(self.data_z32.data, self.data_z32.cols, &x[0], &out[0], <int>self.num_rows)

        elif self.type_string == 'z64':
            if double_or_complex is complex:
                csrlike_spmv(self.data_z64.data, self.data_z64.cols, &x[0], &out[0], <long long>self.num_rows)

        return np.asarray(out)
