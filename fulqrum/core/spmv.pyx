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
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.string cimport memcpy
from libc.math cimport floor, sqrt

from .qubit_operator cimport QubitOperator
from .subspace cimport Subspace
from .element_evaluator cimport build_term_masks
from .tensor_product_cpu cimport (TensorProductCPU, OperatorTables, CrossMode, tensor_product_cpu_operator_tables, build_tensor_product_cpu, tensor_product_cpu_apply, tensor_product_cpu_csr, tensor_product_cpu_csrlike, tensor_product_cpu_diagonal, tensor_product_cpu_residual_groups, tensor_product_cpu_tp_ready, tensor_product_cpu_cross_formed, checked_multiply, checked_add, checked_index)
from cython.operator cimport dereference as deref
from .csrlike cimport CSRLike
from .constants cimport width_t
from .constants import np_width_t
from ..exceptions import FulqrumError


from cython.parallel cimport prange, parallel
import time
import numpy as np
import scipy.sparse as sp
import psutil

import logging
logger = logging.getLogger(__name__)

cimport numpy as np
np.import_array()

include "includes/base_header.pxi"
include "includes/elements_header.pxi"
include "includes/diag_header.pxi"
include "includes/matvec_header.pxi"
include "includes/matvec2_header.pxi"
include "includes/csr_header.pxi"
include "includes/csr_utils_header.pxi"
include "includes/csr2_header.pxi"
include "includes/csrlike_builder_header.pxi"
include "includes/csrlike_builder2_header.pxi"


cdef width_check(width_t ham_width, width_t sub_width):
    """Validate that Hamiltonian width is the same as the subspace width
    """
    if ham_width != sub_width:
        raise FulqrumError(f"Hamiltonian width ({ham_width}) does not match subspace width ({sub_width})")


cdef class FulqrumSpMV():
    def __cinit__(self, QubitOperator diag_hamiltonian,
                  double const_energy,
                  QubitOperator hamiltonian,
                  Subspace subspace,
                  size_t[::1]& group_ptrs,
                  size_t[::1]& group_ladder_ptrs):

        self.real_tensor_product_cpu = NULL
        self.complex_tensor_product_cpu = NULL
        self._tensor_product_cpu_built = False
        logger.info("Initializing FulqrumSpMV")
        spmv_start = time.perf_counter()
        cdef size_t kk
        self.diag_oper = diag_hamiltonian.oper
        self.const_energy = const_energy
        self.oper = hamiltonian.oper
        self.is_real = diag_hamiltonian.is_real() * hamiltonian.is_real()
        self.subspace = subspace
        self.width = self.oper.width
        self.subspace_dim = self.subspace.size()
        self.num_terms = self.oper.terms.size()
        self.num_diag_terms = self.diag_oper.terms.size()
        self.group_ptrs = group_ptrs
        self.group_ladder_ptrs = group_ladder_ptrs
        self.num_groups = group_ptrs.shape[0] - 1
        self.fast_diag = False
        self._disable_fast_diag = False
        if group_ptrs.shape[0] > 1:
            set_group_offdiag_indices(self.oper.terms, self.group_offdiag_inds,
                                      &self.group_ptrs[0], self.num_groups)

        # Log is operator is real or not
        if self.is_real:
            logger.info("Operator is real")
        else:
            logger.info("Operator is complex")
        
        if self.oper.type == 2:
            self.ladder_offset = 2**self.oper.ladder_width
            logger.info("Operator type = 2")
            self.fast_diag = fast_diag_compatible(self.diag_oper)
            if self.oper.terms.size():
                self.group_rowint_length = hamiltonian.group_rowint_length()
                self.ladder_offset = 2**self.oper.ladder_width
            else:
                # Need to set memoryview but not used
                self.group_rowint_length = np.zeros(1, dtype=np_width_t)
            if self.fast_diag:
                fast_diag_term_sort(self.diag_oper)
        else:
            logger.info("Operator type = 1")
        if self.diag_oper.terms.size() > 0 or self.const_energy:
            self.has_nonzero_diag = 1
             # Init diagonal memoryview to None because
             # we only build it when needed
            self.complex_diag_vec = None
        else:
            self.has_nonzero_diag = 0
        self.init_diag = 0
        # We have to init something here otherwise
        # grabbing a pointer to the data is going to complain
        self.real_diag_vec = np.empty(shape=(1,), dtype=float)
        self.complex_diag_vec = np.empty(shape=(1,), dtype=complex)
        self.term_masks = build_term_masks(self.oper.terms, (self.width + 8 * sizeof(size_t) - 1) // (8 * sizeof(size_t)))
        self.diag_masks = build_term_masks(self.diag_oper.terms, (self.width + 8 * sizeof(size_t) - 1) // (8 * sizeof(size_t)))
        spmv_stop = time.perf_counter()
        logger.info("FulqrumSpMV total init time: %s ms", round((spmv_stop - spmv_start)*1000, 3))

    def __dealloc__(self):
        if self.real_tensor_product_cpu != NULL:
            del self.real_tensor_product_cpu
        if self.complex_tensor_product_cpu != NULL:
            del self.complex_tensor_product_cpu

    cdef void _prepare_gpu_tables(self) except *:
        width_check(self.width, self.subspace.width)
        if self.tables.num_groups() != self.num_groups or not self.tables.layout().valid:
            self.tables = tensor_product_cpu_operator_tables(self.oper, self.diag_oper, &self.group_ptrs[0], self.group_offdiag_inds, self.subspace.subspace, False, False)

    cdef bint _prepare_tensor_product_cpu(self) except *:
        if self.oper.type != 2 or not self.subspace.subspace.product:
            return False
        width_check(self.width, self.subspace.width)
        if self._tensor_product_cpu_built:
            return True
        self.tables = tensor_product_cpu_operator_tables(self.oper, self.diag_oper, &self.group_ptrs[0], self.group_offdiag_inds, self.subspace.subspace, False, False)
        if self.is_real:
            self.real_tensor_product_cpu = build_tensor_product_cpu[double](self.tables, &self.group_ptrs[0], self.group_offdiag_inds, self.term_masks, self.subspace.subspace, CrossMode.automatic, False, False)
        else:
            self.complex_tensor_product_cpu = build_tensor_product_cpu[complex](self.tables, &self.group_ptrs[0], self.group_offdiag_inds, self.term_masks, self.subspace.subspace, CrossMode.automatic, False, False)
        self._tensor_product_cpu_built = True
        logger.info("CPU product route: %s", self._cpu_diagnostics())
        return True

    def _cpu_diagnostics(self):
        if not self._prepare_tensor_product_cpu():
            return {"route": "retained", "tp_groups": 0, "residual_groups": 0,
                    "map_materialized": self.subspace.subspace.map_materialized}
        cdef size_t served, residual, allocated
        cdef bool ready, formed
        if self.is_real:
            served = self.real_tensor_product_cpu.tp_groups
            residual = tensor_product_cpu_residual_groups[double](deref(self.real_tensor_product_cpu))
            allocated = self.real_tensor_product_cpu.bytes
            ready = tensor_product_cpu_tp_ready[double](deref(self.real_tensor_product_cpu))
            formed = tensor_product_cpu_cross_formed[double](deref(self.real_tensor_product_cpu))
        else:
            served = self.complex_tensor_product_cpu.tp_groups
            residual = tensor_product_cpu_residual_groups[complex](deref(self.complex_tensor_product_cpu))
            allocated = self.complex_tensor_product_cpu.bytes
            ready = tensor_product_cpu_tp_ready[complex](deref(self.complex_tensor_product_cpu))
            formed = tensor_product_cpu_cross_formed[complex](deref(self.complex_tensor_product_cpu))
        return {"route": "tp+residual" if served else "residual", "tp_groups": served,
                "residual_groups": residual, "tp_ready": ready, "cross_formed": formed,
                "engine_bytes": allocated, "map_materialized": self.subspace.subspace.map_materialized}

    def check_ladder_bins(self):
        """Raises ValueError when the ladder bins do not match the off-diagonal terms.

        RAMPS and the GPU term tables index terms through the bins, so their users
        check them first.
        """
        if self.num_groups == 0:
            return
        check_ladder_bins(self.oper.terms, &self.group_ptrs[0], self.num_groups,
                          &self.group_ladder_ptrs[0], self.group_ladder_ptrs.shape[0], self.ladder_offset)

    def __repr__(self):
        out = f"<FulqrumSpMV(width={self.width}, "
        out += f"num_op_terms={self.num_terms+self.num_diag_terms}({self.num_terms}/{self.num_diag_terms}), "
        out += f"subspace_dim={self.subspace_dim}, "
        out += f"is_real={self.is_real}>"
        return out

    cpdef void update_subspace(self, Subspace subspace):
        """Update the subspace attached to the object
        """
        if self.real_tensor_product_cpu != NULL:
            del self.real_tensor_product_cpu
            self.real_tensor_product_cpu = NULL
        if self.complex_tensor_product_cpu != NULL:
            del self.complex_tensor_product_cpu
            self.complex_tensor_product_cpu = NULL
        self._tensor_product_cpu_built = False
        self.tables = OperatorTables()
        self.subspace = subspace
        self.subspace_dim = self.subspace.size()
        self.init_diag = 0
        self.real_diag_vec = np.empty(shape=(1,), dtype=float)
        self.complex_diag_vec = np.empty(shape=(1,), dtype=complex)

    cdef _compute_diag_via_blas(self,
                                np.uint64_t[::1] alpha_uint,
                                np.uint64_t[::1] beta_uint,
                                size_t a0, # start: alpha str index
                                size_t a1, # end: alpha str index
                                unsigned int num_alpha_qubits):
        """Half-string diagonal slice via factored form + numpy BLAS."""
        if self.is_real != 1:
            raise ValueError("GPU diagonal helpers require real values")
        if not 0 < num_alpha_qubits < self.width or num_alpha_qubits > 64 or self.width - num_alpha_qubits > 64:
            raise ValueError("GPU diagonal helpers require 1 to 64 qubits per sector")
        if a0 > a1 or a1 > <size_t>alpha_uint.shape[0]:
            raise ValueError("Invalid alpha string range")
        cdef size_t na_local = a1 - a0
        cdef size_t n_b = beta_uint.shape[0]
        cdef size_t n_cross
        cdef double[::1] mv_d_alpha
        cdef double[::1] mv_d_beta
        cdef double[:, ::1] mv_Fc
        cdef double[:, ::1] mv_G
        cdef double* Fc_ptr = NULL
        cdef double* G_ptr = NULL

        if na_local == 0 or n_b == 0:
            return np.zeros(checked_multiply(na_local, n_b), dtype=np.float64)

        n_cross = count_diag_cross_terms(self.diag_oper, num_alpha_qubits)
        checked_multiply(checked_multiply(na_local, n_b), sizeof(double))
        checked_multiply(checked_multiply(na_local, n_cross), sizeof(double))
        checked_multiply(checked_multiply(n_b, n_cross), sizeof(double))

        d_alpha_arr = np.empty(na_local, dtype=np.float64)
        d_beta_arr  = np.empty(n_b, dtype=np.float64)
        if n_cross > 0:
            Fc_arr = np.empty((na_local, n_cross), dtype=np.float64)
            G_arr  = np.empty((n_b, n_cross),       dtype=np.float64)
        else:
            Fc_arr = None
            G_arr  = None

        mv_d_alpha = d_alpha_arr
        mv_d_beta  = d_beta_arr
        if n_cross > 0:
            mv_Fc = Fc_arr
            mv_G  = G_arr
            Fc_ptr = &mv_Fc[0, 0]
            G_ptr  = &mv_G[0, 0]

        with nogil:
            compute_diag_factors[double](
                <const unsigned long long*>&alpha_uint[0],
                <const unsigned long long*>&beta_uint[0],
                self.diag_oper,
                num_alpha_qubits, a0, a1, n_b,
                &mv_d_alpha[0], &mv_d_beta[0],
                Fc_ptr, G_ptr)

        if n_cross > 0:
            out_2d = G_arr @ Fc_arr.T   # (n_b, na_local)
        else:
            out_2d = np.zeros((n_b, na_local), dtype=np.float64)
        out_2d += d_beta_arr[:, None]
        out_2d += d_alpha_arr[None, :]
        return out_2d.ravel()

    @cython.boundscheck(False)
    cpdef int compute_diag_vector(self):
        if self.init_diag:
            return 0
        diag_start = time.perf_counter()
        width_check(self.width, self.subspace.width)    
        cdef np.uint64_t[::1] alpha_uint
        cdef np.uint64_t[::1] beta_uint
        cdef bool fast_diag = self.fast_diag
        if self._prepare_tensor_product_cpu():
            checked_multiply(self.subspace_dim, 8 if self.is_real else 16)
            if self.is_real and not self._disable_fast_diag and self.subspace.num_alpha_qubits <= 64 and self.subspace.width - self.subspace.num_alpha_qubits <= 64:
                alpha_arr, beta_arr = self.subspace.to_alpha_beta_arrays()
                alpha_uint = alpha_arr
                beta_uint = beta_arr
                self.real_diag_vec = self._compute_diag_via_blas(alpha_uint, beta_uint, 0, alpha_uint.shape[0], self.subspace.num_alpha_qubits)
                if self.const_energy:
                    np.asarray(self.real_diag_vec)[:] += self.const_energy
            elif self.is_real:
                self.real_diag_vec = np.full(self.subspace_dim, self.const_energy, dtype=float)
                tensor_product_cpu_diagonal(self.subspace.subspace, self.diag_masks, fast_diag and not self._disable_fast_diag, &self.real_diag_vec[0])
            else:
                self.complex_diag_vec = np.full(self.subspace_dim, self.const_energy, dtype=complex)
                tensor_product_cpu_diagonal(self.subspace.subspace, self.diag_masks, fast_diag and not self._disable_fast_diag, &self.complex_diag_vec[0])
            self.init_diag = 1
            return 1
        self.subspace.subspace.ensure_hashmap()
        if self._disable_fast_diag:
            fast_diag = 0
        if self.is_real:
            self.real_diag_vec = np.full(self.subspace_dim, self.const_energy, dtype=float)
            if fast_diag:
                compute_diag_vector_fast(self.subspace.subspace.bitstrings,
                                &self.real_diag_vec[0],
                                self.diag_oper,
                                self.subspace_dim,
                                &self.diag_masks)
            else:
                compute_diag_vector(self.subspace.subspace.bitstrings,
                                &self.real_diag_vec[0],
                                self.diag_oper,
                                self.subspace_dim,
                                &self.diag_masks)
        else:
            self.complex_diag_vec = np.full(self.subspace_dim, self.const_energy, dtype=complex)
            if fast_diag:
                compute_diag_vector_fast(self.subspace.subspace.bitstrings,
                                &self.complex_diag_vec[0],
                                self.diag_oper,
                                self.subspace_dim,
                                &self.diag_masks)
            else:
                compute_diag_vector(self.subspace.subspace.bitstrings,
                                &self.complex_diag_vec[0],
                                self.diag_oper,
                                self.subspace_dim,
                                &self.diag_masks)
        self.init_diag = 1
        diag_stop = time.perf_counter()
        logger.info("Diagonal vector build time: %s ms", round((diag_stop - diag_start)*1000, 3))
        return 1

    def fast_diag_compatible(self):
        return fast_diag_compatible(self.diag_oper)
    
    def diagonal_vector(self, bool disable_fast_mode=False):
        """Diagonal vector of subspace Hamitlonian

        Returns:
            ndarray: Array of complex numbers representing diagonal
        """
        width_check(self.width, self.subspace.width)
        if not self.has_nonzero_diag:
            if self.is_real:
                return np.full(self.subspace_dim, self.const_energy, dtype=float)
            else:
                return np.full(self.subspace_dim, self.const_energy, dtype=complex)
        cdef bool temp = self._disable_fast_diag
        self._disable_fast_diag = disable_fast_mode
        self.compute_diag_vector()
        self._disable_fast_diag = temp
        if self.is_real:
            return np.asarray(self.real_diag_vec)
        return np.asarray(self.complex_diag_vec)


    def minimum_diagonal_energy(self):
        """Return the minimum diagonal energy

        Returns:
            double: Lowest energy value
        """
        return <double>(self.diagonal_vector().min())


    def matvec(self, double_or_complex[::1] x):
        """Matrix-free implementation of SpMV for subspace Hamiltonian

        Parameters:
            x (ndarray): input array

        Returns:
            ndarray: Complex output vector after SpMV on input vector
        """
        width_check(self.width, self.subspace.width)
        if not self._prepare_tensor_product_cpu():
            self.subspace.subspace.ensure_hashmap()
        else:
            checked_multiply(self.subspace_dim, 8 if self.is_real else 16)
        if <size_t>x.shape[0] != self.subspace_dim:
            raise Exception('Incorrect length of input vector.')
        # generate diagonal vector if we have not done so already
        if self.has_nonzero_diag:
            self.compute_diag_vector()

        cdef double_or_complex[::1] out
        if self.is_real:
            out = np.zeros(x.shape[0], dtype=float)
        else:
            out = np.zeros(x.shape[0], dtype=complex)
        if self.oper.type == 2:
            if self.is_real:
                if double_or_complex is double:
                    if self._prepare_tensor_product_cpu():
                        tensor_product_cpu_apply(deref(self.real_tensor_product_cpu), self.subspace.subspace, &self.real_diag_vec[0], self.has_nonzero_diag, &x[0], &out[0])
                    else:
                        omp_matvec2[double](self.oper.terms,
                                self.subspace.subspace.bitstrings,
                                &self.real_diag_vec[0],
                                self.width,
                                self.subspace_dim,
                                self.has_nonzero_diag,
                                &self.group_ptrs[0],
                                &self.group_ladder_ptrs[0],
                                &self.group_rowint_length[0],
                                self.group_offdiag_inds,
                                self.num_groups,
                                self.ladder_offset,
                                &x[0],
                                &out[0],
                                &self.term_masks)
            else:
                if double_or_complex is complex:
                    if self._prepare_tensor_product_cpu():
                        tensor_product_cpu_apply(deref(self.complex_tensor_product_cpu), self.subspace.subspace, &self.complex_diag_vec[0], self.has_nonzero_diag, &x[0], &out[0])
                    else:
                        omp_matvec2[complex](self.oper.terms,
                                self.subspace.subspace.bitstrings,
                                &self.complex_diag_vec[0],
                                self.width,
                                self.subspace_dim,
                                self.has_nonzero_diag,
                                &self.group_ptrs[0],
                                &self.group_ladder_ptrs[0],
                                &self.group_rowint_length[0],
                                self.group_offdiag_inds,
                                self.num_groups,
                                self.ladder_offset,
                                &x[0],
                                &out[0],
                                &self.term_masks)

        else:
            if self.is_real:
                if double_or_complex is double:
                    omp_matvec[double](self.oper.terms,
                            self.subspace.subspace.bitstrings,
                            &self.real_diag_vec[0],
                            self.width,
                            self.subspace_dim,
                            self.has_nonzero_diag,
                            &self.group_ptrs[0],
                            self.group_offdiag_inds,
                            self.num_groups,
                            &x[0],
                            &out[0],
                            &self.term_masks)
            else:
                if double_or_complex is complex:
                    omp_matvec[complex](self.oper.terms,
                            self.subspace.subspace.bitstrings,
                            &self.complex_diag_vec[0],
                            self.width,
                            self.subspace_dim,
                            self.has_nonzero_diag,
                            &self.group_ptrs[0],
                            self.group_offdiag_inds,
                            self.num_groups,
                            &x[0],
                            &out[0],
                            &self.term_masks)

        return np.asarray(out)


    def to_csr_array(self):
        """Convert subspace Hamiltonian to a SciPy CSR array

        Returns:
            csr_array: Sparse representation of subspace Hamiltonian
        """
        logger.info("Building CSR matrix")
        csr_start = time.perf_counter()
        cdef int64 max_int = np.iinfo(np.int32).max
        cdef size_t num_terms = self.oper.terms.size()

        cdef int[::1] indptr32
        cdef int[::1] indices32
        cdef int64[::1] indptr64
        cdef int64[::1] indices64
        cdef double complex[::1] complex_data = np.zeros(1, dtype=complex)
        cdef double[::1] real_data = np.zeros(1, dtype=float)

        # Compute diag vec if we have not done so already
        cdef int did_diag_build = 0
        did_diag_build = self.compute_diag_vector()

        cdef double start, stop
        cdef int compute_values, data_size
        cdef size_t total_bytes
        cdef int64 nnz
        if self.is_real:
            data_size = 8 # size of double
        else:
            data_size = 16 # size of double complex

        cdef int int_64 = 1 # Always start with int64 since we do not know how many nnz there will be in the matrix
        checked_multiply(checked_add(self.subspace_dim, 1), 8)
        checked_index[int64](checked_add(self.subspace_dim, 1))
        indptr64 = np.zeros(checked_add(self.subspace_dim, 1), dtype=np.int64)
        indices64 = np.zeros(1, dtype=np.int64)

        for compute_values in range(2):
            start = time.perf_counter()
            if compute_values:
                # matrix is empty
                if indptr64[self.subspace_dim] == 0:
                    return sp.csr_array((self.subspace_dim, self.subspace_dim), dtype=float if self.is_real else complex)

                # if num_elem < max_int and subspace_dim + 1 < max_int then problem is int32 type
                if (indptr64[self.subspace_dim] < max_int) and (<int64>(checked_add(self.subspace_dim, 1)) < max_int):
                    int_64 = 0
                nnz = indptr64[self.subspace_dim]
                if nnz < 0:
                    raise OverflowError("CSR nonzero count is negative")
                logger.info("CSR NNZ: %s ", nnz)
                logger.info("CSR use int64 indices: %s ", True if int_64 else False)
                # Compute the CSR array sizes
                if int_64:
                    # indptr + indices + data sizes
                    total_bytes = checked_add(checked_multiply(checked_add(self.subspace_dim, 1), 8), checked_multiply(nnz, 8 + data_size))
                else:
                    total_bytes = checked_add(checked_multiply(checked_add(self.subspace_dim, 1), 4), checked_multiply(nnz, 4 + data_size))
                logger.info("Est. CSR matrix size: %s Mb", round(total_bytes/(1024**2), 3))
                mem = psutil.virtual_memory().available
                logger.info("Available memory size: %s Mb", round(mem/(1024**2), 3))
                if mem < total_bytes:
                    raise FulqrumError(f"Sparse matrix of size {round(total_bytes/(1024**2), 3)}Mb does not fit within available memory.")

                if int_64:
                    indices64 = np.zeros(nnz, dtype=np.int64)
                    if self.is_real:
                        real_data = np.zeros(nnz, dtype=float)
                    else:
                        complex_data = np.zeros(nnz, dtype=complex)
                else:
                    indptr32 = np.asarray(indptr64, dtype=np.int32)
                    indptr64 = np.zeros(1, dtype=np.int64) # reset indptr64 since we do not need it anymore
                    indices32 = np.zeros(nnz, dtype=np.int32)
                    if self.is_real:
                        real_data = np.zeros(nnz, dtype=float)
                    else:
                        complex_data = np.zeros(nnz, dtype=complex)
            if int_64:
                if self.oper.type == 2:
                    if self.is_real:
                        if self._prepare_tensor_product_cpu():
                            tensor_product_cpu_csr(deref(self.real_tensor_product_cpu), self.subspace.subspace, &self.real_diag_vec[0], self.has_nonzero_diag, &indptr64[0], &indices64[0], &real_data[0], compute_values)
                        else:
                            csr_matrix_builder2(self.oper.terms,
                                                self.subspace.subspace.bitstrings,
                                                &self.real_diag_vec[0],
                                                self.width,
                                                self.subspace_dim,
                                                self.has_nonzero_diag,
                                                &self.group_ptrs[0],
                                                &self.group_ladder_ptrs[0],
                                                &self.group_rowint_length[0],
                                                self.group_offdiag_inds,
                                                self.num_groups,
                                                self.ladder_offset,
                                                &indptr64[0],
                                                &indices64[0],
                                                &real_data[0],
                                                compute_values,
                                                &self.term_masks)
                    else:
                        if self._prepare_tensor_product_cpu():
                            tensor_product_cpu_csr(deref(self.complex_tensor_product_cpu), self.subspace.subspace, &self.complex_diag_vec[0], self.has_nonzero_diag, &indptr64[0], &indices64[0], &complex_data[0], compute_values)
                        else:
                            csr_matrix_builder2(self.oper.terms,
                                                self.subspace.subspace.bitstrings,
                                                &self.complex_diag_vec[0],
                                                self.width,
                                                self.subspace_dim,
                                                self.has_nonzero_diag,
                                                &self.group_ptrs[0],
                                                &self.group_ladder_ptrs[0],
                                                &self.group_rowint_length[0],
                                                self.group_offdiag_inds,
                                                self.num_groups,
                                                self.ladder_offset,
                                                &indptr64[0],
                                                &indices64[0],
                                                &complex_data[0],
                                                compute_values,
                                                &self.term_masks)
                else:
                    if self.is_real:
                        csr_matrix_builder(self.oper.terms,
                                            self.subspace.subspace.bitstrings,
                                            &self.real_diag_vec[0],
                                            self.width,
                                            self.subspace_dim,
                                            self.has_nonzero_diag,
                                            &self.group_ptrs[0],
                                            self.group_offdiag_inds,
                                            self.num_groups,
                                            &indptr64[0],
                                            &indices64[0],
                                            &real_data[0],
                                            compute_values,
                                            &self.term_masks)
                    else:
                        csr_matrix_builder(self.oper.terms,
                                            self.subspace.subspace.bitstrings,
                                            &self.complex_diag_vec[0],
                                            self.width,
                                            self.subspace_dim,
                                            self.has_nonzero_diag,
                                            &self.group_ptrs[0],
                                            self.group_offdiag_inds,
                                            self.num_groups,
                                            &indptr64[0],
                                            &indices64[0],
                                            &complex_data[0],
                                            compute_values,
                                            &self.term_masks)
            else:
                if self.oper.type == 2:
                    if self.is_real:
                        if self._prepare_tensor_product_cpu():
                            tensor_product_cpu_csr(deref(self.real_tensor_product_cpu), self.subspace.subspace, &self.real_diag_vec[0], self.has_nonzero_diag, &indptr32[0], &indices32[0], &real_data[0], compute_values)
                        else:
                            csr_matrix_builder2(self.oper.terms,
                                                self.subspace.subspace.bitstrings,
                                                &self.real_diag_vec[0],
                                                self.width,
                                                self.subspace_dim,
                                                self.has_nonzero_diag,
                                                &self.group_ptrs[0],
                                                &self.group_ladder_ptrs[0],
                                                &self.group_rowint_length[0],
                                                self.group_offdiag_inds,
                                                self.num_groups,
                                                self.ladder_offset,
                                                &indptr32[0],
                                                &indices32[0],
                                                &real_data[0],
                                                compute_values,
                                                &self.term_masks)
                    else:
                        if self._prepare_tensor_product_cpu():
                            tensor_product_cpu_csr(deref(self.complex_tensor_product_cpu), self.subspace.subspace, &self.complex_diag_vec[0], self.has_nonzero_diag, &indptr32[0], &indices32[0], &complex_data[0], compute_values)
                        else:
                            csr_matrix_builder2(self.oper.terms,
                                                self.subspace.subspace.bitstrings,
                                                &self.complex_diag_vec[0],
                                                self.width,
                                                self.subspace_dim,
                                                self.has_nonzero_diag,
                                                &self.group_ptrs[0],
                                                &self.group_ladder_ptrs[0],
                                                &self.group_rowint_length[0],
                                                self.group_offdiag_inds,
                                                self.num_groups,
                                                self.ladder_offset,
                                                &indptr32[0],
                                                &indices32[0],
                                                &complex_data[0],
                                                compute_values,
                                                &self.term_masks)
                else:
                    if self.is_real:
                        csr_matrix_builder(self.oper.terms,
                                            self.subspace.subspace.bitstrings,
                                            &self.real_diag_vec[0],
                                            self.width,
                                            self.subspace_dim,
                                            self.has_nonzero_diag,
                                            &self.group_ptrs[0],
                                            self.group_offdiag_inds,
                                            self.num_groups,
                                            &indptr32[0],
                                            &indices32[0],
                                            &real_data[0],
                                            compute_values,
                                            &self.term_masks)
                    else:
                        csr_matrix_builder(self.oper.terms,
                                            self.subspace.subspace.bitstrings,
                                            &self.complex_diag_vec[0],
                                            self.width,
                                            self.subspace_dim,
                                            self.has_nonzero_diag,
                                            &self.group_ptrs[0],
                                            self.group_offdiag_inds,
                                            self.num_groups,
                                            &indptr32[0],
                                            &indices32[0],
                                            &complex_data[0],
                                            compute_values,
                                            &self.term_masks)
            stop = time.perf_counter()
            if not compute_values:
                logger.info("CSR structure time: %s ms", round((stop - start)*1000, 3))
            else:
                logger.info("CSR fill time: %s ms", round((stop - start)*1000, 3))
        if int_64:
            if self.is_real:
                mat = sp.csr_array((real_data, indices64, indptr64),
                                    shape=(self.subspace_dim,)*2, dtype=float)
            else:
                mat = sp.csr_array((complex_data, indices64, indptr64),
                                    shape=(self.subspace_dim,)*2, dtype=complex)
        else:
            if self.is_real:
                mat = sp.csr_array((real_data, indices32, indptr32),
                                shape=(self.subspace_dim,)*2, dtype=float)
            else:
                mat = sp.csr_array((complex_data, indices32, indptr32),
                                shape=(self.subspace_dim,)*2, dtype=complex)
        start = time.perf_counter()
        quicksort_indices(mat.indices, mat.indptr, mat.data)
        stop = time.perf_counter()
        logger.info("CSR indices sort time: %s ms", round((stop - start)*1000, 3))
        csr_stop = time.perf_counter()
        logger.info("CSR total matrix build time: %s ms", round((csr_stop - csr_start)*1000, 3))
        return mat

    def to_csrlike(self):
        # This is here to prevent a circular import
        from .linear_operator import CSRLikeLinearOperator
        # Compute diag vec if we have not done so already
        logger.info("Building CSR matrix fast-mode")
        cdef double stop, start
        start = time.perf_counter()
        self.compute_diag_vector()
        checked_multiply(self.subspace_dim, 2 * sizeof(vector[double]))
        cdef CSRLike csrlike = CSRLike(self.subspace_dim, self.is_real)
        if csrlike.type_string == 'd32':
            logger.info("CSR fast-mode double and int32")
            if self.oper.type == 1:
                csrlike_builder(self.oper.terms,
                            self.subspace.subspace.bitstrings,
                            &self.real_diag_vec[0],
                            self.width,
                            self.subspace_dim,
                            self.has_nonzero_diag,
                            &self.group_ptrs[0],
                            self.group_offdiag_inds,
                            self.num_groups,
                            csrlike.data_d32.cols,
                            csrlike.data_d32.data,
                            &self.term_masks)

            else:
                if self._prepare_tensor_product_cpu():
                    tensor_product_cpu_csrlike(deref(self.real_tensor_product_cpu), self.subspace.subspace, &self.real_diag_vec[0], self.has_nonzero_diag, csrlike.data_d32.cols, csrlike.data_d32.data)
                else:
                    csrlike_builder2(self.oper.terms,
                                self.subspace.subspace.bitstrings,
                                &self.real_diag_vec[0],
                                self.width,
                                self.subspace_dim,
                                self.has_nonzero_diag,
                                &self.group_ptrs[0],
                                &self.group_ladder_ptrs[0],
                                &self.group_rowint_length[0],
                                self.group_offdiag_inds,
                                self.num_groups,
                                self.ladder_offset,
                                csrlike.data_d32.cols,
                                csrlike.data_d32.data,
                                &self.term_masks)
        elif csrlike.type_string == 'd64':
            logger.info("CSR fast-mode double and int64")
            if self.oper.type == 1:
                csrlike_builder(self.oper.terms,
                            self.subspace.subspace.bitstrings,
                            &self.real_diag_vec[0],
                            self.width,
                            self.subspace_dim,
                            self.has_nonzero_diag,
                            &self.group_ptrs[0],
                            self.group_offdiag_inds,
                            self.num_groups,
                            csrlike.data_d64.cols,
                            csrlike.data_d64.data,
                            &self.term_masks)

            else:
                if self._prepare_tensor_product_cpu():
                    tensor_product_cpu_csrlike(deref(self.real_tensor_product_cpu), self.subspace.subspace, &self.real_diag_vec[0], self.has_nonzero_diag, csrlike.data_d64.cols, csrlike.data_d64.data)
                else:
                    csrlike_builder2(self.oper.terms,
                                self.subspace.subspace.bitstrings,
                                &self.real_diag_vec[0],
                                self.width,
                                self.subspace_dim,
                                self.has_nonzero_diag,
                                &self.group_ptrs[0],
                                &self.group_ladder_ptrs[0],
                                &self.group_rowint_length[0],
                                self.group_offdiag_inds,
                                self.num_groups,
                                self.ladder_offset,
                                csrlike.data_d64.cols,
                                csrlike.data_d64.data,
                                &self.term_masks)
        elif csrlike.type_string == 'z32':
            logger.info("CSR fast-mode complex and int32")
            if self.oper.type == 1:
                csrlike_builder(self.oper.terms,
                            self.subspace.subspace.bitstrings,
                            &self.complex_diag_vec[0],
                            self.width,
                            self.subspace_dim,
                            self.has_nonzero_diag,
                            &self.group_ptrs[0],
                            self.group_offdiag_inds,
                            self.num_groups,
                            csrlike.data_z32.cols,
                            csrlike.data_z32.data,
                            &self.term_masks)

            else:
                if self._prepare_tensor_product_cpu():
                    tensor_product_cpu_csrlike(deref(self.complex_tensor_product_cpu), self.subspace.subspace, &self.complex_diag_vec[0], self.has_nonzero_diag, csrlike.data_z32.cols, csrlike.data_z32.data)
                else:
                    csrlike_builder2(self.oper.terms,
                                self.subspace.subspace.bitstrings,
                                &self.complex_diag_vec[0],
                                self.width,
                                self.subspace_dim,
                                self.has_nonzero_diag,
                                &self.group_ptrs[0],
                                &self.group_ladder_ptrs[0],
                                &self.group_rowint_length[0],
                                self.group_offdiag_inds,
                                self.num_groups,
                                self.ladder_offset,
                                csrlike.data_z32.cols,
                                csrlike.data_z32.data,
                                &self.term_masks)
        elif csrlike.type_string == 'z64':
            logger.info("CSR fast-mode complex and int64")
            if self.oper.type == 1:
                csrlike_builder(self.oper.terms,
                            self.subspace.subspace.bitstrings,
                            &self.complex_diag_vec[0],
                            self.width,
                            self.subspace_dim,
                            self.has_nonzero_diag,
                            &self.group_ptrs[0],
                            self.group_offdiag_inds,
                            self.num_groups,
                            csrlike.data_z64.cols,
                            csrlike.data_z64.data,
                            &self.term_masks)

            else:
                if self._prepare_tensor_product_cpu():
                    tensor_product_cpu_csrlike(deref(self.complex_tensor_product_cpu), self.subspace.subspace, &self.complex_diag_vec[0], self.has_nonzero_diag, csrlike.data_z64.cols, csrlike.data_z64.data)
                else:
                    csrlike_builder2(self.oper.terms,
                                self.subspace.subspace.bitstrings,
                                &self.complex_diag_vec[0],
                                self.width,
                                self.subspace_dim,
                                self.has_nonzero_diag,
                                &self.group_ptrs[0],
                                &self.group_ladder_ptrs[0],
                                &self.group_rowint_length[0],
                                self.group_offdiag_inds,
                                self.num_groups,
                                self.ladder_offset,
                                csrlike.data_z64.cols,
                                csrlike.data_z64.data,
                                &self.term_masks)

        stop = time.perf_counter()
        logger.info("CSR fast-mode total build time: %s ms", round((stop - start)*1000, 3))
        return CSRLikeLinearOperator(csrlike)

    def diagonal_slice_product_space(self,
            np.uint64_t[::1] alpha_uint,
            np.uint64_t[::1] beta_uint,
            size_t a0, size_t a1,
            unsigned int num_alpha_qubits):
        """Diagonal vector constructor for tensor product Subspace."""
        if self.is_real != 1:
            raise ValueError("GPU diagonal helpers require real values")
        if not 0 < num_alpha_qubits < self.width or num_alpha_qubits > 64 or self.width - num_alpha_qubits > 64:
            raise ValueError("GPU diagonal helpers require 1 to 64 qubits per sector")
        if a0 > a1 or a1 > <size_t>alpha_uint.shape[0]:
            raise ValueError("Invalid alpha string range")
        cdef size_t na_local = a1 - a0
        cdef size_t n_b = beta_uint.shape[0]
        checked_multiply(checked_multiply(na_local, n_b), sizeof(double))
        if not self.has_nonzero_diag:
            return np.zeros(checked_multiply(na_local, n_b), dtype=np.float64)
        return self._compute_diag_via_blas(
            alpha_uint, beta_uint, a0, a1, num_alpha_qubits)

    def diag_separable_pieces(self,
            np.uint64_t[::1] alpha_uint,
            np.uint64_t[::1] beta_uint,
            size_t a0, size_t a1,
            unsigned int num_alpha_qubits):
        """Diagonal recompute (no persistent local rank copy)."""
        if self.is_real != 1:
            raise ValueError("GPU diagonal helpers require real values")
        if not 0 < num_alpha_qubits < self.width or num_alpha_qubits > 64 or self.width - num_alpha_qubits > 64:
            raise ValueError("GPU diagonal helpers require 1 to 64 qubits per sector")
        if a0 > a1 or a1 > <size_t>alpha_uint.shape[0]:
            raise ValueError("Invalid alpha string range")
        cdef size_t na_local = a1 - a0
        cdef size_t n_b = beta_uint.shape[0]
        cdef unsigned int num_beta_qubits = <unsigned int>self.subspace.width - num_alpha_qubits
        cdef double[::1] mv_d_alpha
        cdef double[::1] mv_d_beta
        cdef double[:, ::1] mv_V
        cdef double[::1] mv_da_lin
        cdef double[::1] mv_db_lin
        cdef double[::1] mv_const
        cdef double* c_null = NULL
        cdef bint ok = True

        d_alpha_arr = np.zeros(na_local, dtype=np.float64)
        d_beta_arr = np.zeros(n_b, dtype=np.float64)
        V_arr = np.zeros((num_alpha_qubits, num_beta_qubits), dtype=np.float64)
        da_lin_arr = np.zeros(num_alpha_qubits, dtype=np.float64)
        db_lin_arr = np.zeros(num_beta_qubits, dtype=np.float64)
        const_arr = np.zeros(1, dtype=np.float64)

        if not self.has_nonzero_diag or na_local == 0 or n_b == 0:
            return d_alpha_arr, d_beta_arr, V_arr, True

        mv_d_alpha = d_alpha_arr
        mv_d_beta = d_beta_arr
        mv_V = V_arr
        mv_da_lin = da_lin_arr
        mv_db_lin = db_lin_arr
        mv_const = const_arr

        with nogil:
            compute_diag_factors[double](
                <const unsigned long long*>&alpha_uint[0],
                <const unsigned long long*>&beta_uint[0],
                self.diag_oper,
                num_alpha_qubits, a0, a1, n_b,
                &mv_d_alpha[0], &mv_d_beta[0],
                c_null, c_null)
            ok = compute_diag_cross_factors[double](
                self.diag_oper, num_alpha_qubits, num_beta_qubits,
                &mv_V[0, 0], &mv_da_lin[0], &mv_db_lin[0], &mv_const[0])

        if not ok:
            return d_alpha_arr, d_beta_arr, V_arr, False

        a_slice = np.ascontiguousarray(np.asarray(alpha_uint)[a0:a1])
        b_all = np.asarray(beta_uint)
        pa = np.arange(num_alpha_qubits, dtype=np.uint64)
        qb = np.arange(num_beta_qubits, dtype=np.uint64)
        z_alpha = 1.0 - 2.0 * ((a_slice[:, None] >> pa[None, :]) & np.uint64(1)).astype(np.float64)
        z_beta = 1.0 - 2.0 * ((b_all[:, None] >> qb[None, :]) & np.uint64(1)).astype(np.float64)
        d_alpha_arr = d_alpha_arr + const_arr[0] + z_alpha @ da_lin_arr
        d_beta_arr = d_beta_arr + z_beta @ db_lin_arr

        return d_alpha_arr, d_beta_arr, V_arr, ok


@cython.boundscheck(False)
def quicksort_indices(int32_or_int64[::1] indices,
                      int32_or_int64[::1] indptr,
                      double_or_complex[::1] data):
    cdef int32_or_int64 kk, nrows = indptr.shape[0]-1
    cdef int32_or_int64 start, stop
    for kk in prange(nrows, nogil=True):
        start = indptr[kk]
        stop = indptr[kk+1] - 1
        quicksort_indices_data(&indices[0], &data[0], start, stop)
