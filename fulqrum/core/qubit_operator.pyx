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
from cython.operator cimport dereference
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libc.string cimport memcpy
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map
from libc.math cimport fabs
from libcpp.algorithm cimport sort as stdsort
from cython.operator cimport dereference, preincrement
from .. import __version__ as VERSION

from ..utils.io import dict_to_json, json_to_dict
from ..exceptions import FulqrumError
from .bitset cimport Bitset
from .constants cimport width_t
from .constants import np_width_t

from collections.abc import Iterable
from pathlib import Path
import time
import numbers
import numpy as np
cimport numpy as np

import logging
logger = logging.getLogger(__name__)

include "includes/base_header.pxi"
include "includes/elements_header.pxi"
include "includes/bitset_utils_header.pxi"
include "includes/offdiag_grouping_header.pxi"
include "includes/converters.pxi"
include "includes/io.pxi"
include "includes/diag_header.pxi"


cdef const OperatorTerm_t EmptyOperatorTerm


cdef class QubitOperator():
    """Operator class for qubit terms consisting of Pauli
    operators,projection operators, and ladder operators

    Parameters:
        width (int): Number of qubits
        operators (list): List of tuples for terms in Hamiltonian

    Example:

    .. jupyter-execute::

        import fulqrum as fq
        fq.QubitOperator(5, [("X1", [0, 3], -2), ("XZY", [2, 0, 4], 3)])
    """
    #QubitOperator oper

    def __cinit__(self, unsigned int width,
                  object operators=None):
        self.oper.width = width
        self.oper.sorted = False
        cdef object item
        cdef OperatorTerm_t term
        cdef string op_str
        cdef double complex coeff
        cdef object inds
        cdef size_t kk
        cdef unsigned char op, ind
        if operators is not None:
            for item in operators:
                term = EmptyOperatorTerm
                term.offdiag_weight = 0
                if any(item):
                    if len(item) == 1:
                        term = OperatorTerm_t(item[0])
                    else:
                        op_str = (<string>item[0]).c_str()
                        inds = item[1] if isinstance(item[1], Iterable) else [item[1]]
                        coeff = item[2]
                        for kk in range(<size_t>len(item[0])):
                            if inds[kk] > (self.oper.width - 1):
                                raise FulqrumError(f'Index {item[1]} is out of range for width={self.oper.width}')
                            if op_str[kk] != 73:
                                term.indices.push_back(inds[kk])
                                ind = STR_TO_IND[op_str[kk]]
                                term.values.push_back(ind)
                        term.coeff = coeff
                else:
                    term.coeff = 1
                term.sort_term_data()
                set_offdiag_weight_phase_struct(term)
                term.set_proj_indices()
                self.oper.terms.push_back(term)


    def __dealloc__(self):
        # Clear vectors upon deallocation of class
        (vector[OperatorTerm_t]()).swap(self.oper.terms)

    @classmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def from_label(self, string label, double complex coeff = 1.0):
        cdef QubitOperator_t temp
        cdef QubitOperator out = QubitOperator(label.size())
        out.oper = temp.from_label(label)
        out.oper.terms[0].coeff = coeff
        return out

    @classmethod
    def from_constant(self, unsigned int width, double complex coeff):
        """Generate a constant Hamiltonian term

        Parameters:
            width (unsigned int): Width of operator
            coeff (complex): Operator coefficient

        Returns:
            QubitOperator: Constant operator
        """
        cdef QubitOperator out = QubitOperator(width, [()])
        out.oper.terms[0].coeff = coeff
        return out

    def __len__(self):
        return self.oper.size()

    def set_type(self, unsigned int value):
        """Manually set the type of the operator

        Parameters:
            value (int): Set `1` for a standard qubit Hamiltonian, and `2` for fermionic

        Note:
            This is usually automatically set for you, with `type=2` being set during the
            conversion from `FermionicOperator` to `QubitOperator`
        """
        if (value != 1 and value != 2):
            raise FulqrumError("Type of operator must be '1' or '2'")
        self.oper.type = value

    @property
    def type(self):
        """ Type of QubitOperator

        Type `1` is standard Qubit systems, e.g. Paulis and projectors
        Type `2` is for systems derived from Fermionic systems via extended JW

        Returns:
            int: Type of operator
        """
        return self.oper.type

    @property
    def dtype(self):
        """ The data type of the operator

        Returns:
            type: Data type of operator
        """
        return np.dtype(float) if self.is_real() else np.dtype(complex)

    @property
    def coeff(self):
        """Return the coefficient of a Hamiltonian comprised from a single term

        Returns:
            complex
        """
        if self.oper.terms.size() == 1:
            return self.oper.terms[0].coeff
        else:
            raise FulqrumError("Operator must have a single-term to get coeff.  Otherwise use op.coefficients()")


    def size(self):
        """Return the number of terms in the operator

        Returns:
            int: Number of terms in operator
        """
        return self.oper.size()

    @property
    def width(self):
        """Width (number of qubits) of the operator

        Returns:
            int: Width of operator
        """
        return self.oper.width

    @property
    def sorted(self):
        """Is the operator sorted by off-diagonal structure

        Returns:
            bool: Operator is sorted
        """
        return self.oper.sorted

    @property
    def weight_sorted(self):
        """Is the operator sorted by full operator weight

        Returns:
            bool: Operator is sorted by full weight
        """
        return self.oper.weight_sorted


    @property
    def off_weight_sorted(self):
        """Is the operator sorted by off-diagonal weight

        Returns:
            bool: Operator is sorted by off-diagonal weight
        """
        return self.oper.off_weight_sorted

    @property
    def num_groups(self):
        """Number of off-diagonal groupings

        Returns:
            int : Number of groups in operator
        """
        if self.size() == 0:
            return 0
        if not self.oper.sorted:
            self.oper.group_sort()
        return self.oper.group_ptrs().size() - 1

    @property
    def ladder_width(self):
        """Ladder with

        Returns:
            int : Ladder width
        """

        return self.oper.ladder_width

    @cython.boundscheck(False)
    def offdiag_structures(self):
        """Off-diagonal structures of each term in operator
        """
        cdef size_t kk
        if self.oper.terms.size() == 0:
            raise FulqrumError('QubitOperator has zero terms')
        cdef unsigned int[::1] out = np.empty(self.oper.terms.size(), dtype=np.uint32)
        for kk in range(self.oper.terms.size()):
            out[kk] = self.oper.terms[kk].offdiag_structure
        return np.asarray(out)

    def copy(self):
        """Copy QubitOperator

        Returns:
            QubitOperator
        """
        cdef QubitOperator out = QubitOperator(self.width)
        out.oper = self.oper.copy()
        return out

    @cython.boundscheck(False)
    def is_real(self):
        """Can operator be described via a symmetric matrix

        Returns:
            int: Is operator real-valued
        """
        return self.oper.is_real()

    @cython.boundscheck(False)
    def real_phases(self):
        """The real 'phase' of each term in operator

        Returns:
            ndarray: real phase of each term in operator
        """
        cdef vector[np.int8_t] phases = self.oper.real_phases()
        cdef np.int8_t[::1] out = np.empty(self.oper.terms.size(), dtype=np.int8)
        memcpy(&out[0], &phases[0], phases.size()*sizeof(np.int8_t))
        return np.asarray(out)

    @cython.boundscheck(False)
    def constant_energy(self):
        """Value of the constant energy term(s) in the operator

        Returns:
            float
        """
        return self.oper.constant_energy()

    @cython.boundscheck(False)
    def split_diagonal(self):
        """Spit an operator into diagonal and non-diagonal components

        Returns:
            tuple: QubitOperators for diagonal and non-diagonal parts
        """
        cdef QubitOperator diag = QubitOperator(self.oper.width)
        cdef QubitOperator offdiag = QubitOperator(self.oper.width)
        cdef pair[QubitOperator_t, QubitOperator_t] out = self.oper.split_diagonal()
        diag.oper = out.first
        offdiag.oper = out.second
        return diag, offdiag

    @cython.boundscheck(False)
    def coefficients(self):
        """Return the coefficients for each term in the operator

        Returns:
            ndarray: complex-valued array of coefficients
        """
        
        cdef vector[complex] coeffs = self.oper.coefficients()
        cdef double complex[::1] out = np.empty(self.oper.terms.size(), dtype=complex)
        memcpy(&out[0], &coeffs[0], coeffs.size()*sizeof(complex))
        return np.asarray(out)

    @property
    def operators(self):
        """Return the operators for a single term or empty operator

        Returns:
            list or None : List of operator index tuples, if any, else None

        Notes:
            This returns a list of tuples to allow for multiple indices in
            Fermionic operators
        """
        cdef size_t kk, jj
        cdef OperatorTerm_t * term
        cdef list out = []
        if self.size() > 1:
            raise FulqrumError('Can only grab operators from operators with < 2 terms')
        elif self.size() == 0:
            return None
        else:
            for kk in range(self.oper.terms.size()):
                term = &self.oper.terms[kk]
                for jj in range(term.indices.size()):
                    out.append((IND_TO_STR[term.values[jj]], term.indices[jj]))
            return out

    @property
    def proj_indices(self):
        """Return the projector indices for a single term or empty operator
        """
        if self.oper.terms.size() > 2:
            raise FulqrumError('Can only grab projector indices from operators with < 2 terms')
        cdef size_t kk
        cdef width_t[::1] out
        if self.oper.terms.size() == 0:
            return np.array([], dtype=np_width_t)
        out = np.zeros(self.oper.terms[0].proj_indices.size(), dtype=np_width_t)
        for kk in range(self.oper.terms[0].proj_indices.size()):
            out[kk] = self.oper.terms[0].proj_indices[kk]
        return np.asarray(out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getitem__(self, key):
        """Implements indexing and slicing of terms

        Parameters:
            key (integral or slice or list or tuple): Indices

        Returns:
            QubitOperator: Indexed terms
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.oper.width)
        if isinstance(key, numbers.Integral):
            if key < 0:
                 key = self.oper.terms.size() + key
            kk = <size_t>key
            if kk > self.oper.terms.size() - 1:
                raise FulqrumError(f"Index {kk} is out of range for operator"
                                   f" of length {self.oper.terms.size()}")
            out.oper.terms.push_back(self.oper.terms[kk])
        elif isinstance(key, slice):
            for kk in range(*key.indices(self.oper.terms.size())):
                out.oper.terms.push_back(self.oper.terms[kk])
        elif isinstance(key, (list, tuple)):
            for idx in key:
                kk = <size_t>idx
                out.oper.terms.push_back(self.oper.terms[kk])
        else:
            raise FulqrumError(f"Cannot get operator terms using {type(key)}")
        out.oper.type = self.oper.type
        return out

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        cdef QubitOperator out
        if self._iter_index < self.oper.terms.size():
            self._iter_index += 1
            out = QubitOperator(self.oper.width)
            out.oper.terms.push_back(self.oper.terms[self._iter_index - 1])
            return out
        else:
            raise StopIteration

    def __iadd__(self, QubitOperator other):
        """Inplace addition of QubitOperators
        """
        self.append(other)
        return self

    @cython.boundscheck(False)
    def __imul__(self, double complex other):
        """Inplace multiplication of QubitOperators
        """
        cdef size_t kk
        for kk in range(self.oper.terms.size()):
            self.oper.terms[kk].coeff *= other
        return self

    @cython.boundscheck(False)
    def __add__(self, QubitOperator other):
        """Addition of QubitOperators with copy

        Returns:
            QubitOperator
        """
        if self.oper.type != other.oper.type:
            raise FulqrumError("Operator types do not match")
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.oper.width)
        out.oper.terms = self.oper.terms
        for kk in range(other.oper.terms.size()):
            out.oper.terms.push_back(other.oper.terms[kk])
        out.oper.type = self.oper.type
        return out

    @cython.boundscheck(False)
    def __sub__(self, QubitOperator other):
        """Subtraction of QubitOperators with copy

        Returns:
            QubitOperator
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.oper.width)
        cdef OperatorTerm_t term
        out.oper.terms = self.oper.terms
        for kk in range(other.oper.terms.size()):
            term = other.oper.terms[kk]
            term.coeff *= -1
            out.oper.terms.push_back(term)
        return out

    @cython.boundscheck(False)
    def __mul__(self, double complex other):
        """Multiplication of QubitOperators on left with copy

        Returns:
            QubitOperator
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.oper.width)
        out.oper.terms = self.oper.terms
        for kk in range(out.oper.terms.size()):
            out.oper.terms[kk].coeff *= other
        return out

    @cython.boundscheck(False)
    def __rmul__(self, double complex other):
        """Multiplication of QubitOperators on right with copy

        Returns:
            QubitOperator
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.oper.width)
        out.oper.terms = self.oper.terms
        for kk in range(out.oper.terms.size()):
            out.oper.terms[kk].coeff *= other
        return out

    @cython.boundscheck(False)
    def __repr__(self):
        cdef size_t idx
        cdef list out = []
        cdef str temp_str
        cdef OperatorTerm_t term
        cdef size_t kk
        cdef size_t num_terms = self.oper.terms.size()
        cdef size_t total_terms = num_terms
        cdef int too_many_terms = 0
        if num_terms > 100:
            too_many_terms = 1
            num_terms = 100
        for idx in range(num_terms):
            temp_str = ''
            term = self.oper.terms[idx]
            for kk in range(term.indices.size()):
                if kk:
                    temp_str += ' '
                temp_str += IND_TO_STR[term.values[kk]] + ':'
                temp_str += str(term.indices[kk])
            out.append((temp_str, term.coeff))

        out_strs = ', '.join(str(kk) for kk in out)
        if too_many_terms:
            out_strs += f' + {total_terms-100} more terms'
        add_str = ''
        if num_terms == 1:
            add_str = f", group={self.oper.terms[0].group}"
        return f"<QubitOperator[{out_strs}], width={self.oper.width}{add_str}>"

    @cython.boundscheck(False)
    cpdef void append(self, QubitOperator other):
        cdef size_t kk
        if self.oper.type != other.oper.type:
            raise FulqrumError("Operator types do not match")
        if self.oper.width != other.oper.width:
            raise FulqrumError('Appending number of qubits does not match current number')
        for kk in range(other.oper.terms.size()):
            self.oper.terms.push_back(other.oper.terms[kk])
        self.oper.sorted = 0

    @cython.boundscheck(False)
    def weights(self):
        """Weight of each term in the operator

        Returns:
            ndarray: Array of operator weights
        """
        cdef width_t[::1] out = np.zeros(self.oper.terms.size(), dtype=np_width_t)
        cdef size_t kk
        for kk in range(self.oper.terms.size()):
            out[kk] = self.oper.terms[kk].values.size()
        return np.asarray(out)

    @cython.boundscheck(False)
    def offdiag_weights(self):
        """Off-diagonal weight of each term in the operator

        Returns:
            ndarray: Array of operator off-diagonal weights
        """
        cdef width_t[::1] out = np.zeros(self.oper.terms.size(), dtype=np_width_t)
        cdef size_t kk
        for kk in range(self.oper.terms.size()):
            out[kk] = self.oper.terms[kk].offdiag_weight
        return np.asarray(out)

    def is_diagonal(self):
        """Check if operator is diagonal in computational basis

        Returns:
            bool: True if diagonal, False otherwise
        """
        return self.oper.is_diagonal()

    @cython.boundscheck(False)
    def remove_constant_terms(self, bool return_value=True):
        """Remove constant (identity) terms from operator, optionally
        returning the sum of the coefficients

        Parameters:
            return_value (bool): Return the sum of constant term coefficients, default=True

        Returns:
            QubitOperator: Operator with no identity terms
            double: Sum of identity coefficients, if `return_value=True`
        """
        cdef double val = 0
        if return_value:
            val = self.oper.constant_energy()
        cdef QubitOperator_t temp = self.oper.remove_constant_terms()
        cdef QubitOperator out = QubitOperator(self.width)
        out.oper.terms = temp.terms
        out.oper.type = temp.type
        if return_value:
            return out, val
        return out

    @cython.boundscheck(False)
    def matrix_element(self, object row, object col):
        """Compute matrix element value at given row and column

        Parameters:
            row (str or int): Row index
            col (str or int): Column index

        Returns:
            complex: Element value at H[row, col]
        """
        cdef string row_str, col_str
        if isinstance(row, numbers.Integral):
            row_str = bin(row)[2:].zfill(self.oper.width)
        elif isinstance(row, str):
            row_str = row
        else:
            raise Exception('bad row input')
        if isinstance(col, numbers.Integral):
            col_str = bin(col)[2:].zfill(self.oper.width)
        elif isinstance(col, str):
            col_str = col
        else:
            raise Exception('bad col input')
        if row_str.size() != col_str.size():
            raise Exception('String lengths differ')
        cdef unsigned int bit_len = row_str.size()

        cdef bitset_t row_vec, col_vec, nonzero_vec

        # convert strings to bit arrays
        row_vec = bitset_t(row_str, 0, row_str.size())
        col_vec = bitset_t(col_str, 0, col_str.size())

        if bit_len != self.oper.width:
            raise Exception('Operator width does not match string length')
        cdef size_t num_terms = self.oper.terms.size()
        cdef size_t kk, weight
        cdef double complex out = 0.0
        cdef double complex temp
        cdef OperatorTerm_t * term
        for kk in range(num_terms):
            term = &self.oper.terms[kk]
            weight = term.indices.size()
            nonzero_vec = row_vec #copy row vec into nonzero vec
            get_column_bitset(nonzero_vec, term.indices, term.values, weight)
            # Input col string matches that of nonzero column
            if col_vec == nonzero_vec:
                accum_element(row_vec, nonzero_vec,
                              term.indices, term.values, term.coeff, term.real_phase, weight, out)
        return out

    @cython.boundscheck(False)
    def groups(self):
        """Off-diagonal group structure of terms in operator

        Returns:
            ndarray: Array of ints indicating group of each term
        """
        cdef vector[int] groups = self.oper.groups()
        cdef int[::1] out = np.empty(groups.size(), dtype=np.int32)
        if groups.size():
            memcpy(&out[0], &groups[0], groups.size() * sizeof(int))
        return np.asarray(out)

    @cython.boundscheck(False)
    def group_ptrs(self):
        """Get pointers to start and stop indices for off-diagonal grouping

        Returns:
            ndarray: Array of ints giving pointers to group starts and stops
        """
        cdef vector[size_t] ptrs = self.oper.group_ptrs()
        cdef size_t[::1] out = np.empty(ptrs.size(), dtype=np.uintp)
        if ptrs.size():
            memcpy(&out[0], &ptrs[0], ptrs.size() * sizeof(size_t))
        return np.asarray(out)

    @cython.boundscheck(False)
    def terms_by_group(self, int number):
        """Return terms in operator that correspond to input group number

        Parameters:
            number (int): Group number

        Returns:
            QubitOperator
        """
        cdef size_t kk, ll, start, stop
        cdef size_t[::1] ptrs = self.group_ptrs()
        cdef QubitOperator out = QubitOperator(self.width)
        out.oper = self.oper.terms_by_group(number)
        return out

    def group_sort(self):
        """Inplace sorting of operator terms into groups that represent matrix-elements.
        """
        cdef double st = time.perf_counter()
        self.oper.group_sort()
        cdef double ft = time.perf_counter()
        logger.info("Term grouping time: %s ms", round((ft - st) * 1000, 3))

    def offdiag_weight_sort(self):
        """In-place sort terms by their off-diagonal weight
        """
        self.oper.offdiag_weight_sort()

    def weight_sort(self):
        """In-place sort terms by their standard weight
        """
        self.oper.weight_sort()

    @cython.boundscheck(False)
    def offdiag_weight_ptrs(self):
        """Off-diagonal weight pointers for the operator

        Returns:
            ndarray: Array of off-diagonal weight pointers
        """
        cdef vector[size_t] vec = self.oper.offdiag_weight_ptrs()
        cdef size_t[::1] out = np.empty(vec.size(), dtype=np.uintp)
        if vec.size():
            memcpy(&out[0], &vec[0], vec.size() * sizeof(size_t))
        return np.asarray(out)


    def combine_repeat_terms(self, double atol=1e-12):
        """Combine repeated terms that represent same
        operators, dropping terms smaller than requested tolerance.

        Parameters:
            atol (double): Tolerance for dropping terms, default=1e-12

        Returns:
            QubitOperator: Operator with repeat terms combined
        """
        cdef double st = time.perf_counter()
        cdef QubitOperator out = QubitOperator(self.oper.width)
        out.oper = self.oper.combine_repeat_terms(atol)
        cdef double ft = time.perf_counter()
        logger.info("Combine repeat terms time: %s ms", round((ft - st) * 1000, 3))
        return out

    @cython.boundscheck(False)
    def ladder_ints(self):
        """Compute the ladder operator integer for each term

        If no ladder ops present then default int is max(uint32)

        Parameters:
            ladder_width (int): Number of ladder terms to consider, default = 4

        Returns:
            ndarray: Array of uint32 integers
        """
        if not self.oper.type == 2:
            raise FulqrumError("Operator must be type=2")
        cdef vector[width_t] ladder_ints = self.oper.ladder_integers()
        cdef width_t[::1] out = np.zeros(self.oper.terms.size(), dtype=np_width_t)
        if self.oper.terms.size():
            memcpy(&out[0], &ladder_ints[0], ladder_ints.size() * sizeof(width_t))
        return np.asarray(out)

    @cython.boundscheck(False)
    def group_offdiag_indices(self):
        """Off-diagonal indices for each group in operator
        """
        if not self.oper.sorted:
            self.offdiag_term_grouping()
        cdef size_t[::1] group_ptrs = self.group_ptrs()
        cdef size_t kk, jj
        cdef list out = []
        cdef size_t num_groups = group_ptrs.shape[0] - 1
        cdef vector[vector[width_t]] group_indices
        cdef width_t[::1] temp_inds
        set_group_offdiag_indices(self.oper.terms, group_indices, &group_ptrs[0],
                                 num_groups)
        for kk in range(num_groups):
            temp_inds = np.zeros(group_indices[kk].size(), dtype=np_width_t)
            for jj in range(group_indices[kk].size()):
                temp_inds[jj] = group_indices[kk][jj]
            out.append(np.asarray(temp_inds))
        return out

    @cython.boundscheck(False)
    def group_rowint_length(self):
        """The length (number of bits) in the row int per group
        """
        if not self.oper.type == 2:
            raise FulqrumError("Operator must be type=2")
       
        cdef vector[width_t] group_bit_len = self.oper.group_ladder_int_bit_lengths()
        cdef width_t[::1] out = np.empty(group_bit_len.size(), dtype=np_width_t)
        if group_bit_len.size():
            memcpy(&out[0], &group_bit_len[0], group_bit_len.size()*sizeof(width_t))
        return np.asarray(out)

    def group_term_sort_by_ladder_int(self, unsigned int ladder_width=4):
        """Sort groups by ladder integer if operator is type=2

        Raises:
            FulqrumError: Operator is NOT type=2
        """
        if not self.oper.type == 2:
            raise FulqrumError("Operator must be type=2")
        logger.info("Starting ladder int grouping, ladder_width = %s", ladder_width)
        cdef double st = time.perf_counter()
        self.oper.group_term_sort_by_ladder_int(ladder_width)
        cdef double ft = time.perf_counter()
        logger.info("Ladder int grouping time: %s ms", round((ft - st) * 1000, 3))

    def group_ladder_bin_starts(self):
        if not self.oper.type == 2:
            raise FulqrumError("Operator must be type=2")
        cdef vector[size_t] ptrs = self.oper.group_ladder_int_ptrs()
        cdef size_t[::1] out = np.empty(ptrs.size(), dtype=np.uintp)
        if ptrs.size():
            memcpy(&out[0], &ptrs[0], ptrs.size()*sizeof(size_t))
        return np.asarray(out)

    @cython.boundscheck(False)
    def projector_oper_validation(self, Bitset bits):
        """Return array indicating which terms pass projector validation for a given bit-string

        Parameters:
            bits (Bitset): bit-string of interest

        Returns:
            ndarray
        """
        cdef size_t num_terms = self.oper.terms.size()
        cdef int[::1] out = np.zeros(num_terms, dtype=np.int32)
        cdef size_t kk
        for kk in range(num_terms):
            out[kk] = passes_proj_validation(&self.oper.terms[kk], bits.bits)
        return np.asarray(out)
    
    @cython.boundscheck(False)
    def proj_structures(self):
        """Off-diagonal structures of each term in operator
        """
        cdef size_t kk
        cdef unsigned int[::1] out = np.empty(self.oper.terms.size(), dtype=np.uint32)
        for kk in range(self.oper.terms.size()):
            out[kk] = self.oper.terms[kk].proj_structure
        return np.asarray(out)

    @cython.boundscheck(False)
    def worst_case_offdiag_group_amplitudes(self):
        """Compute the worst case amplitude of off-diagonal groups.

        Requires input operator to be purely off-diagonal

        Parameters:
            off (QubitOperator): Off-diagonal qubit operator

        Returns:
            ndarray: Worse case amplitudes of groups
        """
        diag_op, _ = self.split_diagonal()
        if diag_op.size() != 0:
            raise FulqrumError('Operator must contain off-diagonal terms only')
        cdef size_t[::1] group_ptrs = self.group_ptrs()
        cdef size_t[::1] ladder_starts
        cdef list out = []
        cdef double max_val, temp_val
        cdef size_t max_ladder_int
        cdef size_t idx, kk, jj, start, stop
        if self.type == 2:
            max_ladder_int = 2**self.ladder_width
            self.group_term_sort_by_ladder_int()
            ladder_starts = self.group_ladder_bin_starts()
            for idx in range(<size_t>self.num_groups):
                max_val = 0
                for kk in range(idx*max_ladder_int, (idx+1)*max_ladder_int):
                    start = ladder_starts[kk]
                    stop = ladder_starts[kk+1]
                    temp_val = 0
                    for jj in range(start, stop):
                        temp_val += abs(self[jj].coeff)
                    if temp_val > max_val:
                        max_val = temp_val
                out.append(max_val)
        else:
            for idx in range(<size_t>self.num_groups):
                start = group_ptrs[idx]
                stop = group_ptrs[idx+1]
                temp_val = 0
                for jj in range(start, stop):
                    temp_val += abs(self[jj].coeff)
                out.append(temp_val)
        return np.asarray(out)


    @cython.boundscheck(False)
    def to_dict(self):
        """Dictionary representation of QubitOperator

        Returns:
            dict: Dictionary representation of QubitOperator
        """
        cdef dict out = {'operator-type': 'qubit',
                        'format-version': FORMAT_VERSION,
                        'fulqrum-version': VERSION,
                        'width': self.width,
                        'type':self.oper.type,
                        }
        cdef OperatorTerm_t * term
        cdef size_t kk, jj
        cdef list terms = []
        cdef list temp_inds
        cdef str temp_vals
        for kk in range(self.oper.terms.size()):
            term = &self.oper.terms[kk]
            temp_inds = []
            temp_vals = ''
            for jj in range(term.indices.size()):
                temp_inds.append(term.indices[jj])
                temp_vals += IND_TO_STR[term.values[jj]]
            terms.append([temp_vals, temp_inds, (term.coeff.real, term.coeff.imag)])
        out['terms'] = terms
        return out


    @classmethod
    def from_dict(self, dict dic):
        """QubitOperator from dictionary

        Parameters:
            dic(dict): Dictionary representation of operator

        Returns:
            QubitOperator
        """
        if dic['operator-type'] != 'qubit':
            raise FulqrumError("Dictionary operator-type is not 'qubit'")
        cdef size_t width = dic['width']
        cdef QubitOperator out = QubitOperator(width)
        for term in dic['terms']:
            out += QubitOperator(width, [(term[0], term[1], complex(*term[2]))])
        out.oper.type = dic['type']
        return out


    def to_json(self, filename, overwrite=False):
        """Save operator to a JSON, XZ, or ZST file. File extension can be 'json'
        'xz', or 'zst', the latter two do compression which is
        recommended for large operators.  'zst' is preferred for speed and size.

        Parameters:
            filename (str): File to store to
            overwrite (bool): Overwrite file if it exits, default=False
        """
        self.oper.to_json(str(filename), overwrite)


    @classmethod
    def from_json(self, filename):
        """Load operator from a JSON or XZ file.

        Parameters:
            filename (str): File to load from

        Returns:
            QubitOperator
        """
        if not Path(str(filename)).exists():
            raise FulqrumError("File does not exist")
        cdef QubitOperator out = QubitOperator(1) #dummy width
        out.oper = out.oper.from_json(str(filename))
        return out

    def fast_diag_compatible(self):
        """Is operator compatible with fast diagonal computation

        Returns:
            bool
        """
        return fast_diag_compatible(self.oper)
    
    def fast_diag_term_sort(self):
        """Sort a fast diagonal compatible operator into correct term ordering
        """
        return fast_diag_term_sort(self.oper)

