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
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort as stdsort
from libcpp cimport bool
from libc.math cimport abs

import time
import itertools
import math
cimport cython
import numpy as np
cimport numpy as np

import logging
logger = logging.getLogger(__name__)

from ..exceptions import FulqrumError
from .bitset cimport bitset_t, to_string
from .bitset_view cimport BitsetView
from .bitset cimport Bitset
from .bitset_hashmap cimport BitsetHashMapWrapper

include "includes/base_header.pxi"
include "includes/bitset_utils_header.pxi"
include "includes/types.pxi"

from .constants cimport MAX_SIZE_T

cdef extern from "src/subspace.hpp":
    size_t subspace_find_key(Subspace_t&, const bitset_t&) except +


cdef class Subspace():
    """Hashmap representation of a quantum subspace over bit-strings obtained from
    sampling on a quantum computer (or simulator).

    Parameters:
        subspace_strs: Input bitstrings as either length-1 or length-2 tuple
            (or list of list(s)).

            Two modes of input to subspace is supported:
            1. Half-string mode: In this mode, the input `subspace_strs`
            to the `Subspace()` is a length-2  `tuple[list[str], list[str]]`,
            where the first element represents _alpha_ strings, and the second
            one represents _beta_ strings. Internally, `fulqrum` will sort each
            list and perform a Cartesian product between the two lists to construct
            the full subspace. A full bitstring will be represented by
            `beta half str + alpha half str`, and the final length of subspace
            bitstring will be `len(beta half string) + len(alpha half string)`.
            For example, `[['100', '001, '010'], ['110', '000']]` is a valid input
            `subspace_strs` in half-string mode.
            The final subspace size will be `2 * 3 = 6`, with each bitstrings
            with `3 + 3 = 6` characters/bits. This mode is useful for chemistry applications.

            2. Full-string mode: In this mode, `subspace_strs` has to be length-1
            `tuple[list[str]]`. In this mode, no Cartesian product is taken internally,
            and the supplied _full_ bitstrings are used as is after sorting.

        reserve_multiplier (float): We reserve a capacity for the Hash table that stores the
            subspace bit-strings, typically equal to the number of bit-strings. This
            argument allows a user to reserve more capacity than needed. While it consumes,
            more memory, it reduces collision during Hash table look-up leading to
            minor speed-up.
            Default: 2.

        use_all_bitset_blocks (bool): If `use_all_bitset_blocks=False`, only first block of a
            bitset is used in hashing. If your bitsets are long and rarely share common
            prefixes, setting it to False speeds up execution. However, it likely that
            bitsets for practical cases will share common patterns. In that case, set
            `use_all_bitset_blocks` to True so that the whole bitset is used in the hash
            function. Although hashing n (> 1) blocks is slower than hashing a single block,
            full hashing usually leads to fewer collisions during Hash table look-up.
            Default: `True`.

    Example:

    .. jupyter-execute::

        import fulqrum as fq

        num_qubits = 5
        bitstrings = []
        for val in range(2**num_qubits):
            bitstrings.append(bin(val)[2:].zfill(num_qubits))

        fq.Subspace([bitstrings])

    """
    @cython.boundscheck(False)
    def __cinit__(self, object subspace_strs=None, int reserve_multiplier=2, bool use_all_bitset_blocks=True):
        """
        args:
            subspace_strs: Input bitstrings as either length-1 or length-2 tuple
                (or list of list(s)).

                Two modes of input to subspace is supported:
                1. Half-string mode: In this mode, the input `subspace_strs`
                to the `Subspace()` is a length-2  `tuple[list[str], list[str]]`,
                where the first element represents _alpha_ strings, and the second
                one represents _beta_ strings. Internally, `fulqrum` will sort each
                list and perform a Cartesian product between the two lists to construct
                the full subspace. A full bitstring will be represented by
                `beta half str + alpha half str`, and the final length of subspace
                bitstring will be `len(beta half string) + len(alpha half string)`.
                For example, `[['100', '001, '010'], ['110', '000']]` is a valid input
                `subspace_strs` in half-string mode.
                The final subspace size will be `2 * 3 = 6`, with each bitstrings
                with `3 + 3 = 6` characters/bits. This mode is useful for chemistry applications.

                2. Full-string mode: In this mode, `subspace_strs` has to be length-1
                `tuple[list[str]]`. In this mode, no Cartesian product is taken internally,
                and the supplied _full_ bitstrings are used as is after sorting.

            reserve_multiplier: We reserve a capacity for the Hash table that stores the
                subspace bitstrings, typically equal to the number of bitstrings. This
                argument allows a user to reserve more capacity than needed. While it consumes,
                more memory, it reduces collision during Hash table look-up leading to
                minor speed-up.
                Default: 2.

            use_all_bitset_blocks: If `use_all_bitset_blocks=False`, only first block of a
                bitset is used in hashing. If your bitsets are long and rarely share common
                prefixes, setting it to False speeds up execution. However, it likely that
                bitsets for practical cases will share common patterns. In that case, set
                `use_all_bitset_blocks` to True so that the whole bitset is used in the hash
                function. Although hashing n (> 1) blocks is slower than hashing a single block,
                full hashing usually leads to fewer collisions during Hash table look-up.
                Default: `True`.
        """
        cdef vector[string] full_strs
        cdef vector[string] alpha
        cdef vector[string] beta
        cdef string text
        cdef Bitset bit_key
        self.subspace.configure(reserve_multiplier, use_all_bitset_blocks)
        if not subspace_strs:
            return
        if len(subspace_strs) == 1:
            iterator = subspace_strs[0]
            iterator.sort()
            for value in iterator:
                if isinstance(value, Bitset):
                    bit_key = value
                    to_string(bit_key.bits, text)
                    full_strs.push_back(text)
                elif isinstance(value, str):
                    full_strs.push_back(value)
                else:
                    raise ValueError("Subspace inputs must be bitstrings or Bitset objects")
            self.subspace.initialize_full(full_strs)
        elif len(subspace_strs) == 2:
            alpha = subspace_strs[0]
            beta = subspace_strs[1]
            stdsort(alpha.begin(), alpha.end())
            stdsort(beta.begin(), beta.end())
            self.subspace.initialize_product(alpha, beta)
        else:
            raise ValueError("bitstrings are wrongly formatted")

    @cython.boundscheck(False)
    def __getitem__(self, object key):
        if key < 0:
            key = self.subspace.size + key
        if key < 0 or key >= self.subspace.size:
            raise IndexError("Subspace index is out of range")
        cdef size_t idx = <size_t>key
        cdef bitset_t bits = self.subspace.key_at(idx)
        cdef Bitset out = Bitset()
        out.bits = bits
        return out

    def __len__(self):
        cdef size_t size = self.subspace.size
        return size

    def __repr__(self):
        temp_str = f"size={self.subspace.size}, "
        temp_str += f"width={self.subspace.num_qubits}, "
        temp_str += f"num_buckets={self.subspace.bitstrings.num_buckets()}, "
        temp_str += f"use_all_bitset_blocks={self.subspace.bitstrings.use_all_bitset_blocks()}"
        return f"<Subspace: {temp_str}>"

    def size(self):
        """Size (dimensionality) of subspace

        Returns:
            int
        """
        cdef size_t size = self.subspace.size
        return size
    
    @property
    def width(self):
        return self.subspace.num_qubits

    def copy(self):
        """Return a copy of subspace

        Returns:
            Subspace
        """
        cdef Subspace out = Subspace()
        self.subspace.copy_to(out.subspace)
        return out


    @cython.boundscheck(False)
    def get_orbital_occupancies(self, double[::1] probs, int norb):
        """Computes orbital occupancies of electrons.

        Args:
            probs (np.ndarray): Probabilities of each basis vector of a wavefunction.
                It must be the absolute value squared of an eigenvector, which represents
                amplitudes of each basis vector (Slated determinant) in a wavefunction.
            norb (int): The number of spatial orbitals.

        Returns:
            Orbital occupanies as a length-2 tuple of 1D np.ndarrays.
            It is formatted as ``([a0, ..., aN], [b0, ..., bN])``.
        """
        if norb < 0 or 2 * norb != self.subspace.num_qubits:
            raise ValueError("norb must match the two equal spin sectors")
        if probs.shape[0] != self.subspace.size:
            raise ValueError("Probability vector length does not match the subspace")
        out = np.zeros(2 * norb, dtype=np.float64)
        cdef double[::1] out_mv = out
        cdef bitset_t key
        cdef size_t row, qubit
        for row in range(self.subspace.size):
            key = self.subspace.key_at(row)
            for qubit in range(self.subspace.num_qubits):
                if key[qubit]:
                    out_mv[qubit] += probs[row]

        return np.split(out, 2)


    @cython.boundscheck(False)
    def get_bitstring_index(self, str bitstring):
        """Return the index of the given bitstring.

        Return value is max(size_t) if bitset not in subspace

        Args:
            bitstring (str): The bitstring as Python ``str`` type.

        Returns:
            size_t: Index if the bitstring is found.
            None: If the bitstring is not found.
        """
        cdef size_t index

        cdef bitset_t bitset
        bitset = bitset_t(bitstring, 0, len(bitstring))
        index = subspace_find_key(self.subspace, bitset)

        return None if index == MAX_SIZE_T else index


    @cython.boundscheck(False)
    def interpret_vector(self, double_or_complex[::1] vec, double atol=1e-12, int sort=0, int renormalize=True):
        """Convert solution vector into dict of counts and complex amplitudes

        Parameters:
            vec (ndarray): Complex solution vector
            atol (double): Absolute tolerance for truncation, default=1e-12
            sort (int): Sort output dict by integer representation.
            renormalize (bool): Renormalize values such that probabilities sum to one, default = True

        Returns:
            dict: Dictionary with bit-string keys and complex values

        Notes:
            Truncation can be disabled by calling `atol=0`
        """
        if vec.shape[0] != self.subspace.size:
            raise ValueError("Vector length does not match the subspace")
        cdef size_t kk
        cdef double abs_val
        cdef double reduced_prob = 0
        cdef string s
        # cdef bytes py_bytes
        cdef dict out = {}

        for kk in range(self.subspace.size):
            abs_val = abs(vec[kk])
            if abs_val <= atol:
                continue
            to_string(self.subspace.key_at(kk), s)

            out[s] = vec[kk]
            reduced_prob += abs_val * abs_val

        if renormalize:
            reduced_prob = math.sqrt(reduced_prob)
            for key in out:
                out[key] /= reduced_prob

        if sort:
            out = {k: v for k, v in sorted(out.items(), key=lambda item: int(item[0], 2))}
        return out


    def get_n_th_bitstring(self, size_t n):
        """Return n-th bitstring in the Subspace

        Parameters:
            n (size_t): Index of the expected bitstring.

        Returns:
            str: N-th bitstring in the subspace. Note that, both Python
                dictionary and emhash8::HashMap retains the insertion order.
        """
        cdef string s
        to_string(self.subspace.key_at(n), s)
        return s


    def get_bitset_index(self, Bitset bitset):
        """Return the index of the given Bitset.

        Return value is max(size_t) if bitset not in subspace

        Returns:
            size_t: Index
        """
        return subspace_find_key(self.subspace, bitset.bits)


    @cython.boundscheck(False)
    def to_dict(self, str key_type='bitset'):
        """Converts Subspace to a dictionary

        Parameters:
            key_type (str): Type of key data to return, default='bitset'

        Returns:
            dict
        """
        cdef size_t kk
        cdef string s
        cdef Bitset temp_bits
        cdef bitset_t bits
        cdef dict out = {}


        if key_type not in ['bitset', 'str']:
            raise FulqrumError("key_type must be 'bitset' or 'str'")

        if key_type == 'bitset':
            for kk in range(self.subspace.size):
                bits = self.subspace.key_at(kk)
                temp_bits = Bitset()
                temp_bits.bits = bits
                out[temp_bits] = None

        elif key_type == 'str':
            for kk in range(self.subspace.size):
                to_string(self.subspace.key_at(kk), s)
                out[s] = None
        return out

    @property
    def half_strs_mode(self):
        return self.subspace.product

    @property
    def num_alpha_qubits(self):
        return self.subspace.alpha_width

    @property
    def alpha_strs(self):
        if not self.subspace.product:
            return None
        return [self.subspace.half_string(0, i) for i in range(self.subspace.half_size(0))]

    @property
    def beta_strs(self):
        if not self.subspace.product:
            return None
        return [self.subspace.half_string(1, i) for i in range(self.subspace.half_size(1))]

    def to_alpha_beta_arrays(self):
        if not self.subspace.product:
            raise FulqrumError("to_alpha_beta_arrays() requires a half-string Subspace")
        if self.subspace.alpha_width > 64 or self.subspace.num_qubits - self.subspace.alpha_width > 64:
            raise FulqrumError("Each GPU half must have at most 64 qubits")
        return (np.array([int(s, 2) for s in self.alpha_strs], dtype=np.uint64),
                np.array([int(s, 2) for s in self.beta_strs], dtype=np.uint64))

    def _storage_state(self):
        return {"map_materialized": self.subspace.map_materialized,
                "map_build_count": self.subspace.map_build_count,
                "full_buffer_words": self.subspace.full_keys.size(),
                "product": self.subspace.product}
