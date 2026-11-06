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


import cython
cimport cython

import numpy as np
cimport numpy as np

from cython.parallel cimport prange

cdef extern from "../../core/src/checked_size.hpp":
    size_t checked_multiply(size_t, size_t) except +
    size_t checked_add(size_t, size_t) except +
    I checked_index[I](size_t) except +

cdef extern from "../../core/src/key_words.hpp":
    size_t power_of_two_at_least(size_t) except +

cdef extern from *:
    int __builtin_popcountll(unsigned long long x) nogil
    int __builtin_ctzll(unsigned long long x) nogil

DEF _EMPTY = 0xFFFFFFFFFFFFFFFF


cdef inline unsigned long long _hash64(unsigned long long x) noexcept nogil:

    x = (x ^ (x >> 30)) * <unsigned long long>0xbf58476d1ce4e5b9
    x = (x ^ (x >> 27)) * <unsigned long long>0x94d049bb133111eb
    x = x ^ (x >> 31)
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _ht_get(
    np.uint64_t[::1] ht_keys,
    np.int32_t[::1] ht_vals,
    unsigned long long hmask,
    unsigned long long key,
) noexcept nogil:


    cdef Py_ssize_t slot = <Py_ssize_t>(_hash64(key) & hmask)
    while True:
        if ht_keys[slot] == key:
            return ht_vals[slot]
        if ht_keys[slot] == <unsigned long long>_EMPTY:
            return -1
        slot = (slot + 1) & hmask


cdef _empty_payload(object ptr_arr):

    return (
        ptr_arr,
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.uint8),
        np.zeros(1, dtype=np.uint8),
        np.zeros(1, dtype=np.uint8),
        np.zeros(1, dtype=np.uint8),
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def build_tile_flip_csr_dense(
    np.uint64_t[::1] bra,
    np.uint64_t[::1] src,
    int popcount,
    int nthreads,
):


    cdef Py_ssize_t n_bra = bra.shape[0]
    cdef Py_ssize_t n_src = src.shape[0]
    cdef Py_ssize_t r, s, pos, nnz
    cdef unsigned long long b, x, xx
    cdef int cnt, pc = popcount
    cdef np.int64_t running = 0
    cdef np.int32_t c
    cdef np.int64_t[::1] mv_ptr
    cdef np.int32_t[::1] mv_j, counts
    cdef np.uint8_t[::1] mv_p0, mv_p1, mv_p2, mv_p3

    checked_index[np.int32_t](n_src)
    checked_multiply(checked_add(n_bra, 1), sizeof(np.int64_t))
    ptr_arr = np.zeros(checked_index[Py_ssize_t](checked_add(n_bra, 1)), dtype=np.int64)
    if n_bra == 0 or n_src == 0:
        return _empty_payload(ptr_arr)
    if nthreads < 1:
        nthreads = 1

    mv_ptr = ptr_arr
    counts_arr = np.zeros(n_bra, dtype=np.int32)
    counts = counts_arr

    for r in prange(n_bra, nogil=True, schedule="static", num_threads=nthreads):
        b = bra[r]
        cnt = 0
        for s in range(n_src):
            if __builtin_popcountll(src[s] ^ b) == pc:
                cnt = cnt + 1
        counts[r] = cnt

    for r in range(n_bra):
        c = counts[r]
        mv_ptr[r] = running
        running = checked_index[np.int64_t](checked_add(running, c))
    mv_ptr[n_bra] = running
    nnz = running
    if nnz == 0:
        return _empty_payload(ptr_arr)

    checked_multiply(nnz, sizeof(np.int32_t) + 4 * sizeof(np.uint8_t))
    j_arr = np.empty(nnz, dtype=np.int32)
    p0_arr = np.zeros(nnz, dtype=np.uint8)
    p1_arr = np.zeros(nnz, dtype=np.uint8)
    p2_arr = np.zeros(nnz, dtype=np.uint8)
    p3_arr = np.zeros(nnz, dtype=np.uint8)
    mv_j = j_arr
    mv_p0 = p0_arr
    mv_p1 = p1_arr
    mv_p2 = p2_arr
    mv_p3 = p3_arr

    for r in prange(n_bra, nogil=True, schedule="static", num_threads=nthreads):
        b = bra[r]
        pos = mv_ptr[r]
        for s in range(n_src):
            x = src[s] ^ b
            if __builtin_popcountll(x) == pc:
                mv_j[pos] = <np.int32_t>s
                xx = x
                mv_p0[pos] = <np.uint8_t>__builtin_ctzll(xx)
                xx = xx & (xx - 1)
                if pc >= 2:
                    mv_p1[pos] = <np.uint8_t>__builtin_ctzll(xx)
                    xx = xx & (xx - 1)
                if pc >= 3:
                    mv_p2[pos] = <np.uint8_t>__builtin_ctzll(xx)
                    xx = xx & (xx - 1)
                if pc >= 4:
                    mv_p3[pos] = <np.uint8_t>__builtin_ctzll(xx)
                    xx = xx & (xx - 1)
                pos = pos + 1

    return ptr_arr, j_arr, p0_arr, p1_arr, p2_arr, p3_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def build_tile_flip_csr_masks(
    np.uint64_t[::1] bra,
    np.uint64_t[::1] src,
    np.uint64_t[::1] mask_vals,
    np.uint8_t[:, ::1] mask_pos,
    int nthreads,
):


    cdef Py_ssize_t n_bra = bra.shape[0]
    cdef Py_ssize_t n_src = src.shape[0]
    cdef Py_ssize_t n_masks = mask_vals.shape[0]
    cdef Py_ssize_t r, m, i, pos, cap, slot, nnz
    cdef unsigned long long b, cand, kv, hmask
    cdef int cnt, idx
    cdef np.int64_t running = 0
    cdef np.int32_t c
    cdef np.int64_t[::1] mv_ptr
    cdef np.int32_t[::1] mv_j, counts, ht_vals
    cdef np.uint8_t[::1] mv_p0, mv_p1, mv_p2, mv_p3
    cdef np.uint64_t[::1] ht_keys

    checked_index[np.int32_t](n_src)
    checked_multiply(checked_add(n_bra, 1), sizeof(np.int64_t))
    ptr_arr = np.zeros(checked_index[Py_ssize_t](checked_add(n_bra, 1)), dtype=np.int64)
    if n_bra == 0 or n_src == 0 or n_masks == 0:
        return _empty_payload(ptr_arr)
    if nthreads < 1:
        nthreads = 1
    mv_ptr = ptr_arr

    checked_index[np.int32_t](n_masks)
    if mask_pos.shape[0] != n_masks or mask_pos.shape[1] != 4:
        raise ValueError("Mask positions must have shape (number of masks, 4)")
    cap = checked_index[Py_ssize_t](power_of_two_at_least(checked_multiply(3, n_src)))
    checked_multiply(cap, sizeof(np.uint64_t) + sizeof(np.int32_t))
    hmask = <unsigned long long>(cap - 1)
    keys_arr = np.full(cap, np.uint64(_EMPTY), dtype=np.uint64)
    vals_arr = np.full(cap, np.int32(-1), dtype=np.int32)
    ht_keys = keys_arr
    ht_vals = vals_arr
    with nogil:
        for i in range(n_src):
            kv = src[i]
            slot = _hash64(kv) & hmask
            while ht_keys[slot] != <unsigned long long>_EMPTY:
                slot = (slot + 1) & hmask
            ht_keys[slot] = kv
            ht_vals[slot] = <np.int32_t>i

    counts_arr = np.zeros(n_bra, dtype=np.int32)
    counts = counts_arr

    for r in prange(n_bra, nogil=True, schedule="static", num_threads=nthreads):
        b = bra[r]
        cnt = 0
        for m in range(n_masks):
            cand = b ^ mask_vals[m]
            if _ht_get(ht_keys, ht_vals, hmask, cand) >= 0:
                cnt = cnt + 1
        counts[r] = cnt

    for r in range(n_bra):
        c = counts[r]
        mv_ptr[r] = running
        running = checked_index[np.int64_t](checked_add(running, c))
    mv_ptr[n_bra] = running
    nnz = running
    if nnz == 0:
        return _empty_payload(ptr_arr)

    checked_multiply(nnz, sizeof(np.int32_t) + 4 * sizeof(np.uint8_t))
    j_arr = np.empty(nnz, dtype=np.int32)
    p0_arr = np.zeros(nnz, dtype=np.uint8)
    p1_arr = np.zeros(nnz, dtype=np.uint8)
    p2_arr = np.zeros(nnz, dtype=np.uint8)
    p3_arr = np.zeros(nnz, dtype=np.uint8)
    mv_j = j_arr
    mv_p0 = p0_arr
    mv_p1 = p1_arr
    mv_p2 = p2_arr
    mv_p3 = p3_arr

    for r in prange(n_bra, nogil=True, schedule="static", num_threads=nthreads):
        b = bra[r]
        pos = mv_ptr[r]
        for m in range(n_masks):
            cand = b ^ mask_vals[m]
            idx = _ht_get(ht_keys, ht_vals, hmask, cand)
            if idx >= 0:
                mv_j[pos] = idx
                mv_p0[pos] = mask_pos[m, 0]
                mv_p1[pos] = mask_pos[m, 1]
                mv_p2[pos] = mask_pos[m, 2]
                mv_p3[pos] = mask_pos[m, 3]
                pos = pos + 1

    return ptr_arr, j_arr, p0_arr, p1_arr, p2_arr, p3_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def tile_has_popcount(
    np.uint64_t[::1] bra,
    np.uint64_t[::1] src,
    int popcount,
    int nthreads,
):


    cdef Py_ssize_t n_bra = bra.shape[0]
    cdef Py_ssize_t n_src = src.shape[0]
    cdef Py_ssize_t r, s
    cdef unsigned long long b
    cdef int pc = popcount
    cdef np.int32_t[::1] found_mv

    if n_bra == 0 or n_src == 0:
        return False
    if nthreads < 1:
        nthreads = 1

    found_arr = np.zeros(1, dtype=np.int32)
    found_mv = found_arr

    for r in prange(n_bra, nogil=True, schedule="dynamic", num_threads=nthreads):
        if found_mv[0] == 0:
            b = bra[r]
            for s in range(n_src):
                if __builtin_popcountll(src[s] ^ b) == pc:
                    found_mv[0] = 1
                    break

    return found_mv[0] != 0
