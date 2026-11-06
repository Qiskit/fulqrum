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

# distutils: language = c++
# cython: language_level=3


from libc.stdint cimport int32_t, int64_t, uint64_t, uintptr_t
from libc.string cimport memset


cdef extern from "../kernels/rdm/params.hpp":
    cdef struct RdmSlabPair:
        const double* bra
        const double* ket
        int64_t width
        int64_t bra_offset
        int64_t ket_offset
        int64_t bra_row_stride
        int64_t bra_elem_stride

    cdef struct RdmPairs:
        const int64_t* row_ptr
        const int32_t* cols
        double* overlap
        const uint64_t* keys
        int64_t bra_offset
        int64_t num_bra

    cdef struct RdmDiagonal:
        const double* weight
        const uint64_t* keys
        int64_t bra_offset
        int64_t num_bra

    cdef struct RdmSingles:
        const int32_t* source
        const int32_t* target
        const double* sign
        const int32_t* label
        int64_t count

    cdef struct RdmTensor:
        double* values
        int32_t num_orbitals
        int32_t reserved

    cdef struct RdmAbSlabs:
        const double* own
        const double* source
        int64_t own_width
        int64_t source_width
        int64_t own_row_offset
        int64_t own_column_offset
        int64_t source_row_offset
        int64_t source_column_offset

    cdef struct RdmWeighted:
        double* gamma
        double* alpha_beta
        const uint64_t* column_keys
        int64_t column_offset
        int32_t num_orbitals
        int32_t reserved

    cdef struct RdmRows:
        const uint64_t* row_keys
        int64_t row_offset
        int64_t num_rows

    cdef struct RdmAbLists:
        const int32_t* alpha_source
        const int32_t* alpha_target
        const double* alpha_sign
        const int64_t* label_start
        const int64_t* alpha_offset
        const int32_t* beta_source
        const int32_t* beta_target
        const double* beta_sign
        const int64_t* beta_offset
        int64_t beta_count

    cdef enum RdmParamStruct:
        kRdmSlabPair
        kRdmPairs
        kRdmDiagonal
        kRdmSingles
        kRdmTensor
        kRdmAbSlabs
        kRdmAbLists
        kRdmWeighted
        kRdmRows
        kRdmParamStructCount


cdef extern from "../kernels/rdm/launch.hpp":
    int rdm_launch_pair_overlap(const RdmSlabPair* slabs, const RdmPairs* pairs, void* stream) nogil
    int rdm_launch_same_scatter(const RdmPairs* pairs, const RdmTensor* out, void* stream) nogil
    size_t rdm_same_diagonal_shared_bytes(int32_t num_orbitals)
    int rdm_launch_same_diagonal(const RdmDiagonal* diagonal, const RdmTensor* out, void* stream) nogil
    int rdm_launch_alpha_one_body(const RdmSlabPair* slabs, const RdmSingles* singles, const RdmWeighted* out,
                                  void* stream) nogil
    int rdm_launch_beta_one_body(const RdmSlabPair* slabs, const RdmSingles* singles, const RdmWeighted* out,
                                 void* stream) nogil
    int rdm_launch_alpha_beta(const RdmAbSlabs* slabs, const RdmAbLists* lists, const RdmTensor* out,
                              void* stream) nogil
    int rdm_launch_diagonal(const RdmSlabPair* slabs, const RdmRows* rows, const RdmWeighted* out, int beta_side,
                            void* stream) nogil
    size_t rdm_param_struct_size(int which)
    const char* rdm_error_name(int status)


def _check_struct_sizes():

    sizes = {
        kRdmSlabPair: sizeof(RdmSlabPair),
        kRdmPairs: sizeof(RdmPairs),
        kRdmDiagonal: sizeof(RdmDiagonal),
        kRdmSingles: sizeof(RdmSingles),
        kRdmTensor: sizeof(RdmTensor),
        kRdmAbSlabs: sizeof(RdmAbSlabs),
        kRdmAbLists: sizeof(RdmAbLists),
        kRdmWeighted: sizeof(RdmWeighted),
        kRdmRows: sizeof(RdmRows),
    }
    for which, size in sizes.items():
        if rdm_param_struct_size(which) != size:
            raise RuntimeError(
                f"RDM kernel parameter struct {which} has {rdm_param_struct_size(which)} bytes in "
                f"librdm_cuda.so but {size} bytes here; rebuild the GPU module"
            )


_check_struct_sizes()


def _check(str what, int status):

    if status != 0:
        raise RuntimeError(f"RDM kernel {what} failed: {rdm_error_name(status).decode()} ({status})")


cdef RdmSlabPair _slab_pair(tuple slabs):


    cdef RdmSlabPair out
    out.bra = <const double*><uintptr_t>slabs[0]
    out.ket = <const double*><uintptr_t>slabs[1]
    out.width = slabs[2]
    out.bra_offset = slabs[3]
    out.ket_offset = slabs[4]
    out.bra_row_stride = slabs[5] if len(slabs) > 5 else slabs[2]
    out.bra_elem_stride = slabs[6] if len(slabs) > 5 else 1
    return out


cdef RdmTensor _tensor(uintptr_t values, int num_orbitals):

    cdef RdmTensor out
    memset(&out, 0, sizeof(out))
    out.values = <double*>values
    out.num_orbitals = num_orbitals
    return out


def launch_pair_overlap(tuple slabs, uintptr_t row_ptr, uintptr_t cols, uintptr_t overlap, uintptr_t keys,
                        int64_t bra_offset, int64_t num_bra, uintptr_t stream=0):


    cdef RdmSlabPair s = _slab_pair(slabs)
    cdef RdmPairs p
    p.row_ptr = <const int64_t*>row_ptr
    p.cols = <const int32_t*>cols
    p.overlap = <double*>overlap
    p.keys = <const uint64_t*>keys
    p.bra_offset = bra_offset
    p.num_bra = num_bra
    cdef int status
    with nogil:
        status = rdm_launch_pair_overlap(&s, &p, <void*>stream)
    _check("pair_overlap", status)


def launch_same_scatter(uintptr_t row_ptr, uintptr_t cols, uintptr_t overlap, uintptr_t keys, int64_t bra_offset,
                        int64_t num_bra, uintptr_t out, int num_orbitals, uintptr_t stream=0):


    cdef RdmPairs p
    p.row_ptr = <const int64_t*>row_ptr
    p.cols = <const int32_t*>cols
    p.overlap = <double*>overlap
    p.keys = <const uint64_t*>keys
    p.bra_offset = bra_offset
    p.num_bra = num_bra
    cdef RdmTensor t = _tensor(out, num_orbitals)
    cdef int status
    with nogil:
        status = rdm_launch_same_scatter(&p, &t, <void*>stream)
    _check("same_scatter", status)


def launch_same_diagonal(uintptr_t weight, uintptr_t keys, int64_t bra_offset, int64_t num_bra, uintptr_t out,
                         int num_orbitals, uintptr_t stream=0):


    cdef RdmDiagonal d
    d.weight = <const double*>weight
    d.keys = <const uint64_t*>keys
    d.bra_offset = bra_offset
    d.num_bra = num_bra
    cdef RdmTensor t = _tensor(out, num_orbitals)
    cdef int status
    with nogil:
        status = rdm_launch_same_diagonal(&d, &t, <void*>stream)
    _check("same_diagonal", status)


def launch_one_body(bint beta_side, tuple slabs, uintptr_t source, uintptr_t target, uintptr_t sign, uintptr_t label,
                    int64_t count, uintptr_t gamma, uintptr_t alpha_beta, uintptr_t column_keys,
                    int64_t column_offset, int num_orbitals, uintptr_t stream=0):


    cdef RdmSlabPair s = _slab_pair(slabs)
    cdef RdmSingles x
    cdef RdmWeighted w
    x.source = <const int32_t*>source
    x.target = <const int32_t*>target
    x.sign = <const double*>sign
    x.label = <const int32_t*>label
    x.count = count
    memset(&w, 0, sizeof(w))
    w.gamma = <double*>gamma
    w.alpha_beta = <double*>alpha_beta
    w.column_keys = <const uint64_t*>column_keys
    w.column_offset = column_offset
    w.num_orbitals = num_orbitals
    cdef int status
    with nogil:
        if beta_side:
            status = rdm_launch_beta_one_body(&s, &x, &w, <void*>stream)
        else:
            status = rdm_launch_alpha_one_body(&s, &x, &w, <void*>stream)
    _check("beta_one_body" if beta_side else "alpha_one_body", status)


def launch_diagonal(bint beta_side, tuple slabs, uintptr_t row_keys, int64_t row_offset, int64_t num_rows,
                    uintptr_t gamma, uintptr_t alpha_beta, uintptr_t column_keys, int64_t column_offset,
                    int num_orbitals, uintptr_t stream=0):


    cdef RdmSlabPair s = _slab_pair(slabs)
    cdef RdmRows r
    cdef RdmWeighted w
    r.row_keys = <const uint64_t*>row_keys
    r.row_offset = row_offset
    r.num_rows = num_rows
    memset(&w, 0, sizeof(w))
    w.gamma = <double*>gamma
    w.alpha_beta = <double*>alpha_beta
    w.column_keys = <const uint64_t*>column_keys
    w.column_offset = column_offset
    w.num_orbitals = num_orbitals
    cdef int status
    cdef int side = 1 if beta_side else 0
    with nogil:
        status = rdm_launch_diagonal(&s, &r, &w, side, <void*>stream)
    _check("diagonal", status)


def launch_alpha_beta(dict tiles, dict alpha, uintptr_t beta_source, uintptr_t beta_target, uintptr_t beta_sign,
                      uintptr_t beta_offset, int64_t beta_count, uintptr_t out, int num_orbitals, uintptr_t stream=0):


    cdef RdmAbSlabs s
    cdef RdmAbLists x
    s.own = <const double*><uintptr_t>tiles["own"]
    s.source = <const double*><uintptr_t>tiles["source"]
    s.own_width = tiles["own_width"]
    s.source_width = tiles["source_width"]
    s.own_row_offset = tiles["own_row_offset"]
    s.own_column_offset = tiles["own_column_offset"]
    s.source_row_offset = tiles["source_row_offset"]
    s.source_column_offset = tiles["source_column_offset"]
    x.alpha_source = <const int32_t*><uintptr_t>alpha["source"]
    x.alpha_target = <const int32_t*><uintptr_t>alpha["target"]
    x.alpha_sign = <const double*><uintptr_t>alpha["sign"]
    x.label_start = <const int64_t*><uintptr_t>alpha["label_start"]
    x.alpha_offset = <const int64_t*><uintptr_t>alpha["offset"]
    x.beta_source = <const int32_t*>beta_source
    x.beta_target = <const int32_t*>beta_target
    x.beta_sign = <const double*>beta_sign
    x.beta_offset = <const int64_t*>beta_offset
    x.beta_count = beta_count
    cdef RdmTensor t = _tensor(out, num_orbitals)
    cdef int status
    with nogil:
        status = rdm_launch_alpha_beta(&s, &x, &t, <void*>stream)
    _check("alpha_beta", status)
