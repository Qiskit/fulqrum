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


from libc.stdint cimport int32_t, int64_t, uint16_t, uint32_t, uint64_t, uintptr_t
from libc.string cimport memset


cdef extern from "../kernels/full_strs/params.hpp":
    cdef struct FsKeys:
        const uint64_t* keys
        const uint64_t* halo_keys
        int64_t num_rows
        int32_t num_orbitals
        int32_t reserved

    cdef struct FsTables:
        const double* single_value
        const double* single_delta
        const double* single_reference
        const double* double_value
        const double* mixed_value
        const uint64_t* reference
        int64_t num_pairs
        int64_t num_quads
        int32_t num_orbitals
        int32_t reserved

    cdef struct FsVectors:
        const double* x
        double* y
        const double* diagonal
        int64_t num_rows

    cdef struct FsGroups:
        const uint32_t* members
        const int32_t* offsets
        double* partials
        double* y
        int64_t first_group
        int64_t num_groups
        int32_t cls
        int32_t bucket
        uint32_t output_begin
        uint32_t output_end
        uint32_t source_begin
        uint32_t source_end

    cdef struct FsReduce:
        int32_t* rows
        int32_t* rows_alt
        double* values
        double* values_alt
        void* scratch
        double* y
        int64_t num_entries
        uint64_t scratch_bytes
        int32_t end_bit
        int32_t row_begin
        int32_t row_end
        int32_t reserved

    cdef struct FsRuns:
        const int32_t* perm
        const int32_t* run_rows
        const int32_t* run_starts
        const double* values
        double* y
        int64_t num_runs

    cdef struct FsRdm:
        double* gamma_alpha
        double* gamma_beta
        double* same_alpha
        double* same_beta
        double* other_alpha
        double* other_beta
        double* canonical_alpha
        double* canonical_beta
        double* mixed
        int32_t num_orbitals
        int32_t half

    cdef enum FsParamStruct:
        kFsKeys
        kFsTables
        kFsVectors
        kFsGroups
        kFsReduce
        kFsRuns
        kFsRdm
        kFsParamStructCount


cdef extern from "../kernels/full_strs/launch.hpp":
    int fs_launch_diagonal(const FsVectors* vectors, void* stream) nogil
    int fs_launch_groups_uniform(const FsKeys* keys, const FsTables* tables, const FsGroups* groups, const double* x,
                                 void* stream) nogil
    int fs_launch_groups_binned(const FsKeys* keys, const FsTables* tables, const FsGroups* groups, const double* x,
                                void* stream) nogil
    size_t fs_reduce_scratch_bytes(int64_t num_entries, int end_bit)
    int fs_launch_reduce(const FsReduce* reduce, void* stream) nogil
    size_t fs_reduce_plan_scratch_bytes(int64_t num_entries, int end_bit)
    int fs_build_reduce_plan(int32_t* rows, int32_t* rows_alt, int32_t* positions, int32_t* positions_alt, void* scratch,
                             size_t scratch_bytes, int64_t num_entries, int end_bit, int32_t* sorted_positions,
                             int32_t* run_rows, int32_t* run_counts, int32_t* num_runs, void* stream) nogil
    int fs_launch_gather_runs(const FsRuns* runs, void* stream) nogil
    size_t fs_group_offsets_scratch_bytes(int64_t num_groups)
    int fs_launch_group_offsets(const uint16_t* sizes, int64_t num_groups, int32_t* offsets, void* scratch,
                                size_t scratch_bytes, void* stream) nogil
    int fs_launch_rdm_groups(const FsKeys* keys, const FsGroups* groups, const FsRdm* out, const double* x,
                             void* stream) nogil
    int fs_launch_pair_occupation(const FsKeys* keys, const double* x, double* occupation, int sectors,
                                  int num_orbitals, void* stream) nogil
    size_t fs_param_struct_size(int which)
    const char* fs_error_name(int status)


def _check_struct_sizes():

    sizes = {
        kFsKeys: sizeof(FsKeys),
        kFsTables: sizeof(FsTables),
        kFsVectors: sizeof(FsVectors),
        kFsGroups: sizeof(FsGroups),
        kFsReduce: sizeof(FsReduce),
        kFsRuns: sizeof(FsRuns),
        kFsRdm: sizeof(FsRdm),
    }
    for which, size in sizes.items():
        if fs_param_struct_size(which) != size:
            raise RuntimeError(
                f"full-string kernel parameter struct {which} has {fs_param_struct_size(which)} bytes in "
                f"libfull_strs_cuda.so but {size} bytes here; rebuild the GPU module"
            )


_check_struct_sizes()


def _launch_error(str what, int status):

    return RuntimeError(f"{what} failed: {fs_error_name(status).decode()} ({status})")


def launch_diagonal(uintptr_t x, uintptr_t y, uintptr_t diagonal, int64_t num_rows, uintptr_t stream=0):


    cdef FsVectors vectors
    memset(&vectors, 0, sizeof(vectors))
    vectors.x = <const double*>x
    vectors.y = <double*>y
    vectors.diagonal = <const double*>diagonal
    vectors.num_rows = num_rows
    cdef int status
    with nogil:
        status = fs_launch_diagonal(&vectors, <void*>stream)
    if status != 0:
        raise _launch_error("full-string diagonal launch", status)


def group_offsets_scratch_bytes(int64_t num_groups):

    return fs_group_offsets_scratch_bytes(num_groups)


def launch_group_offsets(uintptr_t sizes, int64_t num_groups, uintptr_t offsets, uintptr_t scratch,
                         size_t scratch_bytes, uintptr_t stream=0):


    if not 0 <= num_groups < 2**31:
        raise ValueError(f"num_groups must be in [0, 2^31); got {num_groups}")
    cdef int status
    with nogil:
        status = fs_launch_group_offsets(<const uint16_t*>sizes, num_groups, <int32_t*>offsets, <void*>scratch,
                                         scratch_bytes, <void*>stream)
    if status != 0:
        raise _launch_error("full-string group offsets", status)


cdef FsTables _tables(dict tables) except *:


    cdef FsTables out
    memset(&out, 0, sizeof(out))
    out.single_value = <const double*><uintptr_t>tables["single_value"]
    out.single_delta = <const double*><uintptr_t>tables["single_delta"]
    out.single_reference = <const double*><uintptr_t>tables["single_reference"]
    out.double_value = <const double*><uintptr_t>tables["double_value"]
    out.mixed_value = <const double*><uintptr_t>tables["mixed_value"]
    out.reference = <const uint64_t*><uintptr_t>tables["reference"]
    out.num_pairs = tables["num_pairs"]
    out.num_quads = tables["num_quads"]
    out.num_orbitals = tables["num_orbitals"]
    return out


def launch_groups_uniform(uintptr_t keys, uintptr_t halo_keys, int64_t num_rows, dict tables, uintptr_t members,
                          uintptr_t offsets, uintptr_t partials, int64_t first_group, int64_t num_groups, int cls,
                          tuple ranges, uintptr_t x, uintptr_t stream=0, uintptr_t y=0):


    _launch_groups(keys, halo_keys, num_rows, tables, members, offsets, partials, first_group, num_groups, cls, -1,
                   ranges, x, stream, y)


def launch_groups_binned(uintptr_t keys, uintptr_t halo_keys, int64_t num_rows, dict tables, uintptr_t members,
                         uintptr_t offsets, uintptr_t partials, int64_t first_group, int64_t num_groups, int cls,
                         int bucket, tuple ranges, uintptr_t x, uintptr_t stream=0, uintptr_t y=0):


    if not 0 <= bucket <= 6:
        raise ValueError(f"bucket must be in 0 to 6; got {bucket}")
    _launch_groups(keys, halo_keys, num_rows, tables, members, offsets, partials, first_group, num_groups, cls, bucket,
                   ranges, x, stream, y)


cdef _launch_groups(uintptr_t keys, uintptr_t halo_keys, int64_t num_rows, dict tables, uintptr_t members,
                    uintptr_t offsets, uintptr_t partials, int64_t first_group, int64_t num_groups, int cls, int bucket,
                    tuple ranges, uintptr_t x, uintptr_t stream, uintptr_t y=0):


    cdef FsKeys key_struct
    memset(&key_struct, 0, sizeof(key_struct))
    key_struct.keys = <const uint64_t*>keys
    key_struct.halo_keys = <const uint64_t*>halo_keys
    key_struct.num_rows = num_rows
    key_struct.num_orbitals = tables["num_orbitals"]
    cdef FsTables table_struct = _tables(tables)
    cdef FsGroups group_struct
    memset(&group_struct, 0, sizeof(group_struct))
    group_struct.members = <const uint32_t*>members
    group_struct.offsets = <const int32_t*>offsets
    group_struct.partials = <double*>partials
    group_struct.y = <double*>y
    group_struct.first_group = first_group
    group_struct.num_groups = num_groups
    group_struct.cls = cls
    group_struct.bucket = bucket if bucket >= 0 else 0
    group_struct.output_begin = ranges[0]
    group_struct.output_end = ranges[1]
    group_struct.source_begin = ranges[2]
    group_struct.source_end = ranges[3]
    cdef int status
    with nogil:
        if bucket < 0:
            status = fs_launch_groups_uniform(&key_struct, &table_struct, &group_struct, <const double*>x, <void*>stream)
        else:
            status = fs_launch_groups_binned(&key_struct, &table_struct, &group_struct, <const double*>x, <void*>stream)
    if status != 0:
        raise _launch_error("full-string group kernel", status)


def reduce_scratch_bytes(int64_t num_entries, int end_bit):

    return fs_reduce_scratch_bytes(num_entries, end_bit)


def launch_reduce(uintptr_t rows, uintptr_t rows_alt, uintptr_t values, uintptr_t values_alt, uintptr_t scratch,
                  size_t scratch_bytes, int64_t num_entries, int end_bit, int32_t row_begin, int32_t row_end, uintptr_t y,
                  uintptr_t stream=0):


    if not 0 <= num_entries < 2**31:
        raise ValueError(f"num_entries must be in [0, 2^31); got {num_entries}")
    cdef FsReduce reduce
    memset(&reduce, 0, sizeof(reduce))
    reduce.rows = <int32_t*>rows
    reduce.rows_alt = <int32_t*>rows_alt
    reduce.values = <double*>values
    reduce.values_alt = <double*>values_alt
    reduce.scratch = <void*>scratch
    reduce.y = <double*>y
    reduce.num_entries = num_entries
    reduce.scratch_bytes = scratch_bytes
    reduce.end_bit = end_bit
    reduce.row_begin = row_begin
    reduce.row_end = row_end
    cdef int status
    with nogil:
        status = fs_launch_reduce(&reduce, <void*>stream)
    if status != 0:
        raise _launch_error("full-string reduce", status)


def reduce_plan_scratch_bytes(int64_t num_entries, int end_bit):

    return fs_reduce_plan_scratch_bytes(num_entries, end_bit)


def build_reduce_plan(uintptr_t rows, uintptr_t rows_alt, uintptr_t positions, uintptr_t positions_alt,
                      uintptr_t scratch, size_t scratch_bytes, int64_t num_entries, int end_bit,
                      uintptr_t sorted_positions, uintptr_t run_rows, uintptr_t run_counts, uintptr_t num_runs,
                      uintptr_t stream=0):


    if not 0 <= num_entries < 2**31:
        raise ValueError(f"num_entries must be in [0, 2^31); got {num_entries}")
    cdef int status
    with nogil:
        status = fs_build_reduce_plan(<int32_t*>rows, <int32_t*>rows_alt, <int32_t*>positions, <int32_t*>positions_alt,
                                      <void*>scratch, scratch_bytes, num_entries, end_bit, <int32_t*>sorted_positions,
                                      <int32_t*>run_rows, <int32_t*>run_counts, <int32_t*>num_runs, <void*>stream)
    if status != 0:
        raise _launch_error("full-string reduce plan", status)


def launch_gather_runs(uintptr_t perm, uintptr_t run_rows, uintptr_t run_starts, int64_t num_runs, uintptr_t values,
                       uintptr_t y, uintptr_t stream=0):


    cdef FsRuns runs
    memset(&runs, 0, sizeof(runs))
    runs.perm = <const int32_t*>perm
    runs.run_rows = <const int32_t*>run_rows
    runs.run_starts = <const int32_t*>run_starts
    runs.values = <const double*>values
    runs.y = <double*>y
    runs.num_runs = num_runs
    cdef int status
    with nogil:
        status = fs_launch_gather_runs(&runs, <void*>stream)
    if status != 0:
        raise _launch_error("full-string gather runs", status)


def launch_rdm_groups(uintptr_t keys, uintptr_t halo_keys, int64_t num_rows, int num_orbitals, uintptr_t members,
                      uintptr_t offsets, int64_t first_group, int64_t num_groups, int cls, tuple ranges, uintptr_t x,
                      dict rdm, uintptr_t stream=0, int bucket=-1, int half=0):


    if not -1 <= bucket <= 6:
        raise ValueError(f"bucket must be in -1 to 6; got {bucket}")
    if half not in (0, 1, 2):
        raise ValueError(f"half must be 0, 1 or 2; got {half}")
    cdef FsKeys key_struct
    memset(&key_struct, 0, sizeof(key_struct))
    key_struct.keys = <const uint64_t*>keys
    key_struct.halo_keys = <const uint64_t*>halo_keys
    key_struct.num_rows = num_rows
    key_struct.num_orbitals = num_orbitals
    cdef FsGroups group_struct
    memset(&group_struct, 0, sizeof(group_struct))
    group_struct.members = <const uint32_t*>members
    group_struct.offsets = <const int32_t*>offsets
    group_struct.first_group = first_group
    group_struct.num_groups = num_groups
    group_struct.cls = cls
    group_struct.bucket = bucket
    group_struct.output_begin = ranges[0]
    group_struct.output_end = ranges[1]
    group_struct.source_begin = ranges[2]
    group_struct.source_end = ranges[3]
    cdef FsRdm out
    memset(&out, 0, sizeof(out))
    out.gamma_alpha = <double*><uintptr_t>rdm["gamma_alpha"]
    out.gamma_beta = <double*><uintptr_t>rdm["gamma_beta"]
    out.same_alpha = <double*><uintptr_t>rdm["same_alpha"]
    out.same_beta = <double*><uintptr_t>rdm["same_beta"]
    out.other_alpha = <double*><uintptr_t>rdm["other_alpha"]
    out.other_beta = <double*><uintptr_t>rdm["other_beta"]
    out.canonical_alpha = <double*><uintptr_t>rdm["canonical_alpha"]
    out.canonical_beta = <double*><uintptr_t>rdm["canonical_beta"]
    out.mixed = <double*><uintptr_t>rdm["mixed"]
    out.num_orbitals = num_orbitals
    out.half = half
    cdef int status
    with nogil:
        status = fs_launch_rdm_groups(&key_struct, &group_struct, &out, <const double*>x, <void*>stream)
    if status != 0:
        raise _launch_error("full-string RDM group launch", status)


def launch_pair_occupation(uintptr_t keys, int64_t num_rows, uintptr_t x, uintptr_t occupation, int sectors,
                           int num_orbitals, uintptr_t stream=0):


    if sectors not in (0, 1, 2):
        raise ValueError(f"sectors must be 0, 1 or 2; got {sectors}")
    cdef FsKeys key_struct
    memset(&key_struct, 0, sizeof(key_struct))
    key_struct.keys = <const uint64_t*>keys
    key_struct.num_rows = num_rows
    key_struct.num_orbitals = num_orbitals
    cdef int status
    with nogil:
        status = fs_launch_pair_occupation(&key_struct, <const double*>x, <double*>occupation, sectors, num_orbitals,
                                           <void*>stream)
    if status != 0:
        raise _launch_error("full-string pair occupation launch", status)
