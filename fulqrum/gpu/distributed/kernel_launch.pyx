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


from libc.string cimport memset
from libc.stdint cimport int8_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t, uintptr_t


cdef extern from "../kernels/formats/params.hpp":
    cdef struct FqTileGeometry:
        int32_t bra_a0
        int32_t n_a_bra
        int32_t bra_b0
        int32_t n_b_bra
        int32_t src_a0
        int32_t n_a_src
        int32_t src_b0
        int32_t n_b_src
        int32_t a_start
        int32_t a_stop
        int32_t b_start
        int32_t b_stop
        int32_t n_alpha_src_compact
        int32_t threads_per_block

    cdef struct FqVectors:
        const double* src_in
        double* out_local
        const int32_t* alpha_src_slot
        const int32_t* beta_src_slot

    cdef struct FqDiagonalTables:
        const double* diag_local
        const double* diag_alpha
        const double* diag_beta
        const double* diag_Mb
        const uint64_t* alpha_strings
        int32_t num_alpha
        int32_t reserved

    cdef struct FqAlphaSinglePairTables:
        const int64_t* sell_offsets
        const int32_t* sell_lengths
        const uint64_t* sell_words
        const double* c0_sell
        const double* pair_table
        int32_t num_pairs
        int32_t rows_per_thread
        int32_t hoist_loads
        int32_t reserved

    cdef struct FqAlphaSingleDeltaTables:
        const int64_t* row_offsets
        const int32_t* columns
        const double* c0
        const double* delta
        const uint64_t* beta_strings
        int32_t num_beta
        int32_t rows_per_thread

    cdef struct FqBetaSinglePairTables:
        const int64_t* row_offsets
        const int32_t* columns
        const double* c0
        const uint32_t* signed_pairs
        const double* pair_table
        int32_t table_columns
        int32_t alphas_per_thread
        int32_t block_group
        int32_t hoist_loads

    cdef struct FqBetaSingleDeltaTables:
        const int64_t* row_offsets
        const int32_t* columns
        const double* c0
        const double* delta
        const uint64_t* alpha_strings
        int32_t num_alpha
        int32_t alphas_per_thread

    cdef struct FqAlphaDoubleTables:
        const int64_t* sell_offsets
        const int32_t* sell_lengths
        const int32_t* sell_columns
        const double* values
        int32_t rows_per_thread
        int32_t min_blocks
        int32_t identity_slot
        int32_t block_group

    cdef struct FqBetaDoubleTables:
        const int64_t* row_offsets
        const int32_t* columns
        const double* values
        int32_t alphas_per_thread
        int32_t block_group

    cdef struct FqBetaDoubleClassTables:
        const double* blocks
        const int64_t* block_offsets
        const int32_t* tile_info
        const int32_t* row_members
        const int32_t* col_members
        int32_t* nonfinite
        int32_t num_tiles
        int32_t alpha_group

    cdef struct FqAlphaDoubleClassTables:
        FqBetaDoubleClassTables classes
        double* xt
        double* yt
        int32_t slab
        int32_t n_alpha_src
        FqBetaDoubleTables leftover
        const int32_t* slab_identity

    cdef struct FqCrossSellTables:
        const int64_t* beta_row_offsets
        const int32_t* beta_columns
        const uint8_t* beta_low
        const uint8_t* beta_high
        const uint64_t* beta_strings
        const double* coefficients
        const int64_t* sell_offsets
        const int32_t* sell_lengths
        const uint64_t* sell_words
        int32_t num_alpha
        int32_t num_beta

    cdef struct FqCrossDirectTables:
        const int64_t* sell_offsets
        const int32_t* sell_lengths
        const uint64_t* sell_words
        const uint64_t* beta_words
        const int32_t* tasks
        const double* coefficients
        int32_t num_pairs
        int32_t num_tasks
        const int32_t* order
        int32_t num_rows
        int32_t gated

    cdef struct FqCrossCliqueTables:
        const uint64_t* edge_words
        const int32_t* group_offsets
        const int32_t* group_pairs
        const int32_t* tile_group_offsets
        const int32_t* parent_offsets
        const uint64_t* members
        const int32_t* parent_ids
        const double* coefficients
        int32_t num_parents
        int32_t num_tiles
        int32_t tile_rows
        int32_t orbitals
        int32_t bucket
        int32_t edges_per_batch
        int32_t transposed
        int32_t parent_group
        int32_t mma
        int32_t distinct_rows
        double* panel
        int32_t* nonfinite
        int32_t panel_columns
        int32_t reserved

    cdef struct FqCrossLbgTables:
        const uint64_t* edge_words
        const int32_t* segment_begin
        const int32_t* segment_end
        const int32_t* segment_pair
        double* out_panel
        const int32_t* nonfinite
        int32_t num_segments
        int32_t out_stride

    cdef struct FqCrossRslTables:
        const int32_t* block_first
        const int32_t* block_count
        const uint64_t* members
        double* panel
        const int32_t* tasks
        const uint64_t* beta_words
        const double* coefficients
        const double* coef_table
        const int32_t* nonfinite
        int32_t num_blocks
        int32_t rows_per_job
        int32_t panel_rows
        int32_t num_pairs
        int32_t orbitals
        int32_t members_per_clique

    cdef struct FqTermTables:
        const double* term_header
        const uint64_t* term_inds_vals
        const uint8_t* term_proj_inds
        const uint8_t* term_proj_bits
        const uint32_t* ladder_offsets
        const int8_t* oper_elems
        const uint32_t* pair_keys
        const int32_t* pair_values
        const uint32_t* pair_empty
        const uint64_t* quad_keys
        const int32_t* quad_values
        const uint64_t* quad_empty
        const uint64_t* alpha_strings
        const uint64_t* beta_strings
        uint64_t quad_mask
        uint32_t pair_mask
        uint32_t ladder_offset
        int32_t term_inds_stride
        int32_t term_proj_stride
        int32_t num_alpha
        int32_t num_beta

    cdef struct FqSinglesBuildTables:
        const int64_t* row_offsets
        const int32_t* columns
        const uint8_t* low
        const uint8_t* high
        const int32_t* alpha_slot
        const int32_t* beta_slot
        double* c0
        double* delta
        int32_t* error

    cdef struct FqDoublesBuildTables:
        const int64_t* row_offsets
        const int32_t* row_lengths
        const int32_t* columns
        const uint8_t* p0
        const uint8_t* p1
        const uint8_t* p2
        const uint8_t* p3
        const int32_t* alpha_slot
        const int32_t* beta_slot
        double* values


cdef extern from "../kernels/formats/params.hpp":
    cdef enum FqParamStruct:
        kFqTileGeometry
        kFqVectors
        kFqDiagonalTables
        kFqAlphaSinglePairTables
        kFqAlphaSingleDeltaTables
        kFqBetaSinglePairTables
        kFqBetaSingleDeltaTables
        kFqAlphaDoubleTables
        kFqBetaDoubleTables
        kFqCrossSellTables
        kFqCrossDirectTables
        kFqCrossCliqueTables
        kFqTermTables
        kFqSinglesBuildTables
        kFqDoublesBuildTables
        kFqBetaDoubleClassTables
        kFqAlphaDoubleClassTables
        kFqCrossLbgTables
        kFqCrossRslTables
        kFqParamStructCount


cdef extern from "../kernels/launch.hpp":
    int fq_launch_diagonal(const FqTileGeometry* geometry, const FqVectors* vectors,
                           const FqDiagonalTables* tables, void* stream) nogil
    int fq_launch_alpha_singles_pair(const FqTileGeometry* geometry, const FqVectors* vectors,
                                     const FqAlphaSinglePairTables* tables, void* stream) nogil
    int fq_launch_alpha_singles_delta(const FqTileGeometry* geometry, const FqVectors* vectors,
                                      const FqAlphaSingleDeltaTables* tables, void* stream) nogil
    int fq_launch_beta_singles_pair(const FqTileGeometry* geometry, const FqVectors* vectors,
                                    const FqBetaSinglePairTables* tables, void* stream) nogil
    int fq_launch_beta_singles_delta(const FqTileGeometry* geometry, const FqVectors* vectors,
                                     const FqBetaSingleDeltaTables* tables, void* stream) nogil
    int fq_launch_alpha_doubles(const FqTileGeometry* geometry, const FqVectors* vectors,
                                const FqAlphaDoubleTables* tables, void* stream) nogil
    int fq_launch_beta_doubles(const FqTileGeometry* geometry, const FqVectors* vectors,
                               const FqBetaDoubleTables* tables, void* stream) nogil
    int fq_launch_beta_doubles_class(const FqTileGeometry* geometry, const FqVectors* vectors,
                                     const FqBetaDoubleClassTables* tables, void* stream) nogil
    int fq_launch_alpha_doubles_class(const FqTileGeometry* geometry, const FqVectors* vectors,
                                      const FqAlphaDoubleClassTables* tables, void* stream) nogil
    int fq_launch_cross_sell(const FqTileGeometry* geometry, const FqVectors* vectors,
                             const FqCrossSellTables* tables, void* stream) nogil
    int fq_launch_cross_direct(const FqTileGeometry* geometry, const FqVectors* vectors,
                               const FqCrossDirectTables* tables, void* stream) nogil
    int fq_prepare_cross_clique(int max_shared_bytes) nogil
    int fq_launch_build_alpha_doubles(const FqTileGeometry* geometry, const FqTermTables* terms,
                                      const FqDoublesBuildTables* tables, void* stream) nogil
    int fq_launch_build_beta_doubles(const FqTileGeometry* geometry, const FqTermTables* terms,
                                     const FqDoublesBuildTables* tables, void* stream) nogil
    int fq_launch_build_alpha_singles(const FqTileGeometry* geometry, const FqTermTables* terms,
                                      const FqSinglesBuildTables* tables, void* stream) nogil
    int fq_launch_build_beta_singles(const FqTileGeometry* geometry, const FqTermTables* terms,
                                     const FqSinglesBuildTables* tables, void* stream) nogil
    int fq_launch_cross_clique(const FqTileGeometry* geometry, const FqVectors* vectors,
                               const FqCrossCliqueTables* tables, void* stream) nogil
    int fq_cross_clique_panel_row(int bucket) nogil
    int fq_launch_cross_clique_gated(const FqTileGeometry* geometry, const FqVectors* vectors,
                                     const FqCrossCliqueTables* tables, void* stream) nogil
    int fq_launch_cross_clique_pack(const FqTileGeometry* geometry, const FqVectors* vectors,
                                    const FqCrossCliqueTables* tables, void* stream) nogil
    int fq_launch_lbg_finite(const double* source, long long n, int32_t* nonfinite, void* stream) nogil
    int fq_launch_cross_lbg(const FqTileGeometry* geometry, const FqVectors* vectors,
                            const FqCrossCliqueTables* tables, const FqCrossLbgTables* lbg, void* stream) nogil
    int fq_launch_cross_rsl(const FqTileGeometry* geometry, const FqVectors* vectors,
                            const FqCrossRslTables* rsl, void* stream) nogil
    size_t fq_param_struct_size(int which)
    const char* fq_error_name(int status)


def _check_struct_sizes():

    sizes = {
        kFqTileGeometry: sizeof(FqTileGeometry),
        kFqVectors: sizeof(FqVectors),
        kFqDiagonalTables: sizeof(FqDiagonalTables),
        kFqAlphaSinglePairTables: sizeof(FqAlphaSinglePairTables),
        kFqAlphaSingleDeltaTables: sizeof(FqAlphaSingleDeltaTables),
        kFqBetaSinglePairTables: sizeof(FqBetaSinglePairTables),
        kFqBetaSingleDeltaTables: sizeof(FqBetaSingleDeltaTables),
        kFqAlphaDoubleTables: sizeof(FqAlphaDoubleTables),
        kFqBetaDoubleTables: sizeof(FqBetaDoubleTables),
        kFqCrossSellTables: sizeof(FqCrossSellTables),
        kFqCrossDirectTables: sizeof(FqCrossDirectTables),
        kFqCrossCliqueTables: sizeof(FqCrossCliqueTables),
        kFqTermTables: sizeof(FqTermTables),
        kFqSinglesBuildTables: sizeof(FqSinglesBuildTables),
        kFqDoublesBuildTables: sizeof(FqDoublesBuildTables),
        kFqBetaDoubleClassTables: sizeof(FqBetaDoubleClassTables),
        kFqAlphaDoubleClassTables: sizeof(FqAlphaDoubleClassTables),
        kFqCrossLbgTables: sizeof(FqCrossLbgTables),
        kFqCrossRslTables: sizeof(FqCrossRslTables),
    }
    for which, size in sizes.items():
        if fq_param_struct_size(which) != size:
            raise RuntimeError(
                f"kernel parameter struct {which} has {fq_param_struct_size(which)} bytes in "
                f"libdist_matvec_cuda.so but {size} bytes here; rebuild the GPU module"
            )


_check_struct_sizes()


def _launch_error(str what, int status):

    return RuntimeError(f"{what} failed: {fq_error_name(status).decode()} ({status})")


def param_struct_digest():

    import hashlib

    sizes = ",".join(str(fq_param_struct_size(which)) for which in range(<int> kFqParamStructCount))
    return hashlib.sha256(sizes.encode()).hexdigest()[:16]


cdef FqTileGeometry _geometry(tuple values) except *:


    cdef FqTileGeometry geometry
    memset(&geometry, 0, sizeof(geometry))
    (geometry.bra_a0, geometry.n_a_bra, geometry.bra_b0, geometry.n_b_bra, geometry.src_a0,
     geometry.n_a_src, geometry.src_b0, geometry.n_b_src, geometry.a_start, geometry.a_stop,
     geometry.b_start, geometry.b_stop, geometry.n_alpha_src_compact) = values[:13]
    geometry.threads_per_block = values[13] if len(values) > 13 else 256
    return geometry


cdef FqVectors _vectors(uintptr_t src_in, uintptr_t out_local, uintptr_t alpha_slot, uintptr_t beta_slot):

    cdef FqVectors vectors
    memset(&vectors, 0, sizeof(vectors))
    vectors.src_in = <const double*> src_in
    vectors.out_local = <double*> out_local
    vectors.alpha_src_slot = <const int32_t*> alpha_slot
    vectors.beta_src_slot = <const int32_t*> beta_slot
    return vectors


def launch_diagonal(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t alpha_slot,
                    uintptr_t beta_slot, uintptr_t diag_local, uintptr_t diag_alpha, uintptr_t diag_beta,
                    uintptr_t diag_Mb, uintptr_t alpha_strings, int num_alpha, uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, alpha_slot, beta_slot)
    cdef FqDiagonalTables tables
    memset(&tables, 0, sizeof(tables))
    tables.diag_local = <const double*> diag_local
    tables.diag_alpha = <const double*> diag_alpha
    tables.diag_beta = <const double*> diag_beta
    tables.diag_Mb = <const double*> diag_Mb
    tables.alpha_strings = <const uint64_t*> alpha_strings
    tables.num_alpha = num_alpha
    tables.reserved = 0
    cdef int status
    with nogil:
        status = fq_launch_diagonal(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("diagonal kernel launch", status)


def launch_alpha_singles_pair(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t alpha_slot,
                              uintptr_t beta_slot, uintptr_t sell_offsets, uintptr_t sell_lengths,
                              uintptr_t sell_words, uintptr_t c0_sell, uintptr_t pair_table, int num_pairs,
                              int rows_per_thread, uintptr_t stream, bint hoist_loads=False):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, alpha_slot, beta_slot)
    cdef FqAlphaSinglePairTables tables
    memset(&tables, 0, sizeof(tables))
    tables.sell_offsets = <const int64_t*> sell_offsets
    tables.sell_lengths = <const int32_t*> sell_lengths
    tables.sell_words = <const uint64_t*> sell_words
    tables.c0_sell = <const double*> c0_sell
    tables.pair_table = <const double*> pair_table
    tables.num_pairs = num_pairs
    tables.rows_per_thread = rows_per_thread
    tables.hoist_loads = 1 if hoist_loads else 0
    cdef int status
    with nogil:
        status = fq_launch_alpha_singles_pair(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("alpha singles (pair table) kernel launch", status)


def launch_alpha_singles_delta(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t alpha_slot,
                               uintptr_t beta_slot, uintptr_t row_offsets, uintptr_t columns, uintptr_t c0,
                               uintptr_t delta, uintptr_t beta_strings, int num_beta, int rows_per_thread,
                               uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, alpha_slot, beta_slot)
    cdef FqAlphaSingleDeltaTables tables
    memset(&tables, 0, sizeof(tables))
    tables.row_offsets = <const int64_t*> row_offsets
    tables.columns = <const int32_t*> columns
    tables.c0 = <const double*> c0
    tables.delta = <const double*> delta
    tables.beta_strings = <const uint64_t*> beta_strings
    tables.num_beta = num_beta
    tables.rows_per_thread = rows_per_thread
    cdef int status
    with nogil:
        status = fq_launch_alpha_singles_delta(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("alpha singles (delta) kernel launch", status)


def launch_beta_singles_pair(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t alpha_slot,
                             uintptr_t beta_slot, uintptr_t row_offsets, uintptr_t columns, uintptr_t c0,
                             uintptr_t signed_pairs, uintptr_t pair_table, int table_columns,
                             int alphas_per_thread, uintptr_t stream, int block_group=0,
                             bint hoist_loads=False):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, alpha_slot, beta_slot)
    cdef FqBetaSinglePairTables tables
    memset(&tables, 0, sizeof(tables))
    tables.row_offsets = <const int64_t*> row_offsets
    tables.columns = <const int32_t*> columns
    tables.c0 = <const double*> c0
    tables.signed_pairs = <const uint32_t*> signed_pairs
    tables.pair_table = <const double*> pair_table
    tables.table_columns = table_columns
    tables.alphas_per_thread = alphas_per_thread
    tables.block_group = block_group
    tables.hoist_loads = 1 if hoist_loads else 0
    cdef int status
    with nogil:
        status = fq_launch_beta_singles_pair(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("beta singles (pair table) kernel launch", status)


def launch_beta_singles_delta(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t alpha_slot,
                              uintptr_t beta_slot, uintptr_t row_offsets, uintptr_t columns, uintptr_t c0,
                              uintptr_t delta, uintptr_t alpha_strings, int num_alpha, int alphas_per_thread,
                              uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, alpha_slot, beta_slot)
    cdef FqBetaSingleDeltaTables tables
    memset(&tables, 0, sizeof(tables))
    tables.row_offsets = <const int64_t*> row_offsets
    tables.columns = <const int32_t*> columns
    tables.c0 = <const double*> c0
    tables.delta = <const double*> delta
    tables.alpha_strings = <const uint64_t*> alpha_strings
    tables.num_alpha = num_alpha
    tables.alphas_per_thread = alphas_per_thread
    cdef int status
    with nogil:
        status = fq_launch_beta_singles_delta(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("beta singles (delta) kernel launch", status)


def launch_alpha_doubles(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t alpha_slot,
                         uintptr_t beta_slot, uintptr_t sell_offsets, uintptr_t sell_lengths,
                         uintptr_t sell_columns, uintptr_t values, int rows_per_thread, int min_blocks,
                         bint identity_slot, uintptr_t stream, int block_group=0):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, alpha_slot, beta_slot)
    cdef FqAlphaDoubleTables tables
    memset(&tables, 0, sizeof(tables))
    tables.sell_offsets = <const int64_t*> sell_offsets
    tables.sell_lengths = <const int32_t*> sell_lengths
    tables.sell_columns = <const int32_t*> sell_columns
    tables.values = <const double*> values
    tables.rows_per_thread = rows_per_thread
    tables.min_blocks = min_blocks
    tables.identity_slot = 1 if identity_slot else 0
    tables.block_group = block_group
    cdef int status
    with nogil:
        status = fq_launch_alpha_doubles(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("alpha doubles kernel launch", status)


def launch_beta_doubles(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t alpha_slot,
                        uintptr_t beta_slot, uintptr_t row_offsets, uintptr_t columns, uintptr_t values,
                        int alphas_per_thread, uintptr_t stream, int block_group=0):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, alpha_slot, beta_slot)
    cdef FqBetaDoubleTables tables
    memset(&tables, 0, sizeof(tables))
    tables.row_offsets = <const int64_t*> row_offsets
    tables.columns = <const int32_t*> columns
    tables.values = <const double*> values
    tables.alphas_per_thread = alphas_per_thread
    tables.block_group = block_group
    cdef int status
    with nogil:
        status = fq_launch_beta_doubles(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("beta doubles kernel launch", status)


def launch_beta_doubles_class(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t alpha_slot,
                              uintptr_t beta_slot, uintptr_t blocks, uintptr_t block_offsets, uintptr_t tile_info,
                              uintptr_t row_members, uintptr_t col_members, uintptr_t nonfinite, int num_tiles,
                              int alpha_group, uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, alpha_slot, beta_slot)
    cdef FqBetaDoubleClassTables tables
    memset(&tables, 0, sizeof(tables))
    tables.blocks = <const double*> blocks
    tables.block_offsets = <const int64_t*> block_offsets
    tables.tile_info = <const int32_t*> tile_info
    tables.row_members = <const int32_t*> row_members
    tables.col_members = <const int32_t*> col_members
    tables.nonfinite = <int32_t*> nonfinite
    tables.num_tiles = num_tiles
    tables.alpha_group = alpha_group
    cdef int status
    with nogil:
        status = fq_launch_beta_doubles_class(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("beta doubles class kernel launch", status)


def launch_alpha_doubles_class(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t alpha_slot,
                               uintptr_t beta_slot, uintptr_t blocks, uintptr_t block_offsets, uintptr_t tile_info,
                               uintptr_t row_members, uintptr_t col_members, uintptr_t nonfinite, int num_tiles,
                               int alpha_group, uintptr_t xt, uintptr_t yt, int slab, int n_alpha_src,
                               uintptr_t stream, uintptr_t leftover_offsets=0, uintptr_t leftover_columns=0,
                               uintptr_t leftover_values=0, uintptr_t slab_identity=0,
                               int leftover_alphas_per_thread=4, int leftover_block_group=0):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, alpha_slot, beta_slot)
    cdef FqAlphaDoubleClassTables tables
    memset(&tables, 0, sizeof(tables))
    tables.classes.blocks = <const double*> blocks
    tables.classes.block_offsets = <const int64_t*> block_offsets
    tables.classes.tile_info = <const int32_t*> tile_info
    tables.classes.row_members = <const int32_t*> row_members
    tables.classes.col_members = <const int32_t*> col_members
    tables.classes.nonfinite = <int32_t*> nonfinite
    tables.classes.num_tiles = num_tiles
    tables.classes.alpha_group = alpha_group
    tables.xt = <double*> xt
    tables.yt = <double*> yt
    tables.slab = slab
    tables.n_alpha_src = n_alpha_src
    tables.leftover.row_offsets = <const int64_t*> leftover_offsets
    tables.leftover.columns = <const int32_t*> leftover_columns
    tables.leftover.values = <const double*> leftover_values
    tables.leftover.alphas_per_thread = leftover_alphas_per_thread
    tables.leftover.block_group = leftover_block_group
    tables.slab_identity = <const int32_t*> slab_identity
    cdef int status
    with nogil:
        status = fq_launch_alpha_doubles_class(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("alpha doubles class launch", status)


def launch_cross_sell(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t beta_slot,
                      uintptr_t beta_row_offsets, uintptr_t beta_columns, uintptr_t beta_low,
                      uintptr_t beta_high, uintptr_t beta_strings, uintptr_t coefficients,
                      uintptr_t sell_offsets, uintptr_t sell_lengths, uintptr_t sell_words, int num_alpha,
                      int num_beta, uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, 0, beta_slot)
    cdef FqCrossSellTables tables
    memset(&tables, 0, sizeof(tables))
    tables.beta_row_offsets = <const int64_t*> beta_row_offsets
    tables.beta_columns = <const int32_t*> beta_columns
    tables.beta_low = <const uint8_t*> beta_low
    tables.beta_high = <const uint8_t*> beta_high
    tables.beta_strings = <const uint64_t*> beta_strings
    tables.coefficients = <const double*> coefficients
    tables.sell_offsets = <const int64_t*> sell_offsets
    tables.sell_lengths = <const int32_t*> sell_lengths
    tables.sell_words = <const uint64_t*> sell_words
    tables.num_alpha = num_alpha
    tables.num_beta = num_beta
    cdef int status
    with nogil:
        status = fq_launch_cross_sell(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("cross SELL kernel launch", status)


def launch_cross_direct(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t sell_offsets,
                        uintptr_t sell_lengths, uintptr_t sell_words, uintptr_t beta_words, uintptr_t tasks,
                        uintptr_t coefficients, int num_pairs, int num_tasks, uintptr_t stream,
                        uintptr_t order=0, int num_rows=0, uintptr_t gate=0):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, 0, gate)
    cdef FqCrossDirectTables tables
    memset(&tables, 0, sizeof(tables))
    tables.sell_offsets = <const int64_t*> sell_offsets
    tables.sell_lengths = <const int32_t*> sell_lengths
    tables.sell_words = <const uint64_t*> sell_words
    tables.beta_words = <const uint64_t*> beta_words
    tables.tasks = <const int32_t*> tasks
    tables.coefficients = <const double*> coefficients
    tables.order = <const int32_t*> order
    tables.num_pairs = num_pairs
    tables.num_tasks = num_tasks
    tables.num_rows = num_rows
    tables.gated = 1 if gate else 0
    cdef int status
    with nogil:
        status = fq_launch_cross_direct(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("cross direct kernel launch", status)


def prepare_cross_clique(int max_shared_bytes):


    cdef int status
    with nogil:
        status = fq_prepare_cross_clique(max_shared_bytes)
    if status:
        raise _launch_error("clique kernel preparation", status)


def launch_cross_clique(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t edge_words,
                        uintptr_t group_offsets, uintptr_t group_pairs, uintptr_t tile_group_offsets,
                        uintptr_t parent_offsets, uintptr_t members, uintptr_t parent_ids, uintptr_t coefficients,
                        int num_parents, int num_tiles, int tile_rows, int orbitals, int bucket,
                        int edges_per_batch, bint transposed, uintptr_t stream, int parent_group=0,
                        bint mma=False, bint distinct_rows=False, uintptr_t panel=0, uintptr_t nonfinite=0,
                        int panel_columns=0):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, 0, 0)
    cdef FqCrossCliqueTables tables
    memset(&tables, 0, sizeof(tables))
    tables.edge_words = <const uint64_t*> edge_words
    tables.group_offsets = <const int32_t*> group_offsets
    tables.group_pairs = <const int32_t*> group_pairs
    tables.tile_group_offsets = <const int32_t*> tile_group_offsets
    tables.parent_offsets = <const int32_t*> parent_offsets
    tables.members = <const uint64_t*> members
    tables.parent_ids = <const int32_t*> parent_ids
    tables.coefficients = <const double*> coefficients
    tables.num_parents = num_parents
    tables.num_tiles = num_tiles
    tables.tile_rows = tile_rows
    tables.orbitals = orbitals
    tables.bucket = bucket
    tables.edges_per_batch = edges_per_batch
    tables.transposed = 1 if transposed else 0
    tables.parent_group = parent_group
    tables.mma = 1 if mma else 0
    tables.distinct_rows = 1 if distinct_rows else 0
    tables.panel = <double*> panel
    tables.nonfinite = <int32_t*> nonfinite
    tables.panel_columns = panel_columns
    cdef int status
    with nogil:
        status = fq_launch_cross_clique(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("cross clique kernel launch", status)


def clique_panel_row(int bucket):

    return fq_cross_clique_panel_row(bucket)


cdef FqCrossCliqueTables _clique_tables(tuple c) except *:


    cdef FqCrossCliqueTables tables
    memset(&tables, 0, sizeof(tables))
    tables.edge_words = <const uint64_t*> <uintptr_t> c[0]
    tables.group_offsets = <const int32_t*> <uintptr_t> c[1]
    tables.group_pairs = <const int32_t*> <uintptr_t> c[2]
    tables.tile_group_offsets = <const int32_t*> <uintptr_t> c[3]
    tables.parent_offsets = <const int32_t*> <uintptr_t> c[4]
    tables.members = <const uint64_t*> <uintptr_t> c[5]
    tables.parent_ids = <const int32_t*> <uintptr_t> c[6]
    tables.coefficients = <const double*> <uintptr_t> c[7]
    tables.num_parents = c[8]
    tables.num_tiles = c[9]
    tables.tile_rows = c[10]
    tables.orbitals = c[11]
    tables.bucket = c[12]
    tables.edges_per_batch = c[13]
    tables.mma = 1
    tables.distinct_rows = 1 if c[14] else 0
    tables.panel = <double*> <uintptr_t> c[15]
    tables.nonfinite = <int32_t*> <uintptr_t> c[16]
    tables.panel_columns = c[17]
    return tables


def launch_lbg_finite(uintptr_t source, long long n, uintptr_t nonfinite, uintptr_t stream):


    cdef int status
    with nogil:
        status = fq_launch_lbg_finite(<const double*> source, n, <int32_t*> nonfinite, <void*> stream)
    if status:
        raise _launch_error("lbg finiteness scan", status)


def launch_cross_clique_gated(tuple geometry, uintptr_t src_in, uintptr_t out_local, tuple clique, uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, 0, 0)
    cdef FqCrossCliqueTables tables = _clique_tables(clique)
    cdef int status
    with nogil:
        status = fq_launch_cross_clique_gated(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("cross clique fallback launch", status)


def launch_cross_clique_pack(tuple geometry, uintptr_t src_in, tuple clique, uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, 0, 0, 0)
    cdef FqCrossCliqueTables tables = _clique_tables(clique)
    cdef int status
    with nogil:
        status = fq_launch_cross_clique_pack(&tile, &vectors, &tables, <void*> stream)
    if status:
        raise _launch_error("cross clique pack launch", status)


def launch_cross_lbg(tuple geometry, uintptr_t src_in, uintptr_t out_local, tuple clique, uintptr_t edge_words,
                     uintptr_t segment_begin, uintptr_t segment_end, uintptr_t segment_pair, int num_segments,
                     uintptr_t out_panel, int out_stride, uintptr_t nonfinite, uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, 0, 0)
    cdef FqCrossCliqueTables tables = _clique_tables(clique)
    cdef FqCrossLbgTables lbg
    memset(&lbg, 0, sizeof(lbg))
    lbg.edge_words = <const uint64_t*> edge_words
    lbg.segment_begin = <const int32_t*> segment_begin
    lbg.segment_end = <const int32_t*> segment_end
    lbg.segment_pair = <const int32_t*> segment_pair
    lbg.out_panel = <double*> out_panel
    lbg.nonfinite = <const int32_t*> nonfinite
    lbg.num_segments = num_segments
    lbg.out_stride = out_stride
    cdef int status
    with nogil:
        status = fq_launch_cross_lbg(&tile, &vectors, &tables, &lbg, <void*> stream)
    if status:
        raise _launch_error("cross lbg launch", status)


def launch_cross_rsl(tuple geometry, uintptr_t src_in, uintptr_t out_local, uintptr_t block_first,
                     uintptr_t block_count, uintptr_t members, uintptr_t panel, uintptr_t tasks, uintptr_t beta_words,
                     uintptr_t coefficients, uintptr_t coef_table, uintptr_t nonfinite, int num_blocks, int rows_per_job,
                     int panel_rows, int num_pairs, int orbitals, int members_per_clique, uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqVectors vectors = _vectors(src_in, out_local, 0, 0)
    cdef FqCrossRslTables rsl
    memset(&rsl, 0, sizeof(rsl))
    rsl.block_first = <const int32_t*> block_first
    rsl.block_count = <const int32_t*> block_count
    rsl.members = <const uint64_t*> members
    rsl.panel = <double*> panel
    rsl.tasks = <const int32_t*> tasks
    rsl.beta_words = <const uint64_t*> beta_words
    rsl.coefficients = <const double*> coefficients
    rsl.coef_table = <const double*> coef_table
    rsl.nonfinite = <const int32_t*> nonfinite
    rsl.num_blocks = num_blocks
    rsl.rows_per_job = rows_per_job
    rsl.panel_rows = panel_rows
    rsl.num_pairs = num_pairs
    rsl.orbitals = orbitals
    rsl.members_per_clique = members_per_clique
    cdef int status
    with nogil:
        status = fq_launch_cross_rsl(&tile, &vectors, &rsl, <void*> stream)
    if status:
        raise _launch_error("cross rsl launch", status)


cdef FqTermTables _terms(tuple values) except *:


    cdef FqTermTables terms
    memset(&terms, 0, sizeof(terms))
    terms.term_header = <const double*> <uintptr_t> values[0]
    terms.term_inds_vals = <const uint64_t*> <uintptr_t> values[1]
    terms.term_proj_inds = <const uint8_t*> <uintptr_t> values[2]
    terms.term_proj_bits = <const uint8_t*> <uintptr_t> values[3]
    terms.ladder_offsets = <const uint32_t*> <uintptr_t> values[4]
    terms.oper_elems = <const int8_t*> <uintptr_t> values[5]
    terms.pair_keys = <const uint32_t*> <uintptr_t> values[6]
    terms.pair_values = <const int32_t*> <uintptr_t> values[7]
    terms.pair_empty = <const uint32_t*> <uintptr_t> values[8]
    terms.quad_keys = <const uint64_t*> <uintptr_t> values[9]
    terms.quad_values = <const int32_t*> <uintptr_t> values[10]
    terms.quad_empty = <const uint64_t*> <uintptr_t> values[11]
    terms.alpha_strings = <const uint64_t*> <uintptr_t> values[12]
    terms.beta_strings = <const uint64_t*> <uintptr_t> values[13]
    terms.quad_mask = values[14]
    terms.pair_mask = values[15]
    terms.ladder_offset = values[16]
    terms.term_inds_stride = values[17]
    terms.term_proj_stride = values[18]
    terms.num_alpha = values[19]
    terms.num_beta = values[20]
    return terms


cdef FqSinglesBuildTables _singles_build(tuple values) except *:


    cdef FqSinglesBuildTables tables
    memset(&tables, 0, sizeof(tables))
    tables.row_offsets = <const int64_t*> <uintptr_t> values[0]
    tables.columns = <const int32_t*> <uintptr_t> values[1]
    tables.low = <const uint8_t*> <uintptr_t> values[2]
    tables.high = <const uint8_t*> <uintptr_t> values[3]
    tables.alpha_slot = <const int32_t*> <uintptr_t> values[4]
    tables.beta_slot = <const int32_t*> <uintptr_t> values[5]
    tables.c0 = <double*> <uintptr_t> values[6]
    tables.delta = <double*> <uintptr_t> values[7]
    tables.error = <int32_t*> <uintptr_t> values[8]
    return tables


cdef FqDoublesBuildTables _doubles_build(tuple values) except *:


    cdef FqDoublesBuildTables tables
    memset(&tables, 0, sizeof(tables))
    tables.row_offsets = <const int64_t*> <uintptr_t> values[0]
    tables.row_lengths = <const int32_t*> <uintptr_t> values[1]
    tables.columns = <const int32_t*> <uintptr_t> values[2]
    tables.p0 = <const uint8_t*> <uintptr_t> values[3]
    tables.p1 = <const uint8_t*> <uintptr_t> values[4]
    tables.p2 = <const uint8_t*> <uintptr_t> values[5]
    tables.p3 = <const uint8_t*> <uintptr_t> values[6]
    tables.alpha_slot = <const int32_t*> <uintptr_t> values[7]
    tables.beta_slot = <const int32_t*> <uintptr_t> values[8]
    tables.values = <double*> <uintptr_t> values[9]
    return tables


def build_alpha_doubles(tuple geometry, tuple terms, tuple tables, uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqTermTables term_tables = _terms(terms)
    cdef FqDoublesBuildTables build = _doubles_build(tables)
    cdef int status
    with nogil:
        status = fq_launch_build_alpha_doubles(&tile, &term_tables, &build, <void*> stream)
    if status:
        raise _launch_error("alpha doubles build launch", status)


def build_beta_doubles(tuple geometry, tuple terms, tuple tables, uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqTermTables term_tables = _terms(terms)
    cdef FqDoublesBuildTables build = _doubles_build(tables)
    cdef int status
    with nogil:
        status = fq_launch_build_beta_doubles(&tile, &term_tables, &build, <void*> stream)
    if status:
        raise _launch_error("beta doubles build launch", status)


def build_alpha_singles(tuple geometry, tuple terms, tuple tables, uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqTermTables term_tables = _terms(terms)
    cdef FqSinglesBuildTables build = _singles_build(tables)
    cdef int status
    with nogil:
        status = fq_launch_build_alpha_singles(&tile, &term_tables, &build, <void*> stream)
    if status:
        raise _launch_error("alpha singles build launch", status)


def build_beta_singles(tuple geometry, tuple terms, tuple tables, uintptr_t stream):


    cdef FqTileGeometry tile = _geometry(geometry)
    cdef FqTermTables term_tables = _terms(terms)
    cdef FqSinglesBuildTables build = _singles_build(tables)
    cdef int status
    with nogil:
        status = fq_launch_build_beta_singles(&tile, &term_tables, &build, <void*> stream)
    if status:
        raise _launch_error("beta singles build launch", status)
