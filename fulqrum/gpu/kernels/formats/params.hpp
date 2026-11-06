/**
 * This code is a part of Fulqrum.
 *
 * (C) Copyright IBM 2026.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 * Modified: descriptive GPU flag and clique names.
 */
#pragma once

#include <cstdint>


struct FqTileGeometry
{
    std::int32_t bra_a0;
    std::int32_t n_a_bra;
    std::int32_t bra_b0;
    std::int32_t n_b_bra;
    std::int32_t src_a0;
    std::int32_t n_a_src;
    std::int32_t src_b0;
    std::int32_t n_b_src;
    std::int32_t a_start;
    std::int32_t a_stop;
    std::int32_t b_start;
    std::int32_t b_stop;
    std::int32_t n_alpha_src_compact;
    std::int32_t threads_per_block;
};


struct FqVectors
{
    const double* src_in;
    double* out_local;
    const std::int32_t* alpha_src_slot;
    const std::int32_t* beta_src_slot;
};


struct FqDiagonalTables
{
    const double* diag_local;
    const double* diag_alpha;
    const double* diag_beta;
    const double* diag_Mb;
    const std::uint64_t* alpha_strings;
    std::int32_t num_alpha;
    std::int32_t reserved;
};


struct FqAlphaSinglePairTables
{
    const std::int64_t* sell_offsets;
    const std::int32_t* sell_lengths;
    const std::uint64_t* sell_words;
    const double* c0_sell;
    const double* pair_table;
    std::int32_t num_pairs;
    std::int32_t rows_per_thread;
    std::int32_t hoist_loads;
    std::int32_t reserved;
};


struct FqAlphaSingleDeltaTables
{
    const std::int64_t* row_offsets;
    const std::int32_t* columns;
    const double* c0;
    const double* delta;
    const std::uint64_t* beta_strings;
    std::int32_t num_beta;
    std::int32_t rows_per_thread;
};


struct FqBetaSinglePairTables
{
    const std::int64_t* row_offsets;
    const std::int32_t* columns;
    const double* c0;
    const std::uint32_t* signed_pairs;
    const double* pair_table;
    std::int32_t table_columns;
    std::int32_t alphas_per_thread;
    std::int32_t block_group;
    std::int32_t hoist_loads;
};


struct FqBetaSingleDeltaTables
{
    const std::int64_t* row_offsets;
    const std::int32_t* columns;
    const double* c0;
    const double* delta;
    const std::uint64_t* alpha_strings;
    std::int32_t num_alpha;
    std::int32_t alphas_per_thread;
};


struct FqAlphaDoubleTables
{
    const std::int64_t* sell_offsets;
    const std::int32_t* sell_lengths;
    const std::int32_t* sell_columns;
    const double* values;
    std::int32_t rows_per_thread;
    std::int32_t min_blocks;
    std::int32_t identity_slot;
    std::int32_t block_group;
};


struct FqBetaDoubleTables
{
    const std::int64_t* row_offsets;
    const std::int32_t* columns;
    const double* values;
    std::int32_t alphas_per_thread;
    std::int32_t block_group;
};


struct FqBetaDoubleClassTables
{
    const double* blocks;
    const std::int64_t* block_offsets;
    const std::int32_t* tile_info;
    const std::int32_t* row_members;
    const std::int32_t* col_members;
    std::int32_t* nonfinite;
    std::int32_t num_tiles;
    std::int32_t alpha_group;
};


struct FqAlphaDoubleClassTables
{
    FqBetaDoubleClassTables classes;
    double* xt;
    double* yt;
    std::int32_t slab;
    std::int32_t n_alpha_src;
    FqBetaDoubleTables leftover;
    const std::int32_t* slab_identity;
};


struct FqCrossSellTables
{
    const std::int64_t* beta_row_offsets;
    const std::int32_t* beta_columns;
    const std::uint8_t* beta_low;
    const std::uint8_t* beta_high;
    const std::uint64_t* beta_strings;
    const double* coefficients;
    const std::int64_t* sell_offsets;
    const std::int32_t* sell_lengths;
    const std::uint64_t* sell_words;
    std::int32_t num_alpha;
    std::int32_t num_beta;
};


struct FqCrossDirectTables
{
    const std::int64_t* sell_offsets;
    const std::int32_t* sell_lengths;
    const std::uint64_t* sell_words;
    const std::uint64_t* beta_words;
    const std::int32_t* tasks;
    const double* coefficients;
    std::int32_t num_pairs;
    std::int32_t num_tasks;
    const std::int32_t* order;
    std::int32_t num_rows;
    std::int32_t gated;
};


struct FqCrossCliqueTables
{
    const std::uint64_t* edge_words;
    const std::int32_t* group_offsets;
    const std::int32_t* group_pairs;
    const std::int32_t* tile_group_offsets;
    const std::int32_t* parent_offsets;
    const std::uint64_t* members;
    const std::int32_t* parent_ids;
    const double* coefficients;
    std::int32_t num_parents;
    std::int32_t num_tiles;
    std::int32_t tile_rows;
    std::int32_t orbitals;
    std::int32_t bucket;
    std::int32_t edges_per_batch;
    std::int32_t transposed;
    std::int32_t parent_group;
    std::int32_t mma;
    std::int32_t distinct_rows;
    double* panel;
    std::int32_t* nonfinite;
    std::int32_t panel_columns;
    std::int32_t reserved;
};


struct FqCrossLbgTables
{
    const std::uint64_t* edge_words;
    const std::int32_t* segment_begin;
    const std::int32_t* segment_end;
    const std::int32_t* segment_pair;
    double* out_panel;
    const std::int32_t* nonfinite;
    std::int32_t num_segments;
    std::int32_t out_stride;
};


struct FqCrossRslTables
{
    const std::int32_t* block_first;
    const std::int32_t* block_count;
    const std::uint64_t* members;
    double* panel;
    const std::int32_t* tasks;
    const std::uint64_t* beta_words;
    const double* coefficients;
    const double* coef_table;
    const std::int32_t* nonfinite;
    std::int32_t num_blocks;
    std::int32_t rows_per_job;
    std::int32_t panel_rows;
    std::int32_t num_pairs;
    std::int32_t orbitals;
    std::int32_t members_per_clique;
};


struct FqTermTables
{
    const double* term_header;
    const std::uint64_t* term_inds_vals;
    const std::uint8_t* term_proj_inds;
    const std::uint8_t* term_proj_bits;
    const std::uint32_t* ladder_offsets;
    const std::int8_t* oper_elems;
    const std::uint32_t* pair_keys;
    const std::int32_t* pair_values;
    const std::uint32_t* pair_empty;
    const std::uint64_t* quad_keys;
    const std::int32_t* quad_values;
    const std::uint64_t* quad_empty;
    const std::uint64_t* alpha_strings;
    const std::uint64_t* beta_strings;
    std::uint64_t quad_mask;
    std::uint32_t pair_mask;
    std::uint32_t ladder_offset;
    std::int32_t term_inds_stride;
    std::int32_t term_proj_stride;
    std::int32_t num_alpha;
    std::int32_t num_beta;
};


struct FqSinglesBuildTables
{
    const std::int64_t* row_offsets;
    const std::int32_t* columns;
    const std::uint8_t* low;
    const std::uint8_t* high;
    const std::int32_t* alpha_slot;
    const std::int32_t* beta_slot;
    double* c0;
    double* delta;
    std::int32_t* error;
};


struct FqDoublesBuildTables
{
    const std::int64_t* row_offsets;
    const std::int32_t* row_lengths;
    const std::int32_t* columns;
    const std::uint8_t* p0;
    const std::uint8_t* p1;
    const std::uint8_t* p2;
    const std::uint8_t* p3;
    const std::int32_t* alpha_slot;
    const std::int32_t* beta_slot;
    double* values;
};


enum FqParamStruct : int
{
    kFqTileGeometry = 0,
    kFqVectors = 1,
    kFqDiagonalTables = 2,
    kFqAlphaSinglePairTables = 3,
    kFqAlphaSingleDeltaTables = 4,
    kFqBetaSinglePairTables = 5,
    kFqBetaSingleDeltaTables = 6,
    kFqAlphaDoubleTables = 7,
    kFqBetaDoubleTables = 8,
    kFqCrossSellTables = 9,
    kFqCrossDirectTables = 10,
    kFqCrossCliqueTables = 11,
    kFqTermTables = 12,
    kFqSinglesBuildTables = 13,
    kFqDoublesBuildTables = 14,
    kFqBetaDoubleClassTables = 15,
    kFqAlphaDoubleClassTables = 16,
    kFqCrossLbgTables = 17,
    kFqCrossRslTables = 18,
    kFqParamStructCount = 19
};

static_assert(sizeof(FqTileGeometry) == 56, "FqTileGeometry layout changed");
static_assert(alignof(FqTileGeometry) == 4, "FqTileGeometry alignment changed");
static_assert(sizeof(FqVectors) == 32, "FqVectors layout changed");
static_assert(alignof(FqVectors) == 8, "FqVectors alignment changed");
static_assert(sizeof(FqDiagonalTables) == 48, "FqDiagonalTables layout changed");
static_assert(sizeof(FqAlphaSinglePairTables) == 56, "FqAlphaSinglePairTables layout changed");
static_assert(sizeof(FqAlphaSingleDeltaTables) == 48, "FqAlphaSingleDeltaTables layout changed");
static_assert(sizeof(FqBetaSinglePairTables) == 56, "FqBetaSinglePairTables layout changed");
static_assert(sizeof(FqBetaSingleDeltaTables) == 48, "FqBetaSingleDeltaTables layout changed");
static_assert(sizeof(FqAlphaDoubleTables) == 48, "FqAlphaDoubleTables layout changed");
static_assert(sizeof(FqBetaDoubleTables) == 32, "FqBetaDoubleTables layout changed");
static_assert(sizeof(FqCrossSellTables) == 80, "FqCrossSellTables layout changed");
static_assert(sizeof(FqCrossDirectTables) == 72, "FqCrossDirectTables layout changed");
static_assert(sizeof(FqCrossCliqueTables) == 128, "FqCrossCliqueTables layout changed");
static_assert(sizeof(FqTermTables) == 144, "FqTermTables layout changed");
static_assert(sizeof(FqSinglesBuildTables) == 72, "FqSinglesBuildTables layout changed");
static_assert(sizeof(FqDoublesBuildTables) == 80, "FqDoublesBuildTables layout changed");
static_assert(sizeof(FqBetaDoubleClassTables) == 56, "FqBetaDoubleClassTables layout changed");
static_assert(sizeof(FqAlphaDoubleClassTables) == 120, "FqAlphaDoubleClassTables layout changed");
static_assert(sizeof(FqCrossLbgTables) == 56, "FqCrossLbgTables layout changed");
static_assert(sizeof(FqCrossRslTables) == 96, "FqCrossRslTables layout changed");
