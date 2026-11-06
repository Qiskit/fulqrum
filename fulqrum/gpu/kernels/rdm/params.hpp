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
 */
#pragma once

#include <cstdint>


struct RdmSlabPair
{
    const double* bra;
    const double* ket;
    std::int64_t width;
    std::int64_t bra_offset;
    std::int64_t ket_offset;
    std::int64_t bra_row_stride;
    std::int64_t bra_elem_stride;
};


struct RdmPairs
{
    const std::int64_t* row_ptr;
    const std::int32_t* cols;
    double* overlap;
    const std::uint64_t* keys;
    std::int64_t bra_offset;
    std::int64_t num_bra;
};


struct RdmDiagonal
{
    const double* weight;
    const std::uint64_t* keys;
    std::int64_t bra_offset;
    std::int64_t num_bra;
};


struct RdmSingles
{
    const std::int32_t* source;
    const std::int32_t* target;
    const double* sign;
    const std::int32_t* label;
    std::int64_t count;
};


struct RdmTensor
{
    double* values;
    std::int32_t num_orbitals;
    std::int32_t reserved;
};


struct RdmWeighted
{
    double* gamma;
    double* alpha_beta;
    const std::uint64_t* column_keys;
    std::int64_t column_offset;
    std::int32_t num_orbitals;
    std::int32_t reserved;
};


struct RdmRows
{
    const std::uint64_t* row_keys;
    std::int64_t row_offset;
    std::int64_t num_rows;
};


struct RdmAbSlabs
{
    const double* own;
    const double* source;
    std::int64_t own_width;
    std::int64_t source_width;
    std::int64_t own_row_offset;
    std::int64_t own_column_offset;
    std::int64_t source_row_offset;
    std::int64_t source_column_offset;
};


struct RdmAbLists
{
    const std::int32_t* alpha_source;
    const std::int32_t* alpha_target;
    const double* alpha_sign;
    const std::int64_t* label_start;
    const std::int64_t* alpha_offset;
    const std::int32_t* beta_source;
    const std::int32_t* beta_target;
    const double* beta_sign;
    const std::int64_t* beta_offset;
    std::int64_t beta_count;
};


enum RdmParamStruct
{
    kRdmSlabPair,
    kRdmPairs,
    kRdmDiagonal,
    kRdmSingles,
    kRdmTensor,
    kRdmAbSlabs,
    kRdmAbLists,
    kRdmWeighted,
    kRdmRows,
    kRdmParamStructCount
};

static_assert(sizeof(RdmSlabPair) == 56, "RdmSlabPair layout");
static_assert(sizeof(RdmPairs) == 48, "RdmPairs layout");
static_assert(sizeof(RdmDiagonal) == 32, "RdmDiagonal layout");
static_assert(sizeof(RdmSingles) == 40, "RdmSingles layout");
static_assert(sizeof(RdmTensor) == 16, "RdmTensor layout");
static_assert(sizeof(RdmAbSlabs) == 64, "RdmAbSlabs layout");
static_assert(sizeof(RdmAbLists) == 80, "RdmAbLists layout");
static_assert(sizeof(RdmWeighted) == 40, "RdmWeighted layout");
static_assert(sizeof(RdmRows) == 24, "RdmRows layout");
