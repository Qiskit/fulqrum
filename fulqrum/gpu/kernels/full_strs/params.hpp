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


struct FsKeys
{
    const std::uint64_t* keys;
    const std::uint64_t* halo_keys;
    std::int64_t num_rows;
    std::int32_t num_orbitals;
    std::int32_t reserved;
};


struct FsTables
{
    const double* single_value;
    const double* single_delta;
    const double* single_reference;
    const double* double_value;
    const double* mixed_value;
    const std::uint64_t* reference;
    std::int64_t num_pairs;
    std::int64_t num_quads;
    std::int32_t num_orbitals;
    std::int32_t reserved;
};


struct FsVectors
{
    const double* x;
    double* y;
    const double* diagonal;
    std::int64_t num_rows;
};


struct FsGroups
{
    const std::uint32_t* members;
    const std::int32_t* offsets;
    double* partials;
    double* y;
    std::int64_t first_group;
    std::int64_t num_groups;
    std::int32_t cls;
    std::int32_t bucket;
    std::uint32_t output_begin;
    std::uint32_t output_end;
    std::uint32_t source_begin;
    std::uint32_t source_end;
};


struct FsReduce
{
    std::int32_t* rows;
    std::int32_t* rows_alt;
    double* values;
    double* values_alt;
    void* scratch;
    double* y;
    std::int64_t num_entries;
    std::uint64_t scratch_bytes;
    std::int32_t end_bit;
    std::int32_t row_begin;
    std::int32_t row_end;
    std::int32_t reserved;
};


struct FsRuns
{
    const std::int32_t* perm;
    const std::int32_t* run_rows;
    const std::int32_t* run_starts;
    const double* values;
    double* y;
    std::int64_t num_runs;
};


struct FsRdm
{
    double* gamma_alpha;
    double* gamma_beta;
    double* same_alpha;
    double* same_beta;
    double* other_alpha;
    double* other_beta;
    double* canonical_alpha;
    double* canonical_beta;
    double* mixed;
    std::int32_t num_orbitals;
    std::int32_t half;
};


enum FsParamStruct
{
    kFsKeys = 0,
    kFsTables = 1,
    kFsVectors = 2,
    kFsGroups = 3,
    kFsReduce = 4,
    kFsRuns = 5,
    kFsRdm = 6,
    kFsParamStructCount = 7
};

static_assert(sizeof(FsKeys) == 32, "FsKeys layout changed");
static_assert(alignof(FsKeys) == 8, "FsKeys alignment changed");
static_assert(sizeof(FsTables) == 72, "FsTables layout changed");
static_assert(alignof(FsTables) == 8, "FsTables alignment changed");
static_assert(sizeof(FsVectors) == 32, "FsVectors layout changed");
static_assert(alignof(FsVectors) == 8, "FsVectors alignment changed");
static_assert(sizeof(FsGroups) == 72, "FsGroups layout changed");
static_assert(alignof(FsGroups) == 8, "FsGroups alignment changed");
static_assert(sizeof(FsReduce) == 80, "FsReduce layout changed");
static_assert(alignof(FsReduce) == 8, "FsReduce alignment changed");
static_assert(sizeof(FsRuns) == 48, "FsRuns layout changed");
static_assert(alignof(FsRuns) == 8, "FsRuns alignment changed");
static_assert(sizeof(FsRdm) == 80, "FsRdm layout changed");
static_assert(alignof(FsRdm) == 8, "FsRdm alignment changed");
