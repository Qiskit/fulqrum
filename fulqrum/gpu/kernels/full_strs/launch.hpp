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

#include <cstddef>
#include <cstdint>

#include "params.hpp"

extern "C"
{


    int fs_launch_diagonal(const FsVectors* vectors, void* stream);


    int fs_launch_groups_uniform(const FsKeys* keys, const FsTables* tables, const FsGroups* groups, const double* x,
                                 void* stream);


    int fs_launch_groups_binned(const FsKeys* keys, const FsTables* tables, const FsGroups* groups, const double* x,
                                void* stream);


    std::size_t fs_group_offsets_scratch_bytes(std::int64_t num_groups);


    int fs_launch_group_offsets(const std::uint16_t* sizes, std::int64_t num_groups, std::int32_t* offsets, void* scratch,
                                std::size_t scratch_bytes, void* stream);


    std::size_t fs_reduce_scratch_bytes(std::int64_t num_entries, int end_bit);


    int fs_launch_reduce(const FsReduce* reduce, void* stream);


    std::size_t fs_reduce_plan_scratch_bytes(std::int64_t num_entries, int end_bit);


    int fs_build_reduce_plan(std::int32_t* rows, std::int32_t* rows_alt, std::int32_t* positions, std::int32_t* positions_alt,
                             void* scratch, std::size_t scratch_bytes, std::int64_t num_entries, int end_bit,
                             std::int32_t* sorted_positions, std::int32_t* run_rows, std::int32_t* run_counts,
                             std::int32_t* num_runs, void* stream);


    int fs_launch_gather_runs(const FsRuns* runs, void* stream);


    std::size_t fs_param_struct_size(int which);


    const char* fs_error_name(int status);


    int fs_launch_rdm_groups(const FsKeys* keys, const FsGroups* groups, const FsRdm* out, const double* x,
                             void* stream);


    int fs_launch_pair_occupation(const FsKeys* keys, const double* x, double* occupation, int sectors,
                                  int num_orbitals, void* stream);
}
