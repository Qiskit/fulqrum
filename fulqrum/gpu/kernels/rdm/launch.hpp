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

constexpr int kRdmThreads = 256;
constexpr int kRdmAbThreads = 128;
constexpr int kRdmAbStage = 256;
constexpr int kRdmMaxGrid = 65535;
constexpr int kRdmMaxOrbitals = 64;

extern "C"
{


    int rdm_launch_pair_overlap(const RdmSlabPair* slabs, const RdmPairs* pairs, void* stream);


    int rdm_launch_same_scatter(const RdmPairs* pairs, const RdmTensor* out, void* stream);


    std::size_t rdm_same_diagonal_shared_bytes(std::int32_t num_orbitals);


    int rdm_launch_same_diagonal(const RdmDiagonal* diagonal, const RdmTensor* out, void* stream);


    int rdm_launch_alpha_one_body(const RdmSlabPair* slabs, const RdmSingles* singles, const RdmWeighted* out,
                                  void* stream);


    int rdm_launch_beta_one_body(const RdmSlabPair* slabs, const RdmSingles* singles, const RdmWeighted* out,
                                 void* stream);


    int rdm_launch_diagonal(const RdmSlabPair* slabs, const RdmRows* rows, const RdmWeighted* out, int beta_side,
                            void* stream);


    int rdm_launch_alpha_beta(const RdmAbSlabs* slabs, const RdmAbLists* lists, const RdmTensor* out, void* stream);


    std::size_t rdm_param_struct_size(int which);


    const char* rdm_error_name(int status);
}
