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
#include <cstdint>
#include <cuda_runtime.h>

#include "launch.hpp"


struct AlphaStage
{
    std::int32_t source[kRdmAbStage];
    std::int32_t target[kRdmAbStage];
    double sign[kRdmAbStage];
};


__device__ __forceinline__ void load_stage(AlphaStage& stage, const RdmAbLists& lists, const std::int64_t first,
                                           const int count)
{
    for(int u = threadIdx.x; u < count; u += blockDim.x)
    {
        stage.source[u] = lists.alpha_source[first + u];
        stage.target[u] = lists.alpha_target[first + u];
        stage.sign[u] = lists.alpha_sign[first + u];
    }
}


__global__ void __launch_bounds__(kRdmAbThreads) alpha_beta_kernel(const __grid_constant__ RdmAbSlabs slabs,
                                                                    const __grid_constant__ RdmAbLists lists,
                                                                    const __grid_constant__ RdmTensor out)
{
    __shared__ AlphaStage stage;
    const std::int64_t label = blockIdx.y;
    const std::int64_t begin = lists.label_start[label];
    const std::int64_t end = lists.label_start[label + 1];
    if(begin == end || label / out.num_orbitals == label % out.num_orbitals)
    {
        return;
    }
    const std::int64_t k = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool live = k < lists.beta_count;
    const std::int64_t ib = live ? std::int64_t(lists.beta_source[k]) - slabs.own_column_offset : 0;
    const std::int64_t jb = live ? std::int64_t(lists.beta_target[k]) - slabs.source_column_offset : 0;
    double acc = 0.0;
    for(std::int64_t base = begin; base < end; base += kRdmAbStage)
    {
        const int count = static_cast<int>(end - base < kRdmAbStage ? end - base : kRdmAbStage);
        load_stage(stage, lists, base, count);
        __syncthreads();
        if(live)
        {
            for(int u = 0; u < count; u++)
            {
                const std::int64_t ia = std::int64_t(stage.source[u]) - slabs.own_row_offset;
                const std::int64_t ja = std::int64_t(stage.target[u]) - slabs.source_row_offset;
                acc = fma(stage.sign[u] * slabs.own[ia * slabs.own_width + ib], slabs.source[ja * slabs.source_width + jb],
                          acc);
            }
        }
        __syncthreads();
    }
    if(live && acc != 0.0)
    {
        atomicAdd(out.values + lists.alpha_offset[label] + lists.beta_offset[k], lists.beta_sign[k] * acc);
    }
}

extern "C"
{

    int rdm_launch_alpha_beta(const RdmAbSlabs* slabs, const RdmAbLists* lists, const RdmTensor* out, void* stream)
    {
        if(lists->beta_count == 0)
        {
            return static_cast<int>(cudaSuccess);
        }
        const std::int64_t blocks = (lists->beta_count + kRdmAbThreads - 1) / kRdmAbThreads;
        const dim3 grid(static_cast<unsigned int>(blocks), static_cast<unsigned int>(out->num_orbitals * out->num_orbitals));
        alpha_beta_kernel<<<grid, kRdmAbThreads, 0, static_cast<cudaStream_t>(stream)>>>(*slabs, *lists, *out);
        return static_cast<int>(cudaGetLastError());
    }
}
