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

namespace fq::device
{


__device__ __forceinline__ std::int64_t block_row()
{
    return std::int64_t(blockIdx.z) * gridDim.y + blockIdx.y;
}


__device__ __forceinline__ std::int64_t thread_column()
{
    return std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
}


struct BlockXY
{
    std::int64_t x;
    std::int64_t y;
    bool valid;
};


__device__ __forceinline__ BlockXY grouped_block(const std::int64_t n_x, const std::int64_t n_y, const int group)
{
    const std::int64_t id = (std::int64_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    if(id >= n_x * n_y)
    {
        return BlockXY{0, 0, false};
    }
    const std::int64_t per_group = std::int64_t(group) * n_y;
    const std::int64_t first = (id / per_group) * group;
    const std::int64_t in_group = id - (id / per_group) * per_group;
    const std::int64_t width = n_x - first < group ? n_x - first : group;
    const std::int64_t y = in_group / width;
    return BlockXY{first + (in_group - y * width), y, true};
}


__device__ __forceinline__ std::int64_t tile_index(const std::int64_t row,
                                                   const std::int64_t width,
                                                   const std::int64_t column)
{
    return row * width + column;
}

}
