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

#include <cooperative_groups.h>

namespace fq::device
{


__device__ __forceinline__ cooperative_groups::thread_block_tile<32> this_warp()
{
    return cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
}


__device__ __forceinline__ bool warp_any(const bool predicate)
{
    return this_warp().any(predicate);
}


template <typename T>
__device__ __forceinline__ T warp_broadcast(const T value, const int lane)
{
    return this_warp().shfl(value, lane);
}

}
