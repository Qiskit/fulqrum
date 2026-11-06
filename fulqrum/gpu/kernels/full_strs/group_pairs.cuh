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

#include "params.hpp"

namespace fq::full_strs::device
{


template <int W>
__device__ __forceinline__ const std::uint64_t* key_of(const FsKeys& keys, const std::uint32_t row)
{
    return row < keys.num_rows ? keys.keys + std::int64_t(row) * 2 * W
                               : keys.halo_keys + (std::int64_t(row) - keys.num_rows) * 2 * W;
}


__device__ __forceinline__ int members_below(const std::uint32_t* members, const int m, const std::uint32_t bound)
{
    int low = 0;
    int high = m;
    while(low < high)
    {
        const int mid = (low + high) >> 1;
        if(members[mid] < bound)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

}
