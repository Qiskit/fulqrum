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


__device__ __forceinline__ std::int64_t four_index(const int m, const int p, const int q, const int r, const int s)
{
    return ((std::int64_t(p) * m + q) * m + r) * m + s;
}


__device__ __forceinline__ int parity_between(const std::uint64_t key, const int low, const int high)
{
    if(high <= low + 1)
    {
        return 0;
    }
    const std::uint64_t mask = ((1ULL << high) - 1ULL) ^ ((1ULL << (low + 1)) - 1ULL);
    return __popcll(key & mask) & 1;
}


__device__ __forceinline__ double double_move_sign(std::uint64_t key, const int q, const int s, const int r,
                                                   const int p)
{
    const int orbital[4] = {q, s, r, p};
    double sign = 1.0;
#pragma unroll
    for(int k = 0; k < 4; k++)
    {
        const bool full = (key >> orbital[k]) & 1ULL;
        if(full == (k >= 2))
        {
            return 0.0;
        }
        if(__popcll(key & ((1ULL << orbital[k]) - 1ULL)) & 1)
        {
            sign = -sign;
        }
        key ^= 1ULL << orbital[k];
    }
    return sign;
}


__device__ __forceinline__ void lowest_four_bits(std::uint64_t x, int* position)
{
    int count = 0;
    while(x && count < 4)
    {
        position[count++] = __ffsll(static_cast<long long>(x)) - 1;
        x &= x - 1;
    }
    while(count < 4)
    {
        position[count++] = 0;
    }
}


__device__ __forceinline__ double warp_sum(double value)
{
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}
