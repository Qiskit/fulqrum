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

#include "../formats/params.hpp"

namespace fq::device
{


template <int K>
struct AlphaColumns
{
    int column[K];
    int alpha[K];
    bool active[K];


    __device__ static AlphaColumns of_thread(const FqTileGeometry& geometry)
    {
        return of_block(geometry, int(blockIdx.x));
    }


    __device__ static AlphaColumns of_block(const FqTileGeometry& geometry, const int block_x)
    {
        AlphaColumns columns;
        const int base = block_x * int(blockDim.x) * K;
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            columns.column[k] = base + k * int(blockDim.x) + int(threadIdx.x);
            columns.alpha[k] = geometry.a_start + columns.column[k];
            columns.active[k] = columns.alpha[k] < geometry.a_stop;
        }
        return columns;
    }


    __device__ bool any() const
    {
        bool found = false;
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            found = found || active[k];
        }
        return found;
    }
};

}
