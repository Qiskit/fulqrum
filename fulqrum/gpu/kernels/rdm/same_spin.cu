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
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

#include "labels.cuh"
#include "launch.hpp"


__device__ __forceinline__ double lane_dot(const double* __restrict__ bra, const double* __restrict__ ket,
                                           const std::int64_t width, const std::int64_t bra_stride, const int lane)
{
    double first = 0.0;
    double second = 0.0;
    std::int64_t t = lane;
    for(; t + 32 < width; t += 64)
    {
        first = fma(bra[t * bra_stride], ket[t], first);
        second = fma(bra[(t + 32) * bra_stride], ket[t + 32], second);
    }
    double sum = first + second;
    for(; t < width; t += 32)
    {
        sum = fma(bra[t * bra_stride], ket[t], sum);
    }
    return sum;
}


__global__ void __launch_bounds__(kRdmThreads) pair_overlap_kernel(const __grid_constant__ RdmSlabPair slabs,
                                                                    const __grid_constant__ RdmPairs pairs)
{
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int warp = static_cast<int>(threadIdx.x >> 5);
    const int warps = static_cast<int>(blockDim.x >> 5);
    for(std::int64_t i = blockIdx.x; i < pairs.num_bra; i += gridDim.x)
    {
        const double* bra = slabs.bra + (pairs.bra_offset + i - slabs.bra_offset) * slabs.bra_row_stride;
        for(std::int64_t e = pairs.row_ptr[i] + warp; e < pairs.row_ptr[i + 1]; e += warps)
        {
            const double* ket = slabs.ket + (std::int64_t(pairs.cols[e]) - slabs.ket_offset) * slabs.width;
            const double sum = warp_sum(lane_dot(bra, ket, slabs.width, slabs.bra_elem_stride, lane));
            if(lane == 0)
            {
                pairs.overlap[e] = sum;
            }
        }
    }
}


__device__ __forceinline__ void add_single_move(const RdmTensor& out, const std::uint64_t bra, const std::uint64_t ket,
                                                const int low, const int high, const double overlap)
{
    const bool low_in_bra = (bra >> low) & 1ULL;
    if(low_in_bra == static_cast<bool>((bra >> high) & 1ULL))
    {
        return;
    }
    const int a = low_in_bra ? low : high;
    const int b = low_in_bra ? high : low;
    const int m = out.num_orbitals;
    const double v = parity_between(bra, low, high) ? -overlap : overlap;
    for(std::uint64_t common = bra & ket; common; common &= common - 1)
    {
        const int k = __ffsll(static_cast<long long>(common)) - 1;
        atomicAdd(out.values + four_index(m, a, b, k, k), v);
        atomicAdd(out.values + four_index(m, a, k, k, b), -v);
        atomicAdd(out.values + four_index(m, k, b, a, k), -v);
        atomicAdd(out.values + four_index(m, k, k, a, b), v);
    }
}


__device__ __forceinline__ void add_double_move(const RdmTensor& out, const std::uint64_t bra, const std::uint64_t ket,
                                                const int* position, const double overlap)
{
    int high[2];
    int low[2];
    int num_high = 0;
    int num_low = 0;
    for(int k = 0; k < 4; k++)
    {
        if((bra >> position[k]) & 1ULL)
        {
            if(num_high < 2)
            {
                high[num_high++] = position[k];
            }
        }
        else if(num_low < 2)
        {
            low[num_low++] = position[k];
        }
    }
    if(num_high != 2 || num_low != 2)
    {
        return;
    }
    const double sign = double_move_sign(ket, low[0], low[1], high[1], high[0]);
    if(sign == 0.0)
    {
        return;
    }
    const int m = out.num_orbitals;
    const double v = sign * overlap;
    atomicAdd(out.values + four_index(m, high[0], low[0], high[1], low[1]), v);
    atomicAdd(out.values + four_index(m, high[0], low[1], high[1], low[0]), -v);
    atomicAdd(out.values + four_index(m, high[1], low[0], high[0], low[1]), -v);
    atomicAdd(out.values + four_index(m, high[1], low[1], high[0], low[0]), v);
}


__global__ void __launch_bounds__(kRdmThreads) same_scatter_kernel(const __grid_constant__ RdmPairs pairs,
                                                                    const __grid_constant__ RdmTensor out)
{
    for(std::int64_t i = blockIdx.x; i < pairs.num_bra; i += gridDim.x)
    {
        const std::uint64_t bra = pairs.keys[pairs.bra_offset + i];
        for(std::int64_t e = pairs.row_ptr[i] + threadIdx.x; e < pairs.row_ptr[i + 1]; e += blockDim.x)
        {
            const double overlap = pairs.overlap[e];
            if(overlap == 0.0)
            {
                continue;
            }
            const std::uint64_t ket = pairs.keys[pairs.cols[e]];
            const std::uint64_t flips = bra ^ ket;
            int position[4];
            lowest_four_bits(flips, position);
            const int distance = __popcll(flips);
            if(distance == 2)
            {
                add_single_move(out, bra, ket, position[0], position[1], overlap);
            }
            else if(distance == 4)
            {
                add_double_move(out, bra, ket, position, overlap);
            }
        }
    }
}


__global__ void __launch_bounds__(kRdmThreads) same_diagonal_kernel(const __grid_constant__ RdmDiagonal diagonal,
                                                                     const __grid_constant__ RdmTensor out)
{
    extern __shared__ double occupation[];
    const int m = out.num_orbitals;
    for(int t = threadIdx.x; t < m * m; t += blockDim.x)
    {
        occupation[t] = 0.0;
    }
    __syncthreads();
    for(std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < diagonal.num_bra;
        i += std::int64_t(gridDim.x) * blockDim.x)
    {
        const double weight = diagonal.weight[i];
        if(weight == 0.0)
        {
            continue;
        }
        const std::uint64_t key = diagonal.keys[diagonal.bra_offset + i];
        for(std::uint64_t x = key; x; x &= x - 1)
        {
            const int u = __ffsll(static_cast<long long>(x)) - 1;
            for(std::uint64_t y = key; y; y &= y - 1)
            {
                const int v = __ffsll(static_cast<long long>(y)) - 1;
                if(u != v)
                {
                    atomicAdd(occupation + u * m + v, weight);
                }
            }
        }
    }
    __syncthreads();
    for(int t = threadIdx.x; t < m * m; t += blockDim.x)
    {
        const double a = occupation[t];
        if(a != 0.0)
        {
            const int u = t / m;
            const int v = t % m;
            atomicAdd(out.values + four_index(m, u, u, v, v), a);
            atomicAdd(out.values + four_index(m, u, v, v, u), -a);
        }
    }
}

extern "C"
{

    int rdm_launch_pair_overlap(const RdmSlabPair* slabs, const RdmPairs* pairs, void* stream)
    {
        const int grid = static_cast<int>(std::min<std::int64_t>(kRdmMaxGrid, std::max<std::int64_t>(1, pairs->num_bra)));
        pair_overlap_kernel<<<grid, kRdmThreads, 0, static_cast<cudaStream_t>(stream)>>>(*slabs, *pairs);
        return static_cast<int>(cudaGetLastError());
    }

    int rdm_launch_same_scatter(const RdmPairs* pairs, const RdmTensor* out, void* stream)
    {
        const int grid = static_cast<int>(std::min<std::int64_t>(kRdmMaxGrid, std::max<std::int64_t>(1, pairs->num_bra)));
        same_scatter_kernel<<<grid, kRdmThreads, 0, static_cast<cudaStream_t>(stream)>>>(*pairs, *out);
        return static_cast<int>(cudaGetLastError());
    }

    std::size_t rdm_same_diagonal_shared_bytes(const std::int32_t num_orbitals)
    {
        return std::size_t(num_orbitals) * std::size_t(num_orbitals) * sizeof(double);
    }

    int rdm_launch_same_diagonal(const RdmDiagonal* diagonal, const RdmTensor* out, void* stream)
    {
        const std::int64_t blocks = (diagonal->num_bra + kRdmThreads - 1) / kRdmThreads;
        const int grid = static_cast<int>(std::min<std::int64_t>(kRdmMaxGrid, std::max<std::int64_t>(1, blocks)));
        same_diagonal_kernel<<<grid, kRdmThreads, rdm_same_diagonal_shared_bytes(out->num_orbitals),
                               static_cast<cudaStream_t>(stream)>>>(*diagonal, *out);
        return static_cast<int>(cudaGetLastError());
    }
}
