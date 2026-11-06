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

constexpr int kOrbitalChunk = 16;


template <bool BETA_SIDE>
__device__ __forceinline__ std::int64_t weighted_index(const int m, const int p, const int q, const int k)
{
    return BETA_SIDE ? four_index(m, k, p, k, q) : four_index(m, p, k, q, k);
}


template <bool BETA_SIDE>
__device__ __forceinline__ void add_chunk(const RdmWeighted& out, const double* __restrict__ bra,
                                          const double* __restrict__ ket, const std::int64_t width,
                                          const std::int64_t bra_stride, const int lane, const int first,
                                          const int p, const int q, const double sign)
{
    double acc[kOrbitalChunk];
#pragma unroll
    for(int j = 0; j < kOrbitalChunk; j++)
    {
        acc[j] = 0.0;
    }
    for(std::int64_t t = lane; t < width; t += 32)
    {
        const double v = bra[t * bra_stride] * ket[t];
        const std::uint64_t key = out.column_keys[out.column_offset + t] >> first;
#pragma unroll
        for(int j = 0; j < kOrbitalChunk; j++)
        {
            acc[j] += ((key >> j) & 1ULL) ? v : 0.0;
        }
    }
    const int m = out.num_orbitals;
#pragma unroll
    for(int j = 0; j < kOrbitalChunk; j++)
    {
        const double sum = warp_sum(acc[j]);
        if(lane == 0 && first + j < m && sum != 0.0)
        {
            atomicAdd(out.alpha_beta + weighted_index<BETA_SIDE>(m, p, q, first + j), sign * sum);
        }
    }
}


template <bool BETA_SIDE>
__global__ void __launch_bounds__(kRdmThreads) weighted_one_body_kernel(const __grid_constant__ RdmSlabPair slabs,
                                                                         const __grid_constant__ RdmSingles singles,
                                                                         const __grid_constant__ RdmWeighted out)
{
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const std::int64_t warps = std::int64_t(gridDim.x) * (blockDim.x >> 5);
    const int m = out.num_orbitals;
    for(std::int64_t k = std::int64_t(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5); k < singles.count;
        k += warps)
    {
        const double* bra = slabs.bra + (std::int64_t(singles.source[k]) - slabs.bra_offset) * slabs.bra_row_stride;
        const double* ket = slabs.ket + (std::int64_t(singles.target[k]) - slabs.ket_offset) * slabs.width;
        const int label = singles.label[k];
        const int p = label / m;
        const int q = label % m;
        const double sign = singles.sign[k];
        double one = 0.0;
        for(std::int64_t t = lane; t < slabs.width; t += 32)
        {
            one = fma(bra[t * slabs.bra_elem_stride], ket[t], one);
        }
        one = warp_sum(one);
        if(lane == 0 && one != 0.0)
        {
            atomicAdd(out.gamma + label, sign * one);
        }
        if(out.alpha_beta == nullptr || (BETA_SIDE && p == q))
        {
            continue;
        }
        for(int first = 0; first < m; first += kOrbitalChunk)
        {
            add_chunk<BETA_SIDE>(out, bra, ket, slabs.width, slabs.bra_elem_stride, lane, first, p, q, sign);
        }
    }
}


__device__ __forceinline__ void add_to_occupied(double* gamma, const std::uint64_t key, const int m, const double value)
{
    for(std::uint64_t x = key; x; x &= x - 1)
    {
        const int q = __ffsll(static_cast<long long>(x)) - 1;
        atomicAdd(gamma + q * m + q, value);
    }
}


__global__ void __launch_bounds__(kRdmThreads) diagonal_kernel(const __grid_constant__ RdmSlabPair slabs,
                                                                const __grid_constant__ RdmRows rows,
                                                                const __grid_constant__ RdmWeighted out)
{
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const std::int64_t warps = std::int64_t(gridDim.x) * (blockDim.x >> 5);
    const int m = out.num_orbitals;
    for(std::int64_t i = std::int64_t(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5); i < rows.num_rows;
        i += warps)
    {
        const double* row = slabs.bra + i * slabs.bra_row_stride;
        const std::int64_t stride = slabs.bra_elem_stride;
        const std::uint64_t key = rows.row_keys[rows.row_offset + i];
        double norm = 0.0;
        for(std::int64_t t = lane; t < slabs.width; t += 32)
        {
            norm = fma(row[t * stride], row[t * stride], norm);
        }
        norm = warp_sum(norm);
        if(lane == 0 && norm != 0.0)
        {
            add_to_occupied(out.gamma, key, m, norm);
        }
        if(out.alpha_beta == nullptr)
        {
            continue;
        }
        for(int first = 0; first < m; first += kOrbitalChunk)
        {
            double acc[kOrbitalChunk];
#pragma unroll
            for(int j = 0; j < kOrbitalChunk; j++)
            {
                acc[j] = 0.0;
            }
            for(std::int64_t t = lane; t < slabs.width; t += 32)
            {
                const double v = row[t * stride] * row[t * stride];
                const std::uint64_t column = out.column_keys[out.column_offset + t] >> first;
#pragma unroll
                for(int j = 0; j < kOrbitalChunk; j++)
                {
                    acc[j] += ((column >> j) & 1ULL) ? v : 0.0;
                }
            }
#pragma unroll
            for(int j = 0; j < kOrbitalChunk; j++)
            {
                const double sum = warp_sum(acc[j]);
                const int r = first + j;
                if(lane == 0 && r < m && sum != 0.0)
                {
                    for(std::uint64_t x = key; x; x &= x - 1)
                    {
                        const int q = __ffsll(static_cast<long long>(x)) - 1;
                        atomicAdd(out.alpha_beta + four_index(m, q, r, q, r), sum);
                    }
                }
            }
        }
    }
}


template <bool BETA_SIDE>
int launch_weighted(const RdmSlabPair* slabs, const RdmSingles* singles, const RdmWeighted* out, void* stream)
{
    if(singles->count == 0)
    {
        return static_cast<int>(cudaSuccess);
    }
    const std::int64_t warps_per_block = kRdmThreads / 32;
    const std::int64_t blocks = (singles->count + warps_per_block - 1) / warps_per_block;
    const int grid = static_cast<int>(std::min<std::int64_t>(kRdmMaxGrid, blocks));
    weighted_one_body_kernel<BETA_SIDE>
        <<<grid, kRdmThreads, 0, static_cast<cudaStream_t>(stream)>>>(*slabs, *singles, *out);
    return static_cast<int>(cudaGetLastError());
}

extern "C"
{

    int rdm_launch_alpha_one_body(const RdmSlabPair* slabs, const RdmSingles* singles, const RdmWeighted* out,
                                  void* stream)
    {
        return launch_weighted<false>(slabs, singles, out, stream);
    }

    int rdm_launch_beta_one_body(const RdmSlabPair* slabs, const RdmSingles* singles, const RdmWeighted* out,
                                 void* stream)
    {
        return launch_weighted<true>(slabs, singles, out, stream);
    }

    int rdm_launch_diagonal(const RdmSlabPair* slabs, const RdmRows* rows, const RdmWeighted* out,
                            const int beta_side, void* stream)
    {
        if(rows->num_rows == 0)
        {
            return static_cast<int>(cudaSuccess);
        }
        RdmWeighted target = *out;
        if(beta_side)
        {
            target.alpha_beta = nullptr;
        }
        const std::int64_t warps_per_block = kRdmThreads / 32;
        const std::int64_t blocks = (rows->num_rows + warps_per_block - 1) / warps_per_block;
        const int grid = static_cast<int>(std::min<std::int64_t>(kRdmMaxGrid, blocks));
        diagonal_kernel<<<grid, kRdmThreads, 0, static_cast<cudaStream_t>(stream)>>>(*slabs, *rows, target);
        return static_cast<int>(cudaGetLastError());
    }
}
