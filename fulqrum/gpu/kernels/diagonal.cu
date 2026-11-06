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

#include "device/tile_index.cuh"
#include "launch.hpp"

namespace
{

constexpr int kDiagonalThreads = 256;
constexpr std::int64_t kMaxDiagonalBlocks = 65535 * 32;


__device__ double diagonal_element(const FqTileGeometry& geometry, const FqDiagonalTables& tables, const int a,
                                   const int b)
{
    if(tables.diag_Mb == nullptr)
    {
        return tables.diag_local[fq::device::tile_index(b, geometry.n_a_bra, a)];
    }
    const std::uint64_t alpha = tables.alpha_strings[geometry.bra_a0 + a];
    double cross = 0.0;
    for(int p = 0; p < tables.num_alpha; ++p)
    {
        cross += (1.0 - 2.0 * double((alpha >> p) & 1ull)) * tables.diag_Mb[std::int64_t(p) * geometry.n_b_bra + b];
    }
    return tables.diag_alpha[a] + tables.diag_beta[b] + cross;
}


__global__ void __launch_bounds__(kDiagonalThreads) diagonal_kernel(const FqTileGeometry geometry,
                                                                    const FqVectors vectors,
                                                                    const FqDiagonalTables tables)
{
    const std::int64_t n_a = geometry.a_stop - geometry.a_start;
    const std::int64_t n_b = geometry.b_stop - geometry.b_start;
    const std::int64_t stride = std::int64_t(gridDim.x) * blockDim.x;
    for(std::int64_t i = fq::device::thread_column(); i < n_a * n_b; i += stride)
    {
        const int a = geometry.a_start + int(i % n_a);
        const int b = geometry.b_start + int(i / n_a);
        const int a_slot = vectors.alpha_src_slot[a];
        const int b_slot = vectors.beta_src_slot[b];
        if(a_slot < 0 || b_slot < 0)
        {
            continue;
        }
        const double value = diagonal_element(geometry, tables, a, b) *
                             vectors.src_in[fq::device::tile_index(b_slot, geometry.n_alpha_src_compact, a_slot)];
        vectors.out_local[i] += value;
    }
}

}

extern "C"
{

    int fq_launch_diagonal(const FqTileGeometry* geometry,
                           const FqVectors* vectors,
                           const FqDiagonalTables* tables,
                           void* stream)
    {
        const std::int64_t n = std::int64_t(geometry->a_stop - geometry->a_start) *
                               (geometry->b_stop - geometry->b_start);
        if(n <= 0)
        {
            return int(cudaSuccess);
        }
        const std::int64_t blocks = (n + kDiagonalThreads - 1) / kDiagonalThreads;
        const unsigned int grid = static_cast<unsigned int>(blocks < kMaxDiagonalBlocks ? blocks : kMaxDiagonalBlocks);
        diagonal_kernel<<<grid, kDiagonalThreads, 0, static_cast<cudaStream_t>(stream)>>>(*geometry, *vectors, *tables);
        return int(cudaGetLastError());
    }
}
