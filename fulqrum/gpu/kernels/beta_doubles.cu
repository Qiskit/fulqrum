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

#include "arms/elements.hpp"
#include "device/alpha_columns.cuh"
#include "device/row_block.cuh"
#include "device/tile_index.cuh"
#include "launch.hpp"
#include "launch_grid.hpp"

namespace
{


template <int K, bool GROUPED>
__global__ void __launch_bounds__(fq::launch::kArmThreads, 4) beta_doubles_kernel(const FqTileGeometry geometry,
                                                              const FqVectors vectors,
                                                              const FqBetaDoubleTables tables)
{
    const int n_a = geometry.a_stop - geometry.a_start;
    std::int64_t block_x = blockIdx.x, block_y = fq::device::block_row();
    if(GROUPED)
    {
        const std::int64_t per_block = std::int64_t(blockDim.x) * K;
        const fq::device::BlockXY xy = fq::device::grouped_block(
            (n_a + per_block - 1) / per_block, geometry.b_stop - geometry.b_start, tables.block_group);
        if(!xy.valid)
        {
            return;
        }
        block_x = xy.x;
        block_y = xy.y;
    }
    const int row = int(block_y);
    const int b = geometry.b_start + row;
    if(b >= geometry.b_stop)
    {
        return;
    }
    const fq::device::AlphaColumns<K> alphas = fq::device::AlphaColumns<K>::of_block(geometry, int(block_x));
    int alpha_slot[K];
#pragma unroll
    for(int k = 0; k < K; ++k)
    {
        alpha_slot[k] = alphas.active[k] ? vectors.alpha_src_slot[alphas.alpha[k]] : -1;
    }
    if(!alphas.any())
    {
        return;
    }
    fq::device::RowAccumulator<K> sums;
    const std::int64_t first_edge = tables.row_offsets[b];
    const int row_edges = int(tables.row_offsets[b + 1] - first_edge);
    for(int k_edge = 0; k_edge < row_edges; ++k_edge)
    {
        const std::int64_t edge = first_edge + k_edge;
        const int beta_slot = vectors.beta_src_slot[tables.columns[edge]];
        if(beta_slot < 0)
        {
            continue;
        }
        const double value = tables.values[edge];
        if(!fq::arms::is_applied(value))
        {
            continue;
        }
        const std::int64_t input_row = std::int64_t(beta_slot) * geometry.n_alpha_src_compact;
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            if(alpha_slot[k] >= 0)
            {
                sums.add(k, value, vectors.src_in[input_row + alpha_slot[k]]);
            }
        }
    }
#pragma unroll
    for(int k = 0; k < K; ++k)
    {
        if(alphas.active[k])
        {
            sums.flush(k, vectors.out_local + fq::device::tile_index(row, n_a, alphas.column[k]));
        }
    }
}

}

extern "C"
{

    int fq_launch_beta_doubles(const FqTileGeometry* geometry,
                               const FqVectors* vectors,
                               const FqBetaDoubleTables* tables,
                               void* stream)
    {
        return fq::launch::with_rows_per_thread(*geometry, tables->alphas_per_thread, [&](auto alphas) {
            constexpr int K = decltype(alphas)::value;
            const cudaStream_t s = static_cast<cudaStream_t>(stream);
            if(tables->block_group >= 1)
            {
                beta_doubles_kernel<K, true><<<fq::launch::linear_grid(fq::launch::alpha_column_blocks<K>(*geometry)),
                                               geometry->threads_per_block, 0, s>>>(*geometry, *vectors, *tables);
                return;
            }
            beta_doubles_kernel<K, false><<<fq::launch::alpha_column_grid<K>(*geometry), geometry->threads_per_block, 0,
                                           s>>>(*geometry, *vectors, *tables);
        });
    }
}
