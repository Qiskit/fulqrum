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
#include "device/row_block.cuh"
#include "device/tile_index.cuh"
#include "device/edge_stream.cuh"
#include "launch.hpp"
#include "launch_grid.hpp"

namespace
{


template <int K, int MIN_BLOCKS, bool IDENTITY_SLOT, bool GROUPED>
__global__ void __launch_bounds__(fq::launch::kArmThreads, MIN_BLOCKS) alpha_doubles_kernel(const FqTileGeometry geometry,
                                                                        const FqVectors vectors,
                                                                        const FqAlphaDoubleTables tables)
{
    const int n_a = geometry.a_stop - geometry.a_start;
    const int n_b = geometry.b_stop - geometry.b_start;
    std::int64_t block_x = blockIdx.x, block_y = fq::device::block_row();
    if(GROUPED)
    {
        const fq::device::BlockXY xy = fq::device::grouped_block((n_a + blockDim.x - 1) / blockDim.x,
                                                                 (std::int64_t(n_b) + K - 1) / K, tables.block_group);
        if(!xy.valid)
        {
            return;
        }
        block_x = xy.x;
        block_y = xy.y;
    }
    const int a_gpu = int(block_x * blockDim.x + threadIdx.x);
    const int a = geometry.a_start + a_gpu;
    if(a >= geometry.a_stop)
    {
        return;
    }
    const int first_row = int(block_y * K);
    if(first_row >= n_b)
    {
        return;
    }
    int beta_slot[K];
#pragma unroll
    for(int k = 0; k < K; ++k)
    {
        const int row = first_row + k;
        beta_slot[k] = row < n_b ? __ldg(vectors.beta_src_slot + geometry.b_start + row) : -1;
    }
    const fq::device::SellRow sell = fq::device::SellRow::at(tables.sell_offsets, tables.sell_lengths, a);
    fq::device::RowAccumulator<K> rows;
    if(sell.length > 0)
    {
        const std::int64_t slot = sell.slot(0);
        int column = __ldg(tables.sell_columns + slot);
        double value = __ldg(tables.values + slot);
        int alpha_slot = IDENTITY_SLOT ? column : __ldg(vectors.alpha_src_slot + column);
        for(int edge = 0; edge < sell.length; ++edge)
        {
            const bool more = edge + 1 < sell.length;
            const std::int64_t next = sell.slot(edge + 1);
            const int next_column = more ? __ldg(tables.sell_columns + next) : 0;
            const double next_value = more ? __ldg(tables.values + next) : 0.0;
            const int next_alpha_slot = more ? (IDENTITY_SLOT ? next_column : __ldg(vectors.alpha_src_slot + next_column)) : -1;
            if(alpha_slot >= 0 && fq::arms::is_applied(value))
            {
#pragma unroll
                for(int k = 0; k < K; ++k)
                {
                    if(beta_slot[k] >= 0)
                    {
                        const std::int64_t input =
                            fq::device::tile_index(beta_slot[k], geometry.n_alpha_src_compact, alpha_slot);
                        rows.add(k, value, __ldg(vectors.src_in + input));
                    }
                }
            }
            column = next_column;
            value = next_value;
            alpha_slot = next_alpha_slot;
        }
    }
#pragma unroll
    for(int k = 0; k < K; ++k)
    {
        const int row = first_row + k;
        if(row < n_b)
        {
            rows.flush(k, vectors.out_local + fq::device::tile_index(row, n_a, a_gpu));
        }
    }
}


template <int K, int MIN_BLOCKS, bool IDENTITY_SLOT>
void launch_order(const FqTileGeometry& geometry, const FqVectors& vectors, const FqAlphaDoubleTables& tables,
                  const cudaStream_t stream)
{
    const int threads = geometry.threads_per_block;
    if(tables.block_group >= 1)
    {
        const long long n_x = (geometry.a_stop - geometry.a_start + threads - 1) / threads;
        const long long n_y = (static_cast<long long>(geometry.b_stop) - geometry.b_start + K - 1) / K;
        alpha_doubles_kernel<K, MIN_BLOCKS, IDENTITY_SLOT, true>
            <<<fq::launch::linear_grid(n_x * n_y), threads, 0, stream>>>(geometry, vectors, tables);
        return;
    }
    alpha_doubles_kernel<K, MIN_BLOCKS, IDENTITY_SLOT, false>
        <<<fq::launch::beta_row_block_grid<K>(geometry), threads, 0, stream>>>(geometry, vectors, tables);
}

template <int K, int MIN_BLOCKS>
void launch_slot(const FqTileGeometry& geometry, const FqVectors& vectors, const FqAlphaDoubleTables& tables,
                 const cudaStream_t stream)
{
    tables.identity_slot != 0 ? launch_order<K, MIN_BLOCKS, true>(geometry, vectors, tables, stream)
                              : launch_order<K, MIN_BLOCKS, false>(geometry, vectors, tables, stream);
}

template <int K>
void launch_instance(const FqTileGeometry& geometry, const FqVectors& vectors, const FqAlphaDoubleTables& tables,
                     const cudaStream_t stream)
{
    if(tables.min_blocks == 6)
    {
        launch_slot<K, 6>(geometry, vectors, tables, stream);
    }
    else if(tables.min_blocks == 5)
    {
        launch_slot<K, 5>(geometry, vectors, tables, stream);
    }
    else
    {
        launch_slot<K, 4>(geometry, vectors, tables, stream);
    }
}

}

extern "C"
{

    int fq_launch_alpha_doubles(const FqTileGeometry* geometry,
                                const FqVectors* vectors,
                                const FqAlphaDoubleTables* tables,
                                void* stream)
    {
        return fq::launch::with_rows_per_thread(*geometry, tables->rows_per_thread, [&](auto rows) {
            launch_instance<decltype(rows)::value>(*geometry, *vectors, *tables, static_cast<cudaStream_t>(stream));
        });
    }
}
