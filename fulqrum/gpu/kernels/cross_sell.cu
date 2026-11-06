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

#include "arms/cross_row.cuh"
#include "device/edge_stream.cuh"
#include "device/row_block.cuh"
#include "device/tile_index.cuh"
#include "formats/pair_index.hpp"
#include "launch.hpp"
#include "launch_grid.hpp"

namespace
{


__device__ double beta_pair_sign(const std::uint64_t beta, const int low, const int high)
{
    return (__popcll(beta & fq::pair_index::between_mask(low, high)) & 1) ? -1.0 : 1.0;
}


__global__ void __launch_bounds__(fq::launch::kArmThreads, 4) cross_sell_kernel(const FqTileGeometry geometry,
                                                                                const FqVectors vectors,
                                                                                const FqCrossSellTables tables)
{
    const int row = int(fq::device::block_row());
    const int b = geometry.b_start + row;
    const int column = int(fq::device::thread_column());
    const int a = geometry.a_start + column;
    if(b >= geometry.b_stop || a >= geometry.a_stop)
    {
        return;
    }
    const std::uint64_t beta = tables.beta_strings[geometry.bra_b0 + b];
    const std::int64_t alpha_pairs = fq::pair_index::num_pairs(tables.num_alpha);
    const auto* words = reinterpret_cast<const unsigned long long*>(tables.sell_words);
    const fq::device::SellRow alpha_row = fq::device::SellRow::at(tables.sell_offsets, tables.sell_lengths, a);
    fq::device::RowAccumulator<1> sum;
    const std::int64_t first_edge = tables.beta_row_offsets[b];
    const int row_edges = int(tables.beta_row_offsets[b + 1] - first_edge);
    for(int k_edge = 0; k_edge < row_edges; ++k_edge)
    {
        const std::int64_t edge = first_edge + k_edge;
        const int beta_slot = vectors.beta_src_slot[tables.beta_columns[edge]];
        if(beta_slot < 0)
        {
            continue;
        }
        const int low = tables.beta_low[edge];
        const int high = tables.beta_high[edge];
        const double* coefficients =
            tables.coefficients + std::int64_t(fq::pair_index::gpu_pair(low, high, tables.num_beta)) * alpha_pairs;
        const double* source = vectors.src_in + std::int64_t(beta_slot) * geometry.n_alpha_src_compact;
        fq::arms::add_cross_row(sum, alpha_row, words, coefficients, beta_pair_sign(beta, low, high), source);
    }
    sum.flush(0, vectors.out_local + fq::device::tile_index(row, geometry.a_stop - geometry.a_start, column));
}

}

extern "C"
{

    int fq_launch_cross_sell(const FqTileGeometry* geometry,
                             const FqVectors* vectors,
                             const FqCrossSellTables* tables,
                             void* stream)
    {
        if(fq::launch::empty_tile(*geometry))
        {
            return int(cudaSuccess);
        }
        cross_sell_kernel<<<fq::launch::alpha_column_grid<1>(*geometry), geometry->threads_per_block, 0,
                            static_cast<cudaStream_t>(stream)>>>(*geometry, *vectors, *tables);
        return int(cudaGetLastError());
    }
}
