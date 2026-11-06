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
 * Modified: descriptive GPU flag and clique names.
 */
#include <cstdint>
#include <cuda_runtime.h>

#include "arms/cross_row.cuh"
#include "device/edge_stream.cuh"
#include "device/row_block.cuh"
#include "device/tile_index.cuh"
#include "formats/edge_word.hpp"
#include "launch.hpp"
#include "launch_grid.hpp"

namespace
{


template <bool REMAPPED>
__global__ void __launch_bounds__(fq::launch::kArmThreads, 6) cross_direct_kernel(const FqTileGeometry geometry,
                                                                                  const FqVectors vectors,
                                                                                  const FqCrossDirectTables tables)
{
    const std::int64_t task = fq::device::block_row();
    const int r = int(fq::device::thread_column());
    if(task >= tables.num_tasks || (REMAPPED ? r >= tables.num_rows : geometry.a_start + r >= geometry.a_stop))
    {
        return;
    }
    if(tables.gated != 0 && __ldg(vectors.beta_src_slot) == 0)
    {

        return;
    }
    const int column = REMAPPED ? __ldg(tables.order + r) : r;
    const int row = REMAPPED ? r : geometry.a_start + r;
    const std::int32_t* range = tables.tasks + task * 3;
    const int b = range[0];
    const auto* words = reinterpret_cast<const unsigned long long*>(tables.sell_words);
    const auto* beta_words = reinterpret_cast<const unsigned long long*>(tables.beta_words);
    const fq::device::SellRow alpha_row = fq::device::SellRow::at(tables.sell_offsets, tables.sell_lengths, row);
    fq::device::RowAccumulator<1> sum;
    for(int edge = range[1]; edge < range[2]; ++edge)
    {
        const unsigned long long beta_word = __ldg(beta_words + edge);
        const double* coefficients =
            tables.coefficients + std::int64_t(fq::edge_word::cross_pair(beta_word)) * tables.num_pairs;
        const double* source =
            vectors.src_in + std::int64_t(fq::edge_word::cross_column(beta_word)) * geometry.n_alpha_src_compact;
        fq::arms::add_cross_row(sum, alpha_row, words, coefficients, fq::edge_word::cross_sign(beta_word), source);
    }
    sum.flush(0, vectors.out_local + fq::device::tile_index(b - geometry.b_start, geometry.a_stop - geometry.a_start,
                                                            column));
}

}

extern "C"
{

    int fq_launch_cross_direct(const FqTileGeometry* geometry,
                               const FqVectors* vectors,
                               const FqCrossDirectTables* tables,
                               void* stream)
    {
        const bool remapped = tables->order != nullptr;
        const long long n_rows = remapped ? tables->num_rows : geometry->a_stop - geometry->a_start;
        if(n_rows <= 0 || tables->num_tasks <= 0)
        {
            return int(cudaSuccess);
        }
        const dim3 grid = fq::launch::beta_grid(
            static_cast<unsigned int>((n_rows + fq::launch::kArmThreads - 1) / fq::launch::kArmThreads),
            tables->num_tasks);
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        remapped ? cross_direct_kernel<true><<<grid, fq::launch::kArmThreads, 0, s>>>(*geometry, *vectors, *tables)
                 : cross_direct_kernel<false><<<grid, fq::launch::kArmThreads, 0, s>>>(*geometry, *vectors, *tables);
        return int(cudaGetLastError());
    }
}
