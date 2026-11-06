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
#include "formats/edge_word.hpp"
#include "launch.hpp"
#include "launch_grid.hpp"

namespace
{


template <int K, bool GROUPED>
__global__ void __launch_bounds__(fq::launch::kArmThreads, 4) beta_singles_pair_kernel(const FqTileGeometry geometry,
                                                                                       const FqVectors vectors,
                                                                                       const FqBetaSinglePairTables tables)
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
        alpha_slot[k] = alphas.active[k] ? __ldg(vectors.alpha_src_slot + alphas.alpha[k]) : -1;
    }
    if(!alphas.any())
    {
        return;
    }
    fq::device::RowAccumulator<K> sums;
    const std::int64_t first = __ldg(tables.row_offsets + b);
    const int row_edges = int(__ldg(tables.row_offsets + b + 1) - first);
    for(int k_edge = 0; k_edge < row_edges; ++k_edge)
    {
        const std::int64_t edge = first + k_edge;
        const int beta_slot = __ldg(vectors.beta_src_slot + __ldg(tables.columns + edge));
        if(beta_slot < 0)
        {
            continue;
        }
        const double c0 = __ldg(tables.c0 + edge);
        const std::uint32_t word = __ldg(tables.signed_pairs + edge);
        const double sign = fq::edge_word::signed_pair_sign(word);
        const double* factors = tables.pair_table + std::int64_t(fq::edge_word::signed_pair_pair(word)) * tables.table_columns;
        const std::int64_t input_row = std::int64_t(beta_slot) * geometry.n_alpha_src_compact;
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            if(alpha_slot[k] < 0)
            {
                continue;
            }
            const double element = fq::arms::pair_element(c0, sign, __ldg(factors + alphas.alpha[k]));
            if(fq::arms::is_applied(element))
            {
                sums.add(k, element, __ldg(vectors.src_in + input_row + alpha_slot[k]));
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


template <int K, int U, bool GROUPED>
__global__ void __launch_bounds__(fq::launch::kArmThreads, 4) beta_singles_pair_hoist_kernel(
    const FqTileGeometry geometry, const FqVectors vectors, const FqBetaSinglePairTables tables)
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
        alpha_slot[k] = alphas.active[k] ? __ldg(vectors.alpha_src_slot + alphas.alpha[k]) : -1;
    }
    if(!alphas.any())
    {
        return;
    }
    fq::device::RowAccumulator<K> sums;
    const std::int64_t first = __ldg(tables.row_offsets + b);
    const int row_edges = int(__ldg(tables.row_offsets + b + 1) - first);
    for(int e0 = 0; e0 < row_edges; e0 += U)
    {
        int column[U];
        double c0[U];
        std::uint32_t word[U];
#pragma unroll
        for(int u = 0; u < U; ++u)
        {
            const bool live = e0 + u < row_edges;
            const std::int64_t edge = first + e0 + u;
            column[u] = live ? __ldg(tables.columns + edge) : -1;
            c0[u] = live ? __ldg(tables.c0 + edge) : 0.0;
            word[u] = live ? __ldg(tables.signed_pairs + edge) : 0u;
        }
        int beta_slot[U];
#pragma unroll
        for(int u = 0; u < U; ++u)
        {
            beta_slot[u] = column[u] >= 0 ? __ldg(vectors.beta_src_slot + column[u]) : -1;
        }
        double factor[U][K];
        double x[U][K];
#pragma unroll
        for(int u = 0; u < U; ++u)
        {
            const double* factors =
                tables.pair_table + std::int64_t(fq::edge_word::signed_pair_pair(word[u])) * tables.table_columns;
            const std::int64_t input_row = std::int64_t(beta_slot[u]) * geometry.n_alpha_src_compact;
#pragma unroll
            for(int k = 0; k < K; ++k)
            {
                const bool use = beta_slot[u] >= 0 && alpha_slot[k] >= 0;
                factor[u][k] = use ? __ldg(factors + alphas.alpha[k]) : 0.0;
                x[u][k] = use ? __ldg(vectors.src_in + input_row + alpha_slot[k]) : 0.0;
            }
        }
#pragma unroll
        for(int u = 0; u < U; ++u)
        {
            if(beta_slot[u] < 0)
            {
                continue;
            }
            const double sign = fq::edge_word::signed_pair_sign(word[u]);
#pragma unroll
            for(int k = 0; k < K; ++k)
            {
                if(alpha_slot[k] < 0)
                {
                    continue;
                }
                const double element = fq::arms::pair_element(c0[u], sign, factor[u][k]);
                if(fq::arms::is_applied(element))
                {
                    sums.add(k, element, x[u][k]);
                }
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


constexpr int kHoistEdges = 2;

template <int K, bool HOIST>
void launch_beta_singles_pair_order(const FqTileGeometry& geometry, const FqVectors& vectors,
                                    const FqBetaSinglePairTables& tables, const cudaStream_t s)
{
    const int threads = geometry.threads_per_block;
    if(tables.block_group >= 1)
    {
        const dim3 grid = fq::launch::linear_grid(fq::launch::alpha_column_blocks<K>(geometry));
        if(HOIST)
        {
            beta_singles_pair_hoist_kernel<K, kHoistEdges, true><<<grid, threads, 0, s>>>(geometry, vectors, tables);
        }
        else
        {
            beta_singles_pair_kernel<K, true><<<grid, threads, 0, s>>>(geometry, vectors, tables);
        }
        return;
    }
    const dim3 grid = fq::launch::alpha_column_grid<K>(geometry);
    if(HOIST)
    {
        beta_singles_pair_hoist_kernel<K, kHoistEdges, false><<<grid, threads, 0, s>>>(geometry, vectors, tables);
    }
    else
    {
        beta_singles_pair_kernel<K, false><<<grid, threads, 0, s>>>(geometry, vectors, tables);
    }
}


template <int K>
__global__ void __launch_bounds__(fq::launch::kArmThreads, 4) beta_singles_delta_kernel(const FqTileGeometry geometry,
                                                                                        const FqVectors vectors,
                                                                                        const FqBetaSingleDeltaTables tables)
{
    const int row = int(fq::device::block_row());
    const int b = geometry.b_start + row;
    if(b >= geometry.b_stop)
    {
        return;
    }
    const int n_a = geometry.a_stop - geometry.a_start;
    const fq::device::AlphaColumns<K> alphas = fq::device::AlphaColumns<K>::of_thread(geometry);
    int alpha_slot[K];
    std::uint64_t alpha[K];
#pragma unroll
    for(int k = 0; k < K; ++k)
    {
        alpha_slot[k] = alphas.active[k] ? vectors.alpha_src_slot[alphas.alpha[k]] : -1;
        alpha[k] = alphas.active[k] ? tables.alpha_strings[geometry.bra_a0 + alphas.alpha[k]] : 0ull;
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
        double element[K];
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            element[k] = tables.c0[edge];
        }
        const double* delta = tables.delta + std::int64_t(edge) * tables.num_alpha;
        for(int p = 0; p < tables.num_alpha; ++p)
        {
            const double value = delta[p];
#pragma unroll
            for(int k = 0; k < K; ++k)
            {
                element[k] += double(int((alpha[k] >> p) & 1ull)) * value;
            }
        }
        const std::int64_t input_row = std::int64_t(beta_slot) * geometry.n_alpha_src_compact;
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            if(alpha_slot[k] >= 0 && fq::arms::is_applied(element[k]))
            {
                sums.add(k, element[k], vectors.src_in[input_row + alpha_slot[k]]);
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

    int fq_launch_beta_singles_pair(const FqTileGeometry* geometry,
                                    const FqVectors* vectors,
                                    const FqBetaSinglePairTables* tables,
                                    void* stream)
    {
        return fq::launch::with_rows_per_thread(*geometry, tables->alphas_per_thread, [&](auto alphas) {
            constexpr int K = decltype(alphas)::value;
            const cudaStream_t s = static_cast<cudaStream_t>(stream);
            if(tables->hoist_loads != 0)
            {
                launch_beta_singles_pair_order<K, true>(*geometry, *vectors, *tables, s);
                return;
            }
            launch_beta_singles_pair_order<K, false>(*geometry, *vectors, *tables, s);
        });
    }

    int fq_launch_beta_singles_delta(const FqTileGeometry* geometry,
                                     const FqVectors* vectors,
                                     const FqBetaSingleDeltaTables* tables,
                                     void* stream)
    {
        return fq::launch::with_rows_per_thread(*geometry, tables->alphas_per_thread, [&](auto alphas) {
            constexpr int K = decltype(alphas)::value;
            beta_singles_delta_kernel<K><<<fq::launch::alpha_column_grid<K>(*geometry), geometry->threads_per_block, 0,
                                          static_cast<cudaStream_t>(stream)>>>(*geometry, *vectors, *tables);
        });
    }
}
