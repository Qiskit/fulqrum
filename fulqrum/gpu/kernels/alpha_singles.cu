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
#include "device/edge_stream.cuh"
#include "device/row_block.cuh"
#include "device/tile_index.cuh"
#include "formats/edge_word.hpp"
#include "launch.hpp"
#include "launch_grid.hpp"

namespace
{


template <int K>
__global__ void __launch_bounds__(fq::launch::kArmThreads, 4) alpha_singles_pair_kernel(const FqTileGeometry geometry,
                                                                                        const FqVectors vectors,
                                                                                        const FqAlphaSinglePairTables tables)
{
    const int a_gpu = int(fq::device::thread_column());
    const int a = geometry.a_start + a_gpu;
    if(a >= geometry.a_stop)
    {
        return;
    }
    const int n_a = geometry.a_stop - geometry.a_start;
    const int n_b = geometry.b_stop - geometry.b_start;
    const int first_row = int(fq::device::block_row() * K);
    if(first_row >= n_b)
    {
        return;
    }
    int beta_slot[K];
    const double* factors[K];
#pragma unroll
    for(int k = 0; k < K; ++k)
    {
        const int row = first_row + k;
        const int b = geometry.b_start + (row < n_b ? row : 0);
        beta_slot[k] = row < n_b ? __ldg(vectors.beta_src_slot + b) : -1;
        factors[k] = tables.pair_table + std::int64_t(b) * tables.num_pairs;
    }
    fq::device::RowAccumulator<K> rows;
    const auto* words = reinterpret_cast<const unsigned long long*>(tables.sell_words);
    for(fq::device::EdgeStream<unsigned long long> edges(
            words, fq::device::SellRow::at(tables.sell_offsets, tables.sell_lengths, a));
        edges.valid(); edges.advance())
    {
        const int column = fq::edge_word::cross_column(edges.current);
        if(column < 0)
        {
            continue;
        }
        const int pair = fq::edge_word::cross_pair(edges.current);
        const double sign = fq::edge_word::cross_sign(edges.current);
        const double c0 = __ldg(tables.c0_sell + edges.slot());
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            if(beta_slot[k] < 0)
            {
                continue;
            }
            const double element = fq::arms::pair_element(c0, sign, __ldg(factors[k] + pair));
            if(fq::arms::is_applied(element))
            {
                const std::int64_t input = fq::device::tile_index(beta_slot[k], geometry.n_alpha_src_compact, column);
                rows.add(k, element, __ldg(vectors.src_in + input));
            }
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


template <int K>
__global__ void __launch_bounds__(fq::launch::kArmThreads, 4) alpha_singles_pair_hoist_kernel(
    const FqTileGeometry geometry, const FqVectors vectors, const FqAlphaSinglePairTables tables)
{
    const int a_gpu = int(fq::device::thread_column());
    const int a = geometry.a_start + a_gpu;
    if(a >= geometry.a_stop)
    {
        return;
    }
    const int n_a = geometry.a_stop - geometry.a_start;
    const int n_b = geometry.b_stop - geometry.b_start;
    const int first_row = int(fq::device::block_row() * K);
    if(first_row >= n_b)
    {
        return;
    }
    int beta_slot[K];
    const double* factors[K];
    std::int64_t input_row[K];
#pragma unroll
    for(int k = 0; k < K; ++k)
    {
        const int row = first_row + k;
        const int b = geometry.b_start + (row < n_b ? row : 0);
        beta_slot[k] = row < n_b ? __ldg(vectors.beta_src_slot + b) : -1;
        factors[k] = tables.pair_table + std::int64_t(b) * tables.num_pairs;
        input_row[k] = std::int64_t(beta_slot[k] < 0 ? 0 : beta_slot[k]) * geometry.n_alpha_src_compact;
    }
    fq::device::RowAccumulator<K> rows;
    const auto* words = reinterpret_cast<const unsigned long long*>(tables.sell_words);
    for(fq::device::EdgeStream<unsigned long long> edges(
            words, fq::device::SellRow::at(tables.sell_offsets, tables.sell_lengths, a));
        edges.valid(); edges.advance())
    {
        const int column = fq::edge_word::cross_column(edges.current);
        const bool live = column >= 0;
        const int pair = live ? fq::edge_word::cross_pair(edges.current) : 0;
        const double sign = fq::edge_word::cross_sign(edges.current);
        const double c0 = __ldg(tables.c0_sell + edges.slot());
        double factor[K];
        double x[K];
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            factor[k] = __ldg(factors[k] + pair);
            x[k] = live && beta_slot[k] >= 0 ? __ldg(vectors.src_in + input_row[k] + column) : 0.0;
        }
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            const double element = fq::arms::pair_element(c0, sign, factor[k]);
            if(live && beta_slot[k] >= 0 && fq::arms::is_applied_bits(element))
            {
                rows.add(k, element, x[k]);
            }
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


template <int K>
__global__ void __launch_bounds__(fq::launch::kArmThreads, 4) alpha_singles_delta_kernel(const FqTileGeometry geometry,
                                                                                         const FqVectors vectors,
                                                                                         const FqAlphaSingleDeltaTables tables)
{
    const int a_gpu = int(fq::device::thread_column());
    const int a = geometry.a_start + a_gpu;
    if(a >= geometry.a_stop)
    {
        return;
    }
    const int n_a = geometry.a_stop - geometry.a_start;
    const int n_b = geometry.b_stop - geometry.b_start;
    const int first_row = int(fq::device::block_row() * K);
    if(first_row >= n_b)
    {
        return;
    }
    int beta_slot[K];
    std::uint64_t beta[K];
#pragma unroll
    for(int k = 0; k < K; ++k)
    {
        const int row = first_row + k;
        const bool live = row < n_b;
        beta_slot[k] = live ? vectors.beta_src_slot[geometry.b_start + row] : -1;
        beta[k] = live ? tables.beta_strings[geometry.bra_b0 + geometry.b_start + row] : 0ull;
    }
    fq::device::RowAccumulator<K> rows;
    const std::int64_t first_edge = tables.row_offsets[a];
    const int row_edges = int(tables.row_offsets[a + 1] - first_edge);
    for(int k_edge = 0; k_edge < row_edges; ++k_edge)
    {
        const std::int64_t edge = first_edge + k_edge;
        const int column = vectors.alpha_src_slot[tables.columns[edge]];
        if(column < 0)
        {
            continue;
        }
        double element[K];
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            element[k] = tables.c0[edge];
        }
        const double* delta = tables.delta + std::int64_t(edge) * tables.num_beta;
        for(int p = 0; p < tables.num_beta; ++p)
        {
            const double value = delta[p];
#pragma unroll
            for(int k = 0; k < K; ++k)
            {
                element[k] += double(int((beta[k] >> p) & 1ull)) * value;
            }
        }
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            if(beta_slot[k] >= 0 && fq::arms::is_applied(element[k]))
            {
                const std::int64_t input = fq::device::tile_index(beta_slot[k], geometry.n_alpha_src_compact, column);
                rows.add(k, element[k], vectors.src_in[input]);
            }
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

}

extern "C"
{

    int fq_launch_alpha_singles_pair(const FqTileGeometry* geometry,
                                     const FqVectors* vectors,
                                     const FqAlphaSinglePairTables* tables,
                                     void* stream)
    {
        return fq::launch::with_rows_per_thread(*geometry, tables->rows_per_thread, [&](auto rows) {
            constexpr int K = decltype(rows)::value;
            const dim3 grid = fq::launch::beta_row_block_grid<K>(*geometry);
            const cudaStream_t s = static_cast<cudaStream_t>(stream);
            if(tables->hoist_loads != 0)
            {
                alpha_singles_pair_hoist_kernel<K><<<grid, geometry->threads_per_block, 0, s>>>(*geometry, *vectors,
                                                                                               *tables);
                return;
            }
            alpha_singles_pair_kernel<K><<<grid, geometry->threads_per_block, 0, s>>>(*geometry, *vectors, *tables);
        });
    }

    int fq_launch_alpha_singles_delta(const FqTileGeometry* geometry,
                                      const FqVectors* vectors,
                                      const FqAlphaSingleDeltaTables* tables,
                                      void* stream)
    {
        return fq::launch::with_rows_per_thread(*geometry, tables->rows_per_thread, [&](auto rows) {
            constexpr int K = decltype(rows)::value;
            alpha_singles_delta_kernel<K><<<fq::launch::beta_row_block_grid<K>(*geometry), geometry->threads_per_block, 0,
                                           static_cast<cudaStream_t>(stream)>>>(*geometry, *vectors, *tables);
        });
    }
}
