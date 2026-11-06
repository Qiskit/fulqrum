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

#include "../arms/elements.hpp"
#include "../device/edge_stream.cuh"
#include "../device/row_block.cuh"
#include "../device/tile_index.cuh"
#include "../device/warp.cuh"
#include "../formats/edge_word.hpp"
#include "../formats/params.hpp"

namespace
{

constexpr int kThreads = 128;


unsigned int blocks_for(const std::int64_t n)
{
    return static_cast<unsigned int>((n + kThreads - 1) / kThreads);
}


__global__ void __launch_bounds__(kThreads) sell_stream_kernel(const std::int64_t* offsets,
                                                               const int* lengths,
                                                               const std::uint64_t* words,
                                                               const int n_rows,
                                                               const int max_length,
                                                               std::uint64_t* out)
{
    const std::int64_t row = fq::device::thread_column();
    if(row >= n_rows)
    {
        return;
    }
    fq::device::EdgeStream<unsigned long long> edges(reinterpret_cast<const unsigned long long*>(words),
                                                      fq::device::SellRow::at(offsets, lengths, int(row)));
    for(; edges.valid(); edges.advance())
    {
        out[fq::device::tile_index(row, max_length, edges.k)] = edges.current;
    }
}


__global__ void __launch_bounds__(kThreads) decode_cross_kernel(const std::uint64_t* words,
                                                                const std::int64_t n,
                                                                int* columns,
                                                                int* pairs,
                                                                double* signs)
{
    const std::int64_t i = fq::device::thread_column();
    if(i >= n)
    {
        return;
    }
    columns[i] = fq::edge_word::cross_column(words[i]);
    pairs[i] = fq::edge_word::cross_pair(words[i]);
    signs[i] = fq::edge_word::cross_sign(words[i]);
}


__global__ void __launch_bounds__(kThreads) pair_element_kernel(const double* c0,
                                                                const double* sign,
                                                                const double* factor,
                                                                const std::int64_t n,
                                                                double* element,
                                                                double* fused)
{
    const std::int64_t i = fq::device::thread_column();
    if(i >= n)
    {
        return;
    }
    element[i] = fq::arms::pair_element(c0[i], sign[i], factor[i]);
    fused[i] = fma(sign[i], factor[i], c0[i]);
}


template <int K>
__global__ void __launch_bounds__(kThreads) row_accumulator_kernel(const double* elements,
                                                                   const double* x,
                                                                   const int n_edges,
                                                                   const int n_rows,
                                                                   double* out)
{
    const std::int64_t first = fq::device::thread_column() * K;
    fq::device::RowAccumulator<K> rows;
    for(int e = 0; e < n_edges; e++)
    {
#pragma unroll
        for(int k = 0; k < K; k++)
        {
            if(first + k < n_rows)
            {
                rows.add(k, elements[e], x[fq::device::tile_index(first + k, n_edges, e)]);
            }
        }
    }
#pragma unroll
    for(int k = 0; k < K; k++)
    {
        if(first + k < n_rows)
        {
            rows.flush(k, out + first + k);
        }
    }
}


__global__ void __launch_bounds__(kThreads) warp_kernel(const int* flags, const int n, int* any, int* lane0)
{
    const std::int64_t i = fq::device::thread_column();
    const int flag = i < n ? flags[i] : 0;
    const bool found = fq::device::warp_any(flag != 0);
    const int first = fq::device::warp_broadcast(flag, 0);
    if(i < n)
    {
        any[i] = found ? 1 : 0;
        lane0[i] = first;
    }
}

}

extern "C"
{


    int fq_test_sell_stream(const std::int64_t* offsets,
                            const int* lengths,
                            const std::uint64_t* words,
                            int n_rows,
                            int max_length,
                            std::uint64_t* out)
    {
        sell_stream_kernel<<<blocks_for(n_rows), kThreads>>>(offsets, lengths, words, n_rows, max_length, out);
        return int(cudaGetLastError());
    }


    int fq_test_decode_cross(const std::uint64_t* words, std::int64_t n, int* columns, int* pairs, double* signs)
    {
        decode_cross_kernel<<<blocks_for(n), kThreads>>>(words, n, columns, pairs, signs);
        return int(cudaGetLastError());
    }


    int fq_test_pair_element(const double* c0,
                             const double* sign,
                             const double* factor,
                             std::int64_t n,
                             double* element,
                             double* fused)
    {
        pair_element_kernel<<<blocks_for(n), kThreads>>>(c0, sign, factor, n, element, fused);
        return int(cudaGetLastError());
    }


    int fq_test_row_accumulator(const double* elements, const double* x, int n_edges, int n_rows, double* out)
    {
        row_accumulator_kernel<4><<<blocks_for((n_rows + 3) / 4), kThreads>>>(elements, x, n_edges, n_rows, out);
        return int(cudaGetLastError());
    }


    int fq_test_warp(const int* flags, int n, int* any, int* lane0)
    {
        warp_kernel<<<blocks_for(n), kThreads>>>(flags, n, any, lane0);
        return int(cudaGetLastError());
    }


    std::size_t fq_test_tile_geometry_size()
    {
        return sizeof(FqTileGeometry);
    }
}
