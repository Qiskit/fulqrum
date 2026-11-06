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
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <algorithm>
#include <cuda_runtime.h>

#include "launch.hpp"

namespace
{

constexpr int kReduceThreads = 256;
constexpr std::int64_t kMaxReduceBlocks = 65535;


__global__ void __launch_bounds__(kReduceThreads) add_runs_kernel(const std::int32_t* __restrict__ rows,
                                                                  const double* __restrict__ values,
                                                                  const std::int64_t num_entries,
                                                                  const std::int32_t row_begin,
                                                                  const std::int32_t row_end,
                                                                  double* __restrict__ y)
{
    const std::int64_t stride = std::int64_t(gridDim.x) * blockDim.x;
    for(std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < num_entries; i += stride)
    {
        const std::int32_t row = rows[i];
        if(row < row_begin || row >= row_end || (i > 0 && rows[i - 1] == row))
        {
            continue;
        }
        double acc = y[row];
        for(std::int64_t j = i; j < num_entries && rows[j] == row; j++)
        {
            acc += values[j];
        }
        y[row] = acc;
    }
}


__global__ void __launch_bounds__(kReduceThreads) iota_kernel(std::int32_t* positions, const std::int64_t count)
{
    const std::int64_t stride = std::int64_t(gridDim.x) * blockDim.x;
    for(std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
    {
        positions[i] = static_cast<std::int32_t>(i);
    }
}


__global__ void __launch_bounds__(kReduceThreads) gather_runs_kernel(const FsRuns runs)
{
    const std::int64_t stride = std::int64_t(gridDim.x) * blockDim.x;
    for(std::int64_t r = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; r < runs.num_runs; r += stride)
    {
        const std::int32_t row = runs.run_rows[r];
        double acc = runs.y[row];
        for(std::int32_t k = runs.run_starts[r]; k < runs.run_starts[r + 1]; k++)
        {
            acc += runs.values[runs.perm[k]];
        }
        runs.y[row] = acc;
    }
}


inline unsigned int reduce_grid(const std::int64_t items)
{
    const std::int64_t blocks = (items + kReduceThreads - 1) / kReduceThreads;
    return static_cast<unsigned int>(blocks < 1 ? 1 : (blocks < kMaxReduceBlocks ? blocks : kMaxReduceBlocks));
}

}

extern "C"
{

    std::size_t fs_reduce_scratch_bytes(const std::int64_t num_entries, const int end_bit)
    {
        std::size_t bytes = 0;
        cub::DoubleBuffer<std::int32_t> rows(nullptr, nullptr);
        cub::DoubleBuffer<double> values(nullptr, nullptr);
        cub::DeviceRadixSort::SortPairs(nullptr, bytes, rows, values, static_cast<int>(num_entries), 0, end_bit);
        return bytes;
    }

    std::size_t fs_reduce_plan_scratch_bytes(const std::int64_t num_entries, const int end_bit)
    {
        std::size_t sort_bytes = 0;
        cub::DoubleBuffer<std::int32_t> rows(nullptr, nullptr);
        cub::DoubleBuffer<std::int32_t> positions(nullptr, nullptr);
        cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, rows, positions, static_cast<int>(num_entries), 0, end_bit);
        std::size_t encode_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(nullptr, encode_bytes, static_cast<std::int32_t*>(nullptr),
                                           static_cast<std::int32_t*>(nullptr), static_cast<std::int32_t*>(nullptr),
                                           static_cast<std::int32_t*>(nullptr), static_cast<int>(num_entries));
        return std::max(sort_bytes, encode_bytes);
    }

    int fs_build_reduce_plan(std::int32_t* rows, std::int32_t* rows_alt, std::int32_t* positions, std::int32_t* positions_alt,
                             void* scratch, std::size_t scratch_bytes, const std::int64_t num_entries, const int end_bit,
                             std::int32_t* sorted_positions, std::int32_t* run_rows, std::int32_t* run_counts,
                             std::int32_t* num_runs, void* stream)
    {
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        if(num_entries <= 0)
        {
            return int(cudaMemsetAsync(num_runs, 0, sizeof(std::int32_t), s));
        }
        iota_kernel<<<reduce_grid(num_entries), kReduceThreads, 0, s>>>(positions, num_entries);
        cudaError_t status = cudaGetLastError();
        if(status != cudaSuccess)
        {
            return int(status);
        }
        cub::DoubleBuffer<std::int32_t> row_buffer(rows, rows_alt);
        cub::DoubleBuffer<std::int32_t> position_buffer(positions, positions_alt);
        std::size_t bytes = scratch_bytes;
        status = cub::DeviceRadixSort::SortPairs(scratch, bytes, row_buffer, position_buffer, static_cast<int>(num_entries), 0,
                                                 end_bit, s);
        if(status != cudaSuccess)
        {
            return int(status);
        }
        status = cudaMemcpyAsync(sorted_positions, position_buffer.Current(), sizeof(std::int32_t) * num_entries,
                                 cudaMemcpyDeviceToDevice, s);
        if(status != cudaSuccess)
        {
            return int(status);
        }
        bytes = scratch_bytes;
        return int(cub::DeviceRunLengthEncode::Encode(scratch, bytes, row_buffer.Current(), run_rows, run_counts, num_runs,
                                                      static_cast<int>(num_entries), s));
    }

    int fs_launch_gather_runs(const FsRuns* runs, void* stream)
    {
        if(runs->num_runs <= 0)
        {
            return int(cudaSuccess);
        }
        gather_runs_kernel<<<reduce_grid(runs->num_runs), kReduceThreads, 0, static_cast<cudaStream_t>(stream)>>>(*runs);
        return int(cudaGetLastError());
    }

    int fs_launch_reduce(const FsReduce* reduce, void* stream)
    {
        if(reduce->num_entries <= 0)
        {
            return int(cudaSuccess);
        }
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        cub::DoubleBuffer<std::int32_t> rows(reduce->rows, reduce->rows_alt);
        cub::DoubleBuffer<double> values(reduce->values, reduce->values_alt);
        std::size_t bytes = static_cast<std::size_t>(reduce->scratch_bytes);
        const cudaError_t sorted = cub::DeviceRadixSort::SortPairs(reduce->scratch, bytes, rows, values,
                                                                   static_cast<int>(reduce->num_entries), 0,
                                                                   reduce->end_bit, s);
        if(sorted != cudaSuccess)
        {
            return int(sorted);
        }
        const std::int64_t blocks = (reduce->num_entries + kReduceThreads - 1) / kReduceThreads;
        const unsigned int grid = static_cast<unsigned int>(blocks < kMaxReduceBlocks ? blocks : kMaxReduceBlocks);
        add_runs_kernel<<<grid, kReduceThreads, 0, s>>>(rows.Current(), values.Current(), reduce->num_entries, reduce->row_begin,
                                                        reduce->row_end, reduce->y);
        return int(cudaGetLastError());
    }
}
