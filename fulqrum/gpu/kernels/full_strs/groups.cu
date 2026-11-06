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
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include "elements.cuh"
#include "group_pairs.cuh"
#include "launch.hpp"

namespace
{

namespace dev = fq::full_strs::device;
using dev::key_of;
using dev::members_below;

constexpr int kGroupThreads = 256;
constexpr int kWarpsPerBlock = kGroupThreads / 32;
constexpr std::int64_t kMaxGroupBlocks = 65535;
constexpr int kKeyWords = 1;
constexpr int kOffsetThreads = 256;
constexpr int kBlockGroupThreads = 256;
constexpr int kMaxSharedMembers = 1024;


template <int W>
__device__ __forceinline__ void load_key(const FsKeys& keys, const std::uint32_t row, std::uint64_t* key)
{
    const std::uint64_t* source = key_of<W>(keys, row);
#pragma unroll
    for(int word = 0; word < 2 * W; word++)
    {
        key[word] = source[word];
    }
}


__device__ __forceinline__ void write_partial(const FsGroups& groups, const std::int64_t position, const double acc)
{
    if(groups.y != nullptr)
    {
        if(acc != 0.0)
        {
            atomicAdd(groups.y + groups.members[position], acc);
        }
        return;
    }
    groups.partials[position] = acc;
}


template <int W, int CLS>
__global__ void __launch_bounds__(kGroupThreads) tiny_group_kernel(const FsKeys keys,
                                                                    const FsTables tables,
                                                                    const FsGroups groups,
                                                                    const double* __restrict__ x)
{
    const std::int64_t stride = std::int64_t(gridDim.x) * blockDim.x;
    for(std::int64_t g = groups.first_group + std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        g < groups.first_group + groups.num_groups;
        g += stride)
    {
        const std::int32_t begin = groups.offsets[g];
        const int m = groups.offsets[g + 1] - begin;
        std::uint64_t key[3][2 * W];
        double value[3];
        bool output[3];
        bool source[3];
#pragma unroll
        for(int j = 0; j < 3; j++)
        {
            output[j] = false;
            source[j] = false;
            if(j < m)
            {
                const std::uint32_t row = groups.members[begin + j];
                load_key<W>(keys, row, key[j]);
                value[j] = x[row];
                output[j] = row >= groups.output_begin && row < groups.output_end;
                source[j] = row >= groups.source_begin && row < groups.source_end;
            }
        }
#pragma unroll
        for(int k = 0; k < 3; k++)
        {
            if(!output[k])
            {
                continue;
            }
            double acc = 0.0;
#pragma unroll
            for(int j = 0; j < 3; j++)
            {
                if(source[j] && j != k && dev::pair_counts<W, CLS>(key[k], key[j]))
                {
                    acc = fma(dev::element<W, CLS>(tables, key[k], key[j]), value[j], acc);
                }
            }
            write_partial(groups, begin + k, acc);
        }
    }
}


template <int W, int CLS, int TILE, int MPL>
__global__ void __launch_bounds__(kGroupThreads) tile_group_kernel(const FsKeys keys,
                                                                    const FsTables tables,
                                                                    const FsGroups groups,
                                                                    const double* __restrict__ x)
{
    constexpr int kTilesPerWarp = 32 / TILE;
    const int lane = threadIdx.x % TILE;
    const int tile_in_warp = (threadIdx.x & 31) / TILE;
    const std::int64_t warp = (std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const std::int64_t num_warps = (std::int64_t(gridDim.x) * blockDim.x) >> 5;
    const std::int64_t last_group = groups.first_group + groups.num_groups;
    for(std::int64_t first = groups.first_group + warp * kTilesPerWarp; first < last_group; first += num_warps * kTilesPerWarp)
    {
        const std::int64_t g = first + tile_in_warp;
        const bool active = g < last_group;
        const std::int32_t begin = active ? groups.offsets[g] : 0;
        const int m = active ? groups.offsets[g + 1] - begin : 0;
        std::uint64_t key[MPL][2 * W];
        double value[MPL];
        int bounds[4] = {0, 0, 0, 0};
#pragma unroll
        for(int slot = 0; slot < MPL; slot++)
        {
            const int j = lane + slot * TILE;
            value[slot] = 0.0;
#pragma unroll
            for(int word = 0; word < 2 * W; word++)
            {
                key[slot][word] = 0;
            }
            if(j < m)
            {
                const std::uint32_t row = groups.members[begin + j];
                load_key<W>(keys, row, key[slot]);
                value[slot] = x[row];
                bounds[0] += row < groups.output_begin ? 1 : 0;
                bounds[1] += row < groups.output_end ? 1 : 0;
                bounds[2] += row < groups.source_begin ? 1 : 0;
                bounds[3] += row < groups.source_end ? 1 : 0;
            }
        }
        const unsigned int mask = 0xffffffffu;
#pragma unroll
        for(int b = 0; b < 4; b++)
        {
#pragma unroll
            for(int offset = TILE / 2; offset > 0; offset >>= 1)
            {
                bounds[b] += __shfl_xor_sync(mask, bounds[b], offset, TILE);
            }
        }
        const int output_first = bounds[0];
        const int output_last = bounds[1];
        const int source_first = bounds[2];
        const int source_last = bounds[3];
        const int warp_first = static_cast<int>(__reduce_min_sync(mask, static_cast<unsigned int>(output_first)));
        const int warp_max = static_cast<int>(__reduce_max_sync(mask, static_cast<unsigned int>(output_last)));
        for(int k = warp_first; k < warp_max; k++)
        {
            std::uint64_t out[2 * W];
#pragma unroll
            for(int word = 0; word < 2 * W; word++)
            {
                std::uint64_t mine = 0;
#pragma unroll
                for(int slot = 0; slot < MPL; slot++)
                {
                    mine = (k / TILE == slot) ? key[slot][word] : mine;
                }
                out[word] = __shfl_sync(mask, mine, k % TILE, TILE);
            }
            double acc = 0.0;
#pragma unroll
            for(int slot = 0; slot < MPL; slot++)
            {
                const int j = lane + slot * TILE;
                const bool output = k >= output_first && k < output_last;
                const bool source = j >= source_first && j < source_last;
                if(output && source && j != k && dev::pair_counts<W, CLS>(out, key[slot]))
                {
                    acc = fma(dev::element<W, CLS>(tables, out, key[slot]), value[slot], acc);
                }
            }
#pragma unroll
            for(int offset = TILE / 2; offset > 0; offset >>= 1)
            {
                acc += __shfl_down_sync(mask, acc, offset, TILE);
            }
            if(lane == 0 && k >= output_first && k < output_last)
            {
                write_partial(groups, begin + k, acc);
            }
        }
    }
}


template <int W, int CLS>
__global__ void __launch_bounds__(kBlockGroupThreads) block_group_kernel(const FsKeys keys,
                                                                          const FsTables tables,
                                                                          const FsGroups groups,
                                                                          const double* __restrict__ x)
{
    __shared__ std::uint64_t shared_keys[kMaxSharedMembers * 2 * W];
    __shared__ double shared_values[kMaxSharedMembers];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int kWarps = kBlockGroupThreads / 32;
    for(std::int64_t g = groups.first_group + blockIdx.x; g < groups.first_group + groups.num_groups; g += gridDim.x)
    {
        const std::int32_t begin = groups.offsets[g];
        const int m = groups.offsets[g + 1] - begin;
        const int output_first = members_below(groups.members + begin, m, groups.output_begin);
        const int output_last = members_below(groups.members + begin, m, groups.output_end);
        const int source_first = members_below(groups.members + begin, m, groups.source_begin);
        const int source_last = members_below(groups.members + begin, m, groups.source_end);
        const bool staged = m <= kMaxSharedMembers;
        __syncthreads();
        if(staged)
        {
            for(int j = threadIdx.x; j < m; j += kBlockGroupThreads)
            {
                const std::uint32_t row = groups.members[begin + j];
                load_key<W>(keys, row, shared_keys + j * 2 * W);
                shared_values[j] = x[row];
            }
        }
        __syncthreads();
        for(int k = output_first + warp; k < output_last; k += kWarps)
        {
            const std::uint64_t* out = staged ? shared_keys + k * 2 * W : key_of<W>(keys, groups.members[begin + k]);
            double acc = 0.0;
            for(int j = lane; j < m; j += 32)
            {
                const std::uint32_t row = staged ? 0u : groups.members[begin + j];
                const std::uint64_t* source = staged ? shared_keys + j * 2 * W : key_of<W>(keys, row);
                const double value = staged ? shared_values[j] : x[row];
                if(j >= source_first && j < source_last && j != k && dev::pair_counts<W, CLS>(out, source))
                {
                    acc = fma(dev::element<W, CLS>(tables, out, source), value, acc);
                }
            }
#pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1)
            {
                acc += __shfl_down_sync(0xffffffffu, acc, offset);
            }
            if(lane == 0)
            {
                write_partial(groups, begin + k, acc);
            }
        }
    }
}


inline unsigned int strided_grid(const std::int64_t items, const std::int64_t per_block)
{
    const std::int64_t blocks = (items + per_block - 1) / per_block;
    return static_cast<unsigned int>(blocks < 1 ? 1 : (blocks < kMaxGroupBlocks ? blocks : kMaxGroupBlocks));
}


template <int CLS>
int launch_binned(const FsKeys& keys, const FsTables& tables, const FsGroups& groups, const double* x, cudaStream_t stream)
{
    const std::int64_t n = groups.num_groups;
    switch(groups.bucket)
    {
    case 0:
        tiny_group_kernel<kKeyWords, CLS><<<strided_grid(n, kGroupThreads), kGroupThreads, 0, stream>>>(keys, tables, groups, x);
        break;
    case 1:
        tile_group_kernel<kKeyWords, CLS, 8, 1><<<strided_grid(n, kGroupThreads / 8), kGroupThreads, 0, stream>>>(keys, tables, groups, x);
        break;
    case 2:
        tile_group_kernel<kKeyWords, CLS, 16, 1><<<strided_grid(n, kGroupThreads / 16), kGroupThreads, 0, stream>>>(keys, tables, groups, x);
        break;
    case 3:
        tile_group_kernel<kKeyWords, CLS, 32, 1><<<strided_grid(n, kGroupThreads / 32), kGroupThreads, 0, stream>>>(keys, tables, groups, x);
        break;
    case 4:
        tile_group_kernel<kKeyWords, CLS, 32, 2><<<strided_grid(n, kGroupThreads / 32), kGroupThreads, 0, stream>>>(keys, tables, groups, x);
        break;
    case 5:
    case 6:
        block_group_kernel<kKeyWords, CLS><<<strided_grid(n, 1), kBlockGroupThreads, 0, stream>>>(keys, tables, groups, x);
        break;
    default:
        return int(cudaErrorInvalidValue);
    }
    return int(cudaGetLastError());
}


template <int W, int CLS>
__global__ void __launch_bounds__(kGroupThreads) uniform_group_kernel(const FsKeys keys,
                                                                       const FsTables tables,
                                                                       const FsGroups groups,
                                                                       const double* __restrict__ x)
{
    const int lane = threadIdx.x & 31;
    const std::int64_t warp = (std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const std::int64_t num_warps = (std::int64_t(gridDim.x) * blockDim.x) >> 5;
    for(std::int64_t g = groups.first_group + warp; g < groups.first_group + groups.num_groups; g += num_warps)
    {
        const std::int32_t begin = groups.offsets[g];
        const std::int32_t end = groups.offsets[g + 1];
        const std::int32_t first_output = begin + members_below(groups.members + begin, end - begin, groups.output_begin);
        const std::int32_t last_output = begin + members_below(groups.members + begin, end - begin, groups.output_end);
        const std::int32_t first_source = begin + members_below(groups.members + begin, end - begin, groups.source_begin);
        const std::int32_t last_source = begin + members_below(groups.members + begin, end - begin, groups.source_end);
        for(std::int32_t k = first_output; k < last_output; k++)
        {
            const std::uint64_t* out = key_of<W>(keys, groups.members[k]);
            double acc = 0.0;
            for(std::int32_t s = begin + lane; s < end; s += 32)
            {
                if(s == k || s < first_source || s >= last_source)
                {
                    continue;
                }
                const std::uint32_t row = groups.members[s];
                const std::uint64_t* source = key_of<W>(keys, row);
                if(dev::pair_counts<W, CLS>(out, source))
                {
                    acc = fma(dev::element<W, CLS>(tables, out, source), x[row], acc);
                }
            }
#pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1)
            {
                acc += __shfl_down_sync(0xffffffffu, acc, offset);
            }
            if(lane == 0)
            {
                write_partial(groups, k, acc);
            }
        }
    }
}


__global__ void __launch_bounds__(kOffsetThreads) widen_sizes_kernel(const std::uint16_t* sizes,
                                                                     const std::int64_t num_groups,
                                                                     std::int32_t* offsets)
{
    const std::int64_t stride = std::int64_t(gridDim.x) * blockDim.x;
    for(std::int64_t g = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; g < num_groups; g += stride)
    {
        offsets[g + 1] = sizes[g];
    }
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        offsets[0] = 0;
    }
}


template <int CLS>
int launch_uniform(const FsKeys& keys, const FsTables& tables, const FsGroups& groups, const double* x, cudaStream_t stream)
{
    const std::int64_t blocks = (groups.num_groups + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const unsigned int grid = static_cast<unsigned int>(blocks < kMaxGroupBlocks ? blocks : kMaxGroupBlocks);
    uniform_group_kernel<kKeyWords, CLS><<<grid, kGroupThreads, 0, stream>>>(keys, tables, groups, x);
    return int(cudaGetLastError());
}

}

extern "C"
{

    int fs_launch_groups_uniform(const FsKeys* keys, const FsTables* tables, const FsGroups* groups, const double* x,
                                 void* stream)
    {
        if(groups->num_groups <= 0)
        {
            return int(cudaSuccess);
        }
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        switch(groups->cls)
        {
        case dev::kAlphaSingle:
            return launch_uniform<dev::kAlphaSingle>(*keys, *tables, *groups, x, s);
        case dev::kAlphaDouble:
            return launch_uniform<dev::kAlphaDouble>(*keys, *tables, *groups, x, s);
        case dev::kBetaSingle:
            return launch_uniform<dev::kBetaSingle>(*keys, *tables, *groups, x, s);
        case dev::kBetaDouble:
            return launch_uniform<dev::kBetaDouble>(*keys, *tables, *groups, x, s);
        case dev::kMixed:
            return launch_uniform<dev::kMixed>(*keys, *tables, *groups, x, s);
        default:
            return int(cudaErrorInvalidValue);
        }
    }

    int fs_launch_groups_binned(const FsKeys* keys, const FsTables* tables, const FsGroups* groups, const double* x,
                                void* stream)
    {
        if(groups->num_groups <= 0)
        {
            return int(cudaSuccess);
        }
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        switch(groups->cls)
        {
        case dev::kAlphaSingle:
            return launch_binned<dev::kAlphaSingle>(*keys, *tables, *groups, x, s);
        case dev::kAlphaDouble:
            return launch_binned<dev::kAlphaDouble>(*keys, *tables, *groups, x, s);
        case dev::kBetaSingle:
            return launch_binned<dev::kBetaSingle>(*keys, *tables, *groups, x, s);
        case dev::kBetaDouble:
            return launch_binned<dev::kBetaDouble>(*keys, *tables, *groups, x, s);
        case dev::kMixed:
            return launch_binned<dev::kMixed>(*keys, *tables, *groups, x, s);
        default:
            return int(cudaErrorInvalidValue);
        }
    }

    std::size_t fs_group_offsets_scratch_bytes(const std::int64_t num_groups)
    {
        std::size_t bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<std::int32_t*>(nullptr),
                                      static_cast<std::int32_t*>(nullptr), static_cast<int>(num_groups));
        return bytes;
    }

    int fs_launch_group_offsets(const std::uint16_t* sizes, const std::int64_t num_groups, std::int32_t* offsets,
                                void* scratch, std::size_t scratch_bytes, void* stream)
    {
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        const std::int64_t blocks = (num_groups + kOffsetThreads - 1) / kOffsetThreads;
        const unsigned int grid = static_cast<unsigned int>(blocks < 1 ? 1 : (blocks < kMaxGroupBlocks ? blocks : kMaxGroupBlocks));
        widen_sizes_kernel<<<grid, kOffsetThreads, 0, s>>>(sizes, num_groups, offsets);
        const cudaError_t launched = cudaGetLastError();
        if(launched != cudaSuccess || num_groups == 0)
        {
            return int(launched);
        }
        return int(cub::DeviceScan::InclusiveSum(scratch, scratch_bytes, offsets + 1, offsets + 1, static_cast<int>(num_groups), s));
    }
}
