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
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

#include "elements.cuh"
#include "group_pairs.cuh"
#include "launch.hpp"

namespace
{

namespace dev = fq::full_strs::device;

constexpr int kRdmThreads = 256;
constexpr int kKeyWords = 1;
constexpr std::int64_t kMaxRdmBlocks = 65535;
constexpr int kRdmSharedMembers = 1024;


__device__ __forceinline__ std::int64_t three_index(const int m, const int p, const int q, const int r)
{
    return (std::int64_t(p) * m + q) * m + r;
}


__device__ __forceinline__ std::int64_t four_index(const int m, const int p, const int q, const int r, const int s)
{
    return ((std::int64_t(p) * m + q) * m + r) * m + s;
}


template <int W>
__device__ __forceinline__ bool apply_ladder(std::uint64_t* words, const int orbital, const bool create, double& sign)
{
    if(dev::has_orbital<W>(words, orbital) == create)
    {
        return false;
    }
    unsigned int below = 0;
#pragma unroll
    for(int word = 0; word < W; word++)
    {
        const int first = word * 64;
        if(orbital >= first + 64)
        {
            below += static_cast<unsigned int>(__popcll(words[word]));
        }
        else if(orbital > first)
        {
            below += static_cast<unsigned int>(__popcll(words[word] & ((1ull << (orbital - first)) - 1ull)));
        }
    }
    if(below & 1u)
    {
        sign = -sign;
    }
    words[orbital >> 6] ^= 1ull << (orbital & 63);
    return true;
}


template <int W>
__device__ __forceinline__ void add_single(const FsRdm& out, const int sector, const std::uint64_t* bra,
                                           const std::uint64_t* ket, const double value)
{
    const int m = out.num_orbitals;
    const std::uint64_t* own = bra + sector * W;
    const std::uint64_t* own_ket = ket + sector * W;
    const std::uint64_t* other = bra + (1 - sector) * W;
    int flipped[2];
    dev::flip_orbitals<W>(own, own_ket, flipped, 2);
    const bool low_in_bra = dev::has_orbital<W>(own, flipped[0]);
    const int a = low_in_bra ? flipped[0] : flipped[1];
    const int b = low_in_bra ? flipped[1] : flipped[0];
    const double v = dev::between_parity<W>(own, flipped[0], flipped[1]) ? -value : value;
    atomicAdd((sector == 0 ? out.gamma_alpha : out.gamma_beta) + a * m + b, v);
    double* same = sector == 0 ? out.same_alpha : out.same_beta;
    double* cross = sector == 0 ? out.other_alpha : out.other_beta;
#pragma unroll
    for(int word = 0; word < W; word++)
    {
        for(std::uint64_t bits = own[word] & own_ket[word]; bits; bits &= bits - 1)
        {
            atomicAdd(same + three_index(m, a, b, word * 64 + __ffsll(static_cast<long long>(bits)) - 1), v);
        }
        for(std::uint64_t bits = other[word]; bits; bits &= bits - 1)
        {
            atomicAdd(cross + three_index(m, a, b, word * 64 + __ffsll(static_cast<long long>(bits)) - 1), v);
        }
    }
}


template <int W>
__device__ __forceinline__ void add_double(const FsRdm& out, const int sector, const std::uint64_t* bra,
                                           const std::uint64_t* ket, const double value)
{
    const std::uint64_t* own = bra + sector * W;
    int flipped[4];
    dev::flip_orbitals<W>(own, ket + sector * W, flipped, 4);
    int high[2];
    int low[2];
    int num_high = 0;
    int num_low = 0;
#pragma unroll
    for(int k = 0; k < 4; k++)
    {
        if(dev::has_orbital<W>(own, flipped[k]))
        {
            high[num_high++ & 1] = flipped[k];
        }
        else
        {
            low[num_low++ & 1] = flipped[k];
        }
    }
    std::uint64_t words[W];
#pragma unroll
    for(int word = 0; word < W; word++)
    {
        words[word] = ket[sector * W + word];
    }
    double sign = 1.0;
    if(!apply_ladder<W>(words, low[0], false, sign) || !apply_ladder<W>(words, low[1], false, sign) ||
       !apply_ladder<W>(words, high[1], true, sign) || !apply_ladder<W>(words, high[0], true, sign))
    {
        return;
    }
    const int m = out.num_orbitals;
    atomicAdd((sector == 0 ? out.canonical_alpha : out.canonical_beta) + four_index(m, high[0], low[0], high[1], low[1]),
              sign * value);
}


template <int W>
__device__ __forceinline__ void add_mixed(const FsRdm& out, const std::uint64_t* bra, const std::uint64_t* ket,
                                          const double value)
{
    int alpha[2];
    int beta[2];
    dev::flip_orbitals<W>(bra, ket, alpha, 2);
    dev::flip_orbitals<W>(bra + W, ket + W, beta, 2);
    const bool alpha_low = dev::has_orbital<W>(bra, alpha[0]);
    const bool beta_low = dev::has_orbital<W>(bra + W, beta[0]);
    const bool odd = dev::between_parity<W>(bra, alpha[0], alpha[1]) != dev::between_parity<W>(bra + W, beta[0], beta[1]);
    const int m = out.num_orbitals;
    atomicAdd(out.mixed + four_index(m, alpha_low ? alpha[0] : alpha[1], alpha_low ? alpha[1] : alpha[0],
                                     beta_low ? beta[0] : beta[1], beta_low ? beta[1] : beta[0]),
              odd ? -value : value);
}


template <int W>
__device__ __forceinline__ bool key_before(const std::uint64_t* a, const std::uint64_t* b)
{
#pragma unroll
    for(int word = 2 * W - 1; word >= 0; word--)
    {
        if(a[word] != b[word])
        {
            return a[word] < b[word];
        }
    }
    return false;
}


template <int W, int CLS>
__device__ __forceinline__ void add_pair(const FsRdm& out, const std::uint64_t* bra, const std::uint64_t* ket,
                                         const double value)
{
    if constexpr(CLS == dev::kAlphaSingle || CLS == dev::kBetaSingle)
    {
        add_single<W>(out, CLS == dev::kAlphaSingle ? 0 : 1, bra, ket, value);
    }
    else if constexpr(CLS == dev::kAlphaDouble || CLS == dev::kBetaDouble)
    {
        add_double<W>(out, CLS == dev::kAlphaDouble ? 0 : 1, bra, ket, value);
    }
    else
    {
        add_mixed<W>(out, bra, ket, value);
    }
}


template <int W>
__device__ __forceinline__ bool counts_once(const FsRdm& out, const int k, const int j, const std::uint64_t* bra,
                                            const std::uint64_t* ket)
{
    if(out.half == 2)
    {
        return k < j;
    }
    return out.half == 0 || key_before<W>(bra, ket);
}


template <int W, int CLS>
__global__ void __launch_bounds__(kRdmThreads) rdm_group_kernel(const FsKeys keys, const FsGroups groups,
                                                                 const FsRdm out, const double* __restrict__ x)
{
    const int lane = threadIdx.x & 31;
    const std::int64_t warp = (std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const std::int64_t num_warps = (std::int64_t(gridDim.x) * blockDim.x) >> 5;
    for(std::int64_t g = groups.first_group + warp; g < groups.first_group + groups.num_groups; g += num_warps)
    {
        const std::int32_t begin = groups.offsets[g];
        const std::int32_t end = groups.offsets[g + 1];
        const std::int32_t first_output = begin + dev::members_below(groups.members + begin, end - begin, groups.output_begin);
        const std::int32_t last_output = begin + dev::members_below(groups.members + begin, end - begin, groups.output_end);
        const std::int32_t first_source = begin + dev::members_below(groups.members + begin, end - begin, groups.source_begin);
        const std::int32_t last_source = begin + dev::members_below(groups.members + begin, end - begin, groups.source_end);
        for(std::int32_t k = first_output; k < last_output; k++)
        {
            const std::uint32_t out_row = groups.members[k];
            const double c_out = x[out_row];
            if(c_out == 0.0)
            {
                continue;
            }
            const std::uint64_t* bra = dev::key_of<W>(keys, out_row);
            for(std::int32_t s = (out.half == 2 ? k + 1 : begin) + lane; s < end; s += 32)
            {
                if(s == k || s < first_source || s >= last_source)
                {
                    continue;
                }
                const std::uint32_t row = groups.members[s];
                const std::uint64_t* ket = dev::key_of<W>(keys, row);
                const double value = c_out * x[row];
                if(value == 0.0 || !counts_once<W>(out, k, s, bra, ket) || !dev::pair_counts<W, CLS>(bra, ket))
                {
                    continue;
                }
                add_pair<W, CLS>(out, bra, ket, value);
            }
        }
    }
}


template <int W, int CLS, int TILE, int MPL>
__global__ void __launch_bounds__(kRdmThreads) rdm_tile_kernel(const FsKeys keys, const FsGroups groups,
                                                                const FsRdm out, const double* __restrict__ x)
{
    constexpr int kTilesPerWarp = 32 / TILE;
    const int lane = threadIdx.x % TILE;
    const int tile_in_warp = (threadIdx.x & 31) / TILE;
    const std::int64_t warp = (std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const std::int64_t num_warps = (std::int64_t(gridDim.x) * blockDim.x) >> 5;
    const std::int64_t last_group = groups.first_group + groups.num_groups;
    const unsigned int mask = 0xffffffffu;
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
                const std::uint64_t* source = dev::key_of<W>(keys, row);
#pragma unroll
                for(int word = 0; word < 2 * W; word++)
                {
                    key[slot][word] = source[word];
                }
                value[slot] = x[row];
                bounds[0] += row < groups.output_begin ? 1 : 0;
                bounds[1] += row < groups.output_end ? 1 : 0;
                bounds[2] += row < groups.source_begin ? 1 : 0;
                bounds[3] += row < groups.source_end ? 1 : 0;
            }
        }
#pragma unroll
        for(int b = 0; b < 4; b++)
        {
#pragma unroll
            for(int offset = TILE / 2; offset > 0; offset >>= 1)
            {
                bounds[b] += __shfl_xor_sync(mask, bounds[b], offset, TILE);
            }
        }
        const int warp_first = static_cast<int>(__reduce_min_sync(mask, static_cast<unsigned int>(bounds[0])));
        const int warp_last = static_cast<int>(__reduce_max_sync(mask, static_cast<unsigned int>(bounds[1])));
        for(int k = warp_first; k < warp_last; k++)
        {
            std::uint64_t bra[2 * W];
            double c_out = 0.0;
#pragma unroll
            for(int word = 0; word < 2 * W; word++)
            {
                std::uint64_t mine = 0;
#pragma unroll
                for(int slot = 0; slot < MPL; slot++)
                {
                    mine = (k / TILE == slot) ? key[slot][word] : mine;
                }
                bra[word] = __shfl_sync(mask, mine, k % TILE, TILE);
            }
            {
                double mine = 0.0;
#pragma unroll
                for(int slot = 0; slot < MPL; slot++)
                {
                    mine = (k / TILE == slot) ? value[slot] : mine;
                }
                c_out = __shfl_sync(mask, mine, k % TILE, TILE);
            }
            if(k < bounds[0] || k >= bounds[1] || c_out == 0.0)
            {
                continue;
            }
#pragma unroll
            for(int slot = 0; slot < MPL; slot++)
            {
                const int j = lane + slot * TILE;
                const double v = c_out * value[slot];
                if(j < m && j >= bounds[2] && j < bounds[3] && j != k && v != 0.0 &&
                   counts_once<W>(out, k, j, bra, key[slot]) && dev::pair_counts<W, CLS>(bra, key[slot]))
                {
                    add_pair<W, CLS>(out, bra, key[slot], v);
                }
            }
        }
    }
}


template <int W, int CLS>
__global__ void __launch_bounds__(kRdmThreads) rdm_block_kernel(const FsKeys keys, const FsGroups groups,
                                                                 const FsRdm out, const double* __restrict__ x)
{
    __shared__ std::uint64_t shared_keys[kRdmSharedMembers * 2 * W];
    __shared__ double shared_values[kRdmSharedMembers];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int kWarps = kRdmThreads / 32;
    for(std::int64_t g = groups.first_group + blockIdx.x; g < groups.first_group + groups.num_groups; g += gridDim.x)
    {
        const std::int32_t begin = groups.offsets[g];
        const int m = groups.offsets[g + 1] - begin;
        const int output_first = dev::members_below(groups.members + begin, m, groups.output_begin);
        const int output_last = dev::members_below(groups.members + begin, m, groups.output_end);
        const int source_first = dev::members_below(groups.members + begin, m, groups.source_begin);
        const int source_last = dev::members_below(groups.members + begin, m, groups.source_end);
        const bool staged = m <= kRdmSharedMembers;
        __syncthreads();
        if(staged)
        {
            for(int j = threadIdx.x; j < m; j += kRdmThreads)
            {
                const std::uint32_t row = groups.members[begin + j];
                const std::uint64_t* source = dev::key_of<W>(keys, row);
#pragma unroll
                for(int word = 0; word < 2 * W; word++)
                {
                    shared_keys[j * 2 * W + word] = source[word];
                }
                shared_values[j] = x[row];
            }
        }
        __syncthreads();
        for(int k = output_first + warp; k < output_last; k += kWarps)
        {
            const std::uint64_t* bra = staged ? shared_keys + k * 2 * W : dev::key_of<W>(keys, groups.members[begin + k]);
            const double c_out = staged ? shared_values[k] : x[groups.members[begin + k]];
            if(c_out == 0.0)
            {
                continue;
            }
            for(int j = (out.half == 2 ? k + 1 : 0) + lane; j < m; j += 32)
            {
                if(j == k || j < source_first || j >= source_last)
                {
                    continue;
                }
                const std::uint32_t row = staged ? 0u : groups.members[begin + j];
                const std::uint64_t* ket = staged ? shared_keys + j * 2 * W : dev::key_of<W>(keys, row);
                const double v = c_out * (staged ? shared_values[j] : x[row]);
                if(v != 0.0 && counts_once<W>(out, k, j, bra, ket) && dev::pair_counts<W, CLS>(bra, ket))
                {
                    add_pair<W, CLS>(out, bra, ket, v);
                }
            }
        }
    }
}


template <int W, int A, int B>
__global__ void __launch_bounds__(kRdmThreads) pair_occupation_kernel(const FsKeys keys, const double* __restrict__ x,
                                                                       double* occupation, const int m)
{
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = entry / m;
    const int v = entry % m;
    const std::int64_t chunk = (keys.num_rows + gridDim.y - 1) / gridDim.y;
    const std::int64_t first = std::int64_t(blockIdx.y) * chunk;
    const std::int64_t last = std::min<std::int64_t>(keys.num_rows, first + chunk);
    double acc = 0.0;
    for(std::int64_t row = first; row < last; row++)
    {
        const std::uint64_t* key = keys.keys + row * 2 * W;
        const double c = x[row];
        if(entry < m * m && dev::has_orbital<W>(key + A * W, u) && dev::has_orbital<W>(key + B * W, v))
        {
            acc = fma(c, c, acc);
        }
    }
    if(entry < m * m && acc != 0.0)
    {
        atomicAdd(occupation + entry, acc);
    }
}


template <int CLS, int TILE, int MPL>
int launch_rdm_tile(const FsKeys& keys, const FsGroups& groups, const FsRdm& out, const double* x,
                    const cudaStream_t stream)
{
    const std::int64_t tiles_per_block = kRdmThreads / TILE;
    const std::int64_t blocks = std::min(kMaxRdmBlocks, (groups.num_groups + tiles_per_block - 1) / tiles_per_block);
    rdm_tile_kernel<kKeyWords, CLS, TILE, MPL><<<static_cast<unsigned int>(std::max<std::int64_t>(1, blocks)),
                                                kRdmThreads, 0, stream>>>(keys, groups, out, x);
    return int(cudaGetLastError());
}


template <int CLS>
int launch_rdm(const FsKeys& keys, const FsGroups& groups, const FsRdm& out, const double* x, const cudaStream_t stream)
{
    switch(groups.bucket)
    {
    case 0:
        return launch_rdm_tile<CLS, 4, 1>(keys, groups, out, x, stream);
    case 1:
        return launch_rdm_tile<CLS, 8, 1>(keys, groups, out, x, stream);
    case 2:
        return launch_rdm_tile<CLS, 16, 1>(keys, groups, out, x, stream);
    case 3:
        return launch_rdm_tile<CLS, 32, 1>(keys, groups, out, x, stream);
    case 4:
        return launch_rdm_tile<CLS, 32, 2>(keys, groups, out, x, stream);
    case 5:
    case 6:
    {
        const std::int64_t blocks = std::min(kMaxRdmBlocks, groups.num_groups);
        rdm_block_kernel<kKeyWords, CLS><<<static_cast<unsigned int>(std::max<std::int64_t>(1, blocks)), kRdmThreads, 0,
                                           stream>>>(keys, groups, out, x);
        return int(cudaGetLastError());
    }
    default:
        break;
    }
    const std::int64_t warps_per_block = kRdmThreads / 32;
    const std::int64_t blocks = std::min(kMaxRdmBlocks, (groups.num_groups + warps_per_block - 1) / warps_per_block);
    rdm_group_kernel<kKeyWords, CLS><<<static_cast<unsigned int>(std::max<std::int64_t>(1, blocks)), kRdmThreads, 0,
                                       stream>>>(keys, groups, out, x);
    return int(cudaGetLastError());
}


template <int A, int B>
int launch_occupation(const FsKeys& keys, const double* x, double* occupation, const int m, const cudaStream_t stream)
{
    const dim3 grid(static_cast<unsigned int>((m * m + kRdmThreads - 1) / kRdmThreads),
                    static_cast<unsigned int>(std::max<std::int64_t>(1, std::min<std::int64_t>(4096, keys.num_rows / 256))));
    pair_occupation_kernel<kKeyWords, A, B><<<grid, kRdmThreads, 0, stream>>>(keys, x, occupation, m);
    return int(cudaGetLastError());
}

}

extern "C"
{

    int fs_launch_rdm_groups(const FsKeys* keys, const FsGroups* groups, const FsRdm* out, const double* x,
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
            return launch_rdm<dev::kAlphaSingle>(*keys, *groups, *out, x, s);
        case dev::kAlphaDouble:
            return launch_rdm<dev::kAlphaDouble>(*keys, *groups, *out, x, s);
        case dev::kBetaSingle:
            return launch_rdm<dev::kBetaSingle>(*keys, *groups, *out, x, s);
        case dev::kBetaDouble:
            return launch_rdm<dev::kBetaDouble>(*keys, *groups, *out, x, s);
        case dev::kMixed:
            return launch_rdm<dev::kMixed>(*keys, *groups, *out, x, s);
        default:
            return int(cudaErrorInvalidValue);
        }
    }

    int fs_launch_pair_occupation(const FsKeys* keys, const double* x, double* occupation, const int sectors,
                                  const int num_orbitals, void* stream)
    {
        if(keys->num_rows <= 0)
        {
            return int(cudaSuccess);
        }
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        switch(sectors)
        {
        case 0:
            return launch_occupation<0, 0>(*keys, x, occupation, num_orbitals, s);
        case 1:
            return launch_occupation<1, 1>(*keys, x, occupation, num_orbitals, s);
        case 2:
            return launch_occupation<0, 1>(*keys, x, occupation, num_orbitals, s);
        default:
            return int(cudaErrorInvalidValue);
        }
    }
}
