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
#pragma once

#include <cstdint>

#include "params.hpp"

namespace fq::full_strs::device
{


enum MoveClass : int
{
    kAlphaSingle = 0,
    kAlphaDouble = 1,
    kBetaSingle = 2,
    kBetaDouble = 3,
    kMixed = 4
};


__device__ __constant__ const signed char kDoubleSlot[16] = {-1, -1, -1, 0, -1, 1, 2, -1, -1, 2, 1, -1, 0, -1, -1, -1};


__device__ __constant__ const signed char kMixedSlot[16] = {-1, -1, -1, -1, -1, 0, 1, -1, -1, 1, 0, -1, -1, -1, -1, -1};


template <int W>
__device__ __forceinline__ bool has_orbital(const std::uint64_t* words, const int orbital)
{
    return ((words[orbital >> 6] >> (orbital & 63)) & 1ull) != 0;
}


__device__ __forceinline__ std::int64_t pair_rank(const int p, const int q)
{
    return std::int64_t(q) * (q - 1) / 2 + p;
}


__device__ __forceinline__ std::int64_t quad_rank(const int* pos)
{
    const std::int64_t z = pos[3];
    const std::int64_t y = pos[2];
    const std::int64_t x = pos[1];
    return z * (z - 1) * (z - 2) * (z - 3) / 24 + y * (y - 1) * (y - 2) / 6 + x * (x - 1) / 2 + pos[0];
}


template <int W>
__device__ __forceinline__ bool between_parity(const std::uint64_t* words, const int lo, const int hi)
{
    unsigned int count = 0;
#pragma unroll
    for(int word = 0; word < W; word++)
    {
        const int first = word * 64;
        const int begin = lo + 1 > first ? lo + 1 : first;
        const int end = hi < first + 64 ? hi : first + 64;
        if(begin < end)
        {
            const int width = end - begin;
            const std::uint64_t mask = (width == 64 ? ~0ull : ((1ull << width) - 1ull)) << (begin - first);
            count += static_cast<unsigned int>(__popcll(words[word] & mask));
        }
    }
    return (count & 1u) != 0;
}


template <int W>
__device__ __forceinline__ int flip_orbitals(const std::uint64_t* a, const std::uint64_t* b, int* out, const int max_count)
{
    int count = 0;
#pragma unroll
    for(int word = 0; word < W; word++)
    {
        std::uint64_t bits = a[word] ^ b[word];
        while(bits != 0)
        {
            if(count < max_count)
            {
                out[count] = word * 64 + __ffsll(static_cast<long long>(bits)) - 1;
            }
            count++;
            bits &= bits - 1;
        }
    }
    return count;
}


template <int W>
__device__ __forceinline__ int flip_count(const std::uint64_t* a, const std::uint64_t* b)
{
    int count = 0;
#pragma unroll
    for(int word = 0; word < W; word++)
    {
        count += __popcll(a[word] ^ b[word]);
    }
    return count;
}


template <int W, int CLS>
__device__ __forceinline__ bool pair_counts(const std::uint64_t* out, const std::uint64_t* source)
{
    if constexpr(CLS == kAlphaSingle)
    {
        return flip_count<W>(out, source) != 0;
    }
    else if constexpr(CLS == kAlphaDouble)
    {
        return flip_count<W>(out, source) == 4;
    }
    else if constexpr(CLS == kBetaSingle)
    {
        return flip_count<W>(out + W, source + W) != 0;
    }
    else if constexpr(CLS == kBetaDouble)
    {
        return flip_count<W>(out + W, source + W) == 4;
    }
    else
    {
        return flip_count<W>(out, source) != 0 && flip_count<W>(out + W, source + W) != 0;
    }
}


template <int W>
__device__ __forceinline__ double single_element(const FsTables& tables, const std::uint64_t* out, const int sector,
                                                 const int p, const int q)
{
    const int n = tables.num_orbitals;
    const std::uint64_t* own = out + sector * W;
    const std::uint64_t* other = out + (1 - sector) * W;
    const int dir = has_orbital<W>(own, p) ? 0 : 1;
    const std::int64_t index = (sector * tables.num_pairs + pair_rank(p, q)) * 2 + dir;
    const double* delta = tables.single_delta + index * 2 * n;
    const double* own_delta = delta + sector * n;
    const double* other_delta = delta + (1 - sector) * n;
    double other_sum = 0.0;
#pragma unroll
    for(int word = 0; word < W; word++)
    {
        std::uint64_t bits = other[word];
        while(bits != 0)
        {
            other_sum += other_delta[word * 64 + __ffsll(static_cast<long long>(bits)) - 1];
            bits &= bits - 1;
        }
    }
    double own_sum = tables.single_reference[index];
    const std::uint64_t* reference = tables.reference + sector * W;
#pragma unroll
    for(int word = 0; word < W; word++)
    {
        std::uint64_t bits = own[word] ^ reference[word];
        while(bits != 0)
        {
            const int r = word * 64 + __ffsll(static_cast<long long>(bits)) - 1;
            bits &= bits - 1;
            if(r != p && r != q)
            {
                own_sum += has_orbital<W>(own, r) ? own_delta[r] : -own_delta[r];
            }
        }
    }
    const double value = tables.single_value[index] + other_sum + own_sum;
    return between_parity<W>(own, p, q) ? -value : value;
}


template <int W>
__device__ __forceinline__ double double_element(const FsTables& tables, const std::uint64_t* out, const int sector,
                                                 const int* pos)
{
    const std::uint64_t* own = out + sector * W;
    const int pattern = int(has_orbital<W>(own, pos[0])) | int(has_orbital<W>(own, pos[1])) << 1 |
                        int(has_orbital<W>(own, pos[2])) << 2 | int(has_orbital<W>(own, pos[3])) << 3;
    const int slot = kDoubleSlot[pattern];
    if(slot < 0)
    {
        return 0.0;
    }
    const double value = tables.double_value[(sector * tables.num_quads + quad_rank(pos)) * 3 + slot];
    const bool odd = between_parity<W>(own, pos[0], pos[1]) != between_parity<W>(own, pos[2], pos[3]);
    return odd ? -value : value;
}


template <int W>
__device__ __forceinline__ double mixed_element(const FsTables& tables, const std::uint64_t* out, const int* alpha,
                                                const int* beta)
{
    const int pattern = int(has_orbital<W>(out, alpha[0])) | int(has_orbital<W>(out, alpha[1])) << 1 |
                        int(has_orbital<W>(out + W, beta[0])) << 2 | int(has_orbital<W>(out + W, beta[1])) << 3;
    const int slot = kMixedSlot[pattern];
    if(slot < 0)
    {
        return 0.0;
    }
    const std::int64_t index = pair_rank(alpha[0], alpha[1]) * tables.num_pairs + pair_rank(beta[0], beta[1]);
    const double value = tables.mixed_value[index * 2 + slot];
    const bool odd = between_parity<W>(out, alpha[0], alpha[1]) != between_parity<W>(out + W, beta[0], beta[1]);
    return odd ? -value : value;
}


template <int W, int CLS>
__device__ __forceinline__ double element(const FsTables& tables, const std::uint64_t* out, const std::uint64_t* source)
{
    int alpha[4];
    int beta[4];
    if constexpr(CLS == kAlphaSingle)
    {
        flip_orbitals<W>(out, source, alpha, 2);
        return single_element<W>(tables, out, 0, alpha[0], alpha[1]);
    }
    else if constexpr(CLS == kBetaSingle)
    {
        flip_orbitals<W>(out + W, source + W, beta, 2);
        return single_element<W>(tables, out, 1, beta[0], beta[1]);
    }
    else if constexpr(CLS == kAlphaDouble)
    {
        flip_orbitals<W>(out, source, alpha, 4);
        return double_element<W>(tables, out, 0, alpha);
    }
    else if constexpr(CLS == kBetaDouble)
    {
        flip_orbitals<W>(out + W, source + W, beta, 4);
        return double_element<W>(tables, out, 1, beta);
    }
    else
    {
        flip_orbitals<W>(out, source, alpha, 2);
        flip_orbitals<W>(out + W, source + W, beta, 2);
        return mixed_element<W>(tables, out, alpha, beta);
    }
}

}
