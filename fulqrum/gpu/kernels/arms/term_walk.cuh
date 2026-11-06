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

#include "../device/gpu_helpers_common.cuh"
#include "../formats/params.hpp"

namespace fq::arms
{


inline constexpr int kFactorMaxOrbitals = 64;


__device__ __forceinline__ int pair_group(const FqTermTables& terms, const std::uint8_t p0, const std::uint8_t p1)
{
    const std::uint32_t key = pack2_d(p0, p1);
    std::uint32_t k = key;
    k = ((k >> 16u) ^ k) * 0x45D9F3Bu;
    k = ((k >> 16u) ^ k) * 0x45D9F3Bu;
    std::uint32_t slot = (k ^ (k >> 16u)) & terms.pair_mask;
    if((terms.pair_empty[slot >> 5u] >> (slot & 31u)) & 1u)
    {
        return -1;
    }
    for(;;)
    {
        const std::uint32_t stored = __ldg(terms.pair_keys + slot);
        if(stored == key)
        {
            return __ldg(terms.pair_values + slot);
        }
        if(stored == 0xFFFFFFFFu)
        {
            return -1;
        }
        slot = (slot + 1u) & terms.pair_mask;
    }
}


__device__ __forceinline__ int quad_group(const FqTermTables& terms,
                                          const std::uint8_t p0,
                                          const std::uint8_t p1,
                                          const std::uint8_t p2,
                                          const std::uint8_t p3)
{
    const unsigned long long key = pack4_d(p0, p1, p2, p3);
    unsigned long long k = key;
    k = (k ^ (k >> 30ULL)) * 0xBF58476D1CE4E5B9ULL;
    k = (k ^ (k >> 27ULL)) * 0x94D049BB133111EBULL;
    unsigned long long slot = (k ^ (k >> 31ULL)) & terms.quad_mask;
    if((terms.quad_empty[slot >> 6ULL] >> (slot & 63ULL)) & 1ULL)
    {
        return -1;
    }
    const auto* keys = reinterpret_cast<const unsigned long long*>(terms.quad_keys);
    for(;;)
    {
        const unsigned long long stored = __ldg(keys + slot);
        if(stored == key)
        {
            return __ldg(terms.quad_values + slot);
        }
        if(stored == 0xFFFFFFFFFFFFFFFFULL)
        {
            return -1;
        }
        slot = (slot + 1ULL) & terms.quad_mask;
    }
}


__device__ __forceinline__ double2 term_header(const FqTermTables& terms, const int t)
{
    return __ldg(reinterpret_cast<const double2*>(terms.term_header) + t);
}


__device__ __forceinline__ double term_element(const FqTermTables& terms,
                                               const unsigned long long a_val,
                                               const unsigned long long b_val,
                                               const unsigned long long* col,
                                               const int t,
                                               const int num_ops,
                                               const double coefficient)
{
    std::int8_t temp = 1;
    const int packed_stride = (terms.term_inds_stride + 3) >> 2;
    const auto* row = reinterpret_cast<const unsigned long long*>(terms.term_inds_vals) + (long long)t * packed_stride;
    unsigned long long packed = 0;
    for(int k = 0; k < num_ops; ++k)
    {
        if((k & 3) == 0)
        {
            packed = __ldg(row + (k >> 2));
        }
        const std::uint16_t iv = (std::uint16_t)(packed & 0xFFFFu);
        packed >>= 16;
        const unsigned int pos = (unsigned int)(iv & 0xFFu);
        const unsigned int val = (unsigned int)((iv >> 8) & 0xFFu);
        const unsigned int row_bit = get_bit_row(a_val, b_val, pos, (unsigned int)terms.num_alpha);
        const unsigned int col_bit = get_bit_col(col, pos, (unsigned int)terms.num_alpha);
        temp *= terms.oper_elems[4u * val + 2u * row_bit + col_bit];
    }
    return coefficient * (double)temp;
}


__device__ __forceinline__ double quad_group_sum(const FqTermTables& terms,
                                                 const unsigned long long a_val,
                                                 const unsigned long long b_val,
                                                 const unsigned long long* col,
                                                 const int group,
                                                 const std::uint8_t p0,
                                                 const std::uint8_t p1,
                                                 const std::uint8_t p2,
                                                 const std::uint8_t p3)
{
    const unsigned int num_alpha = (unsigned int)terms.num_alpha;
    const unsigned int pp = get_bit_row(a_val, b_val, (unsigned int)p0, num_alpha);
    const unsigned int qq = get_bit_row(a_val, b_val, (unsigned int)p1, num_alpha);
    const unsigned int rr = get_bit_row(a_val, b_val, (unsigned int)p2, num_alpha);
    const unsigned int ss = get_bit_row(a_val, b_val, (unsigned int)p3, num_alpha);
    const int g_real = (int)((unsigned int)group >> 8);
    const int n_ladder = (int)((unsigned int)group & 0xFFu);
    const unsigned int row_int = compute_row_int2(pp, qq, rr, ss, n_ladder);
    const std::uint32_t base = (std::uint32_t)g_real * terms.ladder_offset;
    const int t0 = (int)terms.ladder_offsets[base + row_int];
    const int t1 = (int)terms.ladder_offsets[base + row_int + 1u];
    double sum = 0.0;
    for(int t = t0; t < t1; ++t)
    {
        const double2 header = term_header(terms, t);
        const std::uint16_t meta = (std::uint16_t)(__double_as_longlong(header.y) & 0xFFFFull);
        const int num_ops = (int)(meta & 0xFFu);
        const int num_proj = (int)((meta >> 8) & 0xFFu);
        if(passes_proj(a_val, b_val, terms.term_proj_inds, terms.term_proj_bits, t, num_proj, num_alpha,
                       terms.term_proj_stride))
        {
            sum += term_element(terms, a_val, b_val, col, t, num_ops, header.x);
        }
    }
    return sum;
}


template <bool ALPHA>
__device__ __forceinline__ int factor_pair_group(const FqTermTables& terms,
                                                 const unsigned long long a_val,
                                                 const unsigned long long b_val,
                                                 const unsigned long long* col,
                                                 const int group,
                                                 const unsigned int q0,
                                                 const unsigned int q1,
                                                 double* c0_out,
                                                 double* cpb_out)
{
    const unsigned int num_alpha = (unsigned int)terms.num_alpha;
    const unsigned int other = ALPHA ? (unsigned int)terms.num_beta : num_alpha;
    const unsigned int pp = get_bit_row(a_val, b_val, q0, num_alpha);
    const unsigned int qq = get_bit_row(a_val, b_val, q1, num_alpha);
    const unsigned int row_int = compute_row_int2(pp, qq, 0u, 0u, (int)((unsigned int)group & 0xFFu));
    const std::uint32_t base = (std::uint32_t)((int)((unsigned int)group >> 8)) * terms.ladder_offset;
    const int t0 = (int)terms.ladder_offsets[base + row_int];
    const int t1 = (int)terms.ladder_offsets[base + row_int + 1u];
    double c0 = 0.0;
    for(unsigned int p = 0; p < other * 2u; ++p)
    {
        cpb_out[p] = 0.0;
    }
    const int packed_stride = (terms.term_inds_stride + 3) >> 2;
    for(int t = t0; t < t1; ++t)
    {
        const double2 header = term_header(terms, t);
        const std::uint16_t meta = (std::uint16_t)(__double_as_longlong(header.y) & 0xFFFFull);
        const int num_ops = (int)(meta & 0xFFu);
        const int num_proj = (int)((meta >> 8) & 0xFFu);
        int p_seen = -1;
        double f0 = 1.0;
        double f1 = 1.0;
        bool passes = true;
        for(int k = 0; k < num_proj; ++k)
        {
            const unsigned int pos = (unsigned int)terms.term_proj_inds[t * terms.term_proj_stride + k];
            const unsigned int expected = (unsigned int)terms.term_proj_bits[t * terms.term_proj_stride + k];
            if((pos < num_alpha) == ALPHA)
            {
                if(get_bit_row(a_val, b_val, pos, num_alpha) != expected)
                {
                    passes = false;
                    break;
                }
            }
            else
            {
                const int p = ALPHA ? (int)(pos - num_alpha) : (int)pos;
                if(p_seen >= 0 && p_seen != p)
                {
                    return -1;
                }
                p_seen = p;
                if(expected == 0u)
                {
                    f1 = 0.0;
                }
                else
                {
                    f0 = 0.0;
                }
            }
        }
        if(!passes)
        {
            continue;
        }
        std::int8_t temp = 1;
        const auto* row =
            reinterpret_cast<const unsigned long long*>(terms.term_inds_vals) + (long long)t * packed_stride;
        unsigned long long packed = 0;
        for(int k = 0; k < num_ops; ++k)
        {
            if((k & 3) == 0)
            {
                packed = __ldg(row + (k >> 2));
            }
            const std::uint16_t iv = (std::uint16_t)(packed & 0xFFFFu);
            packed >>= 16;
            const unsigned int pos = (unsigned int)(iv & 0xFFu);
            const unsigned int val = (unsigned int)((iv >> 8) & 0xFFu);
            if((pos < num_alpha) == ALPHA)
            {
                const unsigned int bit = ALPHA ? pos : pos - num_alpha;
                const unsigned int row_bit = get_bit(ALPHA ? a_val : b_val, bit);
                const unsigned int col_bit = get_bit(ALPHA ? col[0] : col[1], bit);
                temp *= terms.oper_elems[4u * val + 2u * row_bit + col_bit];
            }
            else
            {
                const int p = ALPHA ? (int)(pos - num_alpha) : (int)pos;
                if(p_seen >= 0 && p_seen != p)
                {
                    return -1;
                }
                p_seen = p;
                f0 *= (double)terms.oper_elems[4u * val + 0u];
                f1 *= (double)terms.oper_elems[4u * val + 3u];
            }
        }
        const double own = header.x * (double)temp;
        if(p_seen < 0)
        {
            c0 += own;
        }
        else
        {
            cpb_out[p_seen * 2 + 0] += own * f0;
            cpb_out[p_seen * 2 + 1] += own * f1;
        }
    }
    *c0_out = c0;
    return 0;
}

}
