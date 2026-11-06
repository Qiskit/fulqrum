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


__device__ __forceinline__ unsigned int get_bit(unsigned long long val, unsigned int pos)
{
    return (unsigned int)((val >> (unsigned long long)pos) & 1ULL);
}

__device__ __forceinline__ unsigned int get_bit_row(unsigned long long a_val, unsigned long long b_val,
                                                    unsigned int pos, unsigned int num_alpha)
{
    return (pos < num_alpha) ? get_bit(a_val, pos)
                             : get_bit(b_val, pos - num_alpha);
}

__device__ __forceinline__ unsigned int get_bit_col(const unsigned long long *col,
                                                    unsigned int pos, unsigned int num_alpha)
{
    return (pos < num_alpha) ? get_bit(col[0], pos)
                             : get_bit(col[1], pos - num_alpha);
}


__device__ __forceinline__ unsigned int compute_row_int(unsigned long long a_val, unsigned long long b_val,
                                                        const uint8_t *__restrict__ grp_offdiag_inds,
                                                        int g, int num_ladder,
                                                        unsigned int num_alpha, int offdiag_stride)
{
    unsigned int out = 0u;
    for (int k = 0; k < num_ladder; ++k)
    {
        unsigned int pos = (unsigned int)grp_offdiag_inds[g * offdiag_stride + k];
        unsigned int bit = get_bit_row(a_val, b_val, pos, num_alpha);
        out |= (bit << (unsigned int)k);
    }
    return out;
}


__device__ __forceinline__ unsigned int compute_row_int2(
    unsigned int _p, unsigned int _q, unsigned int _r, unsigned int _s,
    int num_ladder)
{
    unsigned int out = _p;
    if (num_ladder > 1) out |= (_q << 1u);
    if (num_ladder > 2) out |= (_r << 2u);
    if (num_ladder > 3) out |= (_s << 3u);
    return out;
}


__device__ __forceinline__ void make_col(unsigned long long a_val,
                                         unsigned long long b_val,
                                         unsigned long long *col,
                                         const uint8_t *__restrict__ grp_offdiag_inds,
                                         int g, int num_offdiag,
                                         unsigned int num_alpha, int offdiag_stride)
{
    col[0] = a_val;
    col[1] = b_val;
    for (int k = 0; k < num_offdiag; ++k)
    {
        unsigned int pos = (unsigned int)grp_offdiag_inds[g * offdiag_stride + k];
        if (pos < num_alpha)
            col[0] ^= 1ULL << (unsigned long long)pos;
        else
            col[1] ^= 1ULL << (unsigned long long)(pos - num_alpha);
    }
}


__device__ __forceinline__ int passes_proj(unsigned long long a_val,
                                           unsigned long long b_val,
                                           const uint8_t *__restrict__ term_proj_inds,
                                           const unsigned char *__restrict__ term_proj_bits,
                                           int t, int num_proj,
                                           unsigned int num_alpha, int proj_stride)
{
    for (int k = 0; k < num_proj; ++k)
    {
        unsigned int pos = (unsigned int)term_proj_inds[t * proj_stride + k];
        unsigned int bit = get_bit_row(a_val, b_val, pos, num_alpha);
        if (bit != (unsigned int)term_proj_bits[t * proj_stride + k])
            return 0;
    }
    return 1;
}


__device__ __forceinline__ uint32_t pack2_d(uint8_t p0, uint8_t p1)
{
    return (uint32_t)p0 | ((uint32_t)p1 << 16u);
}

__device__ __forceinline__ uint64_t pack4_d(uint8_t p0, uint8_t p1, uint8_t p2, uint8_t p3)
{
    return (uint64_t)p0 | ((uint64_t)p1 << 16u) |
           ((uint64_t)p2 << 32u) | ((uint64_t)p3 << 48u);
}
