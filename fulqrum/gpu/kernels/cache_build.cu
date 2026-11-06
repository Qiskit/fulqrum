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

#include "arms/term_walk.cuh"
#include "launch.hpp"
#include "launch_grid.hpp"

namespace
{


__device__ __forceinline__ long long first_row()
{
    return (long long)blockIdx.x * blockDim.x + threadIdx.x;
}


__device__ __forceinline__ long long row_stride()
{
    return (long long)gridDim.x * blockDim.x;
}


__global__ void __launch_bounds__(fq::launch::kArmThreads) alpha_doubles_build_kernel(const FqTileGeometry geometry,
                                                                                    const FqTermTables terms,
                                                                                    const FqDoublesBuildTables tables)
{
    const unsigned long long b_val = terms.beta_strings[geometry.bra_b0];
    const int jb_slot = tables.beta_slot[0];
    for(long long a = first_row(); a < geometry.n_a_bra; a += row_stride())
    {
        const unsigned long long a_val = terms.alpha_strings[geometry.bra_a0 + a];
        const std::int64_t base = tables.row_offsets[a >> 5];
        const int length = tables.row_lengths[a];
        for(int e = 0; e < length; ++e)
        {
            const std::int64_t slot = base + (std::int64_t(e) << 5) + (a & 31);
            const int ja = tables.columns[slot];
            const int group =
                fq::arms::quad_group(terms, tables.p0[slot], tables.p1[slot], tables.p2[slot], tables.p3[slot]);
            double value = 0.0;
            if(group >= 0 && tables.alpha_slot[ja] >= 0 && jb_slot >= 0)
            {
                const unsigned long long col[2] = {terms.alpha_strings[geometry.src_a0 + ja], b_val};
                value = fq::arms::quad_group_sum(terms, a_val, b_val, col, group, tables.p0[slot], tables.p1[slot],
                                                 tables.p2[slot], tables.p3[slot]);
            }
            tables.values[slot] = value;
        }
    }
}


__global__ void __launch_bounds__(fq::launch::kArmThreads) beta_doubles_build_kernel(const FqTileGeometry geometry,
                                                                                   const FqTermTables terms,
                                                                                   const FqDoublesBuildTables tables)
{
    const unsigned long long a_val = terms.alpha_strings[geometry.bra_a0];
    const int ja_slot = tables.alpha_slot[0];
    const unsigned int shift = (unsigned int)terms.num_alpha;
    for(long long b = first_row(); b < geometry.n_b_bra; b += row_stride())
    {
        const unsigned long long b_val = terms.beta_strings[geometry.bra_b0 + b];
        const std::int64_t first_edge = tables.row_offsets[b];
        const int row_edges = int(tables.row_offsets[b + 1] - first_edge);
        for(int k_edge = 0; k_edge < row_edges; ++k_edge)
        {
            const std::int64_t edge = first_edge + k_edge;
            const int jb = tables.columns[edge];
            const std::uint8_t q0 = (std::uint8_t)((unsigned int)tables.p0[edge] + shift);
            const std::uint8_t q1 = (std::uint8_t)((unsigned int)tables.p1[edge] + shift);
            const std::uint8_t q2 = (std::uint8_t)((unsigned int)tables.p2[edge] + shift);
            const std::uint8_t q3 = (std::uint8_t)((unsigned int)tables.p3[edge] + shift);
            const int group = fq::arms::quad_group(terms, q0, q1, q2, q3);
            double value = 0.0;
            if(group >= 0 && ja_slot >= 0 && tables.beta_slot[jb] >= 0)
            {
                const unsigned long long col[2] = {a_val, terms.beta_strings[geometry.src_b0 + jb]};
                value = fq::arms::quad_group_sum(terms, a_val, b_val, col, group, q0, q1, q2, q3);
            }
            tables.values[edge] = value;
        }
    }
}


template <bool ALPHA>
__device__ __forceinline__ void write_factor_cache(const FqTermTables& terms,
                                                   const FqSinglesBuildTables& tables,
                                                   const std::int64_t edge,
                                                   const unsigned long long a_val,
                                                   const unsigned long long b_val,
                                                   const unsigned long long* col,
                                                   const int group,
                                                   const unsigned int q0,
                                                   const unsigned int q1,
                                                   const int other)
{
    double c0;
    double cpb[2 * fq::arms::kFactorMaxOrbitals];
    if(fq::arms::factor_pair_group<ALPHA>(terms, a_val, b_val, col, group, q0, q1, &c0, cpb) < 0)
    {
        *tables.error = 1;
        return;
    }
    double c0_total = c0;
    for(int p = 0; p < other; ++p)
    {
        c0_total += cpb[p * 2 + 0];
    }
    const std::int64_t base = edge * other;
    for(int p = 0; p < other; ++p)
    {
        tables.delta[base + p] = cpb[p * 2 + 1] - cpb[p * 2 + 0];
    }
    tables.c0[edge] = c0_total;
}


__global__ void __launch_bounds__(fq::launch::kArmThreads) alpha_singles_build_kernel(const FqTileGeometry geometry,
                                                                                    const FqTermTables terms,
                                                                                    const FqSinglesBuildTables tables)
{
    const unsigned long long b_val = terms.beta_strings[geometry.bra_b0];
    const int jb_slot = tables.beta_slot[0];
    for(long long a = first_row(); a < geometry.n_a_bra; a += row_stride())
    {
        const unsigned long long a_val = terms.alpha_strings[geometry.bra_a0 + a];
        const std::int64_t first_edge = tables.row_offsets[a];
        const int row_edges = int(tables.row_offsets[a + 1] - first_edge);
        for(int k_edge = 0; k_edge < row_edges; ++k_edge)
        {
            const std::int64_t edge = first_edge + k_edge;
            const int ja = tables.columns[edge];
            const int group = fq::arms::pair_group(terms, tables.low[edge], tables.high[edge]);
            if(group >= 0 && tables.alpha_slot[ja] >= 0 && jb_slot >= 0)
            {
                const unsigned long long col[2] = {terms.alpha_strings[geometry.src_a0 + ja], b_val};
                write_factor_cache<true>(terms, tables, edge, a_val, b_val, col, group, tables.low[edge],
                                         tables.high[edge], terms.num_beta);
            }
        }
    }
}


__global__ void __launch_bounds__(fq::launch::kArmThreads) beta_singles_build_kernel(const FqTileGeometry geometry,
                                                                                   const FqTermTables terms,
                                                                                   const FqSinglesBuildTables tables)
{
    const unsigned long long a_val = terms.alpha_strings[geometry.bra_a0];
    const int ja_slot = tables.alpha_slot[0];
    const unsigned int shift = (unsigned int)terms.num_alpha;
    for(long long b = first_row(); b < geometry.n_b_bra; b += row_stride())
    {
        const unsigned long long b_val = terms.beta_strings[geometry.bra_b0 + b];
        const std::int64_t first_edge = tables.row_offsets[b];
        const int row_edges = int(tables.row_offsets[b + 1] - first_edge);
        for(int k_edge = 0; k_edge < row_edges; ++k_edge)
        {
            const std::int64_t edge = first_edge + k_edge;
            const int jb = tables.columns[edge];
            const std::uint8_t q0 = (std::uint8_t)((unsigned int)tables.low[edge] + shift);
            const std::uint8_t q1 = (std::uint8_t)((unsigned int)tables.high[edge] + shift);
            const int group = fq::arms::pair_group(terms, q0, q1);
            if(group >= 0 && ja_slot >= 0 && tables.beta_slot[jb] >= 0)
            {
                const unsigned long long col[2] = {a_val, terms.beta_strings[geometry.src_b0 + jb]};
                write_factor_cache<false>(terms, tables, edge, a_val, b_val, col, group, q0, q1, terms.num_alpha);
            }
        }
    }
}


unsigned int build_blocks(const long long rows)
{
    constexpr long long kMaxBuildBlocks = 65535;
    const long long blocks = (rows + fq::launch::kArmThreads - 1) / fq::launch::kArmThreads;
    return static_cast<unsigned int>(blocks < kMaxBuildBlocks ? blocks : kMaxBuildBlocks);
}

}

extern "C"
{

    int fq_launch_build_alpha_doubles(const FqTileGeometry* geometry,
                                      const FqTermTables* terms,
                                      const FqDoublesBuildTables* tables,
                                      void* stream)
    {
        if(geometry->n_a_bra <= 0)
        {
            return int(cudaSuccess);
        }
        alpha_doubles_build_kernel<<<build_blocks(geometry->n_a_bra), fq::launch::kArmThreads, 0,
                                     static_cast<cudaStream_t>(stream)>>>(*geometry, *terms, *tables);
        return int(cudaGetLastError());
    }

    int fq_launch_build_beta_doubles(const FqTileGeometry* geometry,
                                     const FqTermTables* terms,
                                     const FqDoublesBuildTables* tables,
                                     void* stream)
    {
        if(geometry->n_b_bra <= 0)
        {
            return int(cudaSuccess);
        }
        beta_doubles_build_kernel<<<build_blocks(geometry->n_b_bra), fq::launch::kArmThreads, 0,
                                    static_cast<cudaStream_t>(stream)>>>(*geometry, *terms, *tables);
        return int(cudaGetLastError());
    }

    int fq_launch_build_alpha_singles(const FqTileGeometry* geometry,
                                      const FqTermTables* terms,
                                      const FqSinglesBuildTables* tables,
                                      void* stream)
    {
        if(geometry->n_a_bra <= 0)
        {
            return int(cudaSuccess);
        }
        alpha_singles_build_kernel<<<build_blocks(geometry->n_a_bra), fq::launch::kArmThreads, 0,
                                     static_cast<cudaStream_t>(stream)>>>(*geometry, *terms, *tables);
        return int(cudaGetLastError());
    }

    int fq_launch_build_beta_singles(const FqTileGeometry* geometry,
                                     const FqTermTables* terms,
                                     const FqSinglesBuildTables* tables,
                                     void* stream)
    {
        if(geometry->n_b_bra <= 0)
        {
            return int(cudaSuccess);
        }
        beta_singles_build_kernel<<<build_blocks(geometry->n_b_bra), fq::launch::kArmThreads, 0,
                                    static_cast<cudaStream_t>(stream)>>>(*geometry, *terms, *tables);
        return int(cudaGetLastError());
    }
}
