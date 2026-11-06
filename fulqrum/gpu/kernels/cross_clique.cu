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
 * Modified: descriptive GPU flag and clique names.
 */
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

#include "device/warp.cuh"
#include "formats/clique_word.hpp"
#include "launch.hpp"
#include "launch_grid.hpp"

namespace
{

namespace clique = fq::clique_word;


struct CliqueLane
{
    unsigned long long member;
    int alpha;
    int orbital;
    int orbital_offset;
    bool active;
};


__device__ __forceinline__ CliqueLane lane_member(const FqCrossCliqueTables& tables, const int parent, const int lane)
{
    const int first = __ldg(tables.parent_offsets + parent);
    const int count = __ldg(tables.parent_offsets + parent + 1) - first;
    CliqueLane me;
    me.active = lane < count;
    me.member = me.active ? __ldg(reinterpret_cast<const unsigned long long*>(tables.members) + first + lane) : 0ull;
    me.alpha = clique::member_alpha(me.member);
    me.orbital = clique::member_orbital(me.member);
    me.orbital_offset = me.orbital * tables.orbitals;
    return me;
}


template <int EB>
struct CliqueBatch
{
    double value[EB];
    int row[EB];
    bool not_finite;
};


template <int EB, bool TRANSPOSED>
__device__ __forceinline__ void gather_batch(CliqueBatch<EB>& batch,
                                             const unsigned long long* words,
                                             const int first,
                                             const int last,
                                             const double* source,
                                             const int stride,
                                             const CliqueLane& me)
{
    batch.not_finite = false;
#pragma unroll
    for(int e = 0; e < EB; ++e)
    {
        const bool exists = first + e < last;
        const unsigned long long word = exists ? __ldg(words + first + e) : 0ull;
        batch.row[e] = exists ? clique::tile_word_row(word) : -1;
        const long long column = clique::tile_word_column(word);
        const long long index = TRANSPOSED ? (long long)me.alpha * stride + column : column * stride + me.alpha;
        const double x = me.active && exists ? __ldg(source + index) : 0.0;
        batch.value[e] =
            __longlong_as_double(__double_as_longlong(x) ^ (long long)clique::gather_sign_bit(me.member, word));
        batch.not_finite |= !isfinite(batch.value[e]);
    }
}


template <int NMAX>
__device__ __forceinline__ void load_coefficients(double (&c)[NMAX], const double* block, const int orbital_offset)
{
#pragma unroll
    for(int l = 0; l < NMAX; ++l)
    {
        c[l] = __ldg(block + fq::device::warp_broadcast(orbital_offset, l));
    }
}


template <int NMAX, int EB>
__device__ __forceinline__ void multiply_batch(double (&acc)[EB],
                                               const double (&c)[NMAX],
                                               const double* values,
                                               const bool not_finite)
{
#pragma unroll
    for(int e = 0; e < EB; ++e)
    {
        acc[e] = 0.0;
    }
    if(!not_finite)
    {
#pragma unroll
        for(int l = 0; l < NMAX; ++l)
        {
#pragma unroll
            for(int h = 0; h < EB / 2; ++h)
            {
                const double2 x = reinterpret_cast<const double2*>(values + l * EB)[h];
                acc[2 * h] = fma(c[l], x.x, acc[2 * h]);
                acc[2 * h + 1] = fma(c[l], x.y, acc[2 * h + 1]);
            }
        }
        return;
    }
#pragma unroll 1
    for(int l = 0; l < NMAX; ++l)
    {
        if(c[l] == 0.0)
        {
            continue;
        }
#pragma unroll
        for(int h = 0; h < EB / 2; ++h)
        {
            const double2 x = reinterpret_cast<const double2*>(values + l * EB)[h];
            acc[2 * h] = fma(c[l], x.x, acc[2 * h]);
            acc[2 * h + 1] = fma(c[l], x.y, acc[2 * h + 1]);
        }
    }
}


template <int NMAX, bool TRANSPOSED>
__device__ __forceinline__ void flush_tile(const double* row_tile,
                                           const int tile,
                                           const int tile_rows,
                                           const int n_rows,
                                           const int stride,
                                           double* out,
                                           const CliqueLane& me,
                                           const int lane)
{
    const long long sign = (long long)clique::output_sign_bit(me.member);
    const int row0 = tile * tile_rows;
    for(int r = 0; r < tile_rows && row0 + r < n_rows; ++r)
    {
        const double y = row_tile[r * NMAX + lane];
        if(y != 0.0)
        {
            const long long index =
                TRANSPOSED ? (long long)me.alpha * stride + (row0 + r) : (long long)(row0 + r) * stride + me.alpha;
            atomicAdd(out + index, __longlong_as_double(__double_as_longlong(y) ^ sign));
        }
    }
}


__device__ __forceinline__ void clique_job(const FqCrossCliqueTables& tables, const long long job, int& tile, int& parent)
{
    if(tables.parent_group >= 1)
    {
        const long long per_group = (long long)tables.parent_group * tables.num_tiles;
        const long long first = (job / per_group) * tables.parent_group;
        const long long in_group = job - (job / per_group) * per_group;
        const long long width = tables.num_parents - first < tables.parent_group ? tables.num_parents - first
                                                                                 : tables.parent_group;
        tile = int(in_group / width);
        parent = int(first + (in_group - (long long)tile * width));
    }
    else
    {
        tile = int(job / tables.num_parents);
        parent = int(job % tables.num_parents);
    }
}


template <int NMAX, int EB, bool TRANSPOSED>
__global__ void __launch_bounds__(fq::launch::kCliqueThreads) cross_clique_kernel(const __grid_constant__ FqTileGeometry geometry,
                                                                                const __grid_constant__ FqVectors vectors,
                                                                                const __grid_constant__ FqCrossCliqueTables tables)
{
    static_assert(EB % 2 == 0 && NMAX >= 2 && NMAX <= clique::kMaxMembers, "unsupported clique kernel instance");
    extern __shared__ double shared[];
    const int warp = int(threadIdx.x / 32);
    const int lane = int(threadIdx.x % 32);
    const long long job = (long long)blockIdx.x * clique::kWarpsPerBlock + warp;
    if(job >= (long long)tables.num_parents * tables.num_tiles)
    {
        return;
    }
    int tile, parent;
    clique_job(tables, job, tile, parent);
    double* row_tile = shared + (long long)warp * (tables.tile_rows * NMAX + NMAX * EB);
    double* published = row_tile + tables.tile_rows * NMAX;
    const CliqueLane me = lane_member(tables, __ldg(tables.parent_ids + parent), lane);
    for(int i = lane; i < tables.tile_rows * NMAX; i += 32)
    {
        row_tile[i] = 0.0;
    }
    const auto* words = reinterpret_cast<const unsigned long long*>(tables.edge_words);
    const int stride = geometry.n_alpha_src_compact;
    const int first_group = __ldg(tables.tile_group_offsets + tile);
    const int last_group = __ldg(tables.tile_group_offsets + tile + 1);
    CliqueBatch<EB> next;
    if(first_group < last_group)
    {
        gather_batch<EB, TRANSPOSED>(next, words, __ldg(tables.group_offsets + first_group),
                                     __ldg(tables.group_offsets + first_group + 1), vectors.src_in, stride, me);
    }
    for(int group = first_group; group < last_group; ++group)
    {
        double c[NMAX];
        load_coefficients<NMAX>(c,
                                tables.coefficients +
                                    (long long)__ldg(tables.group_pairs + group) * tables.orbitals * tables.orbitals +
                                    me.orbital,
                                me.orbital_offset);
        const int group_end = __ldg(tables.group_offsets + group + 1);
        const int next_end = group + 1 < last_group ? __ldg(tables.group_offsets + group + 2) : 0;
        for(int e0 = __ldg(tables.group_offsets + group); e0 < group_end; e0 += EB)
        {
            int rows[EB];
            if(lane < NMAX)
            {
#pragma unroll
                for(int e = 0; e < EB; ++e)
                {
                    published[lane * EB + e] = next.value[e];
                }
            }
#pragma unroll
            for(int e = 0; e < EB; ++e)
            {
                rows[e] = next.row[e];
            }
            const bool not_finite = fq::device::warp_any(next.not_finite);
            __syncwarp();
            if(e0 + EB < group_end)
            {
                gather_batch<EB, TRANSPOSED>(next, words, e0 + EB, group_end, vectors.src_in, stride, me);
            }
            else if(group + 1 < last_group)
            {
                gather_batch<EB, TRANSPOSED>(next, words, group_end, next_end, vectors.src_in, stride, me);
            }
            double acc[EB];
            multiply_batch<NMAX, EB>(acc, c, published, not_finite);
            __syncwarp();
            if(lane < NMAX)
            {
#pragma unroll
                for(int e = 0; e < EB; ++e)
                {
                    if(rows[e] >= 0)
                    {
                        row_tile[rows[e] * NMAX + lane] += acc[e];
                    }
                }
            }
        }
    }
    __syncwarp();
    if(me.active)
    {
        flush_tile<NMAX, TRANSPOSED>(row_tile, tile, tables.tile_rows, TRANSPOSED ? geometry.n_a_bra : geometry.n_b_bra,
                                     geometry.n_a_bra, vectors.out_local, me, lane);
    }
}


template <int NMAX, int MT, bool DISTINCT_ROWS>
__device__ __forceinline__ void add_mma_chunk(double* row_tile, const double (&y)[MT][2], const int g, const int t,
                                              const int row0, const int row1)
{
#pragma unroll
    for(int q = 0; q < (DISTINCT_ROWS ? 1 : 4); ++q)
    {
        if(DISTINCT_ROWS || t == q)
        {
#pragma unroll
            for(int i = 0; i < MT; ++i)
            {
                const int m = 8 * i + g;
                if(m < NMAX)
                {
                    if(row0 >= 0)
                    {
                        row_tile[row0 * NMAX + m] += y[i][0];
                    }
                    if(row1 >= 0)
                    {
                        row_tile[row1 * NMAX + m] += y[i][1];
                    }
                }
            }
        }
        __syncwarp();
    }
}


template <int NMAX, int EB, bool TRANSPOSED, bool DISTINCT_ROWS>
__global__ void __launch_bounds__(fq::launch::kCliqueThreads) cross_clique_mma_kernel(const __grid_constant__ FqTileGeometry geometry,
                                                                                    const __grid_constant__ FqVectors vectors,
                                                                                    const __grid_constant__ FqCrossCliqueTables tables)
{
    static_assert(NMAX % 4 == 0 && NMAX >= 4 && NMAX <= clique::kMaxMembers, "unsupported clique kernel instance");
    constexpr int kMTiles = (NMAX + 7) / 8;
    constexpr int kKSteps = NMAX / 4;
    extern __shared__ double shared[];
    const int warp = int(threadIdx.x / 32);
    const int lane = int(threadIdx.x % 32);
    const int g = lane >> 2;
    const int t = lane & 3;
    const long long job = (long long)blockIdx.x * clique::kWarpsPerBlock + warp;
    if(job >= (long long)tables.num_parents * tables.num_tiles)
    {
        return;
    }
    if(tables.panel != nullptr && __ldg(tables.nonfinite) == 0)
    {

        return;
    }
    int tile, parent;
    clique_job(tables, job, tile, parent);
    double* row_tile = shared + (long long)warp * (tables.tile_rows * NMAX + NMAX * EB);
    const int parent_id = __ldg(tables.parent_ids + parent);
    const int first = __ldg(tables.parent_offsets + parent_id);
    const int count = __ldg(tables.parent_offsets + parent_id + 1) - first;
    const auto* members = reinterpret_cast<const unsigned long long*>(tables.members);

    unsigned long long k_member[kKSteps];
    int k_alpha[kKSteps];
    int k_offset[kKSteps];
#pragma unroll
    for(int s = 0; s < kKSteps; ++s)
    {
        const int k = 4 * s + t;
        k_member[s] = k < count ? __ldg(members + first + k) : 0ull;
        k_alpha[s] = clique::member_alpha(k_member[s]);
        k_offset[s] = k < count ? clique::member_orbital(k_member[s]) * tables.orbitals : -1;
    }
    int m_orbital[kMTiles];
#pragma unroll
    for(int i = 0; i < kMTiles; ++i)
    {
        const int m = 8 * i + g;
        m_orbital[i] = m < count ? clique::member_orbital(__ldg(members + first + m)) : -1;
    }
    for(int i = lane; i < tables.tile_rows * NMAX; i += 32)
    {
        row_tile[i] = 0.0;
    }
    __syncwarp();
    const auto* words = reinterpret_cast<const unsigned long long*>(tables.edge_words);
    const int stride = geometry.n_alpha_src_compact;
    const int first_group = __ldg(tables.tile_group_offsets + tile);
    const int last_group = __ldg(tables.tile_group_offsets + tile + 1);
    for(int group = first_group; group < last_group; ++group)
    {
        const double* block =
            tables.coefficients + (long long)__ldg(tables.group_pairs + group) * tables.orbitals * tables.orbitals;
        double a[kMTiles][kKSteps];
#pragma unroll
        for(int i = 0; i < kMTiles; ++i)
        {
#pragma unroll
            for(int s = 0; s < kKSteps; ++s)
            {
                a[i][s] = (m_orbital[i] >= 0 && k_offset[s] >= 0) ? __ldg(block + k_offset[s] + m_orbital[i]) : 0.0;
            }
        }
        const int group_end = __ldg(tables.group_offsets + group + 1);
        for(int e0 = __ldg(tables.group_offsets + group); e0 < group_end; e0 += 8)
        {
            const bool exists = e0 + g < group_end;
            const unsigned long long word = exists ? __ldg(words + e0 + g) : 0ull;
            const long long column = clique::tile_word_column(word);
            double b[kKSteps];
            bool not_finite = false;
#pragma unroll
            for(int s = 0; s < kKSteps; ++s)
            {
                const long long index = TRANSPOSED ? (long long)k_alpha[s] * stride + column : column * stride + k_alpha[s];
                const double x = exists && k_offset[s] >= 0 ? __ldg(vectors.src_in + index) : 0.0;
                b[s] = __longlong_as_double(__double_as_longlong(x) ^ (long long)clique::gather_sign_bit(k_member[s], word));
                not_finite |= !isfinite(b[s]);
            }
            const int e_d0 = e0 + 2 * t;
            const int row_d0 = e_d0 < group_end ? clique::tile_word_row(__ldg(words + e_d0)) : -1;
            const int row_d1 = e_d0 + 1 < group_end ? clique::tile_word_row(__ldg(words + e_d0 + 1)) : -1;
            double y[kMTiles][2];
            if(!fq::device::warp_any(not_finite))
            {
#pragma unroll
                for(int i = 0; i < kMTiles; ++i)
                {
                    y[i][0] = 0.0;
                    y[i][1] = 0.0;
#pragma unroll
                    for(int s = 0; s < kKSteps; ++s)
                    {
                        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                     : "+d"(y[i][0]), "+d"(y[i][1])
                                     : "d"(a[i][s]), "d"(b[s]));
                    }
                }
            }
            else
            {


#pragma unroll
                for(int i = 0; i < kMTiles; ++i)
                {
#pragma unroll
                    for(int half = 0; half < 2; ++half)
                    {
                        const int e = 2 * t + half;
                        double sum = 0.0;
#pragma unroll
                        for(int s = 0; s < kKSteps; ++s)
                        {
#pragma unroll
                            for(int q = 0; q < 4; ++q)
                            {
                                const double x = __shfl_sync(0xffffffffu, b[s], 4 * e + q);
                                const double c = __shfl_sync(0xffffffffu, a[i][s], 4 * g + q);
                                if(c != 0.0)
                                {
                                    sum = fma(c, x, sum);
                                }
                            }
                        }
                        y[i][half] = sum;
                    }
                }
            }
            add_mma_chunk<NMAX, kMTiles, DISTINCT_ROWS>(row_tile, y, g, t, row_d0, row_d1);
        }
    }
    __syncwarp();
    const CliqueLane me = lane_member(tables, parent_id, lane);
    if(me.active)
    {
        flush_tile<NMAX, TRANSPOSED>(row_tile, tile, tables.tile_rows, TRANSPOSED ? geometry.n_a_bra : geometry.n_b_bra,
                                     geometry.n_a_bra, vectors.out_local, me, lane);
    }
}


template <int NMAX>
struct CliquePanel
{
    static constexpr int kMTiles = (NMAX + 7) / 8;
    static constexpr int kKSteps = NMAX / 4;
    static constexpr int kKSP = (kKSteps + 1) & ~1;
    static constexpr int kRow = 4 * kKSP;
};


template <int NMAX, bool TRANSPOSED>
__global__ void __launch_bounds__(256) cross_clique_pack_kernel(const __grid_constant__ FqTileGeometry geometry,
                                                               const double* __restrict__ source,
                                                               const __grid_constant__ FqCrossCliqueTables tables)
{
    using P = CliquePanel<NMAX>;
    __shared__ int s_alpha[P::kRow];
    __shared__ unsigned long long s_sign[P::kRow];
    __shared__ double s_tile[32][P::kRow + 1];
    const auto* members = reinterpret_cast<const unsigned long long*>(tables.members);
    const int stride = geometry.n_alpha_src_compact;
    const int n_cols = tables.panel_columns;
    const int col0 = int(blockIdx.x) * 32;
    const int ncol = min(32, n_cols - col0);
    bool not_finite = false;
    for(int parent = int(blockIdx.y); parent < tables.num_parents; parent += int(gridDim.y))
    {
        const int parent_id = __ldg(tables.parent_ids + parent);
        const int first = __ldg(tables.parent_offsets + parent_id);
        const int count = __ldg(tables.parent_offsets + parent_id + 1) - first;
        __syncthreads();
        for(int q = int(threadIdx.x); q < P::kRow; q += int(blockDim.x))
        {
            const int t = (q % 8) / 2;
            const int s = 2 * (q / 8) + q % 2;
            const int k = 4 * s + t;
            const bool ok = s < P::kKSteps && k < count;
            const unsigned long long member = ok ? __ldg(members + first + k) : 0ull;
            s_alpha[q] = ok ? clique::member_alpha(member) : -1;
            s_sign[q] = member & clique::kSignBit;
        }
        __syncthreads();
        for(int i = int(threadIdx.x); i < 32 * P::kRow; i += int(blockDim.x))
        {
            const int c = TRANSPOSED ? i % 32 : i / P::kRow;
            const int q = TRANSPOSED ? i / 32 : i % P::kRow;
            const int column = col0 + c;
            const int alpha = s_alpha[q];
            double v = 0.0;
            if(alpha >= 0 && c < ncol)
            {
                const long long index = TRANSPOSED ? (long long)alpha * stride + column : (long long)column * stride + alpha;
                v = __longlong_as_double(__double_as_longlong(__ldg(source + index)) ^ (long long)s_sign[q]);
                not_finite |= !isfinite(v);
            }
            s_tile[c][q] = v;
        }
        __syncthreads();
        double* out = tables.panel + ((long long)parent * n_cols + col0) * P::kRow;
        for(int i = int(threadIdx.x); i < ncol * P::kRow; i += int(blockDim.x))
        {
            out[i] = s_tile[i / P::kRow][i % P::kRow];
        }
    }
    if(__syncthreads_or(not_finite) && threadIdx.x == 0)
    {
        atomicExch(tables.nonfinite, 1);
    }
}


template <int NMAX, bool SPLIT>
__global__ void __launch_bounds__(fq::launch::kCliqueThreads) cross_clique_panel_kernel(const __grid_constant__ FqTileGeometry geometry,
                                                                                      const __grid_constant__ FqVectors vectors,
                                                                                      const __grid_constant__ FqCrossCliqueTables tables)
{
    using P = CliquePanel<NMAX>;
    constexpr int MT = P::kMTiles;
    constexpr int KS = P::kKSteps;
    constexpr int KSP = P::kKSP;
    extern __shared__ double shared[];
    const int warp = int(threadIdx.x / 32);
    const int lane = int(threadIdx.x % 32);
    const int g = lane >> 2;
    const int t = lane & 3;
    const long long job = (long long)blockIdx.x * clique::kWarpsPerBlock + warp;
    if(job >= (long long)tables.num_parents * tables.num_tiles || __ldg(tables.nonfinite) != 0)
    {
        return;
    }
    int tile, parent;
    clique_job(tables, job, tile, parent);
    double* row_tile = shared + (long long)warp * tables.tile_rows * NMAX;
    for(int i = lane; i < tables.tile_rows * NMAX; i += 32)
    {
        row_tile[i] = 0.0;
    }
    __syncwarp();
    const int parent_id = __ldg(tables.parent_ids + parent);
    const int first = __ldg(tables.parent_offsets + parent_id);
    const int count = __ldg(tables.parent_offsets + parent_id + 1) - first;
    const auto* members = reinterpret_cast<const unsigned long long*>(tables.members);
    int k_offset[KS];
#pragma unroll
    for(int s = 0; s < KS; ++s)
    {
        const int k = 4 * s + t;
        k_offset[s] = k < count ? clique::member_orbital(__ldg(members + first + k)) * tables.orbitals : -1;
    }
    int m_orbital[MT];
#pragma unroll
    for(int i = 0; i < MT; ++i)
    {
        const int m = 8 * i + g;
        m_orbital[i] = m < count ? clique::member_orbital(__ldg(members + first + m)) : -1;
    }
    const auto* words = reinterpret_cast<const unsigned long long*>(tables.edge_words);
    const double* panel = tables.panel + (long long)parent * tables.panel_columns * P::kRow + 2 * t;
    const long long block_size = (long long)tables.orbitals * tables.orbitals;
    const int first_group = __ldg(tables.tile_group_offsets + tile);
    const int last_group = __ldg(tables.tile_group_offsets + tile + 1);
    if(first_group < last_group)
    {
        int end = __ldg(tables.group_offsets + first_group + 1);
        int e0 = __ldg(tables.group_offsets + first_group);
        unsigned long long word = e0 + g < end ? __ldg(words + e0 + g) : 0ull;
        for(int group = first_group; group < last_group; ++group)
        {
            const double* block = tables.coefficients + (long long)__ldg(tables.group_pairs + group) * block_size;
            double a[MT][KS];
#pragma unroll
            for(int i = 0; i < MT; ++i)
            {
#pragma unroll
                for(int s = 0; s < KS; ++s)
                {
                    a[i][s] = (m_orbital[i] >= 0 && k_offset[s] >= 0) ? __ldg(block + k_offset[s] + m_orbital[i]) : 0.0;
                }
            }
            const int next_end = group + 1 < last_group ? __ldg(tables.group_offsets + group + 2) : end;
            for(; e0 < end; e0 += 8)
            {
                const int ne = e0 + 8 < end ? e0 + 8 : end;
                const int nend = e0 + 8 < end ? end : next_end;
                const unsigned long long next = ne + g < nend ? __ldg(words + ne + g) : 0ull;
                const bool exists = e0 + g < end;
                const double* row = panel + (long long)clique::tile_word_column(word) * P::kRow;
                double b[KSP];
#pragma unroll
                for(int h = 0; h < KSP / 2; ++h)
                {
                    const double2 v = exists ? __ldg(reinterpret_cast<const double2*>(row + 8 * h)) : make_double2(0.0, 0.0);
                    b[2 * h] = v.x;
                    b[2 * h + 1] = v.y;
                }

                const unsigned long long w0 = __shfl_sync(0xffffffffu, word, 8 * t);
                const unsigned long long w1 = __shfl_sync(0xffffffffu, word, 8 * t + 4);
                const int row0 = e0 + 2 * t < end ? clique::tile_word_row(w0) : -1;
                const int row1 = e0 + 2 * t + 1 < end ? clique::tile_word_row(w1) : -1;
                const long long sign0 = (long long)(w0 & clique::kSignBit);
                const long long sign1 = (long long)(w1 & clique::kSignBit);
#pragma unroll
                for(int i = 0; i < MT; ++i)
                {
                    double y0 = 0.0, y1 = 0.0;
                    if constexpr(SPLIT)
                    {
                        double z0 = 0.0, z1 = 0.0;
#pragma unroll
                        for(int s = 0; s < KS; s += 2)
                        {
                            asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                         : "+d"(y0), "+d"(y1)
                                         : "d"(a[i][s]), "d"(b[s]));
                            if(s + 1 < KS)
                            {
                                asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                             : "+d"(z0), "+d"(z1)
                                             : "d"(a[i][s + 1]), "d"(b[s + 1]));
                            }
                        }
                        y0 += z0;
                        y1 += z1;
                    }
                    else
                    {
#pragma unroll
                        for(int s = 0; s < KS; ++s)
                        {
                            asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                         : "+d"(y0), "+d"(y1)
                                         : "d"(a[i][s]), "d"(b[s]));
                        }
                    }
                    const int m = 8 * i + g;
                    if(m < NMAX)
                    {
                        if(row0 >= 0)
                        {
                            row_tile[row0 * NMAX + m] += __longlong_as_double(__double_as_longlong(y0) ^ sign0);
                        }
                        if(row1 >= 0)
                        {
                            row_tile[row1 * NMAX + m] += __longlong_as_double(__double_as_longlong(y1) ^ sign1);
                        }
                    }
                }
                __syncwarp();
                word = next;
            }
            e0 = end;
            end = next_end;
        }
    }
    __syncwarp();
    const CliqueLane me = lane_member(tables, parent_id, lane);
    if(me.active)
    {
        flush_tile<NMAX, false>(row_tile, tile, tables.tile_rows, geometry.n_b_bra, geometry.n_a_bra, vectors.out_local, me,
                                lane);
    }
}


template <typename F>
int with_panel_instance(const int bucket, F f)
{
    switch(bucket)
    {
    case 0:
        return f(cross_clique_panel_kernel<clique::bucket_members(0), true>,
                 cross_clique_pack_kernel<clique::bucket_members(0), false>, clique::bucket_members(0),
                 CliquePanel<clique::bucket_members(0)>::kRow);
    case 1:
        return f(cross_clique_panel_kernel<clique::bucket_members(1), false>,
                 cross_clique_pack_kernel<clique::bucket_members(1), false>, clique::bucket_members(1),
                 CliquePanel<clique::bucket_members(1)>::kRow);
    case 2:
        return f(cross_clique_panel_kernel<clique::bucket_members(2), false>,
                 cross_clique_pack_kernel<clique::bucket_members(2), false>, clique::bucket_members(2),
                 CliquePanel<clique::bucket_members(2)>::kRow);
    case 3:
        return f(cross_clique_panel_kernel<clique::bucket_members(3), false>,
                 cross_clique_pack_kernel<clique::bucket_members(3), false>, clique::bucket_members(3),
                 CliquePanel<clique::bucket_members(3)>::kRow);
    case 4:
        return f(cross_clique_panel_kernel<clique::bucket_members(4), false>,
                 cross_clique_pack_kernel<clique::bucket_members(4), false>, clique::bucket_members(4),
                 CliquePanel<clique::bucket_members(4)>::kRow);
    case 5:
        return f(cross_clique_panel_kernel<clique::bucket_members(5), false>,
                 cross_clique_pack_kernel<clique::bucket_members(5), false>, clique::bucket_members(5),
                 CliquePanel<clique::bucket_members(5)>::kRow);
    case 6:
        return f(cross_clique_panel_kernel<clique::bucket_members(6), false>,
                 cross_clique_pack_kernel<clique::bucket_members(6), false>, clique::bucket_members(6),
                 CliquePanel<clique::bucket_members(6)>::kRow);
    default:
        return int(cudaErrorInvalidValue);
    }
}


template <typename F>
int with_clique_instance(const int bucket, const int edges_per_batch, const bool transposed, const bool mma,
                         const bool distinct_rows, F f)
{
    const auto pick = [&](auto nmax) {
        constexpr int NMAX = decltype(nmax)::value;
        const auto mma_for = [&](auto eb) {
            constexpr int EB = decltype(eb)::value;
            if(distinct_rows)
            {
                return transposed ? f(cross_clique_mma_kernel<NMAX, EB, true, true>, NMAX)
                                  : f(cross_clique_mma_kernel<NMAX, EB, false, true>, NMAX);
            }
            return transposed ? f(cross_clique_mma_kernel<NMAX, EB, true, false>, NMAX)
                              : f(cross_clique_mma_kernel<NMAX, EB, false, false>, NMAX);
        };
        if(edges_per_batch == 4)
        {
            if(mma)
            {
                return mma_for(std::integral_constant<int, 4>{});
            }
            return transposed ? f(cross_clique_kernel<NMAX, 4, true>, NMAX) : f(cross_clique_kernel<NMAX, 4, false>, NMAX);
        }
        if(edges_per_batch == 8)
        {
            if(mma)
            {
                return mma_for(std::integral_constant<int, 8>{});
            }
            return transposed ? f(cross_clique_kernel<NMAX, 8, true>, NMAX) : f(cross_clique_kernel<NMAX, 8, false>, NMAX);
        }
        return int(cudaErrorInvalidValue);
    };
    switch(bucket)
    {
    case 0:
        return pick(std::integral_constant<int, clique::bucket_members(0)>{});
    case 1:
        return pick(std::integral_constant<int, clique::bucket_members(1)>{});
    case 2:
        return pick(std::integral_constant<int, clique::bucket_members(2)>{});
    case 3:
        return pick(std::integral_constant<int, clique::bucket_members(3)>{});
    case 4:
        return pick(std::integral_constant<int, clique::bucket_members(4)>{});
    case 5:
        return pick(std::integral_constant<int, clique::bucket_members(5)>{});
    case 6:
        return pick(std::integral_constant<int, clique::bucket_members(6)>{});
    default:
        return int(cudaErrorInvalidValue);
    }
}

}

extern "C"
{

    int fq_prepare_cross_clique(const int max_shared_bytes)
    {
        for(int bucket = 0; bucket < clique::kBuckets; ++bucket)
        {
            for(const int edges_per_batch : {4, 8})
            {
                for(const bool transposed : {false, true})
                {
                    for(const bool mma : {false, true})
                    {
                        for(const bool distinct_rows : {false, true})
                        {
                            const int status = with_clique_instance(
                                bucket, edges_per_batch, transposed, mma, distinct_rows, [&](auto kernel, int) {
                                    return int(cudaFuncSetAttribute(
                                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes));
                                });
                            if(status != int(cudaSuccess))
                            {
                                return status;
                            }
                        }
                    }
                }
            }
            const int status = with_panel_instance(bucket, [&](auto kernel, auto, int, int) {
                return int(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes));
            });
            if(status != int(cudaSuccess))
            {
                return status;
            }
        }
        return int(cudaSuccess);
    }

    int fq_launch_cross_clique_gated(const FqTileGeometry* geometry,
                                     const FqVectors* vectors,
                                     const FqCrossCliqueTables* tables,
                                     void* stream)
    {


        const long long jobs = (long long)tables->num_parents * tables->num_tiles;
        if(jobs <= 0)
        {
            return int(cudaSuccess);
        }
        if(tables->panel == nullptr || tables->nonfinite == nullptr || tables->mma == 0)
        {
            return int(cudaErrorInvalidValue);
        }
        const unsigned int blocks =
            static_cast<unsigned int>((jobs + clique::kWarpsPerBlock - 1) / clique::kWarpsPerBlock);
        return with_clique_instance(tables->bucket, tables->edges_per_batch, tables->transposed != 0, true,
                                    tables->distinct_rows != 0, [&](auto kernel, const int nmax) {
                                        const std::size_t bytes =
                                            clique::shared_bytes(nmax, tables->edges_per_batch, tables->tile_rows);
                                        kernel<<<blocks, fq::launch::kCliqueThreads, bytes,
                                                 static_cast<cudaStream_t>(stream)>>>(*geometry, *vectors, *tables);
                                        return int(cudaGetLastError());
                                    });
    }

    int fq_launch_cross_clique_pack(const FqTileGeometry* geometry,
                                    const FqVectors* vectors,
                                    const FqCrossCliqueTables* tables,
                                    void* stream)
    {


        if(tables->num_parents <= 0 || tables->panel_columns <= 0)
        {
            return int(cudaSuccess);
        }
        if(tables->panel == nullptr || tables->nonfinite == nullptr || tables->transposed != 0)
        {
            return int(cudaErrorInvalidValue);
        }
        const auto cuda_stream = static_cast<cudaStream_t>(stream);
        return with_panel_instance(tables->bucket, [&](auto, auto pack, int, int) {
            const dim3 pack_grid(static_cast<unsigned int>((tables->panel_columns + 31) / 32),
                                 static_cast<unsigned int>(tables->num_parents < 65535 ? tables->num_parents : 65535));
            pack<<<pack_grid, 256, 0, cuda_stream>>>(*geometry, vectors->src_in, *tables);
            return int(cudaGetLastError());
        });
    }

    int fq_cross_clique_panel_row(const int bucket)
    {
        int row = -1;
        with_panel_instance(bucket, [&](auto, auto, int, const int width) {
            row = width;
            return int(cudaSuccess);
        });
        return row;
    }

    int fq_launch_cross_clique(const FqTileGeometry* geometry,
                               const FqVectors* vectors,
                               const FqCrossCliqueTables* tables,
                               void* stream)
    {
        const long long jobs = (long long)tables->num_parents * tables->num_tiles;
        if(jobs <= 0)
        {
            return int(cudaSuccess);
        }
        const unsigned int blocks =
            static_cast<unsigned int>((jobs + clique::kWarpsPerBlock - 1) / clique::kWarpsPerBlock);
        const auto cuda_stream = static_cast<cudaStream_t>(stream);
        const bool panel = tables->panel != nullptr;
        if(panel)
        {
            if(tables->nonfinite == nullptr || tables->transposed != 0 || tables->mma == 0 || tables->distinct_rows == 0 ||
               tables->panel_columns <= 0)
            {
                return int(cudaErrorInvalidValue);
            }


            cudaMemsetAsync(tables->nonfinite, 0, sizeof(std::int32_t), cuda_stream);
            const int status = with_panel_instance(tables->bucket, [&](auto kernel, auto pack, const int nmax, int) {
                const dim3 pack_grid(static_cast<unsigned int>((tables->panel_columns + 31) / 32),
                                     static_cast<unsigned int>(tables->num_parents < 65535 ? tables->num_parents : 65535));
                pack<<<pack_grid, 256, 0, cuda_stream>>>(*geometry, vectors->src_in, *tables);
                const std::size_t bytes = std::size_t(clique::kWarpsPerBlock) * tables->tile_rows * nmax * sizeof(double);
                kernel<<<blocks, fq::launch::kCliqueThreads, bytes, cuda_stream>>>(*geometry, *vectors, *tables);
                return int(cudaGetLastError());
            });
            if(status != int(cudaSuccess))
            {
                return status;
            }
        }
        return with_clique_instance(tables->bucket, tables->edges_per_batch, tables->transposed != 0, tables->mma != 0,
                                    tables->distinct_rows != 0,
                                    [&](auto kernel, const int nmax) {
                                        const std::size_t bytes =
                                            clique::shared_bytes(nmax, tables->edges_per_batch, tables->tile_rows);
                                        kernel<<<blocks, fq::launch::kCliqueThreads, bytes,
                                                 static_cast<cudaStream_t>(stream)>>>(*geometry, *vectors, *tables);
                                        return int(cudaGetLastError());
                                    });
    }
}
