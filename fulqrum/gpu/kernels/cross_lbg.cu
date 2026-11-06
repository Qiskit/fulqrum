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

#include "formats/clique_word.hpp"
#include "launch.hpp"


namespace
{

namespace clique = fq::clique_word;


template <int NMAX>
struct LbgPanel
{
    static constexpr int kMTiles = (NMAX + 7) / 8;
    static constexpr int kKSteps = NMAX / 4;
    static constexpr int kKSP = (kKSteps + 1) & ~1;
    static constexpr int kRow = 4 * kKSP;
};

__device__ __forceinline__ void lbg_dmma(double& d0, double& d1, const double a, const double b)
{
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+d"(d0), "+d"(d1)
                 : "d"(a), "d"(b));
}


__global__ void __launch_bounds__(256) lbg_finite_kernel(const double* __restrict__ source, const long long n,
                                                       std::int32_t* nonfinite)
{
    bool bad = false;
    for(long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    {
        bad |= !isfinite(__ldg(source + i));
    }
    if(__syncthreads_or(bad) && threadIdx.x == 0)
    {
        atomicExch(nonfinite, 1);
    }
}


template <int NMAX>
__global__ void __launch_bounds__(128) lbg_kernel(const __grid_constant__ FqTileGeometry geometry,
                                                 const __grid_constant__ FqCrossCliqueTables tables,
                                                 const __grid_constant__ FqCrossLbgTables lbg)
{
    using P = LbgPanel<NMAX>;
    constexpr int MT = P::kMTiles;
    constexpr int KS = P::kKSteps;
    constexpr int KSP = P::kKSP;
    const int lane = int(threadIdx.x % 32);
    const int g = lane >> 2;
    const int t = lane & 3;
    const long long item = (long long)blockIdx.x * 4 + threadIdx.x / 32;
    if(item >= (long long)tables.num_parents * lbg.num_segments || __ldg(lbg.nonfinite) != 0)
    {
        return;
    }
    const int parent = int(item / lbg.num_segments);
    const int sg = int(item % lbg.num_segments);
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
    const double* block = tables.coefficients + (long long)__ldg(lbg.segment_pair + sg) * tables.orbitals * tables.orbitals;
    double a[MT][KS];
#pragma unroll
    for(int n = 0; n < MT; ++n)
    {
        const int m = 8 * n + g;
        const int mo = m < count ? clique::member_orbital(__ldg(members + first + m)) : -1;
#pragma unroll
        for(int s = 0; s < KS; ++s)
        {
            a[n][s] = (mo >= 0 && k_offset[s] >= 0) ? __ldg(block + k_offset[s] + mo) : 0.0;
        }
    }
    const auto* words = reinterpret_cast<const unsigned long long*>(lbg.edge_words);
    const double* panel = tables.panel + (long long)parent * tables.panel_columns * P::kRow + 2 * t;
    double* out = lbg.out_panel + (long long)parent * geometry.n_b_bra * lbg.out_stride + t;
    bool lo[MT], hi[MT];
#pragma unroll
    for(int n = 0; n < MT; ++n)
    {
        lo[n] = 8 * n + 2 * t < count;
        hi[n] = 8 * n + 2 * t + 1 < count;
    }
    const int end = __ldg(lbg.segment_end + sg);
    int e0 = __ldg(lbg.segment_begin + sg);
    unsigned long long word = e0 + g < end ? __ldg(words + e0 + g) : 0ull;
    double b[KSP];
    {
        const double* row = panel + (long long)clique::tile_word_column(word) * P::kRow;
#pragma unroll
        for(int h = 0; h < KSP / 2; ++h)
        {
            const double2 v = __ldg(reinterpret_cast<const double2*>(row + 8 * h));
            b[2 * h] = v.x;
            b[2 * h + 1] = v.y;
        }
    }
    for(; e0 < end; e0 += 8)
    {
        const bool exists = e0 + g < end;
        const unsigned long long next = e0 + 8 + g < end ? __ldg(words + e0 + 8 + g) : 0ull;
        const long long sign = (long long)(word & clique::kSignBit);
        double* orow = out + (long long)clique::tile_word_row(word) * lbg.out_stride;
        double y[MT][2];
#pragma unroll
        for(int n = 0; n < MT; ++n)
        {
            y[n][0] = 0.0;
            y[n][1] = 0.0;
#pragma unroll
            for(int s = 0; s < KS; ++s)
            {
                lbg_dmma(y[n][0], y[n][1], b[s], a[n][s]);
            }
            if(exists && lo[n])
            {
                atomicAdd(orow + 8 * n, __longlong_as_double(__double_as_longlong(y[n][0]) ^ sign));
            }
            if(exists && hi[n])
            {
                atomicAdd(orow + 8 * n + 4, __longlong_as_double(__double_as_longlong(y[n][1]) ^ sign));
            }
        }
        {
            const double* row = panel + (long long)clique::tile_word_column(next) * P::kRow;
#pragma unroll
            for(int h = 0; h < KSP / 2; ++h)
            {
                const double2 v = __ldg(reinterpret_cast<const double2*>(row + 8 * h));
                b[2 * h] = v.x;
                b[2 * h + 1] = v.y;
            }
        }
        word = next;
    }
}


__global__ void __launch_bounds__(256) lbg_unpack_kernel(const __grid_constant__ FqTileGeometry geometry,
                                                        const __grid_constant__ FqVectors vectors,
                                                        const __grid_constant__ FqCrossCliqueTables tables,
                                                        const __grid_constant__ FqCrossLbgTables lbg)
{
    const int lane = int(threadIdx.x % 32);
    const long long item = (long long)blockIdx.x * 8 + threadIdx.x / 32;
    const int n_rows = geometry.n_b_bra;
    const int row_blocks = (n_rows + 7) / 8;
    if(item >= (long long)tables.num_parents * row_blocks || __ldg(lbg.nonfinite) != 0)
    {
        return;
    }
    const int parent = int(item / row_blocks);
    const int r0 = int(item % row_blocks) * 8;
    const int parent_id = __ldg(tables.parent_ids + parent);
    const int first = __ldg(tables.parent_offsets + parent_id);
    const int count = __ldg(tables.parent_offsets + parent_id + 1) - first;
    if(lane >= count)
    {
        return;
    }
    const unsigned long long member = __ldg(reinterpret_cast<const unsigned long long*>(tables.members) + first + lane);
    const long long out_sign = (long long)clique::output_sign_bit(member);
    const int alpha = clique::member_alpha(member);
    const int c = lane & 7;
    const double* yp = lbg.out_panel + (long long)parent * n_rows * lbg.out_stride + (lane & ~7) + (c & 1) * 4 + (c >> 1);
    for(int r = r0; r < r0 + 8 && r < n_rows; ++r)
    {
        const double y = yp[(long long)r * lbg.out_stride];
        if(y != 0.0)
        {
            atomicAdd(vectors.out_local + (long long)r * geometry.n_a_bra + alpha,
                      __longlong_as_double(__double_as_longlong(y) ^ out_sign));
        }
    }
}


template <int M>
__global__ void __launch_bounds__(256) rsl_pack_kernel(const __grid_constant__ FqTileGeometry geometry,
                                                      const double* __restrict__ source,
                                                      const __grid_constant__ FqCrossRslTables rsl)
{
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    const long long per_block = (long long)rsl.panel_rows * M * 32;
    if(i >= (long long)rsl.num_blocks * per_block || __ldg(rsl.nonfinite) != 0)
    {
        return;
    }
    const int blk = int(i / per_block);
    const long long rem = i % per_block;
    const int row = int(rem / (M * 32));
    const int k = int((rem / 32) % M);
    const int lane = int(rem % 32);
    double v = 0.0;
    if(lane < __ldg(rsl.block_count + blk))
    {
        const unsigned long long w = __ldg(rsl.members + (long long)(__ldg(rsl.block_first + blk) + lane) * M + k);
        v = __longlong_as_double(__double_as_longlong(__ldg(source + (long long)row * geometry.n_alpha_src_compact +
                                                            clique::member_alpha(w))) ^ (long long)(w & clique::kSignBit));
    }
    rsl.panel[i] = v;
}


template <int M, bool MIXED>
__global__ void __launch_bounds__(128) rsl_kernel(const __grid_constant__ FqTileGeometry geometry,
                                                 const __grid_constant__ FqVectors vectors,
                                                 const __grid_constant__ FqCrossRslTables rsl)
{
    constexpr int NP = M * (M - 1) / 2;
    const int lane = int(threadIdx.x % 32);
    const long long item = (long long)blockIdx.x * 4 + threadIdx.x / 32;
    const int chunks = (geometry.n_b_bra + rsl.rows_per_job - 1) / rsl.rows_per_job;
    if(item >= (long long)rsl.num_blocks * chunks || __ldg(rsl.nonfinite) != 0)
    {
        return;
    }
    const int blk = int(item / chunks);
    const int r0 = int(item % chunks) * rsl.rows_per_job;
    const int count = __ldg(rsl.block_count + blk);
    const int first = __ldg(rsl.block_first + blk);
    const bool active = lane < count;
    const int me = active ? lane : 0;
    int alpha[M];
    long long out_sign[M];
    int orb[M];
#pragma unroll
    for(int i = 0; i < M; ++i)
    {
        const unsigned long long w = __ldg(rsl.members + (long long)(first + me) * M + i);
        alpha[i] = clique::member_alpha(w);
        orb[i] = clique::member_orbital(w);
        out_sign[i] = (long long)clique::output_sign_bit(w);
    }
    int off[NP];
    {
        int q = 0;
#pragma unroll
        for(int i = 0; i < M; ++i)
        {
#pragma unroll
            for(int j = i + 1; j < M; ++j)
            {
                off[q++] = __shfl_sync(0xffffffffu, orb[i], 0) * rsl.orbitals + __shfl_sync(0xffffffffu, orb[j], 0);
            }
        }
    }
    const long long block_size = (long long)rsl.orbitals * rsl.orbitals;
    const auto* words = reinterpret_cast<const unsigned long long*>(rsl.beta_words);
    const double* panel = rsl.panel + (long long)blk * rsl.panel_rows * M * 32 + lane;
    for(int b = r0; b < r0 + rsl.rows_per_job && b < geometry.n_b_bra; ++b)
    {
        const int e1 = __ldg(rsl.tasks + 3 * b + 2);
        double acc[M];
#pragma unroll
        for(int i = 0; i < M; ++i)
        {
            acc[i] = 0.0;
        }
        for(int e = __ldg(rsl.tasks + 3 * b + 1); e < e1; ++e)
        {
            const unsigned long long w = __ldg(words + e);
            const long long pair = (long long)((w >> 32) & 0xffffull);
            const double* gc = MIXED ? rsl.coef_table + (((long long)blk * rsl.num_pairs + pair) * NP) * 32 + lane
                                     : rsl.coefficients + pair * block_size;
            const long long sign = (long long)(w & clique::kSignBit);
            const double* xs = panel + (long long)int(w & 0xffffffffull) * M * 32;
            double x[M];
#pragma unroll
            for(int i = 0; i < M; ++i)
            {
                x[i] = __longlong_as_double(__double_as_longlong(__ldg(xs + i * 32)) ^ sign);
            }
            int q = 0;
#pragma unroll
            for(int i = 0; i < M; ++i)
            {
#pragma unroll
                for(int j = i + 1; j < M; ++j)
                {
                    const double c = MIXED ? __ldg(gc + 32 * q) : __ldg(gc + off[q]);
                    ++q;
                    acc[i] = fma(c, x[j], acc[i]);
                    acc[j] = fma(c, x[i], acc[j]);
                }
            }
        }
        if(active)
        {
#pragma unroll
            for(int i = 0; i < M; ++i)
            {
                if(acc[i] != 0.0)
                {
                    atomicAdd(vectors.out_local + (long long)b * geometry.n_a_bra + alpha[i],
                              __longlong_as_double(__double_as_longlong(acc[i]) ^ out_sign[i]));
                }
            }
        }
    }
}

template <int NMAX>
int launch_lbg_bucket(const FqTileGeometry* geometry, const FqVectors* vectors, const FqCrossCliqueTables* tables,
                      const FqCrossLbgTables* lbg, cudaStream_t s)
{
    const long long items = (long long)tables->num_parents * lbg->num_segments;
    if(items <= 0)
    {
        return int(cudaSuccess);
    }
    cudaMemsetAsync(lbg->out_panel, 0, sizeof(double) * (std::size_t)tables->num_parents * geometry->n_b_bra * lbg->out_stride, s);
    lbg_kernel<NMAX><<<static_cast<unsigned int>((items + 3) / 4), 128, 0, s>>>(*geometry, *tables, *lbg);
    const long long unpack = (long long)tables->num_parents * ((geometry->n_b_bra + 7) / 8);
    lbg_unpack_kernel<<<static_cast<unsigned int>((unpack + 7) / 8), 256, 0, s>>>(*geometry, *vectors, *tables, *lbg);
    return int(cudaGetLastError());
}

template <int M>
int launch_rsl_block(const FqTileGeometry* geometry, const FqVectors* vectors, const FqCrossRslTables* rsl,
                     cudaStream_t s)
{
    const long long n = (long long)rsl->num_blocks * rsl->panel_rows * M * 32;
    rsl_pack_kernel<M><<<static_cast<unsigned int>((n + 255) / 256), 256, 0, s>>>(*geometry, vectors->src_in, *rsl);
    const long long items = (long long)rsl->num_blocks * ((geometry->n_b_bra + rsl->rows_per_job - 1) / rsl->rows_per_job);
    const unsigned int blocks = static_cast<unsigned int>((items + 3) / 4);
    if(rsl->coef_table != nullptr)
    {
        rsl_kernel<M, true><<<blocks, 128, 0, s>>>(*geometry, *vectors, *rsl);
    }
    else
    {
        rsl_kernel<M, false><<<blocks, 128, 0, s>>>(*geometry, *vectors, *rsl);
    }
    return int(cudaGetLastError());
}

}

extern "C"
{

    int fq_launch_lbg_finite(const double* source, const long long n, std::int32_t* nonfinite, void* stream)
    {
        const auto s = static_cast<cudaStream_t>(stream);
        cudaMemsetAsync(nonfinite, 0, sizeof(std::int32_t), s);
        if(n > 0)
        {
            const long long blocks = (n + 255) / 256;
            lbg_finite_kernel<<<static_cast<unsigned int>(blocks < 4096 ? blocks : 4096), 256, 0, s>>>(source, n, nonfinite);
        }
        return int(cudaGetLastError());
    }

    int fq_launch_cross_lbg(const FqTileGeometry* geometry,
                            const FqVectors* vectors,
                            const FqCrossCliqueTables* tables,
                            const FqCrossLbgTables* lbg,
                            void* stream)
    {
        if(tables->transposed != 0 || tables->panel == nullptr || tables->panel_columns <= 0 || lbg->nonfinite == nullptr)
        {
            return int(cudaErrorInvalidValue);
        }
        const auto s = static_cast<cudaStream_t>(stream);
        switch(tables->bucket)
        {
        case 0: return launch_lbg_bucket<clique::bucket_members(0)>(geometry, vectors, tables, lbg, s);
        case 1: return launch_lbg_bucket<clique::bucket_members(1)>(geometry, vectors, tables, lbg, s);
        case 2: return launch_lbg_bucket<clique::bucket_members(2)>(geometry, vectors, tables, lbg, s);
        case 3: return launch_lbg_bucket<clique::bucket_members(3)>(geometry, vectors, tables, lbg, s);
        case 4: return launch_lbg_bucket<clique::bucket_members(4)>(geometry, vectors, tables, lbg, s);
        case 5: return launch_lbg_bucket<clique::bucket_members(5)>(geometry, vectors, tables, lbg, s);
        case 6: return launch_lbg_bucket<clique::bucket_members(6)>(geometry, vectors, tables, lbg, s);
        default: return int(cudaErrorInvalidValue);
        }
    }

    int fq_launch_cross_rsl(const FqTileGeometry* geometry,
                            const FqVectors* vectors,
                            const FqCrossRslTables* rsl,
                            void* stream)
    {
        if(rsl->num_blocks <= 0)
        {
            return int(cudaSuccess);
        }
        if(rsl->nonfinite == nullptr || rsl->panel == nullptr || rsl->rows_per_job <= 0)
        {
            return int(cudaErrorInvalidValue);
        }
        const auto s = static_cast<cudaStream_t>(stream);
        switch(rsl->members_per_clique)
        {
        case 2: return launch_rsl_block<2>(geometry, vectors, rsl, s);
        case 3: return launch_rsl_block<3>(geometry, vectors, rsl, s);
        case 4: return launch_rsl_block<4>(geometry, vectors, rsl, s);
        case 5: return launch_rsl_block<5>(geometry, vectors, rsl, s);
        default: return int(cudaErrorInvalidValue);
        }
    }
}
