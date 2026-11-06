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

#include "launch.hpp"
#include "launch_grid.hpp"

namespace
{

constexpr int kAlphas = 64;
constexpr int kStages = 3;
constexpr int kChunk = 16;
constexpr int kPitch = kAlphas + 4;
constexpr int kBlockDoubles = kChunk / 4 * 8 * 32;
constexpr int kStageDoubles = kBlockDoubles + kChunk * kPitch;
constexpr int kSharedBytes = kStages * kStageDoubles * 8;

__device__ __forceinline__ void copy_async_16(void* shared, const void* global)
{
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(global));
}


__device__ __forceinline__ void copy_async_8(void* shared, const void* global, const bool valid)
{
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(global), "r"(valid ? 8 : 0));
}

__device__ __forceinline__ void copy_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void copy_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void mma_8x8x4(double& c0, double& c1, const double a, const double b)
{
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%0, %1};\n"
                 : "+d"(c0), "+d"(c1)
                 : "d"(a), "d"(b));
}


template <bool FALLBACK>
__global__ void __launch_bounds__(256, 4) beta_doubles_class_kernel(const FqTileGeometry geometry,
                                                                    const FqVectors vectors,
                                                                    const FqBetaDoubleClassTables tables)
{
    if((__ldg(tables.nonfinite) != 0) != FALLBACK)
    {
        return;
    }
    extern __shared__ __align__(128) double stage_memory[];
    const int n_a = geometry.a_stop - geometry.a_start;
    const std::int64_t id = (std::int64_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const std::int64_t per_super = std::int64_t(tables.num_tiles) * tables.alpha_group;
    const std::int64_t super = id / per_super;
    const int tile = int((id - super * per_super) / tables.alpha_group);
    const int column0 = int((super * tables.alpha_group + id % tables.alpha_group) * kAlphas);
    if(column0 >= n_a)
    {
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp & 3, wn = warp >> 2;
    const int g = lane >> 2, tl = lane & 3;
    const int4 info = __ldg(reinterpret_cast<const int4*>(tables.tile_info) + tile);
    const int first_row = info.x, rows = info.y, first_column = info.z, columns = info.w;
    const std::int64_t block_offset = __ldg(tables.block_offsets + tile);
    const int chunks = (columns + kChunk - 1) / kChunk;
    const bool active = wm * 16 < rows;
    constexpr int kSub = kAlphas / 2 / 8;
    double acc[2][kSub][2];
#pragma unroll
    for(int i = 0; i < 2; ++i)
#pragma unroll
        for(int j = 0; j < kSub; ++j)
            acc[i][j][0] = acc[i][j][1] = 0.0;

    const int kk = threadIdx.x >> 4, c16 = threadIdx.x & 15;
    const auto stage = [&](const int chunk, const int buffer) {
        double* sa = stage_memory + buffer * kStageDoubles;
        double* sx = sa + kBlockDoubles;
        const double* ga = tables.blocks + block_offset + std::int64_t(chunk) * kBlockDoubles;
        for(int i = threadIdx.x; i < kBlockDoubles / 2; i += 256)
        {
            copy_async_16(sa + 2 * i, ga + 2 * i);
        }
        const int k = chunk * kChunk + kk;
        const int slot = k < columns ? __ldg(vectors.beta_src_slot + __ldg(tables.col_members + first_column + k)) : -1;
        const double* row = vectors.src_in + (slot >= 0 ? std::int64_t(slot) * geometry.n_alpha_src_compact : 0)
                            + geometry.a_start + column0;
#pragma unroll
        for(int j = 0; j < kAlphas / 16; ++j)
        {
            const int c = c16 + 16 * j;
            const bool valid = slot >= 0 && column0 + c < n_a;
            copy_async_8(sx + kk * kPitch + c, valid ? row + c : vectors.src_in, valid);
        }
        copy_async_commit();
    };
#pragma unroll
    for(int s = 0; s < kStages - 1; ++s)
    {
        if(s < chunks)
        {
            stage(s, s);
        }
        else
        {
            copy_async_commit();
        }
    }
    for(int chunk = 0; chunk < chunks; ++chunk)
    {
        copy_async_wait<kStages - 2>();

        bool not_finite = false;
        if(FALLBACK)
        {
            const double* sx = stage_memory + (chunk % kStages) * kStageDoubles + kBlockDoubles;
#pragma unroll
            for(int j = 0; j < kAlphas / 16; ++j)
            {
                not_finite = not_finite || !isfinite(sx[kk * kPitch + c16 + 16 * j]);
            }
            not_finite = __syncthreads_or(not_finite) != 0;
        }
        else
        {
            __syncthreads();
        }
        const int next = chunk + kStages - 1;
        if(next < chunks)
        {
            stage(next, next % kStages);
        }
        else
        {
            copy_async_commit();
        }
        if(!active)
        {
            continue;
        }
        const double* sa = stage_memory + (chunk % kStages) * kStageDoubles;
        const double* sx = sa + kBlockDoubles;
        if(!FALLBACK || !not_finite)
        {
#pragma unroll
            for(int ks = 0; ks < kChunk / 4; ++ks)
            {
                const double a0 = sa[(ks * 8 + 2 * wm) * 32 + lane];
                const double a1 = sa[(ks * 8 + 2 * wm + 1) * 32 + lane];
                const double* bx = sx + (ks * 4 + tl) * kPitch + wn * (kAlphas / 2) + g;
#pragma unroll
                for(int j = 0; j < kSub; ++j)
                {
                    const double b = bx[8 * j];
                    mma_8x8x4(acc[0][j][0], acc[0][j][1], a0, b);
                    mma_8x8x4(acc[1][j][0], acc[1][j][1], a1, b);
                }
            }
            continue;
        }

#pragma unroll
        for(int i = 0; i < 2; ++i)
        {
            const int sub = 2 * wm + i;
            for(int k = 0; k < kChunk; ++k)
            {
                const double a = sa[((k >> 2) * 8 + sub) * 32 + 4 * g + (k & 3)];
                if(a == 0.0)
                {
                    continue;
                }
#pragma unroll
                for(int j = 0; j < kSub; ++j)
                {
                    const int c = wn * (kAlphas / 2) + 8 * j + 2 * tl;
                    acc[i][j][0] = fma(a, sx[k * kPitch + c], acc[i][j][0]);
                    acc[i][j][1] = fma(a, sx[k * kPitch + c + 1], acc[i][j][1]);
                }
            }
        }
    }
    if(!active)
    {
        return;
    }
#pragma unroll
    for(int i = 0; i < 2; ++i)
    {
        const int r = wm * 16 + i * 8 + g;
        if(r >= rows)
        {
            continue;
        }
        const int b = __ldg(tables.row_members + first_row + r);
        if(b < geometry.b_start || b >= geometry.b_stop)
        {
            continue;
        }
        double* out_row = vectors.out_local + std::int64_t(b - geometry.b_start) * n_a;
#pragma unroll
        for(int j = 0; j < kSub; ++j)
        {
            const int c = column0 + wn * (kAlphas / 2) + 8 * j + 2 * tl;
            if(c < n_a)
            {
                atomicAdd(out_row + c, acc[i][j][0]);
            }
            if(c + 1 < n_a)
            {
                atomicAdd(out_row + c + 1, acc[i][j][1]);
            }
        }
    }
}


__global__ void __launch_bounds__(256) nonfinite_kernel(const FqTileGeometry geometry, const double* src_in,
                                                        int* flag)
{
    const std::int64_t n_a = geometry.a_stop - geometry.a_start;
    const std::int64_t total = n_a * geometry.n_b_src;
    const std::int64_t step = std::int64_t(gridDim.x) * blockDim.x;
    bool found = false;
    for(std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total && !found; i += step)
    {
        const std::int64_t row = i / n_a;
        found = !isfinite(src_in[row * geometry.n_alpha_src_compact + geometry.a_start + (i - row * n_a)]);
    }
    if(__any_sync(0xffffffffu, found) && (threadIdx.x & 31) == 0)
    {
        *flag = 1;
    }
}


template <bool ADD>
__global__ void __launch_bounds__(256) transpose_kernel(const double* __restrict__ src, double* __restrict__ dst,
                                                        const int rows, const int cols, const std::int64_t ld_src,
                                                        const std::int64_t ld_dst)
{
    __shared__ double tile[32][33];
    const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
#pragma unroll
    for(int i = 0; i < 32; i += 8)
    {
        const int r = r0 + threadIdx.y + i, c = c0 + threadIdx.x;
        tile[threadIdx.y + i][threadIdx.x] = (r < rows && c < cols) ? src[r * ld_src + c] : 0.0;
    }
    __syncthreads();
#pragma unroll
    for(int i = 0; i < 32; i += 8)
    {
        const int c = c0 + threadIdx.y + i, r = r0 + threadIdx.x;
        if(r < rows && c < cols)
        {
            double* d = dst + c * ld_dst + r;
            const double v = tile[threadIdx.x][threadIdx.y + i];
            *d = ADD ? *d + v : v;
        }
    }
}

template <bool ADD>
cudaError_t transpose(const double* src, double* dst, const int rows, const int cols, const std::int64_t ld_src,
                      const std::int64_t ld_dst, const cudaStream_t stream)
{
    if(rows <= 0 || cols <= 0)
    {
        return cudaSuccess;
    }
    if((cols + 31) / 32 > 2147483647LL || (rows + 31) / 32 > 65535)
    {
        return cudaErrorInvalidValue;
    }
    transpose_kernel<ADD><<<dim3((cols + 31) / 32, (rows + 31) / 32), dim3(32, 8), 0, stream>>>(src, dst, rows, cols,
                                                                                                ld_src, ld_dst);
    return cudaGetLastError();
}


int launch_class(const FqTileGeometry& geometry, const FqVectors& vectors, const FqBetaDoubleClassTables& tables,
                 const cudaStream_t s)
{
    if(fq::launch::empty_tile(geometry) || tables.num_tiles <= 0)
    {
        return int(cudaSuccess);
    }
    if(tables.alpha_group < 1 || tables.nonfinite == nullptr)
    {
        return int(cudaErrorInvalidValue);
    }
    for(const auto kernel : {beta_doubles_class_kernel<false>, beta_doubles_class_kernel<true>})
    {
        const cudaError_t attribute =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
        if(attribute != cudaSuccess)
        {
            return int(attribute);
        }
    }
    const cudaError_t zeroed = cudaMemsetAsync(tables.nonfinite, 0, sizeof(int), s);
    if(zeroed != cudaSuccess)
    {
        return int(zeroed);
    }
    nonfinite_kernel<<<1024, 256, 0, s>>>(geometry, vectors.src_in, tables.nonfinite);
    const long long n_a = geometry.a_stop - geometry.a_start;
    const long long per_super = static_cast<long long>(tables.alpha_group) * kAlphas;
    const long long supers = (n_a + per_super - 1) / per_super;
    const dim3 grid = fq::launch::linear_grid(supers * tables.num_tiles * tables.alpha_group);
    beta_doubles_class_kernel<false><<<grid, 256, kSharedBytes, s>>>(geometry, vectors, tables);
    beta_doubles_class_kernel<true><<<grid, 256, kSharedBytes, s>>>(geometry, vectors, tables);
    return int(cudaGetLastError());
}

}

extern "C"
{

    int fq_launch_beta_doubles_class(const FqTileGeometry* geometry,
                                     const FqVectors* vectors,
                                     const FqBetaDoubleClassTables* tables,
                                     void* stream)
    {
        return launch_class(*geometry, *vectors, *tables, static_cast<cudaStream_t>(stream));
    }

    int fq_launch_alpha_doubles_class(const FqTileGeometry* geometry,
                                      const FqVectors* vectors,
                                      const FqAlphaDoubleClassTables* tables,
                                      void* stream)
    {


        if(fq::launch::empty_tile(*geometry) || tables->classes.num_tiles <= 0)
        {
            return int(cudaSuccess);
        }


        if(tables->slab < 1 || tables->xt == nullptr || tables->yt == nullptr
           || (tables->leftover.row_offsets != nullptr && tables->slab_identity == nullptr) || geometry->a_start != 0
           || geometry->a_stop != geometry->n_a_bra || geometry->n_b_src != geometry->n_b_bra)
        {
            return int(cudaErrorInvalidValue);
        }
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        const int n_a = geometry->a_stop - geometry->a_start;
        const int n_src_alpha = tables->n_alpha_src;
        const std::int64_t stride = geometry->n_alpha_src_compact;
        for(int b0 = geometry->b_start; b0 < geometry->b_stop; b0 += tables->slab)
        {
            const int rows = geometry->b_stop - b0 < tables->slab ? geometry->b_stop - b0 : tables->slab;

            cudaError_t status = transpose<false>(vectors->src_in + std::int64_t(b0) * stride, tables->xt, rows,
                                                  n_src_alpha, stride, rows, s);
            if(status == cudaSuccess)
            {
                status = cudaMemsetAsync(tables->yt, 0, sizeof(double) * std::size_t(n_a) * rows, s);
            }
            if(status != cudaSuccess)
            {
                return int(status);
            }
            FqTileGeometry slab = *geometry;
            slab.a_start = 0;
            slab.a_stop = rows;
            slab.b_start = 0;
            slab.b_stop = n_a;
            slab.n_b_src = n_src_alpha;
            slab.n_alpha_src_compact = rows;
            FqVectors transposed = *vectors;
            transposed.src_in = tables->xt;
            transposed.out_local = tables->yt;
            transposed.beta_src_slot = vectors->alpha_src_slot;
            const int launched = launch_class(slab, transposed, tables->classes, s);
            if(launched != int(cudaSuccess))
            {
                return launched;
            }
            if(tables->leftover.row_offsets != nullptr)
            {


                FqVectors left = transposed;
                left.alpha_src_slot = tables->slab_identity;
                const int applied = fq_launch_beta_doubles(&slab, &left, &tables->leftover, stream);
                if(applied != int(cudaSuccess))
                {
                    return applied;
                }
            }
            status = transpose<true>(tables->yt, vectors->out_local + std::int64_t(b0 - geometry->b_start) * n_a, n_a,
                                     rows, rows, n_a, s);
            if(status != cudaSuccess)
            {
                return int(status);
            }
        }
        return int(cudaSuccess);
    }
}
