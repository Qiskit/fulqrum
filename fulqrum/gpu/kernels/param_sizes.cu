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
#include <cstddef>

#include "launch.hpp"

extern "C"
{

    const char* fq_error_name(const int status)
    {
        return cudaGetErrorName(static_cast<cudaError_t>(status));
    }

    std::size_t fq_param_struct_size(const int which)
    {
        switch(which)
        {
        case kFqTileGeometry:
            return sizeof(FqTileGeometry);
        case kFqVectors:
            return sizeof(FqVectors);
        case kFqDiagonalTables:
            return sizeof(FqDiagonalTables);
        case kFqAlphaSinglePairTables:
            return sizeof(FqAlphaSinglePairTables);
        case kFqAlphaSingleDeltaTables:
            return sizeof(FqAlphaSingleDeltaTables);
        case kFqBetaSinglePairTables:
            return sizeof(FqBetaSinglePairTables);
        case kFqBetaSingleDeltaTables:
            return sizeof(FqBetaSingleDeltaTables);
        case kFqAlphaDoubleTables:
            return sizeof(FqAlphaDoubleTables);
        case kFqBetaDoubleTables:
            return sizeof(FqBetaDoubleTables);
        case kFqCrossSellTables:
            return sizeof(FqCrossSellTables);
        case kFqCrossDirectTables:
            return sizeof(FqCrossDirectTables);
        case kFqCrossCliqueTables:
            return sizeof(FqCrossCliqueTables);
        case kFqTermTables:
            return sizeof(FqTermTables);
        case kFqSinglesBuildTables:
            return sizeof(FqSinglesBuildTables);
        case kFqDoublesBuildTables:
            return sizeof(FqDoublesBuildTables);
        case kFqBetaDoubleClassTables:
            return sizeof(FqBetaDoubleClassTables);
        case kFqAlphaDoubleClassTables:
            return sizeof(FqAlphaDoubleClassTables);
        case kFqCrossLbgTables:
            return sizeof(FqCrossLbgTables);
        case kFqCrossRslTables:
            return sizeof(FqCrossRslTables);
        default:
            return 0;
        }
    }
}

namespace
{

constexpr int kProbeChains = 8;
constexpr int kProbeThreads = 256;


__global__ void __launch_bounds__(kProbeThreads) probe_mma_kernel(const int iterations, double* out)
{
    const int lane = threadIdx.x & 31;
    const double a = 1.0 + 1e-9 * lane;
    const double b = 1.0 - 1e-9 * lane;
    double y[kProbeChains][2];
#pragma unroll
    for(int c = 0; c < kProbeChains; ++c)
    {
        y[c][0] = 0.0;
        y[c][1] = 0.0;
    }
    for(int i = 0; i < iterations; ++i)
    {
#pragma unroll
        for(int c = 0; c < kProbeChains; ++c)
        {
            asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                         : "+d"(y[c][0]), "+d"(y[c][1])
                         : "d"(a), "d"(b));
        }
    }
    double sum = 0.0;
#pragma unroll
    for(int c = 0; c < kProbeChains; ++c)
    {
        sum += y[c][0] + y[c][1];
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}


__global__ void __launch_bounds__(kProbeThreads) probe_scalar_kernel(const int iterations, double* out)
{
    const int lane = threadIdx.x & 31;
    const double a = 1.0 + 1e-9 * lane;
    const double b = 1.0 - 1e-9 * lane;
    double y[kProbeChains];
#pragma unroll
    for(int c = 0; c < kProbeChains; ++c)
    {
        y[c] = 1e-3 * c;
    }
    for(int i = 0; i < iterations; ++i)
    {
#pragma unroll
        for(int k = 0; k < 8; ++k)
        {
#pragma unroll
            for(int c = 0; c < kProbeChains; ++c)
            {
                y[c] = fma(a, b, y[c]);
            }
        }
    }
    double sum = 0.0;
#pragma unroll
    for(int c = 0; c < kProbeChains; ++c)
    {
        sum += y[c];
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}


__global__ void probe_mma_check_kernel(double* out)
{
    const int lane = threadIdx.x;
    const int row = lane >> 2;
    const int k = lane & 3;
    const double a = double((row + 1) * (k + 2) % 7 - 3);
    const double b = double((k + 3) * (row + 1) % 5 - 2);
    double y0 = 0.0;
    double y1 = 0.0;
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+d"(y0), "+d"(y1)
                 : "d"(a), "d"(b));
    double want[2] = {0.0, 0.0};
    for(int j = 0; j < 2; ++j)
    {
        const int col = 2 * (lane & 3) + j;
        for(int kk = 0; kk < 4; ++kk)
        {
            const double av = double((row + 1) * (kk + 2) % 7 - 3);
            const double bv = double((kk + 3) * (col + 1) % 5 - 2);
            want[j] += av * bv;
        }
    }
    out[4 * lane + 0] = y0;
    out[4 * lane + 1] = y1;
    out[4 * lane + 2] = want[0];
    out[4 * lane + 3] = want[1];
}


__global__ void __launch_bounds__(kProbeThreads) probe_copy_kernel(const double2* in, double2* out, const long long n)
{
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        out[i] = in[i];
    }
}


cudaError_t time_copy(const int blocks, const double2* in, double2* out, const long long n, float* ms)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaError_t status = cudaEventCreate(&start);
    if(status != cudaSuccess)
    {
        return status;
    }
    status = cudaEventCreate(&stop);
    if(status == cudaSuccess)
    {
        probe_copy_kernel<<<blocks, kProbeThreads>>>(in, out, n);
        cudaEventRecord(start);
        probe_copy_kernel<<<blocks, kProbeThreads>>>(in, out, n);
        cudaEventRecord(stop);
        status = cudaEventSynchronize(stop);
        if(status == cudaSuccess)
        {
            status = cudaGetLastError();
        }
        if(status == cudaSuccess)
        {
            status = cudaEventElapsedTime(ms, start, stop);
        }
        cudaEventDestroy(stop);
    }
    cudaEventDestroy(start);
    return status;
}


template <typename Kernel>
cudaError_t time_probe(Kernel kernel, const int blocks, const int iterations, double* out, float* ms)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaError_t status = cudaEventCreate(&start);
    if(status != cudaSuccess)
    {
        return status;
    }
    status = cudaEventCreate(&stop);
    if(status == cudaSuccess)
    {
        kernel<<<blocks, kProbeThreads>>>(iterations, out);
        cudaEventRecord(start);
        kernel<<<blocks, kProbeThreads>>>(iterations, out);
        cudaEventRecord(stop);
        status = cudaEventSynchronize(stop);
        if(status == cudaSuccess)
        {
            status = cudaGetLastError();
        }
        if(status == cudaSuccess)
        {
            status = cudaEventElapsedTime(ms, start, stop);
        }
        cudaEventDestroy(stop);
    }
    cudaEventDestroy(start);
    return status;
}

}

extern "C"
{

    int fq_probe_fp64_mma(const int iterations, float* mma_ms, float* scalar_ms, float* copy_ms, double* flops,
                          double* copy_bytes, int* exact)
    {
        int device = 0;
        int sms = 0;
        int l2 = 0;
        cudaError_t status = cudaGetDevice(&device);
        if(status == cudaSuccess)
        {
            status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        }
        if(status == cudaSuccess)
        {
            status = cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device);
        }
        if(status != cudaSuccess)
        {
            return int(status);
        }
        const int blocks = 4 * sms;
        const int steps = iterations < 1 ? 1 : iterations;
        const long long min_bytes = 64LL << 20;
        const long long buffer_bytes = 4LL * l2 > min_bytes ? 4LL * l2 : min_bytes;
        const long long n = buffer_bytes / static_cast<long long>(sizeof(double2));
        double* out = nullptr;
        double2* copy_in = nullptr;
        double2* copy_out = nullptr;
        status = cudaMalloc(&out, sizeof(double) * blocks * kProbeThreads);
        if(status == cudaSuccess)
        {
            status = cudaMalloc(&copy_in, sizeof(double2) * n);
        }
        if(status == cudaSuccess)
        {
            status = cudaMalloc(&copy_out, sizeof(double2) * n);
        }
        if(status == cudaSuccess)
        {
            status = cudaMemset(copy_in, 0, sizeof(double2) * n);
        }
        if(status == cudaSuccess)
        {
            status = time_probe(probe_mma_kernel, blocks, steps, out, mma_ms);
        }
        if(status == cudaSuccess)
        {
            status = time_probe(probe_scalar_kernel, blocks, steps, out, scalar_ms);
        }
        if(status == cudaSuccess)
        {
            status = time_copy(32 * sms, copy_in, copy_out, n, copy_ms);
        }
        if(status == cudaSuccess)
        {
            const double warps = double(blocks) * (kProbeThreads / 32);
            *flops = warps * steps * kProbeChains * 512.0;
            *copy_bytes = 2.0 * double(n) * sizeof(double2);
            probe_mma_check_kernel<<<1, 32>>>(out);
            double host[128];
            status = cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost);
            if(status == cudaSuccess)
            {
                int same = 1;
                for(int lane = 0; lane < 32; ++lane)
                {
                    if(host[4 * lane] != host[4 * lane + 2] || host[4 * lane + 1] != host[4 * lane + 3])
                    {
                        same = 0;
                    }
                }
                *exact = same;
            }
        }
        cudaFree(copy_out);
        cudaFree(copy_in);
        cudaFree(out);
        return int(status);
    }
}
