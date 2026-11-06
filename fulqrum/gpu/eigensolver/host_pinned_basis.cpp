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


#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace
{

constexpr int kMaxRecords = 64;
constexpr size_t kPageBytes = 4096;

struct Record
{
    void* ptr;
    size_t bytes;
    bool registered;
};

Record g_records[kMaxRecords] = {};
int g_enabled = 0;
bool g_warned = false;
double g_register_seconds = 0.0;
double g_registered_bytes = 0.0;
int g_registered_arrays = 0;
int g_failed_arrays = 0;


void first_touch(char* p, size_t bytes, long long n_local)
{
    const size_t col_bytes = static_cast<size_t>(n_local) * sizeof(double);
    const size_t cols = bytes / col_bytes;
#pragma omp parallel
    {
        const int threads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const size_t block = (col_bytes + threads - 1) / threads;
        const size_t r0 = block * tid;
        const size_t r1 = r0 + block < col_bytes ? r0 + block : col_bytes;
        for(size_t c = 0; c < cols; c++)
        {
            char* col = p + c * col_bytes;
            for(size_t b = r0; b < r1; b += kPageBytes)
            {
                col[b] = 0;
            }
        }
        if(tid == 0)
        {
            for(size_t b = cols * col_bytes; b < bytes; b += kPageBytes)
            {
                p[b] = 0;
            }
        }
    }
}


void warn_once(cudaError_t e, size_t bytes)
{
    cudaGetLastError();
    if(g_warned)
    {
        return;
    }
    g_warned = true;
    std::fprintf(stderr,
                 "PRIMME page-locked basis: cudaHostRegister of %.2f GiB failed: %s (%s); "
                 "the basis stays in pageable memory\n",
                 bytes / double(1ULL << 30),
                 cudaGetErrorName(e),
                 cudaGetErrorString(e));
    std::fflush(stderr);
}

}


extern "C" void fq_primme_host_pinned_set(int enabled)
{
    g_enabled = enabled ? 1 : 0;
    g_register_seconds = 0.0;
    g_registered_bytes = 0.0;
    g_registered_arrays = 0;
    g_failed_arrays = 0;
}


extern "C" void fq_primme_host_pinned_stats(double* seconds, double* gib, int* registered, int* failed)
{
    *seconds = g_register_seconds;
    *gib = g_registered_bytes / double(1ULL << 30);
    *registered = g_registered_arrays;
    *failed = g_failed_arrays;
}


extern "C" int fq_primme_host_basis_malloc(void** x, size_t bytes, long long nLocal, int maxBasisSize)
{
    if(!g_enabled || nLocal <= 0 || maxBasisSize <= 0)
    {
        return 1;
    }
    if(static_cast<long double>(bytes) < static_cast<long double>(nLocal) * maxBasisSize * sizeof(double))
    {
        return 1;
    }
    int slot = -1;
    for(int i = 0; i < kMaxRecords; i++)
    {
        if(!g_records[i].ptr)
        {
            slot = i;
            break;
        }
    }
    if(slot < 0)
    {
        return 1;
    }
    const auto started = std::chrono::steady_clock::now();
    char* p = static_cast<char*>(std::malloc(bytes));
    if(!p)
    {
        return 1;
    }
    first_touch(p, bytes, nLocal);
    const cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterDefault);
    const bool registered = e == cudaSuccess;
    if(registered)
    {
        g_registered_bytes += static_cast<double>(bytes);
        g_registered_arrays++;
    }
    else
    {
        g_failed_arrays++;
        warn_once(e, bytes);
    }
    g_register_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    g_records[slot] = Record{p, bytes, registered};
    *x = p;
    return 0;
}


extern "C" int fq_primme_host_basis_free(void* x)
{
    if(!x)
    {
        return 1;
    }
    for(int i = 0; i < kMaxRecords; i++)
    {
        if(g_records[i].ptr == x)
        {
            if(g_records[i].registered)
            {
                cudaHostUnregister(x);
                cudaGetLastError();
            }
            std::free(x);
            g_records[i] = Record{};
            return 0;
        }
    }
    return 1;
}
