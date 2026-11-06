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
 * Modified: opt-in native memory reports and readable CUDA warnings.
 */


#include <cuda_runtime.h>

#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace
{

constexpr size_t kAlign = 2ULL << 20;
constexpr int kMaxRecords = 64;

struct Record
{
    void* ptr;
    size_t hbm_bytes;
};

Record g_records[kMaxRecords] = {};
long long g_budget = -1;

bool log_enabled()
{

    const char* value = std::getenv("FQ_PRIMME_MANAGED_BASIS_LOG");
    if(!value)
    {
        return false;
    }
    char normalized[16] = {};
    size_t n = 0;
    while(*value && std::isspace(static_cast<unsigned char>(*value)))
    {
        ++value;
    }
    while(*value && !std::isspace(static_cast<unsigned char>(*value)) && n < sizeof(normalized) - 1)
    {
        normalized[n++] = static_cast<char>(std::tolower(static_cast<unsigned char>(*value++)));
    }
    while(*value && std::isspace(static_cast<unsigned char>(*value)))
    {
        ++value;
    }
    if(*value)
    {
        return false;
    }
    return !std::strcmp(normalized, "1") || !std::strcmp(normalized, "true") ||
           !std::strcmp(normalized, "yes") || !std::strcmp(normalized, "on");
}

const char* rank_text()
{
    const char* rank = std::getenv("OMPI_COMM_WORLD_RANK");
    if(!rank)
    {
        rank = std::getenv("SLURM_PROCID");
    }
    return rank ? rank : "unavailable";
}

bool enabled()
{
    static int e = -1;
    if(e < 0)
    {
        const char* v = std::getenv("FQ_PRIMME_MANAGED_BASIS");
        e = (v && v[0] && v[0] != '0' && std::strcmp(v, "false") && std::strcmp(v, "no") &&
             std::strcmp(v, "off"))
                ? 1
                : 0;
    }
    return e == 1;
}

double env_gb(const char* name, double fallback)
{
    const char* v = std::getenv(name);
    if(!v || !v[0])
    {
        return fallback;
    }
    return std::atof(v);
}


cudaError_t advise(const void* p, size_t n, cudaMemoryAdvise adv, cudaMemLocation loc)
{
#if CUDART_VERSION >= 13000
    return cudaMemAdvise(p, n, adv, loc);
#elif CUDART_VERSION >= 12020
    return cudaMemAdvise_v2(p, n, adv, loc);
#else
    return cudaMemAdvise(
        p, n, adv, loc.type == cudaMemLocationTypeDevice ? loc.id : cudaCpuDeviceId);
#endif
}

cudaError_t prefetch(const void* p, size_t n, cudaMemLocation loc)
{
#if CUDART_VERSION >= 13000
    return cudaMemPrefetchAsync(p, n, loc, 0, 0);
#elif CUDART_VERSION >= 12020
    return cudaMemPrefetchAsync_v2(p, n, loc, 0, 0);
#else
    return cudaMemPrefetchAsync(
        p, n, loc.type == cudaMemLocationTypeDevice ? loc.id : cudaCpuDeviceId, 0);
#endif
}


bool warn(const char* what, cudaError_t e)
{
    if(e != cudaSuccess)
    {
        std::fprintf(stderr,
                     "PRIMME managed basis: CUDA operation %s failed: %s (%s)\n",
                     what,
                     cudaGetErrorName(e),
                     cudaGetErrorString(e));
        std::fflush(stderr);
        cudaGetLastError();
        return true;
    }
    return false;
}

}


extern "C" int
fq_primme_gpu_basis_malloc(void** x, size_t bytes, long long nLocal, int maxBasisSize)
{
    if(!enabled() || nLocal <= 0 || maxBasisSize <= 0)
    {
        return 1;
    }

    if(static_cast<long double>(bytes) <
       static_cast<long double>(nLocal) * maxBasisSize * sizeof(double))
    {
        return 1;
    }

    int dev = 0;
    if(warn("cudaGetDevice", cudaGetDevice(&dev)))
    {
        return 1;
    }
    if(g_budget < 0)
    {
        size_t free_b = 0, total_b = 0;
        if(warn("cudaMemGetInfo", cudaMemGetInfo(&free_b, &total_b)))
        {
            return 1;
        }
        const double vec_gb = static_cast<double>(nLocal) * sizeof(double) / (1ULL << 30);
        const double reserve = env_gb("FQ_PRIMME_HBM_RESERVE_GB", 4.0 + 3.0 * vec_gb);
        const double budget = env_gb("FQ_PRIMME_HBM_BUDGET_GB", -1.0);
        long long b = budget >= 0.0 ? static_cast<long long>(budget * (1ULL << 30))
                                    : static_cast<long long>(free_b) -
                                          static_cast<long long>(reserve * (1ULL << 30));
        g_budget = b > 0 ? b : 0;
        if(log_enabled())
        {
            std::fprintf(stderr,
                         "PRIMME managed basis budget: rank=%s, device=%d, free_gpu_memory=%.2f GiB, reserve=%.2f GiB, "
                         "basis_budget=%.2f GiB\n",
                         rank_text(),
                         dev,
                         free_b / double(1ULL << 30),
                         reserve,
                         g_budget / double(1ULL << 30));
        }
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
    if(slot < 0 || cudaMallocManaged(x, bytes, cudaMemAttachGlobal) != cudaSuccess)
    {
        cudaGetLastError();
        return 1;
    }

    size_t hbm = bytes < static_cast<size_t>(g_budget) ? bytes : static_cast<size_t>(g_budget);
    if(hbm < bytes)
    {
        hbm -= hbm % kAlign;
    }
    const size_t host = bytes - hbm;

    cudaMemLocation dloc = {};
    dloc.type = cudaMemLocationTypeDevice;
    dloc.id = dev;
    cudaMemLocation hloc = {};
    int numa = -1;
    if(cudaDeviceGetAttribute(&numa, cudaDevAttrHostNumaId, dev) != cudaSuccess)
    {
        cudaGetLastError();
        numa = -1;
    }
    hloc.type = numa >= 0 ? cudaMemLocationTypeHostNuma : cudaMemLocationTypeHost;
    hloc.id = numa >= 0 ? numa : 0;

    char* p = static_cast<char*>(*x);
    warn("SetAccessedBy", advise(p, bytes, cudaMemAdviseSetAccessedBy, dloc));
    if(hbm > 0)
    {
        warn("SetPreferredLocation(device)",
             advise(p, hbm, cudaMemAdviseSetPreferredLocation, dloc));
        warn("prefetch(device)", prefetch(p, hbm, dloc));
    }
    if(host > 0)
    {
        warn("SetPreferredLocation(host)",
             advise(p + hbm, host, cudaMemAdviseSetPreferredLocation, hloc));
        warn("prefetch(host)", prefetch(p + hbm, host, hloc));
    }
    if(warn("synchronize", cudaDeviceSynchronize()))
    {
        warn("cudaFree", cudaFree(*x));
        *x = nullptr;
        return 1;
    }

    g_budget -= static_cast<long long>(hbm);
    g_records[slot].ptr = *x;
    g_records[slot].hbm_bytes = hbm;
    if(log_enabled())
    {
        char numa_text[24] = "unavailable";
        if(numa >= 0)
        {
            std::snprintf(numa_text, sizeof(numa_text), "%d", numa);
        }
        std::fprintf(stderr,
                     "PRIMME managed basis allocated: rank=%s, total=%.2f GiB, gpu_preferred=%.2f GiB, "
                     "host_preferred=%.2f GiB, host_numa_node=%s, remaining_gpu_budget=%.2f GiB\n",
                     rank_text(),
                     bytes / double(1ULL << 30),
                     hbm / double(1ULL << 30),
                     host / double(1ULL << 30),
                     numa_text,
                     g_budget / double(1ULL << 30));
    }
    return 0;
}


extern "C" void fq_primme_gpu_basis_free(void* x)
{
    for(int i = 0; i < kMaxRecords; i++)
    {
        if(x && g_records[i].ptr == x)
        {
            g_budget += static_cast<long long>(g_records[i].hbm_bytes);
            g_records[i] = Record{};
            return;
        }
    }
}
