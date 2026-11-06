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
#include <condition_variable>
#include <mutex>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct primme_params;
extern "C" int fq_primme_matvec_wait(primme_params* primme, int accept);

namespace
{

constexpr long long kRowBlock = 2048;
constexpr int kMaxCols = 64;
constexpr size_t kPageBytes = 4096;

struct Buffer
{
    double* ptr = nullptr;
    size_t bytes = 0;
    bool registered = false;
};

struct Pending
{
    bool valid = false;
    int col = 0;
    double c[kMaxCols] = {};
    double rho = 1.0;
};

constexpr int kChunks = 8;

int g_enabled = 0;
const double* g_diag = nullptr;


std::mutex g_mutex;
std::condition_variable g_cv;
long long g_rows_ready = 0;
long long g_chunk_rows = kRowBlock;
bool g_hold_output = false;
bool g_abort = false;
Buffer g_t, g_y;
int g_state = 0;
double g_theta = 0.0;
double g_eps = 0.0;
Pending g_pending;
long long g_started = 0;
long long g_taken = 0;
long long g_kept = 0;
long long g_discarded = 0;
long long g_fast = 0;
double g_pass_seconds = 0.0;
double g_window_seconds = 0.0;
double g_alloc_seconds = 0.0;
bool g_warned = false;

double seconds_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}


void first_touch(char* p, size_t bytes)
{
#pragma omp parallel
    {
        const int threads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const size_t block = (bytes + threads - 1) / threads;
        const size_t b0 = block * tid;
        const size_t b1 = b0 + block < bytes ? b0 + block : bytes;
        for(size_t b = b0; b < b1; b += kPageBytes)
        {
            p[b] = 0;
        }
    }
}

void release(Buffer& b)
{
    if(b.ptr)
    {
        if(b.registered)
        {
            cudaHostUnregister(b.ptr);
            cudaGetLastError();
        }
        std::free(b.ptr);
    }
    b = Buffer{};
}


int ensure(Buffer& b, size_t bytes)
{
    if(b.ptr && b.bytes >= bytes)
    {
        return 0;
    }
    release(b);
    char* p = static_cast<char*>(std::malloc(bytes));
    if(!p)
    {
        return 1;
    }
    first_touch(p, bytes);
    const cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterDefault);
    b.registered = e == cudaSuccess;
    if(!b.registered)
    {
        cudaGetLastError();
        if(!g_warned)
        {
            g_warned = true;
            std::fprintf(stderr,
                         "PRIMME speculative matvec: cudaHostRegister of %.2f GiB failed: %s; "
                         "the buffer stays in pageable memory\n",
                         bytes / double(1ULL << 30),
                         cudaGetErrorString(e));
            std::fflush(stderr);
        }
    }
    b.ptr = reinterpret_cast<double*>(p);
    b.bytes = bytes;
    return 0;
}


inline double clamp_denominator(double d, double shift)
{
    const double t = d - shift;
    double a = std::fabs(t);
    a = (a != a || a > 1e-10) ? a : 1e-10;
    return std::copysign(a, t);
}


void reduce_parts(const std::vector<double>& part, int nt, int n, double* out)
{
    for(int j = 0; j < n; j++)
    {
        double s = 0.0;
        for(int i = 0; i < nt; i++)
        {
            s += part[static_cast<size_t>(i) * n + j];
        }
        out[j] = s;
    }
}


void publish_rows(long long rows, bool abort)
{
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_rows_ready = rows;
        g_abort = g_abort || abort;
    }
    g_cv.notify_all();
}


void release_output()
{
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_hold_output = false;
    }
    g_cv.notify_all();
}

}


extern "C" void fq_primme_spec_set_enabled(int enabled, const double* diag)
{
    g_enabled = enabled ? 1 : 0;
    g_diag = diag;
    g_state = 0;
    g_pending.valid = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_rows_ready = 0;
        g_hold_output = false;
        g_abort = false;
    }
    g_started = g_taken = g_kept = g_discarded = g_fast = 0;
    g_pass_seconds = g_window_seconds = g_alloc_seconds = 0.0;
}


extern "C" int fq_primme_spec_enabled(void)
{
    return g_enabled;
}


extern "C" const double* fq_primme_spec_diag(void)
{
    return g_diag;
}


extern "C" void fq_primme_spec_release(void)
{
    publish_rows(0, true);
    release_output();
    release(g_t);
    release(g_y);
    g_state = 0;
    g_pending.valid = false;
    g_diag = nullptr;
}


extern "C" void fq_primme_spec_stats(long long* counts, double* seconds)
{
    counts[0] = g_started;
    counts[1] = g_taken;
    counts[2] = g_kept;
    counts[3] = g_discarded;
    counts[4] = g_fast;
    seconds[0] = g_pass_seconds;
    seconds[1] = g_window_seconds;
    seconds[2] = g_alloc_seconds;
}


extern "C" int fq_primme_spec_buffers(long long n, double** t, double** y)
{
    if(n <= 0)
    {
        return 1;
    }
    const auto started = std::chrono::steady_clock::now();
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    const int err = ensure(g_t, bytes) || ensure(g_y, bytes);
    g_alloc_seconds += seconds_since(started);
    if(err)
    {
        return 1;
    }
    *t = g_t.ptr;
    *y = g_y.ptr;
    return 0;
}


extern "C" int fq_primme_spec_pending(void)
{
    return g_pending.valid ? 1 : 0;
}


extern "C" int fq_primme_spec_set_pending(int col, const double* c, double rho)
{
    if(col < 0 || col > kMaxCols)
    {
        return 1;
    }
    g_pending.valid = true;
    g_pending.col = col;
    for(int j = 0; j < col; j++)
    {
        g_pending.c[j] = c[j];
    }
    g_pending.rho = rho;
    return 0;
}


extern "C" int fq_primme_spec_flush(double* W, long long ldW, long long m)
{
    if(!g_pending.valid)
    {
        release_output();
        return 0;
    }
    const auto started = std::chrono::steady_clock::now();
    const int n = g_pending.col;
    const double inv = 1.0 / g_pending.rho;
    const double* c = g_pending.c;
    const double* y = g_y.ptr;
    double* out = W + static_cast<long long>(n) * ldW;
#pragma omp parallel for schedule(static)
    for(long long r0 = 0; r0 < m; r0 += kRowBlock)
    {
        const long long r1 = r0 + kRowBlock < m ? r0 + kRowBlock : m;
        const long long nr = r1 - r0;
        double acc[kRowBlock];
        for(long long r = 0; r < nr; r++)
        {
            acc[r] = y[r0 + r];
        }
        for(int j = 0; j < n; j++)
        {
            const double* w = W + j * ldW + r0;
            const double cj = c[j];
#pragma omp simd
            for(long long r = 0; r < nr; r++)
            {
                acc[r] -= cj * w[r];
            }
        }
#pragma omp simd
        for(long long r = 0; r < nr; r++)
        {
            out[r0 + r] = acc[r] * inv;
        }
    }
    g_pending.valid = false;
    g_window_seconds += seconds_since(started);
    release_output();
    return 0;
}


extern "C" int fq_primme_spec_begin(long long m, int nV, long long ldV, long long ldW, double theta, double eps)
{
    if(m < 0 || nV < 1 || nV > kMaxCols || ldV < m || ldW < m || (m > 0 && !g_t.ptr) ||
       g_t.bytes < static_cast<size_t>(m) * sizeof(double) || g_state != 0)
    {
        return 1;
    }
    if(g_pending.valid && g_pending.col != nV - 1)
    {
        return 1;
    }
    long long chunk = (m + kChunks - 1) / kChunks;
    chunk = ((chunk + kRowBlock - 1) / kRowBlock) * kRowBlock;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_rows_ready = 0;
        g_abort = false;
        g_chunk_rows = chunk > 0 ? chunk : kRowBlock;
        g_hold_output = g_pending.valid;
    }
    g_state = 1;
    g_theta = theta;
    g_eps = eps;
    g_started++;
    return 0;
}


extern "C" void fq_primme_spec_cancel(void)
{
    if(g_state == 1)
    {
        g_state = 0;
        g_started--;
    }
    publish_rows(0, true);
    release_output();
}


extern "C" long long fq_primme_spec_chunk_rows(void)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_chunk_rows;
}


extern "C" int fq_primme_spec_wait_rows(long long rows)
{
    std::unique_lock<std::mutex> lock(g_mutex);
    g_cv.wait(lock, [rows] { return g_rows_ready >= rows || g_abort; });
    return g_rows_ready >= rows ? 0 : 1;
}


extern "C" void fq_primme_spec_wait_output_free(void)
{
    std::unique_lock<std::mutex> lock(g_mutex);
    g_cv.wait(lock, [] { return !g_hold_output || g_abort; });
}


extern "C" int fq_primme_spec_pass(long long m,
                                   int nV,
                                   const double* V,
                                   long long ldV,
                                   const double* W,
                                   long long ldW,
                                   const double* h,
                                   double theta,
                                   double eps)
{
    if(m < 0 || nV < 1 || nV > kMaxCols || ldV < m || ldW < m || (m > 0 && !g_t.ptr) ||
       g_t.bytes < static_cast<size_t>(m) * sizeof(double))
    {
        return 1;
    }
    const bool pend = g_pending.valid;
    if(pend && g_pending.col != nV - 1)
    {
        return 1;
    }
    const auto started = std::chrono::steady_clock::now();
    const int nW = pend ? nV - 1 : nV;
    double hv[kMaxCols];
    double hw[kMaxCols];
    for(int k = 0; k < nV; k++)
    {
        hv[k] = h[k];
        hw[k] = h[k];
    }
    double hy = 0.0;
    if(pend)
    {

        const double hl = h[nV - 1] / g_pending.rho;
        hy = hl;
        for(int k = 0; k < nW; k++)
        {
            hw[k] = h[k] - hl * g_pending.c[k];
        }
    }
    const double* y = g_y.ptr;
    const double* d = g_diag;
    double* t = g_t.ptr;
    const double sigma = theta;
    long long chunk;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        chunk = g_chunk_rows;
    }
    for(long long c0 = 0; c0 < m; c0 += chunk)
    {
        const long long c1 = c0 + chunk < m ? c0 + chunk : m;
#pragma omp parallel for schedule(static)
        for(long long r0 = c0; r0 < c1; r0 += kRowBlock)
        {
            const long long r1 = r0 + kRowBlock < c1 ? r0 + kRowBlock : c1;
            const long long nr = r1 - r0;
            double x[kRowBlock];
            double w[kRowBlock];
            if(pend)
            {
                const double* yr = y + r0;
#pragma omp simd
                for(long long r = 0; r < nr; r++)
                {
                    x[r] = 0.0;
                    w[r] = hy * yr[r];
                }
            }
            else
            {
                for(long long r = 0; r < nr; r++)
                {
                    x[r] = 0.0;
                    w[r] = 0.0;
                }
            }
            for(int k = 0; k < nV; k++)
            {
                const double* v = V + k * ldV + r0;
                const double hk = hv[k];
#pragma omp simd
                for(long long r = 0; r < nr; r++)
                {
                    x[r] += v[r] * hk;
                }
            }
            for(int k = 0; k < nW; k++)
            {
                const double* u = W + k * ldW + r0;
                const double hk = hw[k];
#pragma omp simd
                for(long long r = 0; r < nr; r++)
                {
                    w[r] += u[r] * hk;
                }
            }
            double* tr = t + r0;
            if(d)
            {
                const double* dr = d + r0;
#pragma omp simd
                for(long long r = 0; r < nr; r++)
                {
                    const double res = w[r] - sigma * x[r];
                    tr[r] = (res + (-eps) * x[r]) / clamp_denominator(dr[r], sigma);
                }
            }
            else
            {
#pragma omp simd
                for(long long r = 0; r < nr; r++)
                {
                    const double res = w[r] - sigma * x[r];
                    tr[r] = res + (-eps) * x[r];
                }
            }
        }
        publish_rows(c1, false);
    }
    if(m == 0)
    {
        publish_rows(0, false);
    }
    g_pass_seconds += seconds_since(started);
    return 0;
}


extern "C" int fq_primme_spec_state(double** t, double** y)
{
    if(t)
    {
        *t = g_state ? g_t.ptr : nullptr;
    }
    if(y)
    {
        *y = g_state ? g_y.ptr : nullptr;
    }
    return g_state;
}


extern "C" int fq_primme_spec_take(double shift, double eps, int precond)
{
    if(g_state != 1 || shift != g_theta || eps != g_eps || (precond != 0) != (g_diag != nullptr))
    {
        return 1;
    }
    g_state = 2;
    g_taken++;
    return 0;
}


extern "C" int fq_primme_spec_restore(double* x, long long m)
{
    if(g_state != 2 || !g_t.ptr || g_t.bytes < static_cast<size_t>(m) * sizeof(double))
    {
        return 1;
    }
    const auto started = std::chrono::steady_clock::now();
    const double* t = g_t.ptr;
#pragma omp parallel for schedule(static)
    for(long long r0 = 0; r0 < m; r0 += kRowBlock)
    {
        const long long r1 = r0 + kRowBlock < m ? r0 + kRowBlock : m;
        std::memcpy(x + r0, t + r0, sizeof(double) * (r1 - r0));
    }
    g_window_seconds += seconds_since(started);
    return 0;
}


extern "C" void fq_primme_spec_finished(int kept, int fast)
{
    if(g_state == 0)
    {
        return;
    }
    if(kept)
    {
        g_kept++;
        if(fast)
        {
            g_fast++;
        }
    }
    else
    {
        g_discarded++;
    }
    g_state = 0;
}


extern "C" int fq_primme_spec_drain(primme_params* primme)
{
    if(g_state == 0)
    {
        return 0;
    }


    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_abort = true;
        g_hold_output = false;
    }
    g_cv.notify_all();
    const int err = fq_primme_matvec_wait(primme, 0);
    fq_primme_spec_finished(0, 0);
    return err;
}


extern "C" int fq_primme_spec_cu(long long m,
                                 int n,
                                 const double* V,
                                 long long ldV,
                                 const double* W,
                                 long long ldW,
                                 const double* t,
                                 double* c,
                                 double* u)
{
    if(m < 0 || n < 1 || n > kMaxCols || ldV < m || ldW < m)
    {
        return 1;
    }
    const auto started = std::chrono::steady_clock::now();
    const int nt = omp_get_max_threads();
    const int stride = 2 * n + 1;
    std::vector<double> part(static_cast<size_t>(nt) * stride, 0.0);
#pragma omp parallel num_threads(nt)
    {
        double* acc = part.data() + static_cast<size_t>(omp_get_thread_num()) * stride;
#pragma omp for schedule(static)
        for(long long r0 = 0; r0 < m; r0 += kRowBlock)
        {
            const long long r1 = r0 + kRowBlock < m ? r0 + kRowBlock : m;
            const long long nr = r1 - r0;
            const double* tr = t + r0;
            for(int j = 0; j < n; j++)
            {
                const double* v = V + j * ldV + r0;
                const double* w = W + j * ldW + r0;
                double sv = 0.0, sw = 0.0;
#pragma omp simd reduction(+ : sv, sw)
                for(long long r = 0; r < nr; r++)
                {
                    sv += v[r] * tr[r];
                    sw += w[r] * tr[r];
                }
                acc[j] += sv;
                acc[n + 1 + j] += sw;
            }
            double st = 0.0;
#pragma omp simd reduction(+ : st)
            for(long long r = 0; r < nr; r++)
            {
                st += tr[r] * tr[r];
            }
            acc[n] += st;
        }
    }
    std::vector<double> total(static_cast<size_t>(stride));
    reduce_parts(part, nt, stride, total.data());
    for(int j = 0; j <= n; j++)
    {
        c[j] = total[static_cast<size_t>(j)];
    }
    for(int j = 0; j < n; j++)
    {
        u[j] = total[static_cast<size_t>(n + 1 + j)];
    }
    g_window_seconds += seconds_since(started);
    return 0;
}


extern "C" int fq_primme_spec_ortho(long long m,
                                    int n,
                                    const double* V,
                                    long long ldV,
                                    const double* t,
                                    const double* c,
                                    double rho,
                                    double* out)
{
    if(m < 0 || n < 1 || n > kMaxCols || ldV < m)
    {
        return 1;
    }
    const auto started = std::chrono::steady_clock::now();
    double cc[kMaxCols];
    for(int j = 0; j < n; j++)
    {
        cc[j] = c[j];
    }
    const double inv = 1.0 / rho;
#pragma omp parallel for schedule(static)
    for(long long r0 = 0; r0 < m; r0 += kRowBlock)
    {
        const long long r1 = r0 + kRowBlock < m ? r0 + kRowBlock : m;
        const long long nr = r1 - r0;
        double v[kRowBlock];
        for(long long r = 0; r < nr; r++)
        {
            v[r] = t[r0 + r];
        }
        for(int j = 0; j < n; j++)
        {
            const double* q = V + j * ldV + r0;
            const double cj = cc[j];
#pragma omp simd
            for(long long r = 0; r < nr; r++)
            {
                v[r] -= cj * q[r];
            }
        }
        double* o = out + r0;
#pragma omp simd
        for(long long r = 0; r < nr; r++)
        {
            o[r] = v[r] * inv;
        }
    }
    g_window_seconds += seconds_since(started);
    return 0;
}


extern "C" int fq_primme_spec_update(long long m,
                                     int n,
                                     const double* a,
                                     long long lda,
                                     const double* c,
                                     double rho,
                                     const double* y,
                                     double* out)
{
    if(m < 0 || n < 0 || n > kMaxCols || (n > 0 && lda < m))
    {
        return 1;
    }
    double cc[kMaxCols];
    for(int j = 0; j < n; j++)
    {
        cc[j] = c[j];
    }
    const double inv = 1.0 / rho;
#pragma omp parallel for schedule(static)
    for(long long r0 = 0; r0 < m; r0 += kRowBlock)
    {
        const long long r1 = r0 + kRowBlock < m ? r0 + kRowBlock : m;
        const long long nr = r1 - r0;
        double acc[kRowBlock];
        for(long long r = 0; r < nr; r++)
        {
            acc[r] = y[r0 + r];
        }
        for(int j = 0; j < n; j++)
        {
            const double* v = a + j * lda + r0;
            const double cj = cc[j];
#pragma omp simd
            for(long long r = 0; r < nr; r++)
            {
                acc[r] -= cj * v[r];
            }
        }
#pragma omp simd
        for(long long r = 0; r < nr; r++)
        {
            out[r0 + r] = acc[r] * inv;
        }
    }
    return 0;
}
