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


#include <omp.h>

#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{

constexpr long long kRowBlock = 2048;
constexpr long long kMaxCols = 64;
constexpr long long kMinRows = 1LL << 20;


void gemv_t(long long m, long long n, const double* a, long long lda, const double* x,
            double* out)
{
    const int nt = omp_get_max_threads();
    std::vector<double> part(static_cast<size_t>(nt) * n, 0.0);
#pragma omp parallel num_threads(nt)
    {
        double* acc = part.data() + static_cast<size_t>(omp_get_thread_num()) * n;
#pragma omp for schedule(static)
        for(long long r0 = 0; r0 < m; r0 += kRowBlock)
        {
            const long long r1 = r0 + kRowBlock < m ? r0 + kRowBlock : m;
            for(long long j = 0; j < n; j++)
            {
                const double* v = a + j * lda;
                double s = 0.0;
#pragma omp simd reduction(+ : s)
                for(long long r = r0; r < r1; r++)
                {
                    s += v[r] * x[r];
                }
                acc[j] += s;
            }
        }
    }
    for(long long j = 0; j < n; j++)
    {
        double s = 0.0;
        for(int t = 0; t < nt; t++)
        {
            s += part[static_cast<size_t>(t) * n + j];
        }
        out[j] = s;
    }
}

}


extern "C" int fq_primme_host_gemv_f64(int trans,
                                       long long m,
                                       long long n,
                                       double alpha,
                                       const double* a,
                                       long long lda,
                                       const double* x,
                                       long long incx,
                                       double beta,
                                       double* y,
                                       long long incy)
{
    if(m < kMinRows || n < 1 || n > kMaxCols || lda < m)
    {
        return 1;
    }

    if(trans)
    {
        if(incx != 1)
        {
            return 1;
        }
        double dots[kMaxCols];
        gemv_t(m, n, a, lda, x, dots);
        for(long long j = 0; j < n; j++)
        {
            double* yj = y + j * incy;
            *yj = (beta == 0.0) ? alpha * dots[j] : alpha * dots[j] + beta * (*yj);
        }
        return 0;
    }

    if(incy != 1)
    {
        return 1;
    }
    double c[kMaxCols];
    for(long long j = 0; j < n; j++)
    {
        c[j] = x[j * incx];
    }
#pragma omp parallel for schedule(static)
    for(long long r0 = 0; r0 < m; r0 += kRowBlock)
    {
        const long long r1 = r0 + kRowBlock < m ? r0 + kRowBlock : m;
        const long long nr = r1 - r0;
        double acc[kRowBlock];
        for(long long r = 0; r < nr; r++)
        {
            acc[r] = 0.0;
        }
        for(long long j = 0; j < n; j++)
        {
            const double* v = a + j * lda + r0;
            const double cj = c[j];
#pragma omp simd
            for(long long r = 0; r < nr; r++)
            {
                acc[r] += cj * v[r];
            }
        }
        double* yr = y + r0;
        if(beta == 0.0)
        {
#pragma omp simd
            for(long long r = 0; r < nr; r++)
            {
                yr[r] = alpha * acc[r];
            }
        }
        else
        {
#pragma omp simd
            for(long long r = 0; r < nr; r++)
            {
                yr[r] = alpha * acc[r] + beta * yr[r];
            }
        }
    }
    return 0;
}
