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

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{

constexpr long long kTileRows = 1024;
constexpr int kMaxCols = 64;
constexpr int kMaxBasis = 256;

inline int imin(int a, int b)
{
    return a < b ? a : b;
}
inline int imax(int a, int b)
{
    return a > b ? a : b;
}


void tile_times_h(const double* A,
                  long long ldA,
                  long long i0,
                  long long m,
                  int nA,
                  const double* h,
                  int ldh,
                  int cb,
                  int nc,
                  double* out)
{
    std::memset(out, 0, sizeof(double) * m * nc);
    for(int k = 0; k < nA; k++)
    {
        const double* a = A + static_cast<long long>(k) * ldA + i0;
        for(int c = 0; c < nc; c++)
        {
            const double hk = h[k + static_cast<long long>(cb + c) * ldh];
            double* o = out + static_cast<long long>(c) * m;
#pragma omp simd
            for(long long r = 0; r < m; r++)
            {
                o[r] += a[r] * hk;
            }
        }
    }
}

double dot(const double* x, const double* y, long long m)
{
    double s = 0.0;
#pragma omp simd reduction(+ : s)
    for(long long r = 0; r < m; r++)
    {
        s += x[r] * y[r];
    }
    return s;
}

}


extern "C" int fq_primme_host_dense_update_f64(const double* V,
                                               const double* W,
                                               const double* BV,
                                               long long mV,
                                               int nV,
                                               long long ldV,
                                               const double* h,
                                               int ldh,
                                               const double* hVals,
                                               double* X0,
                                               int nX0b,
                                               int nX0e,
                                               long long ldX0,
                                               double* X1,
                                               int nX1b,
                                               int nX1e,
                                               long long ldX1,
                                               double* X2,
                                               int nX2b,
                                               int nX2e,
                                               long long ldX2,
                                               double* Wo,
                                               int nWob,
                                               int nWoe,
                                               long long ldWo,
                                               double* R,
                                               int nRb,
                                               int nRe,
                                               long long ldR,
                                               double* Rnorms,
                                               double* G0,
                                               int nG,
                                               int ldG0,
                                               double* H0,
                                               int nH,
                                               int ldH0,
                                               double* rnorms,
                                               int nrb,
                                               int nre,
                                               double* xnorms,
                                               int nxb,
                                               int nxe,
                                               long long tile_rows)
{
    if(BV != nullptr || nV <= 0 || mV <= 0 || nV > kMaxBasis)
    {
        return 1;
    }


    const bool gG = (G0 != nullptr && nG > 0);
    const bool gH = (H0 != nullptr && nH > 0);
    const bool gR = (R != nullptr && nRb < nRe);
    const bool grn = (rnorms != nullptr && nrb < nre);
    const bool gxn = (xnorms != nullptr && nxb < nxe);
    int nXb = INT_MAX, nXe = 0, nYb = INT_MAX, nYe = 0;
    if(X0 && nX0b < nX0e)
    {
        nXb = imin(nXb, nX0b), nXe = imax(nXe, nX0e);
    }
    if(X1 && nX1b < nX1e)
    {
        nXb = imin(nXb, nX1b), nXe = imax(nXe, nX1e);
    }
    if(X2 && nX2b < nX2e)
    {
        nXb = imin(nXb, nX2b), nXe = imax(nXe, nX2e);
    }
    if(gR)
    {
        nXb = imin(nXb, nRb), nXe = imax(nXe, nRe);
    }
    if(gG)
    {
        nXb = imin(nXb, 0), nXe = imax(nXe, nG);
    }
    if(gH)
    {
        nXb = imin(nXb, 0), nXe = imax(nXe, nH);
    }
    if(grn)
    {
        nXb = imin(nXb, nrb), nXe = imax(nXe, nre);
    }
    if(gxn)
    {
        nXb = imin(nXb, nxb), nXe = imax(nXe, nxe);
    }
    nXe = imax(nXe, nXb);
    if(Wo && nWob < nWoe)
    {
        nYb = imin(nYb, nWob), nYe = imax(nYe, nWoe);
    }
    if(gR)
    {
        nYb = imin(nYb, nRb), nYe = imax(nYe, nRe);
    }
    if(grn)
    {
        nYb = imin(nYb, nrb), nYe = imax(nYe, nre);
    }
    if(gH)
    {
        nYb = imin(nYb, 0), nYe = imax(nYe, nH);
    }
    nYe = imax(nYe, nYb);
    const int ncX = nXe - nXb;
    const int ncY = nYe - nYb;
    const int ncR = gR ? nRe - nRb : 0;
    const int nrn = grn ? nre - nrb : 0;
    const int nxn = gxn ? nxe - nxb : 0;
    if(((gG || gH) && nXb != 0) || (gH && nYb != 0) || ncX > kMaxCols || ncY > kMaxCols)
    {
        return 1;
    }

    const long long T = tile_rows > 0 ? tile_rows : kTileRows;
    const int nt = omp_get_max_threads();

    const long long stride = (gG ? static_cast<long long>(nG) * nG : 0) +
                             (gH ? static_cast<long long>(nH) * nH : 0) + ncR + nrn + nxn;
    std::vector<double> part(static_cast<size_t>(nt) * (stride > 0 ? stride : 1), 0.0);

#pragma omp parallel num_threads(nt)
    {
        double* acc = part.data() + static_cast<size_t>(omp_get_thread_num()) * stride;
        double* aG = acc;
        double* aH = aG + (gG ? static_cast<long long>(nG) * nG : 0);
        double* aR = aH + (gH ? static_cast<long long>(nH) * nH : 0);
        double* arn = aR + ncR;
        double* axn = arn + nrn;
        std::vector<double> Xt(static_cast<size_t>(T) * (ncX > 0 ? ncX : 1));
        std::vector<double> Yt(static_cast<size_t>(T) * (ncY > 0 ? ncY : 1));

#pragma omp for schedule(static)
        for(long long i0 = 0; i0 < mV; i0 += T)
        {
            const long long m = i0 + T < mV ? T : mV - i0;
            double* x = Xt.data();
            double* y = Yt.data();

            if(ncX > 0)
            {
                tile_times_h(V, ldV, i0, m, nV, h, ldh, nXb, ncX, x);
            }
            if(ncY > 0)
            {
                tile_times_h(W, ldV, i0, m, nV, h, ldh, nYb, ncY, y);
            }

            auto put = [&](double* dst, int nb, int ne, long long ld, const double* src, int sb) {
                if(!dst)
                {
                    return;
                }
                for(int c = nb; c < ne; c++)
                {
                    std::memcpy(dst + static_cast<long long>(c - nb) * ld + i0,
                                src + static_cast<long long>(c - sb) * m,
                                sizeof(double) * m);
                }
            };
            put(X0, nX0b, nX0e, ldX0, x, nXb);
            put(X1, nX1b, nX1e, ldX1, x, nXb);
            put(X2, nX2b, nX2e, ldX2, x, nXb);
            if(Wo && nWob < nWoe)
            {
                put(Wo, nWob, nWoe, ldWo, y, nYb);
            }

            if(gG)
            {
                for(int b = 0; b < nG; b++)
                {
                    for(int a = 0; a <= b; a++)
                    {
                        aG[a + static_cast<long long>(b) * nG] +=
                            dot(x + static_cast<long long>(a) * m, x + static_cast<long long>(b) * m, m);
                    }
                }
            }
            if(gH)
            {
                for(int b = 0; b < nH; b++)
                {
                    for(int a = 0; a < nH; a++)
                    {
                        aH[a + static_cast<long long>(b) * nH] +=
                            dot(x + static_cast<long long>(a) * m, y + static_cast<long long>(b) * m, m);
                    }
                }
            }
            for(int j = 0; j < nxn; j++)
            {
                const double* xc = x + static_cast<long long>(nxb + j - nXb) * m;
                axn[j] += dot(xc, xc, m);
            }

            for(int c = 0; c < ncR; c++)
            {
                const double* yc = y + static_cast<long long>(nRb - nYb + c) * m;
                const double* xc = x + static_cast<long long>(nRb - nXb + c) * m;
                const double hv = hVals[nRb + c];
                double* rc = R + static_cast<long long>(c) * ldR + i0;
                double s = 0.0;
#pragma omp simd reduction(+ : s)
                for(long long r = 0; r < m; r++)
                {
                    const double v = yc[r] - hv * xc[r];
                    rc[r] = v;
                    s += v * v;
                }
                aR[c] += s;
            }
            for(int c = 0; c < nrn; c++)
            {
                double* yc = y + static_cast<long long>(nrb - nYb + c) * m;
                const double* xc = x + static_cast<long long>(nrb - nXb + c) * m;
                const double hv = hVals[nrb + c];
                double s = 0.0;
#pragma omp simd reduction(+ : s)
                for(long long r = 0; r < m; r++)
                {
                    const double v = yc[r] - hv * xc[r];
                    yc[r] = v;
                    s += v * v;
                }
                arn[c] += s;
            }
        }
    }


    auto total = [&](long long off) {
        double s = 0.0;
        for(int t = 0; t < nt; t++)
        {
            s += part[static_cast<size_t>(t) * stride + off];
        }
        return s;
    };
    long long off = 0;
    if(gG)
    {
        for(int b = 0; b < nG; b++)
        {
            for(int a = 0; a <= b; a++)
            {
                const double s = total(off + a + static_cast<long long>(b) * nG);
                G0[a + static_cast<long long>(b) * ldG0] = s;
                G0[b + static_cast<long long>(a) * ldG0] = s;
            }
        }
        off += static_cast<long long>(nG) * nG;
    }
    if(gH)
    {
        for(int b = 0; b < nH; b++)
        {
            for(int a = 0; a < nH; a++)
            {
                H0[a + static_cast<long long>(b) * ldH0] = total(off + a + static_cast<long long>(b) * nH);
            }
        }
        off += static_cast<long long>(nH) * nH;
    }
    for(int c = 0; c < ncR; c++)
    {
        if(Rnorms)
        {
            Rnorms[c] = total(off + c);
        }
    }
    off += ncR;
    for(int c = 0; c < nrn; c++)
    {
        rnorms[c] = total(off + c);
    }
    off += nrn;
    for(int j = 0; j < nxn; j++)
    {
        xnorms[j] = total(off + j);
    }
    return 0;
}


extern "C" void fq_primme_host_clamped_jacobi(
    const double* d, double shift, const double* x, double* y, long long n)
{
#pragma omp parallel for simd schedule(static)
    for(long long i = 0; i < n; i++)
    {
        const double t = d[i] - shift;
        double a = std::fabs(t);
        a = (a != a || a > 1e-10) ? a : 1e-10;
        y[i] = x[i] / std::copysign(a, t);
    }
}
