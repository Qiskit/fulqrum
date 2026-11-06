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


#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#define FQ_GPU_DENSE_ABI 2

namespace
{

thread_local char g_last_error[512] = "";


void record_error(const char* call, const char* file, int line, const char* name, const char* text)
{
    if(g_last_error[0] != '\0')
    {
        return;
    }
    snprintf(g_last_error,
             sizeof(g_last_error),
             "%s:%d: %s failed: %s (%s)",
             file,
             line,
             call,
             name,
             text);
}


bool failed(cudaError_t status, const char* call, const char* file, int line)
{
    if(status == cudaSuccess)
    {
        return false;
    }
    record_error(call, file, line, cudaGetErrorName(status), cudaGetErrorString(status));
    return true;
}


bool failed(cublasStatus_t status, const char* call, const char* file, int line)
{
    if(status == CUBLAS_STATUS_SUCCESS)
    {
        return false;
    }
    record_error(call, file, line, cublasGetStatusName(status), cublasGetStatusString(status));
    return true;
}


class DeviceFreeGuard
{
public:
    explicit DeviceFreeGuard(std::initializer_list<double**> buffers)
        : buffers_(buffers.begin(), buffers.end())
    {}
    DeviceFreeGuard(const DeviceFreeGuard&) = delete;
    DeviceFreeGuard& operator=(const DeviceFreeGuard&) = delete;
    ~DeviceFreeGuard()
    {
        for(double** buffer : buffers_)
        {
            if(*buffer != nullptr)
            {
                failed(cudaFree(*buffer), "cudaFree", __FILE__, __LINE__);
                *buffer = nullptr;
            }
        }
    }

private:
    std::vector<double**> buffers_;
};

}

#define FQ_CHECK(call)                                                                             \
    do                                                                                             \
    {                                                                                              \
        if(failed((call), #call, __FILE__, __LINE__))                                              \
        {                                                                                          \
            return -1;                                                                             \
        }                                                                                          \
    } while(0)


extern "C" const char* fq_primme_gpu_last_error()
{
    return g_last_error;
}


extern "C" void fq_primme_gpu_last_error_clear()
{
    g_last_error[0] = '\0';
}

static cublasHandle_t g_handle = nullptr;

static int ensure_handle()
{
    if(!g_handle)
    {
        if(failed(cublasCreate(&g_handle), "cublasCreate", __FILE__, __LINE__))
        {
            g_handle = nullptr;
            return -1;
        }
        FQ_CHECK(cublasSetPointerMode(g_handle, CUBLAS_POINTER_MODE_HOST));
    }
    return 0;
}

extern "C" int fq_primme_gpu_dense_abi()
{
    return FQ_GPU_DENSE_ABI;
}


extern "C" void fq_primme_gpu_dense_teardown()
{
    if(g_handle)
    {
        failed(cublasDestroy(g_handle), "cublasDestroy", __FILE__, __LINE__);
        g_handle = nullptr;
    }
}

static inline int imin(int a, int b)
{
    return a < b ? a : b;
}
static inline int imax(int a, int b)
{
    return a > b ? a : b;
}


extern "C" int fq_primme_dense_update_f64(const double* V,
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


    if(BV != nullptr)
    {
        return 1;
    }
    if(nV <= 0 || mV <= 0)
    {
        return 1;
    }
    if(ldV > (long long)INT_MAX)
    {
        return 1;
    }


    if(ensure_handle() != 0)
    {
        return -1;
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
    const int ncR = gR ? (nRe - nRb) : 0;

    if((gG || gH) && nXb != 0)
    {
        return 1;
    }
    if(gH && nYb != 0)
    {
        return 1;
    }


    {

        long long ws_cols = (long long)ncX + (long long)ncY + (long long)ncR +
                            ((ncX > 0 || ncY > 0) ? (long long)nV : 0);
        if(ws_cols < 1)
        {
            ws_cols = 1;
        }
        if(tile_rows <= 0)
        {
            long long budget_mb = 1024;
            const char* e = getenv("FQ_PRIMME_GPU_DENSE_WS_MB");
            if(e && e[0])
            {
                long long v = atoll(e);
                if(v > 0)
                {
                    budget_mb = v;
                }
            }
            tile_rows = (budget_mb << 20) / (ws_cols * 8);
            if(tile_rows < (1LL << 20))
            {
                tile_rows = 1LL << 20;
            }
        }
        if(tile_rows > mV)
        {
            tile_rows = mV;
        }
    }

    const double one = 1.0, zero = 0.0;


    double *dX = nullptr, *dY = nullptr, *dR = nullptr, *dG = nullptr, *dH = nullptr,
           *dVt = nullptr;
    const DeviceFreeGuard guard{&dX, &dY, &dR, &dG, &dH, &dVt};

    if(ncX > 0)
    {
        FQ_CHECK(cudaMalloc(&dX, sizeof(double) * tile_rows * ncX));
    }
    if(ncY > 0)
    {
        FQ_CHECK(cudaMalloc(&dY, sizeof(double) * tile_rows * ncY));
    }
    if(ncR > 0)
    {
        FQ_CHECK(cudaMalloc(&dR, sizeof(double) * tile_rows * ncR));
    }
    if(gG)
    {
        FQ_CHECK(cudaMalloc(&dG, sizeof(double) * nG * nG));
    }
    if(gH)
    {
        FQ_CHECK(cudaMalloc(&dH, sizeof(double) * nH * nH));
    }


    if(ncX > 0 || ncY > 0)
    {
        FQ_CHECK(cudaMalloc(&dVt, sizeof(double) * tile_rows * nV));
    }


    if(Rnorms)
    {
        for(int c = 0; c < ncR; c++)
        {
            Rnorms[c] = 0.0;
        }
    }
    if(grn)
    {
        for(int c = 0; c < (nre - nrb); c++)
        {
            rnorms[c] = 0.0;
        }
    }
    if(gxn)
    {
        for(int c = 0; c < (nxe - nxb); c++)
        {
            xnorms[c] = 0.0;
        }
    }


    bool first = true;
    for(long long i0 = 0; i0 < mV; i0 += tile_rows)
    {
        const int m = (int)imin((long long)tile_rows, mV - i0);
        const int ldx = m, ldy = m, ldr = m;


        if(ncX > 0)
        {
            FQ_CHECK(cudaMemcpy2D(dVt,
                                  sizeof(double) * m,
                                  V + i0,
                                  sizeof(double) * ldV,
                                  sizeof(double) * m,
                                  nV,
                                  cudaMemcpyHostToDevice));
            FQ_CHECK(cublasDgemm(g_handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m,
                                 ncX,
                                 nV,
                                 &one,
                                 dVt,
                                 m,
                                 h + (long long)nXb * ldh,
                                 ldh,
                                 &zero,
                                 dX,
                                 ldx));
        }

        if(ncY > 0)
        {
            FQ_CHECK(cudaMemcpy2D(dVt,
                                  sizeof(double) * m,
                                  W + i0,
                                  sizeof(double) * ldV,
                                  sizeof(double) * m,
                                  nV,
                                  cudaMemcpyHostToDevice));
            FQ_CHECK(cublasDgemm(g_handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m,
                                 ncY,
                                 nV,
                                 &one,
                                 dVt,
                                 m,
                                 h + (long long)nYb * ldh,
                                 ldh,
                                 &zero,
                                 dY,
                                 ldy));
        }


        auto copy_sub = [&](double* dst, int nb, int ne, long long ld) -> cudaError_t {
            if(!dst)
            {
                return cudaSuccess;
            }
            const int cols = ne - nb;
            if(cols <= 0)
            {
                return cudaSuccess;
            }
            cudaError_t e = cudaMemcpy2D(dst + i0,
                                         sizeof(double) * ld,
                                         dX + (long long)(nb - nXb) * ldx,
                                         sizeof(double) * ldx,
                                         sizeof(double) * m,
                                         cols,
                                         cudaMemcpyDeviceToHost);
            return e;
        };
        FQ_CHECK(copy_sub(X0, nX0b, nX0e, ldX0));
        FQ_CHECK(copy_sub(X1, nX1b, nX1e, ldX1));
        FQ_CHECK(copy_sub(X2, nX2b, nX2e, ldX2));


        if(Wo && nWob < nWoe)
        {
            const int cols = nWoe - nWob;
            FQ_CHECK(cudaMemcpy2D(Wo + i0,
                                  sizeof(double) * ldWo,
                                  dY + (long long)(nWob - nYb) * ldy,
                                  sizeof(double) * ldy,
                                  sizeof(double) * m,
                                  cols,
                                  cudaMemcpyDeviceToHost));
        }

        const double beta = first ? 0.0 : 1.0;

        if(gG)
        {
            FQ_CHECK(cublasDgemm(g_handle,
                                 CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 nG,
                                 nG,
                                 m,
                                 &one,
                                 dX,
                                 ldx,
                                 dX,
                                 ldx,
                                 &beta,
                                 dG,
                                 nG));
        }

        if(gH)
        {
            FQ_CHECK(cublasDgemm(g_handle,
                                 CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 nH,
                                 nH,
                                 m,
                                 &one,
                                 dX,
                                 ldx,
                                 dY,
                                 ldy,
                                 &beta,
                                 dH,
                                 nH));
        }


        if(gxn)
        {
            for(int j = nxb; j < nxe; j++)
            {
                double s = 0.0;
                FQ_CHECK(cublasDdot(g_handle,
                                    m,
                                    dX + (long long)(j - nXb) * ldx,
                                    1,
                                    dX + (long long)(j - nXb) * ldx,
                                    1,
                                    &s));
                xnorms[j - nxb] += s;
            }
        }


        if(gR)
        {
            for(int c = 0; c < ncR; c++)
            {
                const long long yc = (long long)(nRb - nYb + c) * ldy;
                const long long xc = (long long)(nRb - nXb + c) * ldx;
                const long long rc2 = (long long)c * ldr;
                FQ_CHECK(cublasDcopy(g_handle, m, dY + yc, 1, dR + rc2, 1));
                const double neg = -hVals[nRb + c];
                FQ_CHECK(cublasDaxpy(g_handle, m, &neg, dX + xc, 1, dR + rc2, 1));
                if(Rnorms)
                {
                    double s = 0.0;
                    FQ_CHECK(cublasDdot(g_handle, m, dR + rc2, 1, dR + rc2, 1, &s));
                    Rnorms[c] += s;
                }
            }
            FQ_CHECK(cudaMemcpy2D(R + i0,
                                  sizeof(double) * ldR,
                                  dR,
                                  sizeof(double) * ldr,
                                  sizeof(double) * m,
                                  ncR,
                                  cudaMemcpyDeviceToHost));
        }


        if(grn)
        {
            for(int c = 0; c < (nre - nrb); c++)
            {
                const long long yc = (long long)(nrb - nYb + c) * ldy;
                const long long xc = (long long)(nrb - nXb + c) * ldx;
                const double neg = -hVals[nrb + c];
                FQ_CHECK(cublasDaxpy(g_handle, m, &neg, dX + xc, 1, dY + yc, 1));
                double s = 0.0;
                FQ_CHECK(cublasDdot(g_handle, m, dY + yc, 1, dY + yc, 1, &s));
                rnorms[c] += s;
            }
        }

        first = false;
    }


    if(gG)
    {
        FQ_CHECK(cudaMemcpy2D(G0,
                              sizeof(double) * ldG0,
                              dG,
                              sizeof(double) * nG,
                              sizeof(double) * nG,
                              nG,
                              cudaMemcpyDeviceToHost));
    }
    if(gH)
    {
        FQ_CHECK(cudaMemcpy2D(H0,
                              sizeof(double) * ldH0,
                              dH,
                              sizeof(double) * nH,
                              sizeof(double) * nH,
                              nH,
                              cudaMemcpyDeviceToHost));
    }

    FQ_CHECK(cudaDeviceSynchronize());
    return 0;
}


extern "C" int fq_primme_ortho_kernel_f64(const double* Q,
                                          long long M,
                                          int nQ,
                                          long long ldQ,
                                          double* V,
                                          int b1,
                                          int b2,
                                          long long ldV,
                                          const double* A,
                                          int ldA,
                                          const double* Y,
                                          int ldY,
                                          int Yortho,
                                          double* Bo,
                                          int ldBo,
                                          int has_DY,
                                          int has_WB,
                                          long long tile_rows)
{


    if(Yortho != 0 && Yortho != 1)
    {
        return 1;
    }
    if(ldV > (long long)INT_MAX || ldQ > (long long)INT_MAX)
    {
        return 1;
    }
    const int nX = b2 - b1;
    if(nX <= 0 || M <= 0 || b2 <= 0)
    {
        return 1;
    }
    if(has_DY && Y == nullptr)
    {
        return 1;
    }
    if(has_DY && (nQ + b1) > 0 && A == nullptr)
    {
        return 1;
    }
    if(has_WB && Bo == nullptr)
    {
        return 1;
    }
    if(!has_DY && !has_WB)
    {
        return 1;
    }
    if(nQ < 0 || b1 < 0)
    {
        return 1;
    }
    if(ensure_handle() != 0)
    {
        return -1;
    }
    if(tile_rows <= 0)
    {
        tile_rows = 1LL << 26;
    }
    if(tile_rows > M)
    {
        tile_rows = M;
    }

    const int nProj = nQ + b1;
    const int nBo = nQ + b2;
    const double one = 1.0, negone = -1.0, zero = 0.0;

    double *dQ = nullptr, *dV = nullptr, *dA = nullptr, *dY = nullptr, *dXo = nullptr,
           *dBo = nullptr;
    const DeviceFreeGuard guard{&dQ, &dV, &dA, &dY, &dXo, &dBo};

    if(nQ > 0)
    {
        FQ_CHECK(cudaMalloc(&dQ, sizeof(double) * tile_rows * nQ));
    }
    FQ_CHECK(cudaMalloc(&dV, sizeof(double) * tile_rows * b2));
    if(has_DY)
    {

        if(Yortho == 1)
        {
            FQ_CHECK(cudaMalloc(&dXo, sizeof(double) * tile_rows * nX));
        }
        FQ_CHECK(cudaMalloc(&dY, sizeof(double) * nX * nX));
        FQ_CHECK(cudaMemcpy2D(dY,
                              sizeof(double) * nX,
                              Y,
                              sizeof(double) * ldY,
                              sizeof(double) * nX,
                              nX,
                              cudaMemcpyHostToDevice));
        if(nProj > 0)
        {
            FQ_CHECK(cudaMalloc(&dA, sizeof(double) * nProj * nX));
            FQ_CHECK(cudaMemcpy2D(dA,
                                  sizeof(double) * nProj,
                                  A,
                                  sizeof(double) * ldA,
                                  sizeof(double) * nProj,
                                  nX,
                                  cudaMemcpyHostToDevice));
        }
    }
    if(has_WB)
    {
        FQ_CHECK(cudaMalloc(&dBo, sizeof(double) * nBo * nX));
    }

    bool first = true;
    for(long long i0 = 0; i0 < M; i0 += tile_rows)
    {
        const int m = (int)(M - i0 < tile_rows ? M - i0 : tile_rows);
        const int ldx = m;
        double* Xdev = dV + (long long)b1 * ldx;


        if(nQ > 0)
        {
            FQ_CHECK(cudaMemcpy2D(dQ,
                                  sizeof(double) * ldx,
                                  Q + i0,
                                  sizeof(double) * ldQ,
                                  sizeof(double) * m,
                                  nQ,
                                  cudaMemcpyHostToDevice));
        }
        FQ_CHECK(cudaMemcpy2D(dV,
                              sizeof(double) * ldx,
                              V + i0,
                              sizeof(double) * ldV,
                              sizeof(double) * m,
                              b2,
                              cudaMemcpyHostToDevice));

        if(has_DY)
        {

            if(nQ > 0)
            {
                FQ_CHECK(cublasDgemm(g_handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     m,
                                     nX,
                                     nQ,
                                     &negone,
                                     dQ,
                                     ldx,
                                     dA,
                                     nProj,
                                     &one,
                                     Xdev,
                                     ldx));
            }

            if(b1 > 0)
            {
                FQ_CHECK(cublasDgemm(g_handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     m,
                                     nX,
                                     b1,
                                     &negone,
                                     dV,
                                     ldx,
                                     dA + nQ,
                                     nProj,
                                     &one,
                                     Xdev,
                                     ldx));
            }
            if(Yortho == 1)
            {

                FQ_CHECK(cublasDgemm(g_handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     m,
                                     nX,
                                     nX,
                                     &one,
                                     Xdev,
                                     ldx,
                                     dY,
                                     nX,
                                     &zero,
                                     dXo,
                                     ldx));
                FQ_CHECK(cudaMemcpy(
                    Xdev, dXo, sizeof(double) * (size_t)m * nX, cudaMemcpyDeviceToDevice));
            }
            else
            {

                FQ_CHECK(cublasDtrsm(g_handle,
                                     CUBLAS_SIDE_RIGHT,
                                     CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_N,
                                     CUBLAS_DIAG_NON_UNIT,
                                     m,
                                     nX,
                                     &one,
                                     dY,
                                     nX,
                                     Xdev,
                                     ldx));
            }

            FQ_CHECK(cudaMemcpy2D(V + i0 + (long long)b1 * ldV,
                                  sizeof(double) * ldV,
                                  Xdev,
                                  sizeof(double) * ldx,
                                  sizeof(double) * m,
                                  nX,
                                  cudaMemcpyDeviceToHost));
        }

        if(has_WB)
        {
            const double beta = first ? 0.0 : 1.0;

            if(nQ > 0)
            {
                FQ_CHECK(cublasDgemm(g_handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     nQ,
                                     nX,
                                     m,
                                     &one,
                                     dQ,
                                     ldx,
                                     Xdev,
                                     ldx,
                                     &beta,
                                     dBo,
                                     nBo));
            }

            FQ_CHECK(cublasDgemm(g_handle,
                                 CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 b2,
                                 nX,
                                 m,
                                 &one,
                                 dV,
                                 ldx,
                                 Xdev,
                                 ldx,
                                 &beta,
                                 dBo + nQ,
                                 nBo));
        }
        first = false;
    }

    if(has_WB)
    {
        FQ_CHECK(cudaMemcpy2D(Bo,
                              sizeof(double) * ldBo,
                              dBo,
                              sizeof(double) * nBo,
                              sizeof(double) * nBo,
                              nX,
                              cudaMemcpyDeviceToHost));
    }

    FQ_CHECK(cudaDeviceSynchronize());
    return 0;
}
