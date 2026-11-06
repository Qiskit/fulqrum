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
#pragma once

namespace fq::device
{


template <int K>
struct RowAccumulator
{
    double sum[K];


    __device__ RowAccumulator()
    {
#pragma unroll
        for(int k = 0; k < K; k++)
        {
            sum[k] = 0.0;
        }
    }


    __device__ void add(const int k, const double element, const double x)
    {
        sum[k] = fma(element, x, sum[k]);
    }


    __device__ void flush(const int k, double* out) const
    {
        *out += sum[k];
    }
};

}
