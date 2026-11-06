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

#include "../device/edge_stream.cuh"
#include "../device/row_block.cuh"
#include "../formats/edge_word.hpp"
#include "elements.hpp"

namespace fq::arms
{


__device__ inline void add_cross_row(fq::device::RowAccumulator<1>& sum,
                                     const fq::device::SellRow& alpha_row,
                                     const unsigned long long* __restrict__ words,
                                     const double* __restrict__ coefficients,
                                     const double beta_sign,
                                     const double* __restrict__ source)
{
    for(fq::device::EdgeStream<unsigned long long> edges(words, alpha_row); edges.valid(); edges.advance())
    {


        const int column = fq::edge_word::cross_column(edges.current);
        const double coefficient = __ldg(coefficients + fq::edge_word::cross_pair(edges.current));
        if(coefficient != 0.0 && column >= 0)
        {
            sum.add(0, cross_element(coefficient, fq::edge_word::cross_sign(edges.current), beta_sign),
                    __ldg(source + column));
        }
    }
}

}
