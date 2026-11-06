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

#include <cstdint>

#include "../formats/sell.hpp"

namespace fq::device
{


struct SellRow
{
    std::int64_t slice_offset;
    int row;
    int length;


    __device__ static SellRow at(const std::int64_t* __restrict__ offsets, const int* __restrict__ lengths, const int row)
    {
        return SellRow{std::int64_t(__ldg(offsets + fq::sell::slice_of(row))), row, __ldg(lengths + row)};
    }


    __device__ std::int64_t slot(const int k) const
    {
        return fq::sell::slot(slice_offset, row, k);
    }
};


template <typename Word>
struct EdgeStream
{
    const Word* __restrict__ words;
    SellRow row;
    int k;
    Word current;
    Word next;


    __device__ EdgeStream(const Word* __restrict__ word_table, const SellRow& sell_row)
        : words(word_table), row(sell_row), k(0),
          current(sell_row.length > 0 ? __ldg(word_table + sell_row.slot(0)) : Word(0)),
          next(sell_row.length > 1 ? __ldg(word_table + sell_row.slot(1)) : Word(0))
    {
    }


    __device__ bool valid() const
    {
        return k < row.length;
    }


    __device__ std::int64_t slot() const
    {
        return row.slot(k);
    }


    __device__ void advance()
    {
        k++;
        current = next;
        next = k + 1 < row.length ? __ldg(words + row.slot(k + 1)) : Word(0);
    }
};

}
