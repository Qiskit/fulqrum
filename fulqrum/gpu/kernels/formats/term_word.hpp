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

namespace fq::term_word
{


inline constexpr int kHeaderWords = 2;


inline constexpr int kHeaderCoefficientWord = 0;


inline constexpr int kHeaderMetaWord = 1;


inline constexpr std::uint64_t kMetaMask = 0xffffu;


inline constexpr int kMetaProjectorShift = 8;


inline constexpr std::uint32_t kFieldMask = 0xffu;


inline constexpr int kOperatorValueShift = 8;


inline constexpr int kOperatorBits = 16;


inline constexpr int kOperatorsPerWord = 4;


inline constexpr std::uint32_t kOperatorZ = 0;


inline constexpr std::uint32_t kOperatorProjector0 = 1;


inline constexpr std::uint32_t kOperatorProjector1 = 2;


inline constexpr std::uint32_t kOperatorX = 3;


inline constexpr std::uint32_t kOperatorY = 4;


inline constexpr std::uint32_t kOperatorMinus = 5;


inline constexpr std::uint32_t kOperatorPlus = 6;


inline constexpr std::uint32_t kOperatorCount = 7;


constexpr std::uint64_t pack_meta(const std::uint32_t num_projectors, const std::uint32_t num_operators)
{
    return (std::uint64_t(num_projectors & kFieldMask) << kMetaProjectorShift) | std::uint64_t(num_operators & kFieldMask);
}


constexpr int meta_num_operators(const std::uint64_t meta_word)
{
    return int((meta_word & kMetaMask) & kFieldMask);
}


constexpr int meta_num_projectors(const std::uint64_t meta_word)
{
    return int(((meta_word & kMetaMask) >> kMetaProjectorShift) & kFieldMask);
}


constexpr std::uint16_t pack_operator(const std::uint32_t position, const std::uint32_t value)
{
    return std::uint16_t(((value & kFieldMask) << kOperatorValueShift) | (position & kFieldMask));
}


constexpr std::uint32_t operator_position(const std::uint16_t word)
{
    return std::uint32_t(word) & kFieldMask;
}


constexpr std::uint32_t operator_value(const std::uint16_t word)
{
    return (std::uint32_t(word) >> kOperatorValueShift) & kFieldMask;
}


constexpr int packed_stride(const int stride)
{
    return (stride + kOperatorsPerWord - 1) / kOperatorsPerWord;
}


constexpr int packed_word_of(const int k)
{
    return k / kOperatorsPerWord;
}


constexpr std::uint64_t pack_operators(const std::uint16_t op0,
                                       const std::uint16_t op1,
                                       const std::uint16_t op2,
                                       const std::uint16_t op3)
{
    return std::uint64_t(op0) | (std::uint64_t(op1) << kOperatorBits) | (std::uint64_t(op2) << (2 * kOperatorBits)) |
           (std::uint64_t(op3) << (3 * kOperatorBits));
}


constexpr std::uint16_t operator_in_word(const std::uint64_t packed, const int k)
{
    return std::uint16_t(packed >> (kOperatorBits * (k % kOperatorsPerWord)));
}


constexpr std::int64_t projector_slot(const std::int64_t term, const int k, const int stride)
{
    return term * stride + k;
}


constexpr std::uint32_t string_bit(const std::uint64_t alpha,
                                   const std::uint64_t beta,
                                   const std::uint32_t position,
                                   const std::uint32_t num_alpha)
{
    return position < num_alpha ? std::uint32_t((alpha >> position) & 1u)
                                : std::uint32_t((beta >> (position - num_alpha)) & 1u);
}


inline constexpr std::int8_t kOperatorElements[kOperatorCount * 4] = {
    1, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0};


constexpr std::uint32_t operator_element_mask(const bool negative)
{
    std::uint32_t mask = 0;
    for(std::uint32_t i = 0; i < kOperatorCount * 4; i++)
    {
        if(negative ? kOperatorElements[i] < 0 : kOperatorElements[i] != 0)
        {
            mask |= std::uint32_t(1) << i;
        }
    }
    return mask;
}


inline constexpr std::uint32_t kNonzeroOperatorElements = operator_element_mask(false);


inline constexpr std::uint32_t kNegativeOperatorElements = operator_element_mask(true);


constexpr int operator_element(const std::uint32_t value, const std::uint32_t row_bit, const std::uint32_t col_bit)
{
    const std::uint32_t entry = 4u * value + 2u * row_bit + col_bit;
    if(((kNonzeroOperatorElements >> entry) & 1u) == 0u)
    {
        return 0;
    }
    return ((kNegativeOperatorElements >> entry) & 1u) != 0u ? -1 : 1;
}

}
