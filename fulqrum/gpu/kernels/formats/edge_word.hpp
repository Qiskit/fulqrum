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

namespace fq::edge_word
{


inline constexpr int kCrossPairShift = 32;


inline constexpr int kCrossSignShift = 48;


inline constexpr std::uint32_t kCrossMaxPair = 0xffffu;


inline constexpr std::uint32_t kSignedPairNegative = 0x80000000u;


constexpr std::uint64_t pack_cross(const std::uint32_t column, const std::uint32_t pair, const int sign)
{
    return std::uint64_t(column) | (std::uint64_t(pair & kCrossMaxPair) << kCrossPairShift) |
           (std::uint64_t(std::uint16_t(std::int16_t(sign))) << kCrossSignShift);
}


constexpr int cross_column(const std::uint64_t word)
{
    return int(word & 0xffffffffull);
}


constexpr int cross_pair(const std::uint64_t word)
{
    return int((word >> kCrossPairShift) & kCrossMaxPair);
}


constexpr double cross_sign(const std::uint64_t word)
{
    return double(std::int16_t(std::uint16_t(word >> kCrossSignShift)));
}


constexpr std::uint32_t pack_signed_pair(const std::uint32_t pair, const int sign)
{
    return pair | (sign < 0 ? kSignedPairNegative : 0u);
}


constexpr std::uint32_t signed_pair_pair(const std::uint32_t word)
{
    return word & ~kSignedPairNegative;
}


constexpr double signed_pair_sign(const std::uint32_t word)
{
    return (word & kSignedPairNegative) ? -1.0 : 1.0;
}

}
