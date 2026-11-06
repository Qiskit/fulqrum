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

#include <cstddef>
#include <cstdint>

namespace fq::clique_word
{


inline constexpr std::uint64_t kColumnMask = 0xffffffffull;


inline constexpr int kOrbitalShift = 32;


inline constexpr std::uint64_t kOrbitalMask = 0xffull;


inline constexpr std::uint64_t kIntersectionBit = 1ull << 47;


inline constexpr int kPhaseShift = 48;


inline constexpr std::uint64_t kSignBit = 1ull << 63;


inline constexpr int kRowShift = 32;


inline constexpr std::uint64_t kRowMask = 0xffffull;


inline constexpr int kBuckets = 7;


inline constexpr int kMaxMembers = 32;


inline constexpr int kWarpsPerBlock = 4;


inline constexpr std::size_t kMaxSharedBytes = 227 * 1024;


constexpr std::uint64_t pack_member(const std::uint32_t alpha, const std::uint32_t orbital, const bool intersection,
                                    const int phase)
{
    return std::uint64_t(alpha) | ((std::uint64_t(orbital) & kOrbitalMask) << kOrbitalShift) |
           (intersection ? kIntersectionBit : 0ull) | (std::uint64_t(std::uint16_t(std::int16_t(phase))) << kPhaseShift);
}


constexpr int member_alpha(const std::uint64_t member)
{
    return int(member & kColumnMask);
}


constexpr int member_orbital(const std::uint64_t member)
{
    return int((member >> kOrbitalShift) & kOrbitalMask);
}


constexpr bool member_is_intersection(const std::uint64_t member)
{
    return (member & kIntersectionBit) != 0;
}


constexpr int member_phase(const std::uint64_t member)
{
    return int(std::int16_t(std::uint16_t(member >> kPhaseShift)));
}


constexpr std::uint64_t gather_sign_bit(const std::uint64_t member, const std::uint64_t word)
{
    return (member ^ word) & kSignBit;
}


constexpr std::uint64_t output_sign_bit(const std::uint64_t member)
{
    return (member ^ (member_is_intersection(member) ? 0ull : kSignBit)) & kSignBit;
}


constexpr std::uint64_t pack_tile_word(const std::uint32_t column, const std::uint32_t row, const bool negative)
{
    return std::uint64_t(column) | ((std::uint64_t(row) & kRowMask) << kRowShift) | (negative ? kSignBit : 0ull);
}


constexpr int tile_word_column(const std::uint64_t word)
{
    return int(word & kColumnMask);
}


constexpr int tile_word_row(const std::uint64_t word)
{
    return int((word >> kRowShift) & kRowMask);
}


constexpr int bucket_members(const int bucket)
{
    return 8 + 4 * bucket;
}


constexpr int bucket_of(const int members)
{
    return members <= 8 ? 0 : (members - 8 + 3) / 4;
}


constexpr std::size_t shared_bytes(const int nmax, const int eb, const int tile_rows)
{
    return std::size_t(kWarpsPerBlock) * (std::size_t(tile_rows) * nmax + std::size_t(nmax) * eb) * sizeof(double);
}

}
