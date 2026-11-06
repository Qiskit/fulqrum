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

namespace fq::group_key
{


inline constexpr int kPositionShift = 16;


inline constexpr std::uint32_t kEmptyKey2 = 0xffffffffu;


inline constexpr std::uint64_t kEmptyKey4 = 0xffffffffffffffffull;


inline constexpr std::int32_t kEmptyValue = -1;


inline constexpr int kGroupShift = 8;


inline constexpr std::uint32_t kRowIntLengthMask = 0xffu;


inline constexpr std::int64_t kMaxGroups = std::int64_t(1) << 24;


inline constexpr int kMaxRowIntLength = 4;


inline constexpr int kBitsetWordSlots2 = 32;


inline constexpr int kBitsetWordSlots4 = 64;


constexpr std::uint32_t pack_key2(const std::uint32_t p0, const std::uint32_t p1)
{
    return p0 | (p1 << kPositionShift);
}


constexpr std::uint64_t pack_key4(const std::uint32_t p0, const std::uint32_t p1, const std::uint32_t p2, const std::uint32_t p3)
{
    return std::uint64_t(p0) | (std::uint64_t(p1) << kPositionShift) | (std::uint64_t(p2) << (2 * kPositionShift)) |
           (std::uint64_t(p3) << (3 * kPositionShift));
}


constexpr std::int32_t pack_group_value(const std::uint32_t group, const std::uint32_t row_int_length)
{
    return std::int32_t((group << kGroupShift) | (row_int_length & kRowIntLengthMask));
}


constexpr int value_group(const std::int32_t value)
{
    return int(std::uint32_t(value) >> kGroupShift);
}


constexpr int value_row_int_length(const std::int32_t value)
{
    return int(std::uint32_t(value) & kRowIntLengthMask);
}


constexpr std::uint32_t row_int(const std::uint32_t bit0,
                                const std::uint32_t bit1,
                                const std::uint32_t bit2,
                                const std::uint32_t bit3,
                                const int length)
{
    const std::uint32_t bits = bit0 | (bit1 << 1u) | (bit2 << 2u) | (bit3 << 3u);
    return bits & ((std::uint32_t(1) << length) - 1u);
}


constexpr std::int64_t ladder_index(const std::int64_t group, const std::int64_t ladder_offset, const std::uint32_t row_int)
{
    return group * ladder_offset + std::int64_t(row_int);
}


constexpr std::uint32_t mix2(const std::uint32_t key)
{
    std::uint32_t k = key;
    k = ((k >> 16u) ^ k) * 0x45d9f3bu;
    k = ((k >> 16u) ^ k) * 0x45d9f3bu;
    return k ^ (k >> 16u);
}


constexpr std::uint64_t mix4(const std::uint64_t key)
{
    std::uint64_t k = key;
    k = (k ^ (k >> 30u)) * 0xbf58476d1ce4e5b9ull;
    k = (k ^ (k >> 27u)) * 0x94d049bb133111ebull;
    return k ^ (k >> 31u);
}


constexpr std::uint32_t first_slot2(const std::uint32_t key, const std::uint32_t mask)
{
    return mix2(key) & mask;
}


constexpr std::uint64_t first_slot4(const std::uint64_t key, const std::uint64_t mask)
{
    return mix4(key) & mask;
}


constexpr std::uint64_t next_slot(const std::uint64_t slot, const std::uint64_t mask)
{
    return (slot + 1u) & mask;
}


constexpr std::uint64_t bitset_word2(const std::uint64_t slot)
{
    return slot / kBitsetWordSlots2;
}


constexpr std::uint64_t bitset_word4(const std::uint64_t slot)
{
    return slot / kBitsetWordSlots4;
}


constexpr bool is_empty2(const std::uint32_t bitset_word, const std::uint64_t slot)
{
    return ((bitset_word >> (slot % kBitsetWordSlots2)) & 1u) != 0u;
}


constexpr bool is_empty4(const std::uint64_t bitset_word, const std::uint64_t slot)
{
    return ((bitset_word >> (slot % kBitsetWordSlots4)) & 1u) != 0u;
}

}
