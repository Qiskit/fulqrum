/**
 * This code is part of Fulqrum.
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

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../subspace.hpp"

inline constexpr std::uint32_t NO_GROUP = 0xffffffffu;

/**
 * Packs a single excitation's column, directed pair and signs into a 64-bit entry.
 *
 * @param column The target string index.
 * @param directed_pair 2 * pair index + direction (0 when the lower qubit is occupied).
 * @param own_sign The shared own-sector factor of the cross terms: 0, +1 or -1.
 * @param jw_sign The sign from the sector qubits between the two flips: +1 or -1.
 */
inline std::uint64_t pack_single(const std::uint32_t column,
                                 const std::uint32_t directed_pair,
                                 const int own_sign,
                                 const int jw_sign)
{
    return std::uint64_t(column) | (std::uint64_t(directed_pair & 0xffffu) << 32) |
           (std::uint64_t(static_cast<std::uint8_t>(own_sign)) << 48) |
           (std::uint64_t(static_cast<std::uint8_t>(jw_sign)) << 56);
}

/**
 * Returns the column of a single-excitation packed entry.
 */
inline std::uint32_t single_column(const std::uint64_t packed_entry)
{
    return static_cast<std::uint32_t>(packed_entry);
}

/**
 * Returns the directed pair of a single-excitation packed entry.
 */
inline std::uint32_t single_directed_pair(const std::uint64_t packed_entry)
{
    return static_cast<std::uint32_t>((packed_entry >> 32) & 0xffffu);
}

/**
 * Returns the own sign of a single-excitation packed entry as a real number.
 */
inline double single_own_sign(const std::uint64_t packed_entry)
{
    return static_cast<double>(static_cast<std::int8_t>(packed_entry >> 48));
}

/**
 * Returns the JW sign of a single-excitation packed entry as a real number.
 */
inline double single_jw_sign(const std::uint64_t packed_entry)
{
    return static_cast<double>(static_cast<std::int8_t>(packed_entry >> 56));
}

/**
 * Returns the string of one half of a tensor-product store, placed at its qubits in a
 * full-width row.
 *
 * @param store The tensor-product store.
 * @param half 0 for alpha (low bits), 1 for beta (high bits).
 * @param index The string index in that half.
 * @param row The full-width row, store.num_words() words. The function sets it.
 */
inline void embed_half_string(const HalfKeyView& store,
                              const int half,
                              const std::size_t index,
                              KeyWord* row)
{
    std::fill_n(row, store.num_words(), KeyWord(0));
    const std::size_t shift = half == 0 ? 0 : store.split_bit();
    or_shifted_key(store.half_key(half, index), store.half_words(half), shift, row, store.num_words());
}

/**
 * Returns the bit of a half string at a rank.
 *
 * @param key The half string words.
 * @param rank The bit position in the half string.
 */
inline bool half_bit(const KeyWord* key, const std::size_t rank)
{
    return (key[rank / KEY_WORD_BITS] >> (rank % KEY_WORD_BITS)) & 1u;
}

/**
 * Returns +1 or -1: the parity of the half string bits strictly between two ranks.
 *
 * @param key The half string words.
 * @param low The lower rank.
 * @param high The higher rank.
 */
inline int between_sign(const KeyWord* key, const std::size_t low, const std::size_t high)
{
    unsigned int count = 0;

    for(std::size_t rank = low + 1; rank < high; rank++)
    {
        count += half_bit(key, rank);
    }

    return (count & 1u) ? -1 : 1;
}
