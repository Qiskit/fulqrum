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

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "key_words.hpp"
#include "checked_size.hpp"

inline constexpr unsigned int POSITION_BITS = 40;
inline constexpr std::uint64_t POSITION_MASK = (std::uint64_t(1) << POSITION_BITS) - 1;
inline constexpr std::size_t PRESENCE_BITS_PER_KEY = 8;
inline constexpr unsigned int PRESENCE_SHIFT = 17;

/**
 * An open-addressing hash table from keys to their positions in a flat key array.
 *
 * Each slot holds the top bits of the key hash and the position plus one, so most
 * failed lookups do not read the key array. A presence bitmap with about
 * PRESENCE_BITS_PER_KEY bits per key rejects most absent keys before the slot array.
 */
class KeyPositionTable
{
public:
    /**
     * Builds the table for a flat key array.
     *
     * @param keys The key words, num_words per key.
     * @param num_keys The number of keys.
     * @param num_words The number of words in each key.
     * @throws std::length_error If there are 2^40 - 1 keys or more.
     */
    void build(const KeyWord* keys, const std::size_t num_keys, const std::size_t num_words)
    {
        if(num_keys >= POSITION_MASK)
        {
            throw std::length_error("A subspace can hold at most 2^40 - 2 bit-strings.");
        }
        const std::size_t capacity = power_of_two_at_least(checked_add(num_keys, num_keys / 2));
        checked_multiply(num_keys, num_words);
        const std::size_t presence_bits = power_of_two_at_least(checked_multiply(PRESENCE_BITS_PER_KEY, num_keys));
        checked_add(checked_multiply(capacity, sizeof(std::uint64_t)), checked_multiply(words_for_width(presence_bits), sizeof(KeyWord)));
        slots_.assign(capacity, 0);
        mask_ = capacity - 1;
        presence_.assign(words_for_width(presence_bits), 0);
        presence_mask_ = presence_bits - 1;
        for(std::size_t position = 0; position < num_keys; position++)
        {
            const std::uint64_t hash = hash_key(keys + position * num_words, num_words);
            const std::size_t presence_bit = (hash >> PRESENCE_SHIFT) & presence_mask_;
            presence_[presence_bit / KEY_WORD_BITS] |= KeyWord(1) << (presence_bit % KEY_WORD_BITS);
            std::size_t slot = hash & mask_;
            while(slots_[slot] != 0)
            {
                slot = (slot + 1) & mask_;
            }
            slots_[slot] = ((hash >> POSITION_BITS) << POSITION_BITS) | (position + 1);
        }
    }

    /**
     * Returns the position of a key.
     *
     * @param keys The flat key array that the table was built for.
     * @param num_words The number of words in each key.
     * @param key The key words.
     * @return The position, or KEY_NOT_FOUND.
     */
    std::size_t find(const KeyWord* keys, const std::size_t num_words, const KeyWord* key) const
    {
        if(slots_.empty())
        {
            return KEY_NOT_FOUND;
        }
        const std::uint64_t hash = hash_key(key, num_words);
        const std::size_t presence_bit = (hash >> PRESENCE_SHIFT) & presence_mask_;
        if(((presence_[presence_bit / KEY_WORD_BITS] >> (presence_bit % KEY_WORD_BITS)) & 1) == 0)
        {
            return KEY_NOT_FOUND;
        }
        const std::uint64_t tag = hash >> POSITION_BITS;
        std::size_t slot = hash & mask_;
        while(slots_[slot] != 0)
        {
            const std::uint64_t entry = slots_[slot];
            if((entry >> POSITION_BITS) == tag)
            {
                const std::size_t position = (entry & POSITION_MASK) - 1;
                if(compare_keys(keys + position * num_words, key, num_words) == 0)
                {
                    return position;
                }
            }
            slot = (slot + 1) & mask_;
        }
        return KEY_NOT_FOUND;
    }

private:
    std::vector<std::uint64_t> slots_;
    std::size_t mask_ = 0;
    std::vector<KeyWord> presence_;
    std::size_t presence_mask_ = 0;
};
