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

#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#    define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "constants.hpp"
#include <boost/dynamic_bitset.hpp>

using KeyWord = std::size_t;
using KeyBitset = boost::dynamic_bitset<KeyWord>;

inline constexpr unsigned int KEY_WORD_BITS = 8 * sizeof(KeyWord);
inline constexpr std::size_t MAX_KEY_WORDS = (MAX_WIDTH + KEY_WORD_BITS - 1) / KEY_WORD_BITS;
inline constexpr std::size_t KEY_NOT_FOUND = MAX_SIZE_T;

/**
 * Returns the number of words that hold a key of the given width.
 *
 * @param width The number of bits in the key.
 * @return The number of words, zero for width zero.
 */
inline std::size_t words_for_width(const std::size_t width)
{
    return width / KEY_WORD_BITS + (width % KEY_WORD_BITS != 0);
}

/**
 * Returns the number of OpenMP threads that a parallel region can use.
 *
 * @return One without OpenMP.
 */
inline int max_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/**
 * Compares two keys as unsigned numbers, from the highest word down.
 *
 * @param a The first key.
 * @param b The second key.
 * @param num_words The number of words in each key.
 * @return A negative value, zero or a positive value when a is less than,
 * equal to or greater than b.
 */
template <typename Word>
inline int compare_keys(const Word* a, const Word* b, const std::size_t num_words)
{
    for(std::size_t word = num_words; word-- > 0;)
    {
        if(a[word] != b[word])
        {
            return a[word] < b[word] ? -1 : 1;
        }
    }
    return 0;
}

/**
 * Returns the highest 64 bits of a key, the sort prefix.
 *
 * @param key The key words.
 * @param width The number of bits in the key.
 * @return Bits [width - 64, width) of the key, or the whole key when width <= 64.
 */
template <typename Word>
inline Word key_prefix(const Word* key, const std::size_t width)
{
    if(width <= KEY_WORD_BITS)
    {
        return width == 0 ? 0 : key[0];
    }
    const std::size_t first_bit = width - KEY_WORD_BITS;
    const std::size_t word = first_bit / KEY_WORD_BITS;
    const unsigned int shift = first_bit % KEY_WORD_BITS;
    if(shift == 0)
    {
        return key[word];
    }
    return (key[word] >> shift) | (key[word + 1] << (KEY_WORD_BITS - shift));
}

/**
 * Copies a range of bits of a key to the low bits of another key.
 *
 * @param source The source key.
 * @param first_bit The first bit of the range.
 * @param num_bits The number of bits in the range.
 * @param target The target key. The function sets every one of its words.
 * @param target_words The number of words in the target key.
 */
inline void copy_bit_range(const KeyWord* source,
                           const std::size_t first_bit,
                           const std::size_t num_bits,
                           KeyWord* target,
                           const std::size_t target_words)
{
    const std::size_t first_word = first_bit / KEY_WORD_BITS;
    const unsigned int shift = first_bit % KEY_WORD_BITS;
    const std::size_t source_end = words_for_width(first_bit + num_bits);
    for(std::size_t word = 0; word < target_words; word++)
    {
        const std::size_t low = first_word + word;
        KeyWord value = low < source_end ? source[low] >> shift : 0;
        if(shift != 0 && low + 1 < source_end)
        {
            value |= source[low + 1] << (KEY_WORD_BITS - shift);
        }
        target[word] = value;
    }
    const unsigned int tail_bits = num_bits % KEY_WORD_BITS;
    if(tail_bits != 0 && target_words > 0)
    {
        target[(num_bits - 1) / KEY_WORD_BITS] &= (KeyWord(1) << tail_bits) - 1;
    }
}

/**
 * Adds a key, shifted left by a number of bits, to another key with OR.
 *
 * @param source The source key.
 * @param source_words The number of words in the source key.
 * @param shift The number of bits to shift the source key.
 * @param target The target key.
 * @param target_words The number of words in the target key.
 */
inline void or_shifted_key(const KeyWord* source,
                           const std::size_t source_words,
                           const std::size_t shift,
                           KeyWord* target,
                           const std::size_t target_words)
{
    const std::size_t word_shift = shift / KEY_WORD_BITS;
    const unsigned int bit_shift = shift % KEY_WORD_BITS;
    for(std::size_t word = 0; word < source_words; word++)
    {
        const std::size_t low = word + word_shift;
        if(low < target_words)
        {
            target[low] |= source[word] << bit_shift;
        }
        if(bit_shift != 0 && low + 1 < target_words)
        {
            target[low + 1] |= source[word] >> (KEY_WORD_BITS - bit_shift);
        }
    }
}

/**
 * Returns the mask of the bits at and above the split bit in one word of a key.
 *
 * @param word The word number.
 * @param split_bit The split bit.
 * @return All ones above the split word, zero below it.
 */
inline KeyWord high_part_mask(const std::size_t word, const std::size_t split_bit)
{
    const std::size_t split_word = split_bit / KEY_WORD_BITS;
    if(word != split_word)
    {
        return word > split_word ? ~KeyWord(0) : 0;
    }
    return ~KeyWord(0) << (split_bit % KEY_WORD_BITS);
}

/**
 * Tells whether two keys have the same bits at and above the split bit.
 *
 * @param a The first key.
 * @param b The second key.
 * @param split_bit The split bit.
 * @param num_words The number of words in each key.
 * @return True when the high parts are equal.
 */
inline bool high_parts_equal(const KeyWord* a,
                             const KeyWord* b,
                             const std::size_t split_bit,
                             const std::size_t num_words)
{
    for(std::size_t word = split_bit / KEY_WORD_BITS; word < num_words; word++)
    {
        if(((a[word] ^ b[word]) & high_part_mask(word, split_bit)) != 0)
        {
            return false;
        }
    }
    return true;
}

/**
 * Mixes a 64-bit value into a well-spread hash value (SplitMix64 finalizer).
 *
 * @param value The input value.
 * @return The mixed value.
 */
inline std::uint64_t mix_bits(std::uint64_t value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

/**
 * Hashes the bits of a key at and above the split bit.
 *
 * @param key The key.
 * @param split_bit The split bit.
 * @param num_words The number of words in the key.
 * @return The hash value.
 */
inline std::uint64_t hash_high_part(const KeyWord* key,
                                    const std::size_t split_bit,
                                    const std::size_t num_words)
{
    std::uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for(std::size_t word = split_bit / KEY_WORD_BITS; word < num_words; word++)
    {
        hash = mix_bits(hash ^ (key[word] & high_part_mask(word, split_bit)));
    }
    return hash;
}

/**
 * Hashes all words of a key.
 *
 * @param key The key.
 * @param num_words The number of words in the key.
 * @return The hash value.
 */
inline std::uint64_t hash_key(const KeyWord* key, const std::size_t num_words)
{
    std::uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for(std::size_t word = 0; word < num_words; word++)
    {
        hash = mix_bits(hash ^ key[word]);
    }
    return hash;
}

/**
 * Returns the smallest power of two that is at least a minimum value and at least 2.
 *
 * @param minimum The minimum value.
 */
inline std::size_t power_of_two_at_least(const std::size_t minimum)
{
    std::size_t value = 2;
    while(value < minimum)
    {
        if(value > std::numeric_limits<std::size_t>::max() / 2)
            throw std::overflow_error("Hash capacity exceeds size_t.");
        value *= 2;
    }
    return value;
}
