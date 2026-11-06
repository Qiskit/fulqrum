/**
 * This code is part of Fulqrum.
 *
 * (C) Copyright IBM 2024.
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

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "key_words.hpp"
#include "checked_size.hpp"
#include "qubit_term.hpp"

/**
 * Cached data for evaluating operator terms with bit operations.
 *
 * Each term contributes to <row|term|column>. Here row is the output bit string
 * (the bra), and column is row with the term's flip bits toggled. X, Y, + and -
 * flip bits; Z and the 0/1 projectors do not. Flip masks and column membership
 * are handled by callers, not stored in this structure.
 *
 * For term i, words contains three consecutive arrays of num_words KeyWords,
 * starting at i * 3 * num_words:
 * - mask: positions where a projector or ladder operator restricts the row.
 * - expect: required row bits at those positions; other bits are zero.
 * - parity: positions whose set row bits contribute a minus sign.
 *
 * A term is zero unless (row & mask) == expect in every word. Otherwise its
 * cached coefficient is multiplied by (-1)^popcount(row & parity). The cached
 * coefficient includes the term coefficient and the row-independent Y phase.
 * real_values follows Fulqrum's real_phase convention; complex_values retains
 * the complex coefficient and phase. Both arrays use the original term order.
 *
 * This structure owns its arrays. It must be rebuilt when term coefficients,
 * factors, ordering or the number of key words changes. Once built, it can be
 * read concurrently. Bit q is stored in word q / KEY_WORD_BITS, at position
 * q % KEY_WORD_BITS; word zero holds the lowest qubit indices.
 */
struct TermMasks
{
    std::size_t num_words = 0;
    std::vector<KeyWord> words;
    std::vector<double> real_values;
    std::vector<std::complex<double>> complex_values;
};

/**
 * Count the set bits in a 64-bit unsigned integer.
 *
 * Uses fixed-width arithmetic and bit masks, without allocating memory or
 * depending on a hardware population-count instruction.
 *
 * @param word Integer whose bits are counted.
 * @return Number of set bits, from 0 to 64.
 */
inline unsigned int count_bits(std::uint64_t word)
{
    word = word - ((word >> 1) & 0x5555555555555555ULL);
    word = (word & 0x3333333333333333ULL) + ((word >> 2) & 0x3333333333333333ULL);
    word = (word + (word >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return static_cast<unsigned int>((word * 0x0101010101010101ULL) >> 56);
}

/**
 * Precompute row constraints, parity masks and coefficients for every term.
 *
 * Z and Y add their positions to the parity mask. Projector 0 and ladder -
 * require row bit 0; projector 1 and ladder + require row bit 1. X imposes no
 * row constraint or sign. The ladder conventions are - = |0><1| and
 * + = |1><0|, so these constraints apply to the output row.
 *
 * Each Y contributes a constant factor -i to complex_values and a row-dependent
 * sign through parity. For example, a unit-coefficient Y gives <0|Y|1> = -i
 * and <1|Y|0> = +i. For n_y Y factors, the complex cached coefficient is
 * term.coeff * (-i)^n_y. The real cached coefficient is
 * term.real_phase * real(term.coeff) * (-1)^(n_y mod 2), matching the retained
 * real evaluator. It is not obtained by taking the real part of complex_values.
 *
 * Allocates 3 * num_words * terms.size() mask words and one real and one
 * complex coefficient per term. Size arithmetic is checked for overflow;
 * allocation failures propagate. Large term sets are processed with OpenMP.
 *
 * @param terms Valid, normalized terms with matching indices/values arrays and
 *     initialized real_phase metadata. Each qubit appears at most once per term,
 *     and its index must fit num_words * KEY_WORD_BITS.
 *     This function does not validate these input invariants.
 * @param num_words Number of KeyWords needed for a complete row, normally
 *     words_for_width(operator_width). Zero is valid for factor-free constants.
 * @return Owning cache in the same order as terms. No references to terms remain.
 */
inline TermMasks build_term_masks(const std::vector<OperatorTerm_t>& terms, const std::size_t num_words)
{
    TermMasks masks;
    masks.num_words = num_words;
    const std::size_t mask_words = checked_multiply(checked_multiply(3, num_words), terms.size());
    checked_add(checked_add(checked_multiply(mask_words, sizeof(KeyWord)), checked_multiply(terms.size(), sizeof(double))), checked_multiply(terms.size(), sizeof(std::complex<double>)));
    masks.words.assign(mask_words, 0);
    masks.real_values.assign(terms.size(), 0.0);
    masks.complex_values.assign(terms.size(), 0.0);
    const std::array<std::complex<double>, 4> minus_i_powers = {
        std::complex<double>(1, 0), std::complex<double>(0, -1), std::complex<double>(-1, 0), std::complex<double>(0, 1)};
#pragma omp parallel for schedule(static) if(terms.size() > 4096)
    for(std::size_t index = 0; index < terms.size(); index++)
    {
        const OperatorTerm_t& term = terms[index];
        KeyWord* mask = num_words ? masks.words.data() + 3 * num_words * index : nullptr;
        KeyWord* expect = num_words ? mask + num_words : nullptr;
        KeyWord* parity = num_words ? expect + num_words : nullptr;
        unsigned int num_y = 0;
        for(std::size_t factor = 0; factor < term.indices.size(); factor++)
        {
            const width_t qubit = term.indices[factor];
            const std::size_t word = qubit / KEY_WORD_BITS;
            const KeyWord bit = KeyWord(1) << (qubit % KEY_WORD_BITS);
            const unsigned char value = term.values[factor];
            if(value == OPER_VALUE_Z)
            {
                parity[word] |= bit;
            }
            else if(value == OPER_VALUE_Y)
            {
                parity[word] |= bit;
                num_y++;
            }
            else if(value == OPER_VALUE_PROJ_0 || value == OPER_VALUE_MINUS)
            {
                mask[word] |= bit;
            }
            else if(value == OPER_VALUE_PROJ_1 || value == OPER_VALUE_PLUS)
            {
                mask[word] |= bit;
                expect[word] |= bit;
            }
        }
        masks.real_values[index] = term.real_phase * term.coeff.real() * ((num_y & 1u) ? -1.0 : 1.0);
        masks.complex_values[index] = term.coeff * minus_i_powers[num_y % 4];
    }
    return masks;
}

/**
 * Evaluate and sum cached terms in the half-open range [begin, end) on an output row.
 *
 * Tests each term's row constraints, skips terms that fail them, and applies
 * the parity sign to the cached coefficient. Parity is combined across all
 * row words, so multiword bit strings follow the same rule as single-word rows.
 * The function allocates no memory and does not modify the cache or row.
 *
 * For a matrix element, callers supply a group whose terms share the same flip
 * mask, and determine column = row XOR flips separately. This function does
 * not check the column, subspace membership or whether terms share their flips.
 *
 * @tparam T double for Fulqrum's real evaluator, or std::complex<double> for
 *     complex evaluation. The choice selects real_values or complex_values.
 * @tparam Word Row storage word type, normally KeyWord, with the same bit layout.
 * @param masks Cache produced by build_term_masks, matching the current terms.
 * @param begin First term index, inclusive.
 * @param end Last term index, exclusive; begin <= end <= cached term count.
 * @param row Readable array of at least masks.num_words words. May be nullptr
 *     for an empty term range or when masks.num_words == 0.
 * @return Sum of the selected contributions; zero for an empty range or when
 *     all row constraints fail. Bounds and pointer validity are caller obligations.
 */
template <typename T, typename Word = KeyWord>
inline T term_sum(const TermMasks& masks, const std::size_t begin, const std::size_t end, const Word* row)
{
    if(begin == end)
    {
        return T(0);
    }
    const std::size_t num_words = masks.num_words;
    const std::size_t stride = 3 * num_words;
    const KeyWord* words = num_words ? masks.words.data() + begin * stride : nullptr;
    T total = T(0);
    for(std::size_t index = begin; index < end; index++, words = num_words ? words + stride : words)
    {
        KeyWord mismatch = 0;
        KeyWord parity = 0;
        for(std::size_t word = 0; word < num_words; word++)
        {
            mismatch |= (row[word] & words[word]) ^ words[num_words + word];
            parity ^= row[word] & words[2 * num_words + word];
        }
        if(mismatch != 0)
        {
            continue;
        }
        const bool odd = __builtin_parityll(parity);
        if constexpr(std::is_same_v<T, double>)
        {
            total += odd ? -masks.real_values[index] : masks.real_values[index];
        }
        else
        {
            total += odd ? -masks.complex_values[index] : masks.complex_values[index];
        }
    }
    return total;
}

/**
 * Allow a literal nullptr for constant-only caches or empty term ranges.
 *
 * Forwards to the pointer overload without allocating a dummy row. The same
 * preconditions apply: a nonempty range requires masks.num_words == 0.
 */
template <typename T>
inline T term_sum(const TermMasks& masks, const std::size_t begin, const std::size_t end, std::nullptr_t)
{
    return term_sum<T>(masks, begin, end, static_cast<const KeyWord*>(nullptr));
}


/**
 * A lookup-table representation of a group with at most four flipped positions.
 *
 * positions[0..num_flips) lists qubit indices in increasing order. Extracting
 * those row bits gives a small integer pattern: pattern bit i is row bit
 * positions[i]. values[pattern] stores the summed contribution of all terms,
 * including row constraints and parity signs at the flipped positions.
 * Only the first 2^num_flips entries are used; a zero-flip group uses entry zero.
 *
 * Parity outside the flipped positions is shared by all terms and applied
 * separately by compact_element. parity_offset is a word index into a caller-owned
 * parity_words vector, not a byte offset or pointer. This record does not own
 * that vector; callers must retain its mask segment while using the record.
 */
template <typename T>
struct CompactGroup
{
    std::array<width_t, 4> positions = {0, 0, 0, 0};
    std::uint32_t num_flips = 0;
    std::size_t parity_offset = 0;
    std::array<T, 16> values = {};
};

/**
 * Try to replace a term-group evaluation with a small row-pattern lookup table.
 *
 * A group qualifies when it is nonempty, has at most four flipped positions,
 * every term's row constraints lie within those positions, and all terms have
 * the same parity mask outside them. Inside the flipped positions, constraints
 * and parity may differ: they are folded into the lookup table by summing the
 * terms for each of the 2^num_flips possible row patterns.
 *
 * Refusal returns false without modifying record or parity_words; callers can
 * continue using term_sum. On success, appends num_words of shared outer parity
 * and stores their starting word index in record.parity_offset. Vector growth
 * can throw: record is assigned only after appending succeeds, but parity_words
 * may contain a partially appended segment if an allocation fails.
 *
 * @tparam T Same real or complex coefficient convention as term_sum<T>.
 * @param terms Cache for the operator's current term order.
 * @param term_begin First term of the group, inclusive.
 * @param term_end Last term of the group, exclusive; indices must be in bounds.
 * @param flips Readable array of terms.num_words words containing the group's
 *     common flip mask. All selected terms must share these flips; not verified here.
 * @param num_flips Exact number of set bits in flips. The caller must supply the
 *     correct count; the function does not check it against the mask.
 * @param record Receives the compact group on success; unchanged on refusal.
 * @param parity_words Owns shared parity-mask segments for compact records.
 * @return True if the compact representation was built; false if the group is
 *     empty, has more than four flips, or fails the constraint/parity conditions.
 */
template <typename T>
bool build_compact_group(const TermMasks& terms,
                         const std::size_t term_begin,
                         const std::size_t term_end,
                         const KeyWord* flips,
                         const std::uint32_t num_flips,
                         CompactGroup<T>& record,
                         std::vector<KeyWord>& parity_words)
{
    const std::size_t num_words = terms.num_words;
    if(num_flips > 4 || term_end <= term_begin)
    {
        return false;
    }
    CompactGroup<T> built;
    built.num_flips = num_flips;
    std::uint32_t found = 0;
    for(std::size_t word = 0; word < num_words; word++)
    {
        for(KeyWord bits = flips[word]; bits != 0; bits &= bits - 1)
        {
            built.positions[found++] = static_cast<width_t>(word * KEY_WORD_BITS + __builtin_ctzll(bits));
        }
    }
    auto pattern_bits = [&](const KeyWord* words) {
        unsigned int bits = 0;
        for(std::uint32_t index = 0; index < built.num_flips; index++)
        {
            const width_t qubit = built.positions[index];
            bits |= static_cast<unsigned int>((words[qubit / KEY_WORD_BITS] >> (qubit % KEY_WORD_BITS)) & 1u) << index;
        }
        return bits;
    };
    const std::size_t stride = 3 * num_words;
    const KeyWord* shared = terms.words.data() + term_begin * stride + 2 * num_words;
    for(std::size_t term = term_begin; term < term_end; term++)
    {
        const KeyWord* words = terms.words.data() + term * stride;
        for(std::size_t word = 0; word < num_words; word++)
        {
            const bool mask_outside = (words[word] & ~flips[word]) != 0;
            const bool parity_differs = ((words[2 * num_words + word] ^ shared[word]) & ~flips[word]) != 0;
            if(mask_outside || parity_differs)
            {
                return false;
            }
        }
        const unsigned int mask = pattern_bits(words);
        const unsigned int expect = pattern_bits(words + num_words);
        const unsigned int parity = pattern_bits(words + 2 * num_words);
        T magnitude;
        if constexpr(std::is_same_v<T, double>)
        {
            magnitude = terms.real_values[term];
        }
        else
        {
            magnitude = terms.complex_values[term];
        }
        for(unsigned int pattern = 0; pattern < (1u << built.num_flips); pattern++)
        {
            if((pattern & mask) == expect)
            {
                const bool odd = count_bits(pattern & parity) & 1u;
                built.values[pattern] += odd ? -magnitude : magnitude;
            }
        }
    }
    built.parity_offset = parity_words.size();
    for(std::size_t word = 0; word < num_words; word++)
    {
        parity_words.push_back(shared[word] & ~flips[word]);
    }
    record = built;
    return true;
}

/**
 * Evaluate a compact group's contribution on an output row.
 *
 * Extracts the row bits at record.positions to select values[pattern], then
 * applies the sign from the shared parity mask outside those positions. For a
 * successfully built record, this equals term_sum<T> for the original group.
 * Callers still determine the column and check subspace membership separately.
 * The function allocates no memory and does not modify its inputs.
 *
 * @param record Record produced by a successful build_compact_group<T> call.
 * @param parity_words Vector containing the record's shared parity segment at
 *     [record.parity_offset, record.parity_offset + num_words).
 * @param num_words Row word count used when building the record.
 * @param row Readable array of num_words KeyWords in the cache's bit layout.
 * @return The group's matrix-element contribution, including its parity sign.
 *     Record validity, array bounds and buffer lifetimes are caller obligations.
 */
template <typename T>
inline T compact_element(const CompactGroup<T>& record,
                         const std::vector<KeyWord>& parity_words,
                         const std::size_t num_words,
                         const KeyWord* row)
{
    unsigned int pattern = 0;
    for(std::uint32_t index = 0; index < record.num_flips; index++)
    {
        const width_t qubit = record.positions[index];
        pattern |= static_cast<unsigned int>((row[qubit / KEY_WORD_BITS] >> (qubit % KEY_WORD_BITS)) & 1u) << index;
    }
    const KeyWord* parity = parity_words.data() + record.parity_offset;
    KeyWord bits = 0;
    for(std::size_t word = 0; word < num_words; word++)
    {
        bits ^= row[word] & parity[word];
    }
    const T value = record.values[pattern];
    return __builtin_parityll(bits) ? -value : value;
}
