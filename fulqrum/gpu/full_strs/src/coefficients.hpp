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

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../../core/src/base.hpp"
#include "../../../core/src/element_evaluator.hpp"
#include "../../../core/src/qubit_oper.hpp"
#include "keys.hpp"

namespace fq::full_strs
{


enum MoveClass : int
{
    kAlphaSingle = 0,
    kAlphaDouble = 1,
    kBetaSingle = 2,
    kBetaDouble = 3,
    kMixed = 4,
    kNumClasses = 5
};


inline constexpr std::array<std::int8_t, 16> kDoubleSlot = {-1, -1, -1, 0, -1, 1, 2, -1,
                                                            -1, 2, 1, -1, 0, -1, -1, -1};


inline constexpr std::array<std::int8_t, 16> kMixedSlot = {-1, -1, -1, -1, -1, 0, 1, -1,
                                                           -1, 1, 0, -1, -1, -1, -1, -1};


inline constexpr double kTableTolerance = 1e-14;


inline constexpr int kVerifySamples = 6;


inline std::int64_t binomial(const std::int64_t m, const int k)
{
    if(m < k)
    {
        return 0;
    }
    std::int64_t value = 1;
    for(int j = 0; j < k; j++)
    {
        value = value * (m - j) / (j + 1);
    }
    return value;
}


inline std::int64_t pair_rank(const int p, const int q)
{
    return static_cast<std::int64_t>(q) * (q - 1) / 2 + p;
}


inline std::int64_t quad_rank(const int* positions)
{
    return binomial(positions[3], 4) + binomial(positions[2], 3) + binomial(positions[1], 2) + positions[0];
}


inline std::pair<int, int> pair_orbitals(const std::int64_t rank)
{
    int q = 1;
    while(static_cast<std::int64_t>(q) * (q + 1) / 2 <= rank)
    {
        q++;
    }
    return {static_cast<int>(rank - static_cast<std::int64_t>(q) * (q - 1) / 2), q};
}


inline void quad_orbitals(std::int64_t rank, int* positions)
{
    for(int k = 4; k >= 1; k--)
    {
        int z = k - 1;
        while(binomial(z + 1, k) <= rank)
        {
            z++;
        }
        positions[k - 1] = z;
        rank -= binomial(z, k);
    }
}


template <std::size_t W>
inline bool has_orbital(const Word* words, const int orbital)
{
    return ((words[orbital / kWordBits] >> (orbital % kWordBits)) & 1u) != 0;
}


template <std::size_t W, typename OutWord>
inline void set_orbital(OutWord* words, const int orbital)
{
    words[orbital / kWordBits] |= Word(1) << (orbital % kWordBits);
}


template <std::size_t W>
inline int list_orbitals(const Word* words, int* out, const int max_count)
{
    int count = 0;
    for(std::size_t word = 0; word < W; word++)
    {
        Word bits = words[word];
        while(bits != 0)
        {
            if(count < max_count)
            {
                out[count] = static_cast<int>(word) * kWordBits + __builtin_ctzll(bits);
            }
            count++;
            bits &= bits - 1;
        }
    }
    return count;
}


template <std::size_t W>
inline bool between_parity(const Word* words, const int lo, const int hi)
{
    unsigned int count = 0;
    for(std::size_t word = 0; word < W; word++)
    {
        const int first = static_cast<int>(word) * kWordBits;
        const int begin = std::max(lo + 1, first);
        const int end = std::min(hi, first + kWordBits);
        if(begin >= end)
        {
            continue;
        }
        const int width = end - begin;
        const Word mask = (width == kWordBits ? ~Word(0) : (Word(1) << width) - 1) << (begin - first);
        count += static_cast<unsigned int>(__builtin_popcountll(words[word] & mask));
    }
    return (count & 1u) != 0;
}


template <typename T>
inline T conjugate(const T& value)
{
    if constexpr(std::is_same_v<T, double>)
    {
        return value;
    }
    else
    {
        return std::conj(value);
    }
}


template <typename T>
inline bool values_agree(const T& a, const T& b, const double scale)
{
    const double size = std::max({1.0, std::abs(a), std::abs(b), scale});
    return std::abs(a - b) <= kTableTolerance * size;
}


template <std::size_t W>
struct OperatorTerms
{
    int num_orbitals = 0;
    TermMasks offdiag;
    TermMasks diagonal;
    std::vector<std::int64_t> group_start;
    std::vector<std::int32_t> group_class;
    std::vector<std::int64_t> group_index;


    std::int64_t num_groups() const
    {
        return static_cast<std::int64_t>(group_class.size());
    }


    std::int64_t find_group(const int cls, const std::int64_t index) const
    {
        std::int64_t lo = 0;
        std::int64_t hi = num_groups();
        while(lo < hi)
        {
            const std::int64_t mid = lo + (hi - lo) / 2;
            const int mid_class = group_class[static_cast<std::size_t>(mid)];
            const std::int64_t mid_index = group_index[static_cast<std::size_t>(mid)];
            if(mid_class < cls || (mid_class == cls && mid_index < index))
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        if(lo < num_groups() && group_class[static_cast<std::size_t>(lo)] == cls &&
           group_index[static_cast<std::size_t>(lo)] == index)
        {
            return lo;
        }
        return -1;
    }


    template <typename T>
    T group_value(const std::int64_t g, const Word* ket) const
    {
        return term_sum<T>(offdiag,
                           static_cast<std::size_t>(group_start[static_cast<std::size_t>(g)]),
                           static_cast<std::size_t>(group_start[static_cast<std::size_t>(g) + 1]),
                           ket);
    }
};


template <std::size_t W, typename OutWord>
inline void qubit_to_sector_key(const KeyWord* qubit_words,
                                const std::size_t num_qubit_words,
                                const int num_orbitals,
                                OutWord* out)
{
    std::fill(out, out + 2 * W, Word(0));
    for(std::size_t word = 0; word < num_qubit_words; word++)
    {
        Word bits = qubit_words[word];
        while(bits != 0)
        {
            const int qubit = static_cast<int>(word) * kWordBits + __builtin_ctzll(bits);
            const int sector = qubit >= num_orbitals ? 1 : 0;
            set_orbital<W>(out + sector * W, qubit - sector * num_orbitals);
            bits &= bits - 1;
        }
    }
}


template <std::size_t W>
inline TermMasks sector_masks(const TermMasks& source, const std::vector<std::int64_t>& order, const int num_orbitals)
{
    TermMasks out;
    out.num_words = 2 * W;
    out.words.assign(checked_multiply(3 * 2 * W, order.size()), 0);
    out.real_values.resize(order.size());
    out.complex_values.resize(order.size());
    const std::size_t in_stride = 3 * source.num_words;
    const std::size_t out_stride = 3 * 2 * W;
#pragma omp parallel for schedule(static)
    for(std::size_t index = 0; index < order.size(); index++)
    {
        const std::size_t term = static_cast<std::size_t>(order[index]);
        for(std::size_t part = 0; part < 3; part++)
        {
            qubit_to_sector_key<W>(source.words.data() + term * in_stride + part * source.num_words,
                                   source.num_words,
                                   num_orbitals,
                                   out.words.data() + index * out_stride + part * 2 * W);
        }
        out.real_values[index] = source.real_values[term];
        out.complex_values[index] = source.complex_values[term];
    }
    return out;
}


template <std::size_t W>
inline std::string flips_text(const Word* flips)
{
    std::string text;
    for(int sector = 0; sector < 2; sector++)
    {
        int orbitals[64 * W];
        const int count = list_orbitals<W>(flips + sector * W, orbitals, 64 * W);
        text += sector == 0 ? "alpha {" : " beta {";
        for(int k = 0; k < count; k++)
        {
            text += (k > 0 ? "," : "") + std::to_string(orbitals[k]);
        }
        text += "}";
    }
    return text;
}


template <std::size_t W>
inline std::pair<int, std::int64_t> classify_flips(const Word* flips, const int num_orbitals)
{
    int alpha[4];
    int beta[4];
    const int num_alpha = list_orbitals<W>(flips, alpha, 4);
    const int num_beta = list_orbitals<W>(flips + W, beta, 4);
    const std::int64_t num_pairs = binomial(num_orbitals, 2);
    if(num_alpha == 2 && num_beta == 0)
    {
        return {kAlphaSingle, pair_rank(alpha[0], alpha[1])};
    }
    if(num_alpha == 0 && num_beta == 2)
    {
        return {kBetaSingle, pair_rank(beta[0], beta[1])};
    }
    if(num_alpha == 4 && num_beta == 0)
    {
        return {kAlphaDouble, quad_rank(alpha)};
    }
    if(num_alpha == 0 && num_beta == 4)
    {
        return {kBetaDouble, quad_rank(beta)};
    }
    if(num_alpha == 2 && num_beta == 2)
    {
        return {kMixed, pair_rank(alpha[0], alpha[1]) * num_pairs + pair_rank(beta[0], beta[1])};
    }
    return {-1, -1};
}


template <std::size_t W>
inline OperatorTerms<W> split_operator(const QubitOperator& op, const bool real_only)
{
    OperatorTerms<W> out;
    out.num_orbitals = orbitals_for_width(op.width, kWordBits * static_cast<int>(W));
    const int n = out.num_orbitals;
    if(op.type != 2)
    {
        throw std::invalid_argument("The operator must be type 2 (fermionic ladder terms), got type " +
                                    std::to_string(op.type));
    }
    const std::size_t num_terms = op.terms.size();
    struct TermRecord
    {
        std::int64_t key;
        std::int64_t term;
    };
    const std::int64_t num_pairs = binomial(n, 2);
    const std::int64_t num_quads = binomial(n, 4);
    const std::array<std::int64_t, kNumClasses> class_offset = {
        0, num_pairs, num_pairs + num_quads, 2 * num_pairs + num_quads, 2 * num_pairs + 2 * num_quads};
    std::vector<TermRecord> records;
    std::vector<std::int64_t> diagonal_terms;
    for(std::size_t term = 0; term < num_terms; term++)
    {
        const OperatorTerm_t& t = op.terms[term];
        if(real_only && (t.real_phase == 0 || std::abs(t.coeff.imag()) > ATOL))
        {
            throw std::invalid_argument("Operator term " + std::to_string(term) +
                                        " has a complex value; only real operators are supported");
        }
        std::array<Word, 2 * W> flips{};
        bool has_flip = false;
        for(std::size_t factor = 0; factor < t.indices.size(); factor++)
        {
            const unsigned char value = t.values[factor];
            if(value == OPER_VALUE_X || value == OPER_VALUE_Y)
            {
                throw std::invalid_argument("Operator term " + std::to_string(term) +
                                            " has an X or Y factor; only ladder terms are supported");
            }
            if(value == OPER_VALUE_PLUS || value == OPER_VALUE_MINUS)
            {
                const int qubit = static_cast<int>(t.indices[factor]);
                const int sector = qubit >= n ? 1 : 0;
                set_orbital<W>(flips.data() + sector * W, qubit - sector * n);
                has_flip = true;
            }
        }
        if(!has_flip)
        {
            diagonal_terms.push_back(static_cast<std::int64_t>(term));
            continue;
        }
        const auto [cls, index] = classify_flips<W>(flips.data(), n);
        if(cls < 0)
        {
            throw std::invalid_argument("Operator term " + std::to_string(term) + " flips " +
                                        flips_text<W>(flips.data()) +
                                        "; only spin-conserving single and double moves are supported");
        }
        records.push_back({class_offset[static_cast<std::size_t>(cls)] + index, static_cast<std::int64_t>(term)});
    }
    parallel_sort(records, [](const TermRecord& a, const TermRecord& b) {
        return a.key != b.key ? a.key < b.key : a.term < b.term;
    });
    std::vector<std::int64_t> order(records.size());
    for(std::size_t k = 0; k < records.size(); k++)
    {
        order[k] = records[k].term;
        if(k == 0 || records[k].key != records[k - 1].key)
        {
            const std::int64_t key = records[k].key;
            int cls = kNumClasses - 1;
            while(class_offset[static_cast<std::size_t>(cls)] > key)
            {
                cls--;
            }
            out.group_start.push_back(static_cast<std::int64_t>(k));
            out.group_class.push_back(cls);
            out.group_index.push_back(key - class_offset[static_cast<std::size_t>(cls)]);
        }
    }
    out.group_start.push_back(static_cast<std::int64_t>(records.size()));
    if(records.size() + diagonal_terms.size() != num_terms)
    {
        throw std::logic_error("The operator groups hold " + std::to_string(records.size()) +
                               " off-diagonal terms; the operator has " +
                               std::to_string(num_terms - diagonal_terms.size()));
    }
    const TermMasks qubit_masks = build_term_masks(op.terms, words_for_width(op.width));
    out.offdiag = sector_masks<W>(qubit_masks, order, n);
    out.diagonal = sector_masks<W>(qubit_masks, diagonal_terms, n);
    return out;
}


template <std::size_t W, typename T>
struct CoefficientTables
{
    int num_orbitals = 0;
    std::int64_t num_pairs = 0;
    std::int64_t num_quads = 0;
    std::array<Word, 2 * W> reference{};
    std::vector<T> single_value;
    std::vector<T> single_delta;
    std::vector<T> single_reference;
    std::vector<T> double_value;
    std::vector<T> mixed_value;


    std::int64_t single_index(const int sector, const std::int64_t pair, const int dir) const
    {
        return (sector * num_pairs + pair) * 2 + dir;
    }


    const T* deltas(const std::int64_t index) const
    {
        return single_delta.data() + index * 2 * num_orbitals;
    }


    T single_element(const Word* out, const int sector, const int p, const int q) const
    {
        const Word* own = out + sector * W;
        const Word* other = out + (1 - sector) * W;
        const int dir = has_orbital<W>(own, p) ? 0 : 1;
        const std::int64_t index = single_index(sector, pair_rank(p, q), dir);
        const T* delta = deltas(index);
        const T* own_delta = delta + sector * num_orbitals;
        const T* other_delta = delta + (1 - sector) * num_orbitals;
        T other_sum = T(0);
        for(std::size_t word = 0; word < W; word++)
        {
            Word bits = other[word];
            while(bits != 0)
            {
                other_sum += other_delta[static_cast<int>(word) * kWordBits + __builtin_ctzll(bits)];
                bits &= bits - 1;
            }
        }
        T own_sum = single_reference[static_cast<std::size_t>(index)];
        const Word* ref = reference.data() + sector * W;
        for(std::size_t word = 0; word < W; word++)
        {
            Word bits = own[word] ^ ref[word];
            while(bits != 0)
            {
                const int r = static_cast<int>(word) * kWordBits + __builtin_ctzll(bits);
                bits &= bits - 1;
                if(r == p || r == q)
                {
                    continue;
                }
                own_sum += has_orbital<W>(own, r) ? own_delta[r] : -own_delta[r];
            }
        }
        const T value = single_value[static_cast<std::size_t>(index)] + other_sum + own_sum;
        return between_parity<W>(own, p, q) ? -value : value;
    }


    T double_element(const Word* out, const int sector, const int* positions) const
    {
        const Word* own = out + sector * W;
        int pattern = 0;
        for(int k = 0; k < 4; k++)
        {
            pattern |= static_cast<int>(has_orbital<W>(own, positions[k])) << k;
        }
        const int slot = kDoubleSlot[static_cast<std::size_t>(pattern)];
        if(slot < 0)
        {
            return T(0);
        }
        const T value = double_value[static_cast<std::size_t>((sector * num_quads + quad_rank(positions)) * 3 + slot)];
        const bool odd = between_parity<W>(own, positions[0], positions[1]) !=
                         between_parity<W>(own, positions[2], positions[3]);
        return odd ? -value : value;
    }


    T mixed_element(const Word* out, const int* alpha, const int* beta) const
    {
        const int pattern = static_cast<int>(has_orbital<W>(out, alpha[0])) |
                            static_cast<int>(has_orbital<W>(out, alpha[1])) << 1 |
                            static_cast<int>(has_orbital<W>(out + W, beta[0])) << 2 |
                            static_cast<int>(has_orbital<W>(out + W, beta[1])) << 3;
        const int slot = kMixedSlot[static_cast<std::size_t>(pattern)];
        if(slot < 0)
        {
            return T(0);
        }
        const std::int64_t index = pair_rank(alpha[0], alpha[1]) * num_pairs + pair_rank(beta[0], beta[1]);
        const T value = mixed_value[static_cast<std::size_t>(index * 2 + slot)];
        const bool odd = between_parity<W>(out, alpha[0], alpha[1]) != between_parity<W>(out + W, beta[0], beta[1]);
        return odd ? -value : value;
    }


    T element(const Word* out, const Word* source) const
    {
        std::array<Word, 2 * W> flips;
        for(std::size_t word = 0; word < 2 * W; word++)
        {
            flips[word] = out[word] ^ source[word];
        }
        int alpha[4];
        int beta[4];
        const int num_alpha = list_orbitals<W>(flips.data(), alpha, 4);
        const int num_beta = list_orbitals<W>(flips.data() + W, beta, 4);
        if(num_alpha == 2 && num_beta == 0)
        {
            return single_element(out, 0, alpha[0], alpha[1]);
        }
        if(num_alpha == 0 && num_beta == 2)
        {
            return single_element(out, 1, beta[0], beta[1]);
        }
        if(num_alpha == 4 && num_beta == 0)
        {
            return double_element(out, 0, alpha);
        }
        if(num_alpha == 0 && num_beta == 4)
        {
            return double_element(out, 1, beta);
        }
        if(num_alpha == 2 && num_beta == 2)
        {
            return mixed_element(out, alpha, beta);
        }
        return T(0);
    }
};


template <std::size_t W>
struct TableSpec
{
    int num_alpha = 0;
    int num_beta = 0;
    std::array<Word, 2 * W> reference{};
};


template <std::size_t W>
inline TableSpec<W> store_spec(const KeyStore<W>& store)
{
    TableSpec<W> spec;
    spec.num_alpha = store.num_alpha;
    spec.num_beta = store.num_beta;
    std::int64_t best_column = 0;
    for(std::int64_t column = 1; column < store.num_columns(); column++)
    {
        const std::size_t c = static_cast<std::size_t>(column);
        const std::size_t b = static_cast<std::size_t>(best_column);
        if(store.column_start[c + 1] - store.column_start[c] > store.column_start[b + 1] - store.column_start[b])
        {
            best_column = column;
        }
    }
    std::int64_t best_line = 0;
    for(std::int64_t line = 1; line < store.num_lines(); line++)
    {
        const std::size_t l = static_cast<std::size_t>(line);
        const std::size_t b = static_cast<std::size_t>(best_line);
        if(store.line_start[l + 1] - store.line_start[l] > store.line_start[b + 1] - store.line_start[b])
        {
            best_line = line;
        }
    }
    std::copy_n(store.column_alpha(best_column), W, spec.reference.data());
    std::copy_n(store.line_beta(best_line), W, spec.reference.data() + W);
    return spec;
}


struct TableFailure
{
    std::int64_t group = -1;
    std::string reason;


    void record(const std::int64_t g, const std::string& text)
    {
#pragma omp critical(full_strs_table_failure)
        {
            if(group < 0 || g < group)
            {
                group = g;
                reason = text;
            }
        }
    }
};


inline std::uint64_t mix64(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}


template <std::size_t W>
inline void fill_random(Word* words,
                        const int num_orbitals,
                        const int* excluded,
                        const int num_excluded,
                        int count,
                        std::uint64_t& state)
{
    int free[64 * W];
    int num_free = 0;
    for(int orbital = 0; orbital < num_orbitals; orbital++)
    {
        if(std::find(excluded, excluded + num_excluded, orbital) == excluded + num_excluded)
        {
            free[num_free++] = orbital;
        }
    }
    if(count < 0 || count > num_free)
    {
        count = num_free;
    }
    for(int k = 0; k < count; k++)
    {
        state = mix64(state);
        const int pick = k + static_cast<int>(state % static_cast<std::uint64_t>(num_free - k));
        std::swap(free[k], free[pick]);
        set_orbital<W>(words, free[k]);
    }
}


template <std::size_t W>
inline void minimal_state(const std::array<std::pair<int, int>, 4>& orbitals, const int count, const int pattern, Word* out)
{
    std::fill(out, out + 2 * W, Word(0));
    for(int k = 0; k < count; k++)
    {
        if((pattern >> k) & 1)
        {
            set_orbital<W>(out + orbitals[static_cast<std::size_t>(k)].first * W, orbitals[static_cast<std::size_t>(k)].second);
        }
    }
}


inline std::array<std::pair<int, int>, 4> group_orbitals(const int cls,
                                                         const std::int64_t index,
                                                         const std::int64_t num_pairs,
                                                         int& count)
{
    std::array<std::pair<int, int>, 4> out{};
    if(cls == kAlphaSingle || cls == kBetaSingle)
    {
        const auto [p, q] = pair_orbitals(index);
        const int sector = cls == kAlphaSingle ? 0 : 1;
        out[0] = {sector, p};
        out[1] = {sector, q};
        count = 2;
    }
    else if(cls == kAlphaDouble || cls == kBetaDouble)
    {
        int positions[4];
        quad_orbitals(index, positions);
        const int sector = cls == kAlphaDouble ? 0 : 1;
        for(int k = 0; k < 4; k++)
        {
            out[static_cast<std::size_t>(k)] = {sector, positions[k]};
        }
        count = 4;
    }
    else
    {
        const auto [wa, xa] = pair_orbitals(index / num_pairs);
        const auto [yb, zb] = pair_orbitals(index % num_pairs);
        out = {{{0, wa}, {0, xa}, {1, yb}, {1, zb}}};
        count = 4;
    }
    return out;
}


template <std::size_t W, typename T>
inline std::string build_single_group(CoefficientTables<W, T>& tables, const OperatorTerms<W>& terms, const std::int64_t g)
{
    const int n = tables.num_orbitals;
    const int cls = terms.group_class[static_cast<std::size_t>(g)];
    const int sector = cls == kAlphaSingle ? 0 : 1;
    const std::int64_t pair = terms.group_index[static_cast<std::size_t>(g)];
    const auto [p, q] = pair_orbitals(pair);
    std::array<Word, 2 * W> ket{};
    for(int dir = 0; dir < 2; dir++)
    {
        const int annihilated = dir == 0 ? q : p;
        const std::int64_t index = tables.single_index(sector, pair, dir);
        ket.fill(0);
        set_orbital<W>(ket.data() + sector * W, annihilated);
        const T base = terms.template group_value<T>(g, ket.data());
        tables.single_value[static_cast<std::size_t>(index)] = base;
        T* delta = tables.single_delta.data() + index * 2 * n;
        for(int k = 0; k < 2 * n; k++)
        {
            const int k_sector = k >= n ? 1 : 0;
            const int orbital = k - k_sector * n;
            if(k_sector == sector && (orbital == p || orbital == q))
            {
                continue;
            }
            std::array<Word, 2 * W> spectator = ket;
            set_orbital<W>(spectator.data() + k_sector * W, orbital);
            const T value = terms.template group_value<T>(g, spectator.data());
            const bool inside = k_sector == sector && orbital > p && orbital < q;
            delta[k] = (inside ? -value : value) - base;
        }
    }
    const std::int64_t first = tables.single_index(sector, pair, 0);
    const T* d0 = tables.single_delta.data() + first * 2 * n;
    const T* d1 = d0 + 2 * n;
    if(!values_agree(tables.single_value[static_cast<std::size_t>(first)],
                     conjugate(tables.single_value[static_cast<std::size_t>(first) + 1]),
                     0.0))
    {
        return "the two directions of the single move have different values (operator is not Hermitian)";
    }
    for(int k = 0; k < 2 * n; k++)
    {
        if(!values_agree(d0[k], conjugate(d1[k]), 0.0))
        {
            return "the two directions of the single move have different spectator terms for spin-orbital " +
                   std::to_string(k) + " (operator is not Hermitian)";
        }
    }
    for(int dir = 0; dir < 2; dir++)
    {
        const std::int64_t index = first + dir;
        const T* delta = tables.single_delta.data() + index * 2 * n;
        const Word* ref = tables.reference.data() + sector * W;
        T sum = T(0);
        for(int orbital = 0; orbital < n; orbital++)
        {
            if(orbital != p && orbital != q && has_orbital<W>(ref, orbital))
            {
                sum += delta[sector * n + orbital];
            }
        }
        tables.single_reference[static_cast<std::size_t>(index)] = sum;
    }
    return std::string();
}


template <std::size_t W, typename T>
inline std::string build_four_flip_group(CoefficientTables<W, T>& tables, const OperatorTerms<W>& terms, const std::int64_t g)
{
    const int cls = terms.group_class[static_cast<std::size_t>(g)];
    const std::int64_t index = terms.group_index[static_cast<std::size_t>(g)];
    int count = 0;
    const std::array<std::pair<int, int>, 4> orbitals = group_orbitals(cls, index, tables.num_pairs, count);
    const bool mixed = cls == kMixed;
    const std::array<std::int8_t, 16>& slots = mixed ? kMixedSlot : kDoubleSlot;
    std::array<Word, 2 * W> ket{};
    for(int pattern = 0; pattern < 16; pattern++)
    {
        const int slot = slots[static_cast<std::size_t>(pattern)];
        if(slot < 0 || pattern > 15 - pattern)
        {
            continue;
        }
        minimal_state<W>(orbitals, 4, 15 - pattern, ket.data());
        const T value = terms.template group_value<T>(g, ket.data());
        minimal_state<W>(orbitals, 4, pattern, ket.data());
        const T mirror = terms.template group_value<T>(g, ket.data());
        if(!values_agree(value, conjugate(mirror), 0.0))
        {
            return "output patterns " + std::to_string(pattern) + " and " + std::to_string(15 - pattern) +
                   " have different values (operator is not Hermitian)";
        }
        if(mixed)
        {
            tables.mixed_value[static_cast<std::size_t>(index * 2 + slot)] = value;
        }
        else
        {
            const int sector = cls == kAlphaDouble ? 0 : 1;
            tables.double_value[static_cast<std::size_t>((sector * tables.num_quads + index) * 3 + slot)] = value;
        }
    }
    return std::string();
}


template <std::size_t W, typename T>
inline std::string verify_group(const CoefficientTables<W, T>& tables,
                                const OperatorTerms<W>& terms,
                                const TableSpec<W>& spec,
                                const std::int64_t g)
{
    const int n = tables.num_orbitals;
    const int cls = terms.group_class[static_cast<std::size_t>(g)];
    const std::int64_t index = terms.group_index[static_cast<std::size_t>(g)];
    int count = 0;
    const std::array<std::pair<int, int>, 4> orbitals = group_orbitals(cls, index, tables.num_pairs, count);
    std::array<int, 6> patterns{};
    int num_patterns = 0;
    if(count == 2)
    {
        patterns = {1, 2, 0, 0, 0, 0};
        num_patterns = 2;
    }
    else
    {
        const std::array<std::int8_t, 16>& slots = cls == kMixed ? kMixedSlot : kDoubleSlot;
        for(int pattern = 0; pattern < 16; pattern++)
        {
            if(slots[static_cast<std::size_t>(pattern)] >= 0)
            {
                patterns[static_cast<std::size_t>(num_patterns++)] = pattern;
            }
        }
    }
    std::array<std::array<int, 4>, 2> excluded{};
    std::array<int, 2> num_excluded = {0, 0};
    for(int k = 0; k < count; k++)
    {
        const auto [sector, orbital] = orbitals[static_cast<std::size_t>(k)];
        excluded[static_cast<std::size_t>(sector)][static_cast<std::size_t>(num_excluded[static_cast<std::size_t>(sector)]++)] = orbital;
    }
    std::uint64_t state = mix64(static_cast<std::uint64_t>(g) * 0x100000001B3ULL + 0x5EED);
    std::array<Word, 2 * W> out{};
    std::array<Word, 2 * W> ket{};
    for(int sample = 0; sample <= kVerifySamples; sample++)
    {
        state = mix64(state);
        const int pattern = sample == 0 ? patterns[0] : patterns[static_cast<std::size_t>(state % static_cast<std::uint64_t>(num_patterns))];
        minimal_state<W>(orbitals, count, pattern, out.data());
        for(int sector = 0; sector < 2; sector++)
        {
            const int electrons = sector == 0 ? spec.num_alpha : spec.num_beta;
            int held = 0;
            for(int k = 0; k < count; k++)
            {
                held += (orbitals[static_cast<std::size_t>(k)].first == sector && ((pattern >> k) & 1)) ? 1 : 0;
            }
            fill_random<W>(out.data() + sector * W,
                           n,
                           excluded[static_cast<std::size_t>(sector)].data(),
                           num_excluded[static_cast<std::size_t>(sector)],
                           sample == 0 ? -1 : std::max(0, electrons - held),
                           state);
        }
        minimal_state<W>(orbitals, count, (1 << count) - 1, ket.data());
        for(std::size_t word = 0; word < 2 * W; word++)
        {
            ket[word] ^= out[word];
        }
        const T expected = terms.template group_value<T>(g, ket.data());
        const T got = tables.element(out.data(), ket.data());
        double scale = std::abs(got);
        if(count == 2)
        {
            const int sector = orbitals[0].first;
            const int dir = (pattern & 1) ? 0 : 1;
            const std::int64_t entry = tables.single_index(sector, index, dir);
            const T* delta = tables.deltas(entry);
            scale = std::abs(tables.single_value[static_cast<std::size_t>(entry)]) +
                    std::abs(tables.single_reference[static_cast<std::size_t>(entry)]);
            for(int k = 0; k < 2 * n; k++)
            {
                scale += std::abs(delta[k]);
            }
        }
        if(!values_agree(got, expected, scale))
        {
            return "the table element differs from the operator terms on a sample state (sample " +
                   std::to_string(sample) + ", difference " + std::to_string(std::abs(got - expected)) +
                   "); the group has terms that the single and double tables cannot express";
        }
    }
    return std::string();
}


template <std::size_t W, typename T>
inline CoefficientTables<W, T> build_coefficients(const OperatorTerms<W>& terms, const TableSpec<W>& spec)
{
    CoefficientTables<W, T> tables;
    const int n = terms.num_orbitals;
    tables.num_orbitals = n;
    tables.num_pairs = binomial(n, 2);
    tables.num_quads = binomial(n, 4);
    tables.reference = spec.reference;
    tables.single_value.assign(static_cast<std::size_t>(4 * tables.num_pairs), T(0));
    tables.single_delta.assign(static_cast<std::size_t>(4 * tables.num_pairs * 2 * n), T(0));
    tables.single_reference.assign(static_cast<std::size_t>(4 * tables.num_pairs), T(0));
    tables.double_value.assign(static_cast<std::size_t>(2 * tables.num_quads * 3), T(0));
    tables.mixed_value.assign(static_cast<std::size_t>(tables.num_pairs * tables.num_pairs * 2), T(0));
    const std::int64_t num_groups = terms.num_groups();
    TableFailure failure;
    for(int pass = 0; pass < 2; pass++)
    {
#pragma omp parallel for schedule(dynamic, 64)
        for(std::int64_t g = 0; g < num_groups; g++)
        {
            const int cls = terms.group_class[static_cast<std::size_t>(g)];
            std::string reason;
            if(pass == 0)
            {
                reason = (cls == kAlphaSingle || cls == kBetaSingle) ? build_single_group(tables, terms, g)
                                                                     : build_four_flip_group(tables, terms, g);
            }
            else
            {
                reason = verify_group(tables, terms, spec, g);
            }
            if(!reason.empty())
            {
                failure.record(g, reason);
            }
        }
        if(failure.group >= 0)
        {
            int count = 0;
            const std::size_t g = static_cast<std::size_t>(failure.group);
            const std::array<std::pair<int, int>, 4> orbitals =
                group_orbitals(terms.group_class[g], terms.group_index[g], tables.num_pairs, count);
            std::array<Word, 2 * W> flips{};
            for(int k = 0; k < count; k++)
            {
                set_orbital<W>(flips.data() + orbitals[static_cast<std::size_t>(k)].first * W,
                               orbitals[static_cast<std::size_t>(k)].second);
            }
            throw std::invalid_argument("Operator group " + std::to_string(failure.group) + " (flips " +
                                        flips_text<W>(flips.data()) + ") fails the coefficient check: " +
                                        failure.reason);
        }
    }
    return tables;
}


template <std::size_t W, typename T>
inline std::vector<T> diagonal_values(const OperatorTerms<W>& terms, const KeyStore<W>& store)
{
    const std::int64_t num_rows = store.num_rows();
    const std::size_t num_terms = terms.diagonal.real_values.size();
    std::vector<T> out(static_cast<std::size_t>(num_rows));
#pragma omp parallel for schedule(static)
    for(std::int64_t row = 0; row < num_rows; row++)
    {
        out[static_cast<std::size_t>(row)] = term_sum<T>(terms.diagonal, 0, num_terms, store.alpha(row));
    }
    return out;
}

}
