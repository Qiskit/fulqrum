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
#include "../fulqrum/core/src/element_evaluator.hpp"
#include "../fulqrum/core/src/elements.hpp"
#include "doctest.h"
#include <complex>
#include <random>
#include <string>
#include <vector>

typedef std::complex<double> complex;

namespace
{

/**
 * Returns the row with the flip bits of a term.
 *
 * @param term The term.
 * @param row The row.
 */
KeyBitset column_of(const OperatorTerm_t& term, const KeyBitset& row)
{
    KeyBitset col = row;
    for(std::size_t factor = 0; factor < term.indices.size(); factor++)
    {
        if(term.values[factor] > OPER_VALUE_PROJ_1)
        {
            col.flip(term.indices[factor]);
        }
    }
    return col;
}

/**
 * Returns <row|term|column> from accum_element, the reference evaluator.
 *
 * @param term The term.
 * @param row The row.
 */
template <typename T>
T reference_element(const OperatorTerm_t& term, const KeyBitset& row)
{
    T out = 0;
    accum_element(row,
                  column_of(term, row),
                  term.indices,
                  term.values,
                  term.coeff,
                  term.real_phase,
                  static_cast<unsigned int>(term.indices.size()),
                  out);
    return out;
}

/**
 * Returns a random row of a width.
 *
 * @param width The number of qubits.
 * @param engine The random engine.
 */
KeyBitset random_row(const width_t width, std::mt19937_64& engine)
{
    KeyBitset row(width);
    for(width_t qubit = 0; qubit < width; qubit++)
    {
        row[qubit] = engine() % 2 == 1;
    }
    return row;
}

/**
 * Returns the flip mask of a term.
 *
 * @param term The term.
 * @param num_words The number of key words.
 */
std::vector<KeyWord> flip_mask(const OperatorTerm_t& term, const std::size_t num_words)
{
    std::vector<KeyWord> flips(num_words, 0);
    for(std::size_t factor = 0; factor < term.indices.size(); factor++)
    {
        if(term.values[factor] > OPER_VALUE_PROJ_1)
        {
            const width_t qubit = term.indices[factor];
            flips[qubit / KEY_WORD_BITS] |= KeyWord(1) << (qubit % KEY_WORD_BITS);
        }
    }
    return flips;
}

} // namespace

TEST_CASE("Term masks evaluate matrix elements using the row bits")
{
    CHECK(OPER_ELEMENTS[4 * OPER_VALUE_PLUS + 2] == complex(1, 0));
    CHECK(OPER_ELEMENTS[4 * OPER_VALUE_PLUS + 1] == complex(0, 0));
    const width_t width = 140;
    for(const width_t qubit :
        {width_t(2), width_t(63), width_t(64), width_t(127), width_t(128), width_t(139)})
    {
        const std::vector<OperatorTerm_t> terms = {OperatorTerm_t("+", {qubit}, complex(1, 0))};
        const TermMasks masks = build_term_masks(terms, words_for_width(width));
        KeyBitset row(width);
        CHECK(term_sum<complex>(masks, 0, 1, row.m_bits.data()) == complex(0, 0));
        CHECK(reference_element<complex>(terms[0], row) == complex(0, 0));
        row[qubit] = true;
        CHECK(term_sum<complex>(masks, 0, 1, row.m_bits.data()) == complex(1, 0));
        CHECK(reference_element<complex>(terms[0], row) == complex(1, 0));
        CHECK(term_sum<double>(masks, 0, 1, row.m_bits.data()) == 1.0);
    }
}

TEST_CASE("Term masks give the Y sign on both row bits")
{
    const width_t width = 200;
    for(const width_t qubit : {width_t(0), width_t(64), width_t(199)})
    {
        const std::vector<OperatorTerm_t> terms = {OperatorTerm_t("Y", {qubit}, complex(1, 0))};
        const TermMasks masks = build_term_masks(terms, words_for_width(width));
        KeyBitset row(width);
        CHECK(term_sum<complex>(masks, 0, 1, row.m_bits.data()) == complex(0, -1));
        CHECK(reference_element<complex>(terms[0], row) == complex(0, -1));
        row[qubit] = true;
        CHECK(term_sum<complex>(masks, 0, 1, row.m_bits.data()) == complex(0, 1));
        CHECK(reference_element<complex>(terms[0], row) == complex(0, 1));
    }
}

TEST_CASE("Term masks match accum_element on multiword keys")
{
    std::mt19937_64 engine(5);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    const std::string letters = "XYZ01+-";
    for(const width_t width :
        {width_t(63), width_t(64), width_t(65), width_t(70), width_t(140), width_t(200)})
    {
        const std::vector<width_t> qubits = {
            0, 1, 62, 63, 64, 65, 69, 127, 128, 129, 139, 191, 192, 199};
        std::vector<width_t> valid;
        for(const width_t qubit : qubits)
        {
            if(qubit < width)
            {
                valid.push_back(qubit);
            }
        }
        for(int real = 0; real < 2; real++)
        {
            std::vector<OperatorTerm_t> terms;
            for(int index = 0; index < 300; index++)
            {
                std::vector<width_t> shuffled = valid;
                std::shuffle(shuffled.begin(), shuffled.end(), engine);
                const std::size_t size = 1 + engine() % std::min<std::size_t>(6, valid.size());
                std::string ops;
                for(std::size_t factor = 0; factor < size; factor++)
                {
                    ops.push_back(letters[engine() % letters.size()]);
                }
                if(real == 1)
                {
                    std::size_t num_y = 0;
                    for(const char letter : ops)
                    {
                        num_y += letter == 'Y';
                    }
                    if(num_y % 2 == 1)
                    {
                        ops[ops.find('Y')] = 'X';
                    }
                }
                const complex coeff(uniform(engine), real == 1 ? 0.0 : uniform(engine));
                terms.emplace_back(
                    ops, std::vector<width_t>(shuffled.begin(), shuffled.begin() + size), coeff);
            }
            const TermMasks masks = build_term_masks(terms, words_for_width(width));
            for(std::size_t index = 0; index < terms.size(); index++)
            {
                for(int sample = 0; sample < 4; sample++)
                {
                    const KeyBitset row = random_row(width, engine);
                    const complex value =
                        term_sum<complex>(masks, index, index + 1, row.m_bits.data());
                    CHECK(std::abs(value - reference_element<complex>(terms[index], row)) <= 1e-14);
                    if(real == 1)
                    {
                        const double real_value =
                            term_sum<double>(masks, index, index + 1, row.m_bits.data());
                        CHECK(std::abs(real_value - reference_element<double>(terms[index], row)) <=
                              1e-14);
                    }
                }
            }
        }
    }
}

TEST_CASE("Compact groups match term sums on every flip pattern")
{
    const width_t width = 200;
    const std::size_t num_words = words_for_width(width);
    const std::vector<width_t> flips = {5, 64, 130, 199};
    const std::vector<width_t> outer = {0, 70, 150, 198};
    std::vector<OperatorTerm_t> terms;
    const std::vector<std::string> patterns = {"+-XY", "-+YX", "XX+-", "YY-+", "++--"};
    double scale = 0.3;
    for(const std::string& pattern : patterns)
    {
        std::vector<width_t> inds = flips;
        inds.insert(inds.end(), outer.begin(), outer.end());
        terms.emplace_back(pattern + "ZZZZ", inds, complex(scale, -0.5 * scale));
        scale += 0.17;
    }
    const TermMasks masks = build_term_masks(terms, num_words);
    const std::vector<KeyWord> flip_words = flip_mask(terms[0], num_words);
    CompactGroup<complex> record;
    std::vector<KeyWord> parity_words(3, 0);
    REQUIRE(build_compact_group<complex>(
        masks, 0, terms.size(), flip_words.data(), 4, record, parity_words));
    CHECK(record.parity_offset == 3);
    CHECK(parity_words.size() == 3 + num_words);
    std::mt19937_64 engine(11);
    for(unsigned int pattern = 0; pattern < 16; pattern++)
    {
        for(int sample = 0; sample < 8; sample++)
        {
            KeyBitset row = random_row(width, engine);
            for(std::size_t index = 0; index < flips.size(); index++)
            {
                row[flips[index]] = (pattern >> index) & 1u;
            }
            const complex expected = term_sum<complex>(masks, 0, terms.size(), row.m_bits.data());
            const complex value =
                compact_element<complex>(record, parity_words, num_words, row.m_bits.data());
            CHECK(std::abs(value - expected) <= 1e-14);
        }
    }
}

TEST_CASE("Compact groups refuse groups outside the compact form")
{
    const width_t width = 140;
    const std::size_t num_words = words_for_width(width);
    std::vector<KeyWord> parity_words;
    CompactGroup<double> record;
    SUBCASE("projector outside the flips")
    {
        const std::vector<OperatorTerm_t> terms = {
            OperatorTerm_t("+-", {3, 100}, complex(1, 0)),
            OperatorTerm_t("+-1", {3, 100, 70}, complex(0.5, 0))};
        const TermMasks masks = build_term_masks(terms, num_words);
        const std::vector<KeyWord> flips = flip_mask(terms[0], num_words);
        CHECK_FALSE(
            build_compact_group<double>(masks, 0, 2, flips.data(), 2, record, parity_words));
    }
    SUBCASE("outer parity differs between terms")
    {
        const std::vector<OperatorTerm_t> terms = {
            OperatorTerm_t("+-Z", {3, 100, 70}, complex(1, 0)),
            OperatorTerm_t("+-Z", {3, 100, 71}, complex(0.5, 0))};
        const TermMasks masks = build_term_masks(terms, num_words);
        const std::vector<KeyWord> flips = flip_mask(terms[0], num_words);
        CHECK_FALSE(
            build_compact_group<double>(masks, 0, 2, flips.data(), 2, record, parity_words));
    }
    SUBCASE("more than four flips")
    {
        const std::vector<OperatorTerm_t> terms = {
            OperatorTerm_t("XXXXX", {1, 2, 3, 70, 130}, complex(1, 0))};
        const TermMasks masks = build_term_masks(terms, num_words);
        const std::vector<KeyWord> flips = flip_mask(terms[0], num_words);
        CHECK_FALSE(
            build_compact_group<double>(masks, 0, 1, flips.data(), 5, record, parity_words));
    }
    CHECK(parity_words.empty());
}
