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
#include "../fulqrum/core/src/diag.hpp"
#include "../fulqrum/core/src/element_evaluator.hpp"
#include "../fulqrum/core/src/elements.hpp"
#include "doctest.h"
#include <algorithm>
#include <complex>
#include <random>
#include <vector>

namespace
{

template <typename T>
T reference_diagonal(const KeyBitset& row, const std::vector<OperatorTerm_t>& terms)
{
    T val = 0;
    for(const auto& term : terms)
    {
        accum_element(row,
                      row,
                      term.indices,
                      term.values,
                      term.coeff,
                      term.real_phase,
                      term.indices.size(),
                      val);
    }
    return val;
}

template <typename T>
void check_diagonal_paths(const width_t width)
{
    std::mt19937_64 engine(width);
    std::uniform_real_distribution<double> uniform(-0.1, 0.1);
    QubitOperator_t op;
    op.width = width;
    op.set_type(2);
    for(width_t p = 0; p < width; p++)
    {
        for(width_t q = p; q < width; q++)
        {
            const std::complex<double> c(uniform(engine),
                                         std::is_same_v<T, double> ? 0 : uniform(engine));
            op.terms.emplace_back(p == q ? "1" : "11",
                                  p == q ? std::vector<width_t>{p} : std::vector<width_t>{p, q},
                                  c);
        }
    }
    std::shuffle(op.terms.begin(), op.terms.end(), engine);
    REQUIRE(fast_diag_compatible(op));
    fast_diag_term_sort(op);
    const TermMasks masks = build_term_masks(op.terms, words_for_width(width));
    std::vector<std::size_t> row_ptrs(1, 0);
    for(width_t p = 0; p < width; p++)
    {
        row_ptrs.push_back(row_ptrs.back() + width - p);
    }
    for(unsigned int sample = 0; sample < 12; sample++)
    {
        KeyBitset row(width);
        for(width_t p = 0; p < width; p++)
        {
            row[p] = engine() % 3 == 0;
        }
        const T expected = reference_diagonal<T>(row, op.terms);
        T general, cached, fast, fast_cached;
        single_bitstring_diagonal(row, op.terms, general);
        single_bitstring_diagonal(row, masks, cached);
        single_bitstring_diagonal_fast(row, op.terms, row_ptrs, fast);
        single_bitstring_diagonal_fast(row, masks, row_ptrs, fast_cached);
        CHECK(std::abs(general - expected) <= 1e-12);
        CHECK(std::abs(cached - expected) <= 1e-12);
        CHECK(std::abs(fast - expected) <= 1e-12);
        CHECK(std::abs(fast_cached - expected) <= 1e-12);
        for(const std::vector<width_t>& flips :
            {std::vector<width_t>{0, width_t(width - 1)},
             std::vector<width_t>{1, 2, width_t(width - 2), width_t(width - 1)}})
        {
            KeyBitset col = row;
            for(const width_t p : flips)
            {
                col.flip(p);
            }
            const T col_expected = reference_diagonal<T>(col, op.terms);
            const auto occupied = set_bit_indices(row);
            const T delta = single_bitstring_diagonal_delta_fast(
                row, col, occupied, op.terms, row_ptrs, flips.data(), flips.size(), expected);
            const T delta_cached = single_bitstring_diagonal_delta_fast(
                row, col, occupied, masks, row_ptrs, flips.data(), flips.size(), expected);
            CHECK(std::abs(delta - col_expected) <= 1e-12);
            CHECK(std::abs(delta_cached - col_expected) <= 1e-12);
        }
    }
}

}

TEST_CASE("Cached and standalone diagonal evaluators preserve fast and delta paths")
{
    for(const width_t width : {width_t(8), width_t(65), width_t(140)})
    {
        check_diagonal_paths<double>(width);
        check_diagonal_paths<std::complex<double>>(width);
    }
}

TEST_CASE("Term mask ranges preserve cancellation identity and empty operators")
{
    using complex = std::complex<double>;
    const std::vector<OperatorTerm_t> terms = {
        OperatorTerm_t("I", {0}, complex(2, 0)),
        OperatorTerm_t("Z", {64}, complex(0.25, 0)),
        OperatorTerm_t("Z", {64}, complex(-0.25, 0)),
        OperatorTerm_t("0", {63}, complex(0, 0)),
    };
    const TermMasks masks = build_term_masks(terms, 2);
    KeyBitset row(65);
    CHECK(term_sum<double>(masks, 0, 4, row.m_bits.data()) == 2.0);
    CHECK(term_sum<complex>(masks, 1, 3, row.m_bits.data()) == complex(0, 0));
    CHECK(term_sum<double>(masks, 3, 4, row.m_bits.data()) == 0.0);
    CHECK(term_sum<double>(masks, 2, 2, row.m_bits.data()) == 0.0);
    const TermMasks empty = build_term_masks({}, 0);
    CHECK(term_sum<complex>(empty, 0, 0, nullptr) == complex(0, 0));
    const TermMasks constant = build_term_masks({OperatorTerm_t(complex(2, -3))}, 0);
    CHECK(term_sum<complex>(constant, 0, 1, nullptr) == complex(2, -3));
}

TEST_CASE("Parallel term mask construction preserves term order")
{
    std::vector<OperatorTerm_t> terms;
    for(std::size_t i = 0; i < 5000; i++)
    {
        terms.emplace_back(i % 2 ? "Z" : "1",
                           std::vector<width_t>{width_t(i % 140)},
                           std::complex<double>(double(i % 19) / 19, 0));
    }
    const TermMasks masks = build_term_masks(terms, words_for_width(140));
    KeyBitset row(140);
    row.set();
    CHECK(std::abs(term_sum<double>(masks, 0, terms.size(), row.m_bits.data()) -
                   reference_diagonal<double>(row, terms)) <= 1e-12);
}
