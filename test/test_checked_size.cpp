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
#include "doctest.h"
#include "../fulqrum/core/src/checked_size.hpp"
#include "../fulqrum/core/src/row_outputs.hpp"
#include "../fulqrum/core/src/element_evaluator.hpp"
#include "../fulqrum/core/src/diag.hpp"
#include "../fulqrum/core/src/key_words.hpp"

TEST_CASE("Size arithmetic rejects overflow before allocation")
{
    CHECK_THROWS_AS(checked_multiply(std::numeric_limits<std::size_t>::max(), 2), std::overflow_error);
    CHECK_THROWS_AS(checked_add(std::numeric_limits<std::size_t>::max(), 1), std::overflow_error);
    CHECK(checked_multiply(0, std::numeric_limits<std::size_t>::max()) == 0);
    CHECK(checked_add(std::numeric_limits<std::size_t>::max(), 0) == std::numeric_limits<std::size_t>::max());
    CHECK(ceiling_divide(std::numeric_limits<std::size_t>::max(), 2) == std::numeric_limits<std::size_t>::max() / 2 + 1);
    CHECK_THROWS_AS(ceiling_divide(1, 0), std::invalid_argument);
    CHECK_THROWS_AS(power_of_two_at_least(std::numeric_limits<std::size_t>::max()), std::overflow_error);
    CHECK_THROWS_AS(checked_index<std::size_t>(std::int64_t(-1)), std::invalid_argument);
    CHECK_THROWS_AS(checked_multiply(checked_multiply(checked_index<std::size_t>(std::numeric_limits<std::int64_t>::max()), 2), 8), std::overflow_error);
    std::vector<OperatorTerm_t> terms(1);
    CHECK_THROWS_AS(build_term_masks(terms, std::numeric_limits<std::size_t>::max()), std::overflow_error);
}

TEST_CASE("CSR counts cannot wrap their index type")
{
    auto make_walk = []() { return [](std::size_t, auto&& emit) {
        emit(0, 1.0);
        emit(1, 1.0);
    }; };
    std::vector<std::uint8_t> offsets(201);
    CHECK_THROWS_AS((csr_from_rows<std::uint8_t, double>(200, nullptr, make_walk, offsets.data(), nullptr, nullptr, 0)), std::overflow_error);
    auto repeated = []() { return [](std::size_t, auto&& emit) {
        for(int n = 0; n < 256; n++) emit(0, 1.0);
    }; };
    CHECK_THROWS_AS((csr_from_rows<std::uint8_t, double>(1, nullptr, repeated, offsets.data(), nullptr, nullptr, 0)), std::overflow_error);
    CHECK_THROWS_AS(checked_index<std::uint8_t>(256), std::overflow_error);
    auto excess = []() { return [](std::size_t, auto&& emit) {
        for(int n = 0; n < 128; n++) emit(0, 1.0);
    }; };
    std::int8_t starts[2] = {0, 127};
    std::vector<std::int8_t> columns(128, 17);
    std::vector<double> values(128, -1.0);
    CHECK_THROWS_AS((csr_from_rows<std::int8_t, double>(1, nullptr, excess, starts, columns.data(), values.data(), 1)), std::runtime_error);
    CHECK(values.back() == -1.0);

}


TEST_CASE("Diagonal factor dimensions are checked before output access")
{
    QubitOperator diagonal(0);
    double constant = 1.0;
    CHECK_THROWS_AS(compute_diag_cross_factors<double>(diagonal, std::numeric_limits<unsigned int>::max(), std::numeric_limits<unsigned int>::max(), nullptr, nullptr, nullptr, &constant), std::overflow_error);
    CHECK(constant == 1.0);
    CHECK_THROWS_AS(compute_diag_factors<double>(nullptr, nullptr, diagonal, 1, 3, 1, 0, nullptr, nullptr, nullptr, nullptr), std::invalid_argument);
}
