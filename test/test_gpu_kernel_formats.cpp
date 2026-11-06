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
#include "../fulqrum/gpu/kernels/arms/elements.hpp"
#include "../fulqrum/gpu/kernels/formats/edge_word.hpp"
#include "../fulqrum/gpu/kernels/formats/pair_index.hpp"
#include "../fulqrum/gpu/kernels/formats/params.hpp"
#include "../fulqrum/gpu/kernels/formats/sell.hpp"
#include <cstdint>
#include <set>
#include <vector>

TEST_CASE("Cross edge words keep the column, the pair id and the sign")
{
    for(const int sign : {-1, 1})
    {
        for(const std::uint32_t pair : {0u, 1u, 629u, 0xffffu})
        {
            const std::uint64_t word = fq::edge_word::pack_cross(123456u, pair, sign);
            CHECK(fq::edge_word::cross_column(word) == 123456);
            CHECK(fq::edge_word::cross_pair(word) == int(pair));
            CHECK(fq::edge_word::cross_sign(word) == double(sign));
        }
    }
    const std::uint64_t legacy =
        std::uint64_t(7) | (std::uint64_t(5) << 32) | ((std::uint64_t(std::int64_t(-1)) & 0xffffu) << 48);
    CHECK(fq::edge_word::pack_cross(7u, 5u, -1) == legacy);
}

TEST_CASE("Signed pair words keep the pair id and the sign")
{
    CHECK(fq::edge_word::pack_signed_pair(41u, -1) == (41u | (1u << 31)));
    CHECK(fq::edge_word::signed_pair_pair(fq::edge_word::pack_signed_pair(41u, -1)) == 41u);
    CHECK(fq::edge_word::signed_pair_sign(fq::edge_word::pack_signed_pair(41u, -1)) == -1.0);
    CHECK(fq::edge_word::signed_pair_sign(fq::edge_word::pack_signed_pair(41u, 1)) == 1.0);
}

TEST_CASE("GPU pair ids and core pair indices each number every pair once")
{
    for(const int orbitals : {2, 5, 36})
    {
        std::set<int> gpu;
        std::set<int> core;
        int expected = 0;
        for(int low = 0; low < orbitals; low++)
        {
            for(int high = low + 1; high < orbitals; high++)
            {
                CHECK(fq::pair_index::gpu_pair(low, high, orbitals) == expected);
                expected++;
                gpu.insert(fq::pair_index::gpu_pair(low, high, orbitals));
                core.insert(fq::pair_index::core_pair(low, high));
            }
        }
        CHECK(int(gpu.size()) == fq::pair_index::num_pairs(orbitals));
        CHECK(int(core.size()) == fq::pair_index::num_pairs(orbitals));
        CHECK(*core.rbegin() == fq::pair_index::num_pairs(orbitals) - 1);
    }
    CHECK(fq::pair_index::core_pair(0, 1) == 0);
    CHECK(fq::pair_index::core_pair(0, 2) == 1);
    CHECK(fq::pair_index::core_pair(1, 2) == 2);
    CHECK(fq::pair_index::gpu_pair(1, 2, 4) == 3);
}

TEST_CASE("The between mask holds the bits strictly between two orbitals")
{
    CHECK(fq::pair_index::between_mask(1, 2) == 0u);
    CHECK(fq::pair_index::between_mask(0, 2) == 0b10u);
    CHECK(fq::pair_index::between_mask(1, 5) == 0b11100u);
    CHECK(fq::pair_index::between_mask(0, 63) == ((std::uint64_t(1) << 63) - 2));
}

TEST_CASE("SELL slots interleave the rows of a slice edge by edge")
{
    CHECK(fq::sell::slice_of(31) == 0);
    CHECK(fq::sell::slice_of(32) == 1);
    CHECK(fq::sell::lane_of(33) == 1);
    CHECK(fq::sell::slot(0, 0, 1) == 32);
    CHECK(fq::sell::slot(96, 34, 2) == 96 + 64 + 2);
    CHECK(fq::sell::kMaxSlots == 9223372036854775807);
}

TEST_CASE("Element functions give the pair-table and cross forms")
{
    CHECK(fq::arms::pair_element(0.5, -1.0, 0.25) == 0.25);
    CHECK(fq::arms::cross_element(0.75, -1.0, -1.0) == 0.75);
    CHECK(fq::arms::cross_element(0.75, 1.0, -1.0) == -0.75);
    CHECK_FALSE(fq::arms::is_applied(1e-14));
    CHECK(fq::arms::is_applied(-2e-14));
}

TEST_CASE("Parameter structs have their fixed layout")
{
    CHECK(sizeof(FqTileGeometry) == 56);
    CHECK(sizeof(FqVectors) == 32);
    constexpr std::uint64_t word = fq::edge_word::pack_cross(3u, 4u, 1);
    static_assert(fq::edge_word::cross_pair(word) == 4, "constexpr decode");
}
