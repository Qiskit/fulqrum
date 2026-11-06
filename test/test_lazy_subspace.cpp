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
#include "../fulqrum/core/src/subspace.hpp"
#include "doctest.h"
#include <thread>
#include <vector>

TEST_CASE("Lazy product view preserves half membership and row order")
{
    Subspace_t space;
    space.initialize_product({"001", "010", "100"}, {"01", "10"});
    HalfKeyView view(space);
    CHECK(view.size() == 6);
    CHECK(view.split_bit() == 3);
    const KeyWord key = 4;
    CHECK(view.find_half_key(0, &key) == 2);
    CHECK(view.key(5) == KeyBitset(std::string("10100")));
    CHECK(!space.map_materialized);
    space.ensure_hashmap();
    CHECK(space.map_build_count == 1);
    for(std::size_t i = 0; i < space.size; i++)
    {
        CHECK(space.bitstrings.get(view.key(i)) == i);
        REQUIRE(space.bitstrings.get_ptr(view.key(i)) != nullptr);
        CHECK(*space.bitstrings.get_ptr(view.key(i)) == i);
    }
    CHECK(view.find_half_key(0, &key) == 2);
    space.ensure_hashmap();
    CHECK(space.map_build_count == 1);
    Subspace_t copy(space);
    CHECK(!copy.map_materialized);
    CHECK(copy.product);
    CHECK(copy.key_at(5) == space.key_at(5));
    copy.prepare_mutation();
    CHECK(!copy.product);
    CHECK(copy.half_keys[0].empty());
    CHECK(copy.key_at(5) == space.key_at(5));
}

TEST_CASE("Full lazy buffer is released only after successful map construction")
{
    Subspace_t space;
    space.configure(2, false);
    space.initialize_full({"00", "01", "11"});
    const auto failing_insert = [](auto& pending, const KeyBitset& key, std::size_t row) {
        pending.insert_unique(key, row);
        throw std::bad_alloc();
    };
    CHECK_THROWS_AS(space.ensure_hashmap_with(failing_insert), std::bad_alloc);
    CHECK(!space.map_materialized);
    CHECK(space.full_keys.size() == 3);
    CHECK(space.key_at(2) == KeyBitset(std::string("11")));
    space.ensure_hashmap();
    CHECK(space.full_keys.capacity() == 0);
    CHECK(space.key_at(2) == KeyBitset(std::string("11")));
    REQUIRE(space.bitstrings.get_ptr(KeyBitset(std::string("01"))) != nullptr);
    CHECK(*space.bitstrings.get_ptr(KeyBitset(std::string("01"))) == 1);
    Subspace_t copy(space);
    CHECK(copy.full_keys.size() == 3);
    CHECK(!copy.map_materialized);
    CHECK(copy.key_at(1) == KeyBitset(std::string("01")));
    CHECK_THROWS_AS(Subspace_t::checked_product(MAX_SIZE_T, 2), std::length_error);
}

TEST_CASE("Concurrent retained CPU entry materializes a full map once")
{
    Subspace_t space;
    space.initialize_full({"00", "01", "10", "11"});
    std::vector<std::thread> threads;
    for(int i = 0; i < 4; i++)
        threads.emplace_back([&]() { space.ensure_hashmap(); });
    for(auto& thread : threads)
        thread.join();
    CHECK(space.map_build_count == 1);
    CHECK(space.full_keys.empty());
    CHECK(space.key_at(3) == KeyBitset(std::string("11")));
}

TEST_CASE("Hashmap reserve rejects arithmetic and backend capacity overflow")
{
    bitset_map_namespace::BitsetHashMapWrapper map;
    CHECK_THROWS_AS(map.reserve(MAX_SIZE_T), std::length_error);
    CHECK_THROWS_AS(map.reserve(std::size_t(1) << 32), std::length_error);
    CHECK(map.size() == 0);
    map.reserve(2);
    map.insert_unique(KeyBitset(std::string("01")), 0);
    REQUIRE(map.get_ptr(KeyBitset(std::string("01"))) != nullptr);
    CHECK(*map.get_ptr(KeyBitset(std::string("01"))) == 0);
    map.reserve(3);
    REQUIRE(map.get_ptr(KeyBitset(std::string("01"))) != nullptr);
    CHECK(*map.get_ptr(KeyBitset(std::string("01"))) == 0);
}
