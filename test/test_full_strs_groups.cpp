#include "doctest.h"
#include "../fulqrum/gpu/full_strs/src/chunks.hpp"
#include "../fulqrum/gpu/full_strs/src/groups.hpp"
#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace
{

using fq::full_strs::GroupKinds;
using fq::full_strs::KeyStore;
using fq::full_strs::Word;

/**
 * Returns a random string of norb orbitals with count electrons, drawn near the lowest orbitals.
 *
 * @param engine The random engine.
 * @param norb The number of orbitals.
 * @param count The number of electrons.
 * @param spread The number of low orbitals that the electrons can use (at least count).
 */
Word clustered_string(std::mt19937_64& engine, const int norb, const int count, const int spread)
{
    std::vector<int> orbitals;
    for(int k = 0; k < std::min(norb, spread); k++)
    {
        orbitals.push_back(k);
    }
    std::shuffle(orbitals.begin(), orbitals.end(), engine);
    Word out = 0;
    for(int k = 0; k < count; k++)
    {
        out |= Word(1) << orbitals[static_cast<std::size_t>(k)];
    }
    return out;
}

/**
 * Makes a store of random rows whose strings come from small pools, so lines and columns repeat.
 */
KeyStore<1> random_store(const int norb, const int num_alpha, const int num_beta, const int rows, const int pool, const unsigned int seed)
{
    std::mt19937_64 engine(seed);
    std::vector<Word> alphas;
    std::vector<Word> betas;
    for(int k = 0; k < pool; k++)
    {
        alphas.push_back(clustered_string(engine, norb, num_alpha, num_alpha + 4));
        betas.push_back(clustered_string(engine, norb, num_beta, num_beta + 4));
    }
    std::vector<Word> keys;
    for(int row = 0; row < rows; row++)
    {
        keys.push_back(alphas[engine() % alphas.size()]);
        keys.push_back(betas[engine() % betas.size()]);
    }
    return fq::full_strs::build_key_store<1>(keys.data(), rows, norb);
}

/**
 * Checks that the stored groups give every connected ordered pair of every class exactly once.
 *
 * @param store The key store.
 * @param kinds The group kinds.
 * @return The number of connected ordered pairs.
 */
std::size_t check_pair_sets(const KeyStore<1>& store, const GroupKinds& kinds)
{
    const fq::full_strs::GroupStore groups = fq::full_strs::build_groups(store, kinds);
    std::size_t total = 0;
    for(int cls = 0; cls < fq::full_strs::kNumClasses; cls++)
    {
        const auto stored = fq::full_strs::group_pairs(store, groups, cls);
        const auto expected = fq::full_strs::brute_force_pairs(store, cls);
        CHECK(stored.size() == expected.size());
        CHECK(stored == expected);
        total += expected.size();
    }
    std::int64_t slots = 0;
    for(const auto& c : groups.classes)
    {
        slots += c.num_slots();
        for(std::int64_t g = 0; g < c.num_groups(); g++)
        {
            const std::int64_t first = c.member_start[static_cast<std::size_t>(g)];
            const std::int64_t last = c.member_start[static_cast<std::size_t>(g) + 1];
            CHECK(last - first >= 2);
            CHECK(std::is_sorted(c.members.begin() + first, c.members.begin() + last));
        }
    }
    CHECK(groups.row_slot_start.back() == slots);
    std::vector<std::int64_t> seen(groups.row_slots);
    std::sort(seen.begin(), seen.end());
    for(std::int64_t s = 0; s < slots; s++)
    {
        REQUIRE(seen[static_cast<std::size_t>(s)] == s);
    }
    return total;
}

}

TEST_CASE("Full-string groups: every connected pair exactly once, all kind combinations")
{
    struct Shape
    {
        int norb;
        int num_alpha;
        int num_beta;
        int rows;
        int pool;
    };
    const std::vector<Shape> shapes = {{10, 5, 5, 1500, 60}, {10, 4, 3, 1500, 40}, {12, 9, 8, 1200, 50}, {9, 2, 7, 800, 30}};
    unsigned int seed = 31;
    for(const Shape& shape : shapes)
    {
        const KeyStore<1> store = random_store(shape.norb, shape.num_alpha, shape.num_beta, shape.rows, shape.pool, seed++);
        for(int alpha_kind = 0; alpha_kind < 2; alpha_kind++)
        {
            for(int beta_kind = 0; beta_kind < 2; beta_kind++)
            {
                CAPTURE(shape.norb);
                CAPTURE(alpha_kind);
                CAPTURE(beta_kind);
                CHECK(check_pair_sets(store, {alpha_kind, beta_kind}) > 1000);
            }
        }
    }
}

TEST_CASE("Full-string groups: the key slices partition the pairs")
{
    const KeyStore<1> store = random_store(10, 5, 4, 1500, 60, 77);
    for(int alpha_kind = 0; alpha_kind < 2; alpha_kind++)
    {
        const GroupKinds kinds{alpha_kind, 1 - alpha_kind};
        const auto whole = fq::full_strs::sliced_stats(store, kinds, 1);
        const auto sliced = fq::full_strs::sliced_stats(store, kinds, 5);
        for(int cls = 0; cls < fq::full_strs::kNumClasses; cls++)
        {
            std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
            for(int index = 0; index < 5; index++)
            {
                const auto groups = fq::full_strs::build_groups(store, kinds, fq::full_strs::GroupSlice{5, index});
                const auto part = fq::full_strs::group_pairs(store, groups, cls);
                pairs.insert(pairs.end(), part.begin(), part.end());
            }
            std::sort(pairs.begin(), pairs.end());
            CHECK(pairs == fq::full_strs::brute_force_pairs(store, cls));
            CHECK(whole[static_cast<std::size_t>(cls)].pairs == sliced[static_cast<std::size_t>(cls)].pairs);
            CHECK(whole[static_cast<std::size_t>(cls)].outputs == sliced[static_cast<std::size_t>(cls)].outputs);
        }
    }
}

TEST_CASE("Full-string groups: chunks hold every group once, binned by size")
{
    const KeyStore<1> store = random_store(10, 5, 4, 2000, 50, 91);
    const GroupKinds kinds{fq::full_strs::kIntersection, fq::full_strs::kUnion};
    const fq::full_strs::GroupStore whole = fq::full_strs::build_groups(store, kinds);
    const auto chunks = fq::full_strs::build_chunks(store, kinds, 3);
    REQUIRE(chunks.size() == 3);
    for(int cls = 0; cls < fq::full_strs::kNumClasses; cls++)
    {
        std::vector<std::vector<std::int32_t>> expected;
        const auto& c = whole.classes[static_cast<std::size_t>(cls)];
        for(std::int64_t g = 0; g < c.num_groups(); g++)
        {
            expected.emplace_back(c.members.begin() + c.member_start[static_cast<std::size_t>(g)],
                                  c.members.begin() + c.member_start[static_cast<std::size_t>(g) + 1]);
        }
        std::vector<std::vector<std::int32_t>> stored;
        for(const auto& chunk : chunks)
        {
            const auto& cc = chunk.classes[static_cast<std::size_t>(cls)];
            CHECK(cc.bucket_start[0] == 0);
            CHECK(cc.bucket_start[fq::full_strs::kSizeBuckets] == cc.num_groups());
            std::int64_t offset = 0;
            for(int bucket = 0; bucket < fq::full_strs::kSizeBuckets; bucket++)
            {
                for(std::int64_t g = cc.bucket_start[static_cast<std::size_t>(bucket)]; g < cc.bucket_start[static_cast<std::size_t>(bucket) + 1]; g++)
                {
                    const int size = cc.sizes[static_cast<std::size_t>(g)];
                    CHECK(fq::full_strs::size_bucket(size) == bucket);
                    stored.emplace_back(cc.members.begin() + offset, cc.members.begin() + offset + size);
                    offset += size;
                }
            }
            CHECK(offset == static_cast<std::int64_t>(cc.members.size()));
        }
        std::sort(expected.begin(), expected.end());
        std::sort(stored.begin(), stored.end());
        CHECK(stored == expected);
    }
}

TEST_CASE("Full-string groups: default kinds")
{
    const KeyStore<1> store = random_store(10, 7, 3, 100, 10, 5);
    const GroupKinds kinds = fq::full_strs::default_kinds(store);
    CHECK(kinds.alpha == fq::full_strs::kUnion);
    CHECK(kinds.beta == fq::full_strs::kIntersection);
    const KeyStore<1> half = random_store(10, 5, 5, 100, 10, 6);
    CHECK(fq::full_strs::default_kinds(half).alpha == fq::full_strs::kIntersection);
}

TEST_CASE("Full-string groups: degenerate shapes")
{
    struct Shape
    {
        int norb;
        int num_alpha;
        int num_beta;
    };
    const std::vector<Shape> shapes = {{6, 6, 1}, {6, 1, 5}, {5, 0, 2}, {4, 1, 1}, {64, 64, 1}, {64, 1, 63}};
    unsigned int seed = 41;
    for(const Shape& shape : shapes)
    {
        const KeyStore<1> store = random_store(shape.norb, shape.num_alpha, shape.num_beta, 300, 12, seed++);
        for(int alpha_kind = 0; alpha_kind < 2; alpha_kind++)
        {
            for(int beta_kind = 0; beta_kind < 2; beta_kind++)
            {
                CAPTURE(shape.norb);
                CAPTURE(shape.num_alpha);
                CAPTURE(alpha_kind);
                CAPTURE(beta_kind);
                check_pair_sets(store, {alpha_kind, beta_kind});
            }
        }
    }
    const std::vector<Word> one_row = {0b011, 0b101};
    const KeyStore<1> single = fq::full_strs::build_key_store<1>(one_row.data(), 1, 3);
    const auto groups = fq::full_strs::build_groups(single, fq::full_strs::default_kinds(single));
    CHECK(groups.row_slots.empty());
}
