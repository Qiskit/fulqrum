#include "doctest.h"
#include "../fulqrum/gpu/full_strs/src/keys.hpp"
#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using fq::full_strs::KeyStore;
using fq::full_strs::Word;

/**
 * Makes a random string of norb bits with count bits set.
 *
 * @param engine The random engine.
 * @param norb The number of orbitals.
 * @param count The number of occupied orbitals.
 * @return The string.
 */
Word random_string(std::mt19937_64& engine, const int norb, const int count)
{
    std::vector<int> orbitals(static_cast<std::size_t>(norb));
    for(int orbital = 0; orbital < norb; orbital++)
    {
        orbitals[static_cast<std::size_t>(orbital)] = orbital;
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
 * Makes random (alpha, beta) rows from small string pools, with duplicates, in random order.
 *
 * @param num_rows The number of rows.
 * @param norb The number of orbitals per sector.
 * @param num_alpha The alpha electron count.
 * @param num_beta The beta electron count.
 * @param pool The number of distinct strings per sector.
 * @param seed The random seed.
 * @return The flat keys, alpha word then beta word per row.
 */
std::vector<Word> random_keys(const std::size_t num_rows,
                              const int norb,
                              const int num_alpha,
                              const int num_beta,
                              const std::size_t pool,
                              const unsigned int seed)
{
    std::mt19937_64 engine(seed);
    std::vector<Word> alphas(pool);
    std::vector<Word> betas(pool);
    for(std::size_t k = 0; k < pool; k++)
    {
        alphas[k] = random_string(engine, norb, num_alpha);
        betas[k] = random_string(engine, norb, num_beta);
    }
    std::vector<Word> keys;
    keys.reserve(2 * num_rows);
    for(std::size_t row = 0; row < num_rows; row++)
    {
        if(row > 0 && engine() % 10 == 0)
        {
            const std::size_t copy = engine() % row;
            keys.push_back(keys[2 * copy]);
            keys.push_back(keys[2 * copy + 1]);
            continue;
        }
        keys.push_back(alphas[engine() % pool]);
        keys.push_back(betas[engine() % pool]);
    }
    return keys;
}

/**
 * Checks every property of a key store against the input it was built from.
 *
 * @param store The store.
 * @param input The input keys.
 */
void check_store(const KeyStore<1>& store, const std::vector<Word>& input)
{
    std::set<std::pair<Word, Word>> expected;
    for(std::size_t row = 0; row < input.size() / 2; row++)
    {
        expected.insert({input[2 * row + 1], input[2 * row]});
    }
    const std::int64_t n = store.num_rows();
    REQUIRE(static_cast<std::size_t>(n) == expected.size());
    std::int64_t row = 0;
    for(const auto& [beta, alpha] : expected)
    {
        CHECK(store.beta(row)[0] == beta);
        CHECK(store.alpha(row)[0] == alpha);
        row++;
    }

    REQUIRE(store.line_start.front() == 0);
    REQUIRE(store.line_start.back() == n);
    REQUIRE(store.line_of_row.size() == static_cast<std::size_t>(n));
    for(std::int64_t line = 0; line < store.num_lines(); line++)
    {
        const std::int64_t begin = store.line_start[static_cast<std::size_t>(line)];
        const std::int64_t end = store.line_start[static_cast<std::size_t>(line) + 1];
        REQUIRE(begin < end);
        if(line > 0)
        {
            CHECK(store.line_beta(line - 1)[0] < store.line_beta(line)[0]);
        }
        for(std::int64_t r = begin; r < end; r++)
        {
            CHECK(store.line_of_row[static_cast<std::size_t>(r)] == line);
            CHECK(store.beta(r)[0] == store.line_beta(line)[0]);
        }
    }

    REQUIRE(store.column_start.front() == 0);
    REQUIRE(store.column_start.back() == n);
    std::vector<std::int64_t> seen(store.column_rows);
    std::sort(seen.begin(), seen.end());
    for(std::int64_t r = 0; r < n; r++)
    {
        REQUIRE(seen[static_cast<std::size_t>(r)] == r);
    }
    for(std::int64_t column = 0; column < store.num_columns(); column++)
    {
        const std::int64_t begin = store.column_start[static_cast<std::size_t>(column)];
        const std::int64_t end = store.column_start[static_cast<std::size_t>(column) + 1];
        REQUIRE(begin < end);
        if(column > 0)
        {
            CHECK(store.column_alpha(column - 1)[0] < store.column_alpha(column)[0]);
        }
        for(std::int64_t entry = begin; entry < end; entry++)
        {
            const std::int64_t r = store.column_rows[static_cast<std::size_t>(entry)];
            CHECK(store.alpha(r)[0] == store.column_alpha(column)[0]);
            CHECK(store.column_of_row[static_cast<std::size_t>(r)] == column);
            if(entry > begin)
            {
                const std::int64_t previous = store.column_rows[static_cast<std::size_t>(entry - 1)];
                CHECK(store.line_of_row[static_cast<std::size_t>(previous)] <
                      store.line_of_row[static_cast<std::size_t>(r)]);
            }
        }
    }
}

}

TEST_CASE("Full-string key store: sorted, unique, line and column views")
{
    struct Shape
    {
        std::size_t rows;
        int norb;
        int num_alpha;
        int num_beta;
        std::size_t pool;
    };
    const std::vector<Shape> shapes = {{1, 4, 2, 2, 1},
                                       {500, 10, 5, 5, 30},
                                       {500, 10, 4, 3, 30},
                                       {3000, 36, 27, 27, 200},
                                       {200000, 64, 16, 16, 2000},
                                       {200000, 64, 64, 1, 1}};
    unsigned int seed = 7;
    for(const Shape& shape : shapes)
    {
        const std::vector<Word> input =
            random_keys(shape.rows, shape.norb, shape.num_alpha, shape.num_beta, shape.pool, seed++);
        const KeyStore<1> store = fq::full_strs::build_key_store<1>(
            input.data(), static_cast<std::int64_t>(shape.rows), shape.norb);
        CHECK(store.num_orbitals == shape.norb);
        CHECK(store.num_alpha == shape.num_alpha);
        CHECK(store.num_beta == shape.num_beta);
        check_store(store, input);
    }
}

TEST_CASE("Full-string key store: refusal cases")
{
    const std::vector<Word> good = {0b0011, 0b0101, 0b0110, 0b1001};
    CHECK_NOTHROW(fq::full_strs::build_key_store<1>(good.data(), 2, 4));
    CHECK_THROWS_AS(fq::full_strs::build_key_store<1>(good.data(), 2, 0), std::invalid_argument);
    CHECK_THROWS_AS(fq::full_strs::build_key_store<1>(good.data(), 2, 65), std::invalid_argument);
    CHECK_THROWS_AS(fq::full_strs::build_key_store<1>(good.data(), 0, 4), std::invalid_argument);
    CHECK_THROWS_AS(fq::full_strs::build_key_store<1>(good.data(), 2, 3), std::invalid_argument);

    const std::vector<Word> outside_alpha = {0b0011, 0b0101, 0b10010, 0b1001};
    CHECK_THROWS_AS(fq::full_strs::build_key_store<1>(outside_alpha.data(), 2, 4), std::invalid_argument);
    const std::vector<Word> outside_beta = {0b0011, 0b0101, 0b0110, 0b10001};
    CHECK_THROWS_AS(fq::full_strs::build_key_store<1>(outside_beta.data(), 2, 4), std::invalid_argument);
    const std::vector<Word> mixed_alpha = {0b0011, 0b0101, 0b0111, 0b1001};
    CHECK_THROWS_AS(fq::full_strs::build_key_store<1>(mixed_alpha.data(), 2, 4), std::invalid_argument);
    const std::vector<Word> mixed_beta = {0b0011, 0b0101, 0b0110, 0b1000};
    CHECK_THROWS_AS(fq::full_strs::build_key_store<1>(mixed_beta.data(), 2, 4), std::invalid_argument);

    const std::vector<Word> full_word = {~Word(0), Word(1) << 63};
    CHECK_NOTHROW(fq::full_strs::build_key_store<1>(full_word.data(), 1, 64));
    CHECK_THROWS_AS(fq::full_strs::build_key_store<1>(full_word.data(), 1, 63), std::invalid_argument);

    CHECK(fq::full_strs::orbitals_for_width(72, 64) == 36);
    CHECK_THROWS_AS(fq::full_strs::orbitals_for_width(0, 64), std::invalid_argument);
    CHECK_THROWS_AS(fq::full_strs::orbitals_for_width(71, 64), std::invalid_argument);
    CHECK_THROWS_AS(fq::full_strs::orbitals_for_width(130, 64), std::invalid_argument);
}

TEST_CASE("Full-string key store: two words per sector")
{
    const std::vector<Word> rows = {Word(0), Word(1) << 3, Word(1), Word(0),
                                    Word(1), Word(0),      Word(0), Word(1) << 5,
                                    Word(0), Word(1) << 3, Word(1), Word(0)};
    const KeyStore<2> store = fq::full_strs::build_key_store<2>(rows.data(), 3, 100);
    REQUIRE(store.num_rows() == 2);
    CHECK(store.beta(0)[0] == 1);
    CHECK(store.beta(1)[1] == Word(1) << 5);
    CHECK(store.num_lines() == 2);
    CHECK(store.num_columns() == 2);
    CHECK(store.column_alpha(0)[0] == 1);
    CHECK(store.column_alpha(1)[1] == Word(1) << 3);
    CHECK_THROWS_AS(fq::full_strs::build_key_store<2>(rows.data(), 3, 67), std::invalid_argument);
}


TEST_CASE("Full-string key size overflow precedes input access")
{
    CHECK_THROWS_AS(fq::full_strs::build_key_store<1>(nullptr, std::numeric_limits<std::int64_t>::max(), 4), std::overflow_error);
}
