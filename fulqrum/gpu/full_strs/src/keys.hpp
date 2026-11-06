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
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../../../core/src/key_sort.hpp"

namespace fq::full_strs
{


using Word = std::uint64_t;

static_assert(sizeof(Word) == sizeof(KeyWord) && std::is_unsigned_v<Word>, "The key word must be a 64-bit unsigned integer");


inline constexpr int kWordBits = 64;


template <std::size_t W>
inline int compare_sector(const Word* a, const Word* b)
{
    for(std::size_t word = W; word-- > 0;)
    {
        if(a[word] != b[word])
        {
            return a[word] < b[word] ? -1 : 1;
        }
    }
    return 0;
}


template <std::size_t W>
inline int sector_popcount(const Word* words)
{
    int count = 0;
    for(std::size_t word = 0; word < W; word++)
    {
        count += __builtin_popcountll(words[word]);
    }
    return count;
}


template <std::size_t W>
inline bool has_outside_bits(const Word* words, const int num_orbitals)
{
    for(std::size_t word = 0; word < W; word++)
    {
        const int first = static_cast<int>(word) * kWordBits;
        if(num_orbitals >= first + kWordBits)
        {
            continue;
        }
        const Word allowed = num_orbitals <= first ? Word(0) : (Word(1) << (num_orbitals - first)) - 1;
        if((words[word] & ~allowed) != 0)
        {
            return true;
        }
    }
    return false;
}


template <std::size_t W>
struct KeyStore
{
    static constexpr std::size_t kWords = 2 * W;

    int num_orbitals = 0;
    int num_alpha = 0;
    int num_beta = 0;
    std::vector<Word> keys;
    std::vector<std::int64_t> line_start;
    std::vector<std::int64_t> line_of_row;
    std::vector<std::int64_t> column_start;
    std::vector<std::int64_t> column_rows;
    std::vector<std::int64_t> column_of_row;


    std::int64_t num_rows() const
    {
        return static_cast<std::int64_t>(keys.size() / kWords);
    }


    std::int64_t num_lines() const
    {
        return static_cast<std::int64_t>(line_start.size()) - 1;
    }


    std::int64_t num_columns() const
    {
        return static_cast<std::int64_t>(column_start.size()) - 1;
    }


    const Word* alpha(const std::int64_t row) const
    {
        return keys.data() + static_cast<std::size_t>(row) * kWords;
    }


    const Word* beta(const std::int64_t row) const
    {
        return alpha(row) + W;
    }


    const Word* line_beta(const std::int64_t line) const
    {
        return beta(line_start[static_cast<std::size_t>(line)]);
    }


    const Word* column_alpha(const std::int64_t column) const
    {
        return alpha(column_rows[static_cast<std::size_t>(column_start[static_cast<std::size_t>(column)])]);
    }
};


inline int orbitals_for_width(const std::size_t width, const int max_orbitals)
{
    if(width == 0 || width % 2 != 0)
    {
        throw std::invalid_argument("The operator width must be even and positive, got " +
                                    std::to_string(width));
    }
    if(width / 2 > static_cast<std::size_t>(max_orbitals))
    {
        throw std::invalid_argument("The operator has " + std::to_string(width / 2) +
                                    " orbitals per spin sector; the limit is " +
                                    std::to_string(max_orbitals));
    }
    return static_cast<int>(width / 2);
}


template <std::size_t W>
inline std::int64_t first_bad_row(const Word* keys, const std::int64_t num_rows, const int num_orbitals)
{
    const int num_alpha = sector_popcount<W>(keys);
    const int num_beta = sector_popcount<W>(keys + W);
    std::int64_t first = num_rows;
#pragma omp parallel for schedule(static) reduction(min : first)
    for(std::int64_t row = 0; row < num_rows; row++)
    {
        const Word* key = keys + static_cast<std::size_t>(row) * 2 * W;
        const bool bad = has_outside_bits<W>(key, num_orbitals) ||
                         has_outside_bits<W>(key + W, num_orbitals) ||
                         sector_popcount<W>(key) != num_alpha ||
                         sector_popcount<W>(key + W) != num_beta;
        if(bad && row < first)
        {
            first = row;
        }
    }
    return first == num_rows ? -1 : first;
}


template <std::size_t W>
inline void build_columns(KeyStore<W>& store)
{
    const std::int64_t num_rows = store.num_rows();
    struct ColumnRecord
    {
        Word prefix;
        std::int64_t row;
    };
    std::vector<ColumnRecord> records(static_cast<std::size_t>(num_rows));
#pragma omp parallel for schedule(static)
    for(std::int64_t row = 0; row < num_rows; row++)
    {
        records[static_cast<std::size_t>(row)] = {store.alpha(row)[W - 1], row};
    }
    const KeyStore<W>& view = store;
    parallel_sort(records, [&view](const ColumnRecord& a, const ColumnRecord& b) {
        if(a.prefix != b.prefix)
        {
            return a.prefix < b.prefix;
        }
        const int order = compare_sector<W>(view.alpha(a.row), view.alpha(b.row));
        if(order != 0)
        {
            return order < 0;
        }
        return a.row < b.row;
    });
    store.column_rows.resize(static_cast<std::size_t>(num_rows));
    store.column_of_row.resize(static_cast<std::size_t>(num_rows));
    store.column_start.clear();
    for(std::int64_t entry = 0; entry < num_rows; entry++)
    {
        const std::int64_t row = records[static_cast<std::size_t>(entry)].row;
        if(entry == 0 ||
           compare_sector<W>(store.alpha(row), store.alpha(store.column_rows[static_cast<std::size_t>(entry - 1)])) != 0)
        {
            store.column_start.push_back(entry);
        }
        store.column_rows[static_cast<std::size_t>(entry)] = row;
        store.column_of_row[static_cast<std::size_t>(row)] = static_cast<std::int64_t>(store.column_start.size()) - 1;
    }
    store.column_start.push_back(num_rows);
}


template <std::size_t W>
inline KeyStore<W> build_key_store(const Word* keys, const std::int64_t num_rows, const int num_orbitals)
{
    if(num_orbitals < 1 || num_orbitals > kWordBits * static_cast<int>(W))
    {
        throw std::invalid_argument("The number of orbitals per spin sector must be 1 to " +
                                    std::to_string(kWordBits * static_cast<int>(W)) + ", got " +
                                    std::to_string(num_orbitals));
    }
    if(num_rows <= 0)
    {
        throw std::invalid_argument("The key array has no rows");
    }
    const std::size_t words = checked_multiply(static_cast<std::size_t>(num_rows), 2 * W);
    checked_multiply(words, sizeof(Word));
    const std::int64_t bad = first_bad_row<W>(keys, num_rows, num_orbitals);
    if(bad >= 0)
    {
        const Word* key = keys + static_cast<std::size_t>(bad) * 2 * W;
        if(has_outside_bits<W>(key, num_orbitals) || has_outside_bits<W>(key + W, num_orbitals))
        {
            throw std::invalid_argument("Input row " + std::to_string(bad) +
                                        " occupies an orbital at or above " + std::to_string(num_orbitals));
        }
        throw std::invalid_argument(
            "Input row " + std::to_string(bad) + " has " + std::to_string(sector_popcount<W>(key)) +
            " alpha and " + std::to_string(sector_popcount<W>(key + W)) + " beta electrons; row 0 has " +
            std::to_string(sector_popcount<W>(keys)) + " and " + std::to_string(sector_popcount<W>(keys + W)));
    }
    KeyStore<W> store;
    store.num_orbitals = num_orbitals;
    store.num_alpha = sector_popcount<W>(keys);
    store.num_beta = sector_popcount<W>(keys + W);
    store.keys.assign(keys, keys + words);
    sort_unique_keys(store.keys, 2 * W * kWordBits);
    const std::int64_t num_unique = store.num_rows();
    store.line_of_row.resize(static_cast<std::size_t>(num_unique));
    for(std::int64_t row = 0; row < num_unique; row++)
    {
        if(row == 0 || compare_sector<W>(store.beta(row), store.beta(row - 1)) != 0)
        {
            store.line_start.push_back(row);
        }
        store.line_of_row[static_cast<std::size_t>(row)] = static_cast<std::int64_t>(store.line_start.size()) - 1;
    }
    store.line_start.push_back(num_unique);
    build_columns(store);
    return store;
}

}
