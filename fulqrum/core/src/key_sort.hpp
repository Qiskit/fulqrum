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
#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "key_words.hpp"
#include "checked_size.hpp"

inline constexpr std::size_t PARALLEL_SORT_MIN_ITEMS = std::size_t(1) << 16;

/**
 * Sorts a vector in parallel: one chunk per thread, then pairwise merges.
 *
 * @param items The items to sort.
 * @param less The strict weak order.
 */
template <typename T, typename Less>
void parallel_sort(std::vector<T>& items, const Less& less)
{
    const std::size_t num_items = items.size();
    const std::size_t num_chunks = static_cast<std::size_t>(max_threads());
    if(num_items < PARALLEL_SORT_MIN_ITEMS || num_chunks < 2)
    {
        std::sort(items.begin(), items.end(), less);
        return;
    }
    const std::size_t chunk_size = ceiling_divide(num_items, num_chunks);
#pragma omp parallel for schedule(static)
    for(std::size_t chunk = 0; chunk < num_chunks; chunk++)
    {
        const std::size_t begin = std::min(chunk * chunk_size, num_items);
        const std::size_t end = std::min(begin + chunk_size, num_items);
        std::sort(items.begin() + begin, items.begin() + end, less);
    }
    std::vector<T> buffer(num_items);
    for(std::size_t run = chunk_size; run < num_items; run = run > num_items / 2 ? num_items : run * 2)
    {
        const std::size_t num_pairs = ceiling_divide(num_items, run > num_items / 2 ? num_items : 2 * run);
#pragma omp parallel for schedule(dynamic)
        for(std::size_t pair = 0; pair < num_pairs; pair++)
        {
            const std::size_t begin = pair * 2 * run;
            const std::size_t middle = begin + std::min(run, num_items - begin);
            const std::size_t end = middle + std::min(run, num_items - middle);
            std::merge(items.begin() + begin,
                       items.begin() + middle,
                       items.begin() + middle,
                       items.begin() + end,
                       buffer.begin() + begin,
                       less);
        }
        items.swap(buffer);
    }
}

/**
 * Moves the keys of a flat key array into a new order, in place.
 *
 * @param keys The key words. Key i moves to the position where order holds i.
 * @param order For each new position, the old position of its key. The function
 * overwrites it.
 * @param num_words The number of words in each key.
 */
template <typename Word>
inline void permute_keys(std::vector<Word>& keys,
                         std::vector<std::size_t>& order,
                         const std::size_t num_words)
{
    if(keys.size() != checked_multiply(order.size(), num_words))
        throw std::invalid_argument("Key permutation shape mismatch.");
    if(order.empty()) return;
    std::vector<Word> held(num_words);
    for(std::size_t start = 0; start < order.size(); start++)
    {
        if(order[start] == start)
        {
            continue;
        }
        std::copy_n(keys.data() + start * num_words, num_words, held.data());
        std::size_t position = start;
        while(order[position] != start)
        {
            const std::size_t source = order[position];
            std::copy_n(keys.data() + source * num_words, num_words, keys.data() + position * num_words);
            order[position] = position;
            position = source;
        }
        std::copy_n(held.data(), num_words, keys.data() + position * num_words);
        order[position] = position;
    }
}

/**
 * Sorts a flat key array in numeric order and removes duplicate keys.
 *
 * @param keys The key words, num_words per key. The function resizes it.
 * @param width The number of bits in each key.
 * @return The number of unique keys.
 */
template <typename Word>
inline std::size_t sort_unique_keys(std::vector<Word>& keys, const std::size_t width)
{
    const std::size_t num_words = words_for_width(width);
    if(num_words == 0)
    {
        keys.clear();
        return 0;
    }
    if(keys.size() % num_words != 0)
        throw std::invalid_argument("Flat key array does not match its width.");
    const std::size_t num_keys = keys.size() / num_words;
    struct SortRecord
    {
        Word prefix;
        std::size_t position;
    };
    std::vector<SortRecord> records(num_keys);
#pragma omp parallel for schedule(static)
    for(std::size_t key = 0; key < num_keys; key++)
    {
        records[key] = {key_prefix(keys.data() + key * num_words, width), key};
    }
    const Word* key_words = keys.data();
    parallel_sort(records, [key_words, num_words](const SortRecord& a, const SortRecord& b) {
        if(a.prefix != b.prefix)
        {
            return a.prefix < b.prefix;
        }
        return compare_keys(key_words + a.position * num_words,
                            key_words + b.position * num_words,
                            num_words) < 0;
    });
    std::vector<std::size_t> order(num_keys);
#pragma omp parallel for schedule(static)
    for(std::size_t key = 0; key < num_keys; key++)
    {
        order[key] = records[key].position;
    }
    std::vector<SortRecord>().swap(records);
    permute_keys(keys, order, num_words);

    std::size_t num_unique = 0;
    for(std::size_t key = 0; key < num_keys; key++)
    {
        const Word* current = keys.data() + key * num_words;
        if(num_unique > 0 &&
           compare_keys(keys.data() + (num_unique - 1) * num_words, current, num_words) == 0)
        {
            continue;
        }
        std::copy_n(current, num_words, keys.data() + num_unique * num_words);
        num_unique++;
    }
    keys.resize(checked_multiply(num_unique, num_words));
    keys.shrink_to_fit();
    return num_unique;
}
