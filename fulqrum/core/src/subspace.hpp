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
#pragma once
#include "bitset_hashmap.hpp"
#include "constants.hpp"
#include "key_position_table.hpp"
#include "key_words.hpp"
#include <algorithm>
#include <array>
#include <boost/dynamic_bitset.hpp>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

/** @struct subspace
 * @brief Data structure for subspace defined by counts
 *
 * @var bitstrings The subspace bit-strings stored in a hash table
 * @var num_qubits The number of qubits, i.e length of bitstrings
 * @var size Dimension / number of bit-strings in the subspace
 */
typedef struct Subspace
{
    bitset_map_namespace::BitsetHashMapWrapper bitstrings;
    width_t num_qubits = 0;
    std::size_t size = 0;
    width_t alpha_width = 0;
    bool product = false;
    bool map_materialized = false;
    std::size_t map_build_count = 0;
    int reserve_multiplier = 2;
    std::vector<KeyWord> full_keys;
    std::array<std::vector<KeyWord>, 2> half_keys;
    std::array<std::size_t, 2> half_sizes = {0, 0};
    std::array<KeyPositionTable, 2> half_positions;
    mutable std::mutex storage_mutex;

    Subspace() = default;
    Subspace(const Subspace& other)
    {
        other.copy_to(*this);
    }
    Subspace& operator=(const Subspace& other)
    {
        if(this != &other)
            other.copy_to(*this);
        return *this;
    }

    static std::size_t checked_product(std::size_t a, std::size_t b)
    {
        if(b && a > MAX_SIZE_T / b)
            throw std::length_error("Subspace size exceeds size_t.");
        return a * b;
    }

    static void validate_strings(const std::vector<std::string>& strings, std::size_t width)
    {
        if(width == 0 || width > MAX_WIDTH)
            throw std::invalid_argument("Bitstring width must be between 1 and 65535.");
        for(std::size_t i = 0; i < strings.size(); i++)
        {
            if(strings[i].size() != width ||
               strings[i].find_first_not_of("01") != std::string::npos)
                throw std::invalid_argument(
                    "Bitstrings must have consistent widths and contain only 0 and 1.");
            if(i && strings[i] == strings[i - 1])
                throw std::invalid_argument("Subspace bitstrings must be unique.");
        }
    }

    void configure(int multiplier, bool all_blocks)
    {
        if(multiplier < 1)
            throw std::invalid_argument("reserve_multiplier must be >= 1.");
        reserve_multiplier = multiplier;
        bitstrings.use_all_blocks = all_blocks;
    }

    void initialize_full(const std::vector<std::string>& strings)
    {
        if(strings.empty())
            return;
        validate_strings(strings, strings[0].size());
        num_qubits = static_cast<width_t>(strings[0].size());
        size = strings.size();
        full_keys.resize(checked_product(size, words_for_width(num_qubits)));
        for(std::size_t i = 0; i < size; i++)
        {
            KeyBitset key(strings[i]);
            std::copy(
                key.m_bits.begin(), key.m_bits.end(), full_keys.begin() + i * key.num_blocks());
        }
    }

    void initialize_product(const std::vector<std::string>& alpha,
                            const std::vector<std::string>& beta)
    {
        if(alpha.empty() || beta.empty())
            throw std::invalid_argument("Half-string lists must be nonempty.");
        validate_strings(alpha, alpha[0].size());
        validate_strings(beta, beta[0].size());
        const std::size_t width = alpha[0].size() + beta[0].size();
        if(width > MAX_WIDTH)
            throw std::invalid_argument("Subspace width must be <= 65535.");
        num_qubits = static_cast<width_t>(width);
        alpha_width = static_cast<width_t>(alpha[0].size());
        size = checked_product(alpha.size(), beta.size());
        product = true;
        half_sizes = {alpha.size(), beta.size()};
        for(int sector = 0; sector < 2; sector++)
        {
            const auto& strings = sector == 0 ? alpha : beta;
            const std::size_t words = words_for_width(strings[0].size());
            half_keys[sector].resize(checked_product(strings.size(), words));
            for(std::size_t i = 0; i < strings.size(); i++)
            {
                KeyBitset key(strings[i]);
                std::copy(
                    key.m_bits.begin(), key.m_bits.end(), half_keys[sector].begin() + i * words);
            }
            half_positions[sector].build(half_keys[sector].data(), strings.size(), words);
        }
    }

    void load_key_unlocked(std::size_t index, KeyBitset& key) const
    {
        if(index >= size)
            throw std::out_of_range("Subspace index is out of range.");
        if(map_materialized && !product)
        {
            key = bitstrings.get_n_th_bitset(index);
            return;
        }
        key.resize(num_qubits);
        if(product)
        {
            std::fill(key.m_bits.begin(), key.m_bits.end(), KeyWord(0));
            const std::size_t aw = words_for_width(alpha_width);
            const std::size_t bw = words_for_width(num_qubits - alpha_width);
            std::copy_n(half_keys[0].data() + (index % half_sizes[0]) * aw, aw, key.m_bits.data());
            or_shifted_key(half_keys[1].data() + (index / half_sizes[0]) * bw,
                           bw,
                           alpha_width,
                           key.m_bits.data(),
                           key.num_blocks());
        }
        else
        {
            std::copy_n(
                full_keys.data() + index * key.num_blocks(), key.num_blocks(), key.m_bits.data());
        }
    }

    void load_key(std::size_t index, KeyBitset& key) const
    {
        if(product)
        {
            load_key_unlocked(index, key);
            return;
        }
        std::lock_guard<std::mutex> lock(storage_mutex);
        load_key_unlocked(index, key);
    }

    KeyBitset key_at(std::size_t index) const
    {
        KeyBitset key;
        load_key(index, key);
        return key;
    }

    void ensure_hashmap()
    {
        ensure_hashmap_with([](auto& pending, const KeyBitset& key, std::size_t row) {
            pending.insert_unique(key, row);
        });
    }

    template <typename Insert>
    void ensure_hashmap_with(const Insert& insert)
    {
        std::lock_guard<std::mutex> lock(storage_mutex);
        if(map_materialized)
            return;
#ifdef _OPENMP
        if(omp_in_parallel())
            throw std::runtime_error("Hashmap materialization must precede parallel execution.");
#endif
        const std::size_t capacity =
            checked_product(size, static_cast<std::size_t>(reserve_multiplier));
        if(capacity == MAX_SIZE_T)
            throw std::length_error("Hashmap reserve exceeds size_t.");
        bitset_map_namespace::BitsetHashMapWrapper pending(bitstrings.use_all_blocks);
        pending.reserve(capacity + 1);
        KeyBitset key;
        for(std::size_t i = 0; i < size; i++)
        {
            load_key_unlocked(i, key);
            insert(pending, key, i);
        }
        bitstrings.map.swap(pending.map);
        bitstrings.map2.swap(pending.map2);
        map_materialized = true;
        map_build_count++;
        if(!product)
            std::vector<KeyWord>().swap(full_keys);
    }

    void prepare_mutation()
    {
        ensure_hashmap();
        product = false;
        half_sizes = {0, 0};
        half_keys = {};
        half_positions = {};
        alpha_width = 0;
    }

    void copy_to(Subspace& out) const
    {
        if(this == &out)
            return;
        std::scoped_lock lock(storage_mutex, out.storage_mutex);
        out.num_qubits = num_qubits;
        out.size = size;
        out.alpha_width = alpha_width;
        out.product = product;
        out.reserve_multiplier = reserve_multiplier;
        out.half_keys = half_keys;
        out.half_sizes = half_sizes;
        out.half_positions = half_positions;
        bitset_map_namespace::BitsetHashMapWrapper empty(bitstrings.use_all_blocks);
        out.bitstrings.map.swap(empty.map);
        out.bitstrings.map2.swap(empty.map2);
        out.bitstrings.use_all_blocks = bitstrings.use_all_blocks;
        out.map_materialized = false;
        out.map_build_count = 0;
        out.full_keys.clear();
        if(!product)
        {
            const std::size_t words = words_for_width(num_qubits);
            out.full_keys.resize(checked_product(size, words));
            KeyBitset key;
            for(std::size_t i = 0; i < size; i++)
            {
                load_key_unlocked(i, key);
                std::copy_n(key.m_bits.data(), words, out.full_keys.data() + i * words);
            }
        }
    }

    std::string half_string(int sector, std::size_t index) const
    {
        if(!product || index >= half_sizes.at(sector))
            throw std::out_of_range("Half-string index out of range.");
        KeyBitset bits(sector == 0 ? alpha_width : num_qubits - alpha_width);
        std::copy_n(half_keys.at(sector).data() + index * bits.num_blocks(),
                    bits.num_blocks(),
                    bits.m_bits.data());
        std::string text;
        boost::to_string(bits, text);
        return text;
    }

    KeyWord half_word(int sector, std::size_t index) const
    {
        if(!product || index >= half_sizes.at(sector))
            throw std::out_of_range("Half-string index out of range.");
        return half_keys.at(
            sector)[index * words_for_width(sector == 0 ? alpha_width : num_qubits - alpha_width)];
    }

    std::size_t half_size(int sector) const
    {
        return half_sizes.at(sector);
    }
} Subspace_t;

class HalfKeyView
{
public:
    explicit HalfKeyView(const Subspace_t& subspace)
        : owner_(&subspace)
    {}
    width_t width() const
    {
        return owner_->num_qubits;
    }
    width_t split_bit() const
    {
        return owner_->alpha_width;
    }
    bool is_tensor_product() const
    {
        return owner_->product;
    }
    std::size_t size() const
    {
        return owner_->size;
    }
    std::size_t num_words() const
    {
        return words_for_width(width());
    }
    std::size_t num_half_keys(int sector) const
    {
        return owner_->half_sizes.at(sector);
    }
    std::size_t half_words(int sector) const
    {
        return words_for_width(sector == 0 ? split_bit() : width() - split_bit());
    }
    const KeyWord* half_key(int sector, std::size_t index) const
    {
        if(!is_tensor_product() || index >= num_half_keys(sector))
            throw std::out_of_range("Half-key index out of range.");
        return owner_->half_keys.at(sector).data() + index * half_words(sector);
    }
    std::size_t find_half_key(int sector, const KeyWord* key) const
    {
        return owner_->half_positions.at(sector).find(
            owner_->half_keys.at(sector).data(), half_words(sector), key);
    }
    KeyBitset key(std::size_t index) const
    {
        return owner_->key_at(index);
    }
    void load_key(std::size_t index, KeyBitset& key) const
    {
        owner_->load_key(index, key);
    }

private:
    const Subspace_t* owner_;
};

inline std::size_t subspace_find_key(Subspace_t& space, const KeyBitset& key)
{
    if(space.size == 0)
        return MAX_SIZE_T;
    if(key.size() != space.num_qubits)
        return MAX_SIZE_T;
    if(space.product)
    {
        const HalfKeyView view(space);
        std::vector<KeyWord> a(view.half_words(0)), b(view.half_words(1));
        copy_bit_range(key.m_bits.data(), 0, space.alpha_width, a.data(), a.size());
        copy_bit_range(key.m_bits.data(),
                       space.alpha_width,
                       space.num_qubits - space.alpha_width,
                       b.data(),
                       b.size());
        const std::size_t ai = view.find_half_key(0, a.data()),
                          bi = view.find_half_key(1, b.data());
        return ai == MAX_SIZE_T || bi == MAX_SIZE_T ? MAX_SIZE_T : bi * view.num_half_keys(0) + ai;
    }
    space.ensure_hashmap();
    return space.bitstrings.get(key);
}
