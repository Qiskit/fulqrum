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
#include "../fulqrum/core/src/subspace.hpp"
#include <algorithm>
#include <string>
#include <vector>

struct SubspaceFixture
{
    mutable Subspace_t storage;
    HalfKeyView view;
    SubspaceFixture()
        : view(storage)
    {}
    SubspaceFixture(const SubspaceFixture& other)
        : storage(other.storage)
        , view(storage)
    {}
    operator const HalfKeyView&() const
    {
        return view;
    }
    std::size_t size() const
    {
        return storage.size;
    }
    std::size_t num_words() const
    {
        return view.num_words();
    }
    std::size_t half_words(int sector) const
    {
        return view.half_words(sector);
    }
    std::size_t num_half_keys(int sector) const
    {
        return view.num_half_keys(sector);
    }
    KeyBitset key(std::size_t i) const
    {
        return view.key(i);
    }
    std::size_t find(const KeyBitset& key) const
    {
        return subspace_find_key(storage, key);
    }

    static std::vector<std::string> strings(const std::vector<KeyWord>& keys, std::size_t width)
    {
        std::vector<std::string> out;
        const std::size_t words = words_for_width(width);
        for(std::size_t i = 0; i < keys.size(); i += words)
        {
            KeyBitset key(width);
            std::copy_n(keys.data() + i, words, key.m_bits.data());
            std::string text;
            boost::to_string(key, text);
            out.push_back(text);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    static SubspaceFixture from_halves(const std::vector<KeyWord>& alpha,
                                     std::size_t aw,
                                     const std::vector<KeyWord>& beta,
                                     std::size_t bw)
    {
        SubspaceFixture fixture;
        fixture.storage.initialize_product(strings(alpha, aw), strings(beta, bw));
        return fixture;
    }

    static SubspaceFixture
    from_keys(const std::vector<KeyWord>& keys, std::size_t width, std::size_t split)
    {
        SubspaceFixture fixture;
        fixture.storage.initialize_full(strings(keys, width));
        fixture.storage.alpha_width = split;
        return fixture;
    }
};
