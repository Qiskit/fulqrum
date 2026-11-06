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

#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#    define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif

#include <boost/dynamic_bitset.hpp>
#include <cassert>

#include "./external/hash_table8.hpp"
#include "./external/rapidhash.h"
#include "constants.hpp"

struct BitsetHasherRapid
{
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& bs) const
    {
        const std::size_t num_bytes = bs.num_blocks() * sizeof(std::size_t);

        return rapidhashMicro(bs.m_bits.data(), num_bytes);
    }
};

struct BitsetHasherRapidFirstBlock
{
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& bs) const
    {
        return rapidhashNano(bs.m_bits.data(), 8);
    }
};

namespace bitset_map_namespace
{
using Bitset = boost::dynamic_bitset<std::size_t>;

using BitsetMap = emhash8::HashMap<Bitset, std::size_t, BitsetHasherRapid>;
using BitsetMap2 = emhash8::HashMap<Bitset, std::size_t, BitsetHasherRapidFirstBlock>;

class BitsetHashMapWrapper
{
public:
    // hashes all bitset blocks.
    // Can be slower for long bitsets.
    // Has better hash value distribution,
    // i.e., less hash collision
    BitsetMap map;
    // hashes only the first bitset block.
    // Faster for long bitsets.
    // May lead to more hash collisions
    BitsetMap2 map2;
    bool use_all_blocks;

    BitsetHashMapWrapper(bool use_all_bitset_blocks = true)
    {
        use_all_blocks = use_all_bitset_blocks;
    }

    // Gets a pointer to the internal
    // data-structure that holds all
    // (key, value) pairs.
    const auto* get_bitsets() const
    {
        if(use_all_blocks)
        {
            return map.values();
        }
        return map2.values();
    }

    // Reserves capacity for specified
    // number of items (key, value) pairs.
    // We typically know the number of
    // bitsets in a Subspace beforehand
    // (exception: RAMPS routine).
    // Reserving fixed capacity serves
    // two purposes:
    // (1) prevents resizing of the HashMap
    //	during insertion.
    // (2) allows us to fix number of bits
    //	in ``_bucket_occupancy`` member
    //	variable in the HashMap (see notes
    //	in external/hash_table8.hpp
    //	for details)
    // Note that emhash8::HashMap is a power
    // of 2-based hashmap. Therefore, the
    // actually reserved space is larger
    // (some power-of-2) ``num_items``
    void reserve(const uint64_t num_items)
    {
        if(use_all_blocks)
        {
            const bool occupied = map.size() != 0;
            if(!map.reserve(num_items, true) || occupied) map.set_bucket_occupancy();
        }
        else
        {
            const bool occupied = map2.size() != 0;
            if(!map2.reserve(num_items, true) || occupied) map2.set_bucket_occupancy();
        }
    }

    // Inserts (key, value) pairs where each key must be unique.
    // Offers slightly faster insertion when we know our
    // keys are unique.
    void insert_unique(const Bitset& bs, std::size_t value)
    {
        if(use_all_blocks)
        {
            map.emplace_unique(bs, value);
        }
        else
        {
            map2.emplace_unique(bs, value);
        }
    }

    // Inserts a (key, value) pair into the HashMap
    // Replaces values of existing keys.
    void emplace(const Bitset& bs, std::size_t value)
    {
        if(use_all_blocks)
        {
            map.emplace(std::make_pair(bs, value));
        }
        else
        {
            map2.emplace(std::make_pair(bs, value));
        }
    }

    // Takes a key (Bitset) as input and returns
    // a pointer to the value if the key is found.
    // Else, returns a nullptr.
    // Uses the custom try_get_using_bucket_occ(<Key>)
    // method to reject non-existing keys faster.
    // This method can only be used if reserve()
    // method is called on the HashMap beforehand
    // to fix capacity.
    // See notes in external/hash_table8.hpp
    // for details.
    std::size_t* get_ptr(const Bitset& bs) const
    {
        if(use_all_blocks)
        {
            return map.try_get_using_bucket_occ(bs);
        }
        return map2.try_get_using_bucket_occ(bs);
    }

    // Takes a key (Bitset) as input and returns
    // a pointer to the value if the key is found.
    // Else, returns a nullptr.
    // Works even if capacity is not fixed with
    // reserve(). Slower than ``get_ptr()``.
    std::size_t* get_ptr2(const Bitset& bs) const
    {
        if(use_all_blocks)
        {
            return map.try_get(bs);
        }
        return map2.try_get(bs);
    }

    // Returns the value if the key exists.
    // Returns MAX_SIZE_T if the key is not
    // found.
    std::size_t get(const Bitset& bs) const
    {
        if(use_all_blocks)
        {
            auto it = map.find(bs);
            if(it == map.end())
                return MAX_SIZE_T;
            return it->second;
        }
        auto it = map2.find(bs);
        if(it == map2.end())
            return MAX_SIZE_T;
        return it->second;
    }

    // emhash8::HashMap preserves insertion order
    // of (key, value) pairs.
    // Therefore, it is possible to access n-th
    // pair in the HashMap similar to arrays/vectors.
    // This method takes an index ``n`` and returns
    // the n-th key (Bitset).
    Bitset get_n_th_bitset(std::size_t n) const
    {
        if(use_all_blocks)
        {
            assert(n < map.size());
            const auto* keys = map.values();
            return keys[n].first;
        }
        assert(n < map2.size());
        const auto* keys = map2.values();
        return keys[n].first;
    }

    // Utility function to set
    // _bucket_occupancy member
    // in the HashMap manually
    // after the map is fully
    // constructed.
    // When we do not know the
    // number of items in the
    // HashMap a priori, we can
    // first construct the map
    // and then call this method
    // to set bucket occupancy.
    // Setting bucket occupancy
    // allows us to use the
    // faster ``get_ptr()``
    // method during lookup.
    void set_bucket_occupancy()
    {
        if(use_all_blocks)
        {
            map.set_bucket_occupancy();
        }
        else
        {
            map2.set_bucket_occupancy();
        }
    }

    // Utility function to get
    // the size (number of (key
    // value) pairs) of the
    // HashMap.
    std::size_t size() const
    {
        if(use_all_blocks)
        {
            return map.size();
        }
        return map2.size();
    }

    // Utility function to check
    // whether the HashMap is
    // hashing all Bitset blocks
    // or only the first one.
    bool use_all_bitset_blocks() const
    {
        if(use_all_blocks)
        {
            return true;
        }
        return false;
    }

    // Utility function to get
    // the number of buckets
    // in the HashMap.
    std::size_t num_buckets() const
    {
        if(use_all_blocks)
        {
            return map.get_num_buckets();
        }
        return map2.get_num_buckets();
    }
};
} // namespace bitset_map_namespace
