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

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../checked_size.hpp"
#include "../operator_tables.hpp"
#include "packed_entries.hpp"
#include "../tier_utils.hpp"

/**
 * Maps a key to a group with open addressing. Its size follows the number of keys.
 */
struct GroupHash
{
    static constexpr std::uint64_t EMPTY = ~std::uint64_t(0);
    std::vector<std::uint64_t> keys;
    std::vector<std::uint32_t> groups;
    unsigned int shift = 64;

    /**
     * Makes an empty map with room for a number of keys at a load of at most one half.
     *
     * @param num_keys The number of keys that the map will hold.
     */
    void reserve(const std::size_t num_keys)
    {
        std::size_t capacity = 2;
        shift = 63;

        while(capacity < 2 * num_keys)
        {
            capacity *= 2;
            shift--;
        }

        keys.assign(capacity, EMPTY);
        groups.assign(capacity, NO_GROUP);
    }

    /**
     * Returns the first slot to probe for a key.
     */
    std::size_t home(const std::uint64_t key) const
    {
        return static_cast<std::size_t>((key * 0x9e3779b97f4a7c15ull) >> shift);
    }

    /**
     * Sets the group of a key.
     *
     * @param key The key. It must not be EMPTY.
     * @param group The group.
     */
    void insert(const std::uint64_t key, const std::uint32_t group)
    {
        const std::size_t mask = keys.size() - 1;
        std::size_t slot = home(key);

        while(keys[slot] != EMPTY && keys[slot] != key)
        {
            slot = (slot + 1) & mask;
        }

        keys[slot] = key;
        groups[slot] = group;
    }

    /**
     * Returns the group of a key, or NO_GROUP.
     *
     * @param key The key.
     */
    std::uint32_t find(const std::uint64_t key) const
    {
        if(keys.empty())
        {
            return NO_GROUP;
        }

        const std::size_t mask = keys.size() - 1;

        for(std::size_t slot = home(key);; slot = (slot + 1) & mask)
        {
            if(keys[slot] == key)
            {
                return groups[slot];
            }

            if(keys[slot] == EMPTY)
            {
                return NO_GROUP;
            }
        }
    }
};

/**
 * Maps the key of a double excitation to its group: one bit per possible key tells
 * whether a group exists, and a hash over the actual groups gives it.
 */
struct DoubleGroupIndex
{
    std::vector<std::uint64_t> present;
    GroupHash groups;

    /**
     * Makes an empty index for keys below a bound, with room for a number of groups.
     *
     * @param num_keys The bound of the keys.
     * @param num_groups The number of groups that the index will hold.
     */
    void reserve(const std::size_t num_keys, const std::size_t num_groups)
    {
        present.assign(ceiling_divide(num_keys, 64), 0);
        groups.reserve(num_groups);
    }

    /**
     * Sets the group of a key.
     *
     * @param key The key.
     * @param group The group.
     */
    void insert(const std::uint64_t key, const std::uint32_t group)
    {
        present[key / 64] |= std::uint64_t(1) << (key % 64);
        groups.insert(key, group);
    }

    /**
     * Returns the group of a key, or NO_GROUP.
     *
     * @param key The key, below the bound.
     */
    std::uint32_t find(const std::uint64_t key) const
    {
        if(((present[key / 64] >> (key % 64)) & 1u) == 0)
        {
            return NO_GROUP;
        }

        return groups.find(key);
    }
};

struct SectorGroupIndex
{
    std::array<std::vector<std::uint32_t>, 2> single;
    std::array<DoubleGroupIndex, 2> double_excitation;
};

inline SectorGroupIndex index_sector_groups(const OperatorTables& tables,
                                            const std::array<std::size_t, 2>& num_pairs)
{
    SectorGroupIndex index;
    std::array<std::size_t, 2> num_doubles = {0, 0};

    for(std::size_t group = 0; group < tables.num_groups(); group++)
    {
        const GroupData& plan = tables.group(group);

        if(plan.tier == GroupTier::same_sector_double)
        {
            num_doubles[plan.sector]++;
        }
    }

    for(int sector = 0; sector < 2; sector++)
    {
        index.single[sector].assign(num_pairs[sector], NO_GROUP);
        index.double_excitation[sector].reserve(checked_multiply(num_pairs[sector], num_pairs[sector]), num_doubles[sector]);
    }

    const std::vector<width_t>& rank = tables.layout().rank_in_sector;

    for(std::size_t group = 0; group < tables.num_groups(); group++)
    {
        const GroupData& plan = tables.group(group);

        if(plan.tier == GroupTier::same_sector_single)
        {
            index.single[plan.sector][rank_pair_index(rank[plan.flips[0]], rank[plan.flips[1]])] =
                static_cast<std::uint32_t>(group);
        }
        else if(plan.tier == GroupTier::same_sector_double)
        {
            const std::size_t low = rank_pair_index(rank[plan.flips[0]], rank[plan.flips[1]]);
            const std::size_t high = rank_pair_index(rank[plan.flips[2]], rank[plan.flips[3]]);
            index.double_excitation[plan.sector].insert(low * num_pairs[plan.sector] + high,
                                                        static_cast<std::uint32_t>(group));
        }
    }

    return index;
}
