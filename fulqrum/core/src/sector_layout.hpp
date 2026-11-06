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
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "base.hpp"
#include "constants.hpp"
#include "subspace.hpp"

/**
 * How the qubits split into the two spin sectors.
 */
enum class SpinLayout : int
{
    automatic = 0,
    block = 1,
    interleaved = 2,
    spinless = 3,
};

/**
 * The split of the qubits into sector 0 (alpha) and sector 1 (beta).
 */
struct SectorLayout
{
    bool valid = false;
    std::vector<std::uint8_t> sector_of;
    std::vector<width_t> rank_in_sector;
    std::array<width_t, 2> num_orbitals = {0, 0};

    /**
     * Tells whether sector 0 holds the low qubits and sector 1 the high qubits.
     */
    bool is_block() const
    {
        return std::is_sorted(sector_of.begin(), sector_of.end());
    }
};

/**
 * Tells whether a group keeps its flips inside the sectors that the sector tiers support.
 *
 * Supported: two flips in one sector, four flips in one sector, or two flips in each.
 *
 * @param flips The flip qubits of the group.
 * @param sector_of The sector of each qubit.
 * @return True when the group is supported.
 */
inline bool group_fits_sectors(const std::vector<width_t>& flips,
                               const std::vector<std::uint8_t>& sector_of)
{
    std::size_t in_sector_one = 0;
    for(const width_t qubit : flips)
    {
        in_sector_one += sector_of[qubit];
    }
    const std::size_t in_sector_zero = flips.size() - in_sector_one;
    if(flips.size() == 2)
    {
        return in_sector_zero != 1;
    }
    if(flips.size() == 4)
    {
        return in_sector_zero % 2 == 0;
    }
    return false;
}

/**
 * Tells whether no two-flip or four-flip group crosses the sectors.
 *
 * A group crosses when its two flips are in different sectors, or when its four
 * flips split three and one. Groups of other sizes are ignored.
 *
 * @param group_flips The flip qubits of each group.
 * @param sector_of The sector of each qubit.
 */
inline bool no_group_crosses_sectors(const std::vector<std::vector<width_t>>& group_flips,
                                     const std::vector<std::uint8_t>& sector_of)
{
    for(const std::vector<width_t>& flips : group_flips)
    {
        const bool counted = flips.size() == 2 || flips.size() == 4;
        if(counted && !group_fits_sectors(flips, sector_of))
        {
            return false;
        }
    }
    return true;
}

/**
 * Returns the sectors of a block layout: qubits below the split are sector 0.
 *
 * @param width The number of qubits.
 * @param split The first qubit of sector 1.
 */
inline std::vector<std::uint8_t> block_sectors(const width_t width, const width_t split)
{
    std::vector<std::uint8_t> sector_of(width, 0);
    std::fill(sector_of.begin() + split, sector_of.end(), std::uint8_t(1));
    return sector_of;
}

/**
 * Returns the sectors of an interleaved layout: even qubits are sector 0.
 *
 * @param width The number of qubits.
 */
inline std::vector<std::uint8_t> interleaved_sectors(const width_t width)
{
    std::vector<std::uint8_t> sector_of(width);
    for(width_t qubit = 0; qubit < width; qubit++)
    {
        sector_of[qubit] = static_cast<std::uint8_t>(qubit % 2);
    }
    return sector_of;
}

/**
 * Derives the sectors from the operator by union-find over its two-flip groups.
 *
 * @param group_flips The flip qubits of each group.
 * @param width The number of qubits.
 * @return The sectors, or an empty vector when a group crosses every candidate.
 */
inline std::vector<std::uint8_t>
derive_sectors(const std::vector<std::vector<width_t>>& group_flips, const width_t width)
{
    std::vector<width_t> parent(width);
    for(width_t qubit = 0; qubit < width; qubit++)
    {
        parent[qubit] = qubit;
    }
    auto root_of = [&parent](width_t qubit) {
        while(parent[qubit] != qubit)
        {
            parent[qubit] = parent[parent[qubit]];
            qubit = parent[qubit];
        }
        return qubit;
    };
    for(const std::vector<width_t>& flips : group_flips)
    {
        if(flips.size() == 2)
        {
            parent[root_of(flips[0])] = root_of(flips[1]);
        }
    }
    std::vector<std::size_t> members(width, 0);
    for(width_t qubit = 0; qubit < width; qubit++)
    {
        members[root_of(qubit)]++;
    }
    std::vector<width_t> joined_roots;
    for(width_t qubit = 0; qubit < width; qubit++)
    {
        const width_t root = root_of(qubit);
        const bool seen = std::find(joined_roots.begin(), joined_roots.end(), root) != joined_roots.end();
        if(members[root] > 1 && !seen)
        {
            joined_roots.push_back(root);
        }
    }
    if(joined_roots.size() == 2)
    {
        std::vector<std::uint8_t> sector_of(width);
        for(width_t qubit = 0; qubit < width; qubit++)
        {
            const width_t root = root_of(qubit);
            if(root == joined_roots[0] || root == joined_roots[1])
            {
                sector_of[qubit] = static_cast<std::uint8_t>(root == joined_roots[1]);
            }
            else
            {
                sector_of[qubit] = static_cast<std::uint8_t>(qubit >= width / 2);
            }
        }
        if(no_group_crosses_sectors(group_flips, sector_of))
        {
            return sector_of;
        }
    }
    std::vector<std::uint8_t> midpoint = block_sectors(width, width / 2);
    if(no_group_crosses_sectors(group_flips, midpoint))
    {
        return midpoint;
    }
    return {};
}

/**
 * Makes a valid layout from the sector of each qubit.
 *
 * @param sector_of The sector of each qubit.
 */
inline SectorLayout make_sector_layout(std::vector<std::uint8_t> sector_of)
{
    SectorLayout layout;
    layout.valid = true;
    layout.rank_in_sector.resize(sector_of.size());
    for(std::size_t qubit = 0; qubit < sector_of.size(); qubit++)
    {
        layout.rank_in_sector[qubit] = layout.num_orbitals[sector_of[qubit]]++;
    }
    layout.sector_of = std::move(sector_of);
    return layout;
}

/**
 * Tells whether every key of a store has the same number of set bits in each sector.
 *
 * @param store The subspace store.
 * @param sector_of The sector of each qubit.
 */
inline bool sector_electrons_are_constant(const HalfKeyView& store,
                                          const std::vector<std::uint8_t>& sector_of)
{
    if(store.size() == 0)
    {
        return true;
    }
    if(store.is_tensor_product() && sector_of == block_sectors(store.width(), store.split_bit()))
    {
        for(int sector = 0; sector < 2; sector++)
        {
            const std::size_t words = store.half_words(sector);
            auto count = [&](const KeyWord* key) {
                std::size_t value = 0;
                for(std::size_t word = 0; word < words; word++) value += __builtin_popcountll(key[word]);
                return value;
            };
            const std::size_t first = count(store.half_key(sector, 0));
            for(std::size_t i = 1; i < store.num_half_keys(sector); i++)
                if(count(store.half_key(sector, i)) != first) return false;
        }
        return true;
    }
    std::vector<KeyWord> sector_one_mask(store.num_words(), 0);
    for(std::size_t qubit = 0; qubit < sector_of.size(); qubit++)
    {
        if(sector_of[qubit] == 1)
        {
            sector_one_mask[qubit / KEY_WORD_BITS] |= KeyWord(1) << (qubit % KEY_WORD_BITS);
        }
    }
    auto counts = [&](const KeyBitset& key) {
        std::size_t total = 0;
        std::size_t in_sector_one = 0;
        for(std::size_t word = 0; word < store.num_words(); word++)
        {
            total += static_cast<std::size_t>(__builtin_popcountll(key.m_bits[word]));
            in_sector_one +=
                static_cast<std::size_t>(__builtin_popcountll(key.m_bits[word] & sector_one_mask[word]));
        }
        return std::array<std::size_t, 2>{total - in_sector_one, in_sector_one};
    };
    const std::array<std::size_t, 2> first = counts(store.key(0));
    int mismatch = 0;
#pragma omp parallel reduction(|| : mismatch)
    {
        KeyBitset key;
#pragma omp for schedule(static)
        for(std::size_t index = 1; index < store.size(); index++)
        {
            store.load_key(index, key);
            mismatch = mismatch || counts(key) != first;
        }
    }
    return !mismatch;
}

/**
 * Tells whether a sector split has no crossing group and constant sector electron counts.
 *
 * @param group_flips The flip qubits of each group.
 * @param store The subspace store.
 * @param sector_of The sector of each qubit.
 */
inline bool sectors_fit(const std::vector<std::vector<width_t>>& group_flips,
                        const HalfKeyView& store,
                        const std::vector<std::uint8_t>& sector_of)
{
    return no_group_crosses_sectors(group_flips, sector_of) &&
           sector_electrons_are_constant(store, sector_of);
}

/**
 * Returns the lowest block split that fits the operator and the subspace.
 *
 * The split width / 2 is tried first.
 *
 * @param group_flips The flip qubits of each group.
 * @param store The subspace store.
 * @param width The number of qubits.
 * @return The split, or 0 when no split from 1 to width - 1 fits.
 */
inline width_t find_block_split(const std::vector<std::vector<width_t>>& group_flips,
                                const HalfKeyView& store,
                                const width_t width)
{
    if(width >= 2 && sectors_fit(group_flips, store, block_sectors(width, width / 2)))
    {
        return width / 2;
    }
    for(width_t split = 1; split < width; split++)
    {
        if(sectors_fit(group_flips, store, block_sectors(width, split)))
        {
            return split;
        }
    }
    return 0;
}

/**
 * Chooses the sector layout of an operator on a subspace.
 *
 * A tensor-product store fixes a block layout at its split bit. Otherwise the requested
 * layout is used, or derived from the operator when it is automatic. The layout is
 * not valid when it is spinless, when no candidate fits the operator, or when a
 * sector does not have a constant electron count in the subspace.
 *
 * @param requested The requested layout.
 * @param store The subspace store.
 * @param group_flips The flip qubits of each group.
 * @param width The number of qubits.
 * @return The layout.
 * @throws std::invalid_argument If a tensor-product store gets a layout other than
 * automatic or block, or if a block layout has no split that fits.
 */
inline SectorLayout choose_sector_layout(const SpinLayout requested,
                                         const HalfKeyView& store,
                                         const std::vector<std::vector<width_t>>& group_flips,
                                         const width_t width)
{
    std::vector<std::uint8_t> sector_of;
    if(store.is_tensor_product())
    {
        if(requested != SpinLayout::automatic && requested != SpinLayout::block)
        {
            throw std::invalid_argument(
                "A half-string subspace uses the block spin layout: alpha in the low bits.");
        }
        sector_of = block_sectors(width, store.split_bit());
    }
    else if(requested == SpinLayout::block)
    {
        const width_t split = find_block_split(group_flips, store, width);
        if(split == 0)
        {
            throw std::invalid_argument(
                "No block split keeps every single excitation inside one spin sector and "
                "gives a constant electron count per sector.");
        }
        sector_of = block_sectors(width, split);
    }
    else if(requested == SpinLayout::interleaved)
    {
        sector_of = interleaved_sectors(width);
    }
    else if(requested == SpinLayout::automatic)
    {
        sector_of = derive_sectors(group_flips, width);
    }
    if(sector_of.empty() || !sector_electrons_are_constant(store, sector_of))
    {
        return SectorLayout{};
    }
    return make_sector_layout(std::move(sector_of));
}
