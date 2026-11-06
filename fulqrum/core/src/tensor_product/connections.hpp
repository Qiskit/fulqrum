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
#include <vector>

#include "../checked_size.hpp"
#include "../operator_tables.hpp"
#include "../subspace.hpp"
#include "groups.hpp"
#include "packed_entries.hpp"
#include "../tier_utils.hpp"

/**
 * The connections of the strings of one sector to the strings of the same sector.
 */
template <typename T>
struct SectorConnections
{
    std::vector<std::uint64_t> single_offsets;
    std::vector<std::uint64_t> single_packed_entries;
    std::vector<T> single_values;
    std::vector<std::uint64_t> single_cross_index;
    std::vector<std::uint64_t> double_offsets;
    std::vector<std::uint32_t> double_columns;
    std::vector<T> double_values;
    std::vector<std::uint32_t> pair_slot;
    std::vector<std::uint32_t> slot_pair;
};

template <typename T>
struct UpperConnections
{
    std::vector<std::uint64_t> single_offsets;
    std::vector<std::uint64_t> single_packed_entries;
    std::vector<T> single_values;
    std::vector<std::uint64_t> double_offsets;
    std::vector<std::uint32_t> double_columns;
    std::vector<T> double_values;
};

template <typename T>
UpperConnections<T> upper_connections(const SectorConnections<T>& connections)
{
    const std::size_t num_strings = connections.single_offsets.size() - 1;
    UpperConnections<T> upper;
    upper.single_offsets.assign(checked_add(num_strings, 1), 0);
    upper.double_offsets.assign(checked_add(num_strings, 1), 0);

    for(std::size_t string = 0; string < num_strings; string++)
    {
        std::uint64_t singles = 0;
        std::uint64_t doubles = 0;

        for(std::uint64_t k = connections.single_offsets[string]; k < connections.single_offsets[string + 1]; k++)
        {
            singles += single_column(connections.single_packed_entries[k]) >= string;
        }

        for(std::uint64_t k = connections.double_offsets[string]; k < connections.double_offsets[string + 1]; k++)
        {
            doubles += connections.double_columns[k] >= string;
        }

        upper.single_offsets[string + 1] = checked_add(upper.single_offsets[string], singles);
        upper.double_offsets[string + 1] = checked_add(upper.double_offsets[string], doubles);
    }

    const std::size_t num_singles = upper.single_offsets.back();
    const std::size_t num_doubles = upper.double_offsets.back();
    const std::size_t bytes = checked_add(checked_multiply(num_singles, sizeof(std::uint64_t) + sizeof(T)),
                                             checked_multiply(num_doubles, sizeof(std::uint32_t) + sizeof(T)));
    bytes;
    upper.single_packed_entries.reserve(num_singles);
    upper.single_values.reserve(num_singles);
    upper.double_columns.reserve(num_doubles);
    upper.double_values.reserve(num_doubles);

    for(std::size_t string = 0; string < num_strings; string++)
    {
        for(std::uint64_t k = connections.single_offsets[string]; k < connections.single_offsets[string + 1]; k++)
        {
            if(single_column(connections.single_packed_entries[k]) >= string)
            {
                upper.single_packed_entries.push_back(connections.single_packed_entries[k]);
                upper.single_values.push_back(connections.single_values[k]);
            }
        }

        for(std::uint64_t k = connections.double_offsets[string]; k < connections.double_offsets[string + 1]; k++)
        {
            if(connections.double_columns[k] >= string)
            {
                upper.double_columns.push_back(connections.double_columns[k]);
                upper.double_values.push_back(connections.double_values[k]);
            }
        }
    }

    return upper;
}

template <typename T>
void assign_pair_slots(const std::size_t num_pairs, SectorConnections<T>& connections)
{
    connections.pair_slot.assign(checked_multiply(2, num_pairs), NO_GROUP);

    for(const std::uint64_t packed_entry : connections.single_packed_entries)
    {
        connections.pair_slot[single_directed_pair(packed_entry)] = 0;
    }

    connections.slot_pair.clear();

    for(std::size_t pair = 0; pair < connections.pair_slot.size(); pair++)
    {
        if(connections.pair_slot[pair] != NO_GROUP)
        {
            connections.pair_slot[pair] = static_cast<std::uint32_t>(connections.slot_pair.size());
            connections.slot_pair.push_back(static_cast<std::uint32_t>(pair));
        }
    }
}

/**
 * The connections of one string of a sector, before they are packed.
 */
template <typename T>
struct StringConnections
{
    std::vector<std::uint64_t> single_packed_entries;
    std::vector<T> single_values;
    std::vector<std::uint32_t> double_columns;
    std::vector<T> double_values;
};

/**
 * Adds the single excitations of one sector string that stay in the store.
 *
 * @param tables The operator tables.
 * @param store The tensor-product store.
 * @param groups The sector group index.
 * @param sector 0 for alpha, 1 for beta.
 * @param key The sector string.
 * @param row The sector string embedded in a full key.
 * @param occupied The ranks of the set bits of the string.
 * @param empty The ranks of the clear bits of the string.
 * @param target The buffer for a target string, half_words(sector) words.
 * @param found The connections of the string that receive the packed entries and values.
 */
template <typename T>
void find_string_singles(const OperatorTables& tables,
                         const HalfKeyView& store,
                         const SectorGroupIndex& groups,
                         const int sector,
                         const KeyWord* key,
                         const KeyWord* row,
                         const std::vector<std::size_t>& occupied,
                         const std::vector<std::size_t>& empty,
                         std::vector<KeyWord>& target,
                         StringConnections<T>& found)
{
    const FactorList& factors = tables.factors();
    const std::size_t half_words = store.half_words(sector);

    for(const std::size_t from : occupied)
    {
        for(const std::size_t to : empty)
        {
            std::copy_n(key, half_words, target.data());
            target[from / KEY_WORD_BITS] ^= KeyWord(1) << (from % KEY_WORD_BITS);
            target[to / KEY_WORD_BITS] ^= KeyWord(1) << (to % KEY_WORD_BITS);
            const std::size_t column = store.find_half_key(sector, target.data());

            if(column == KEY_NOT_FOUND)
            {
                continue;
            }

            const std::size_t low = std::min(from, to);
            const std::size_t high = std::max(from, to);
            const std::size_t pair = rank_pair_index(low, high);
            const int direction = from == low ? 0 : 1;
            T value = 0;
            int own_sign = 0;
            const std::uint32_t group = groups.single[sector][pair];

            if(group != NO_GROUP)
            {
                const SingleDirectionData& single = tables.group(group).single[direction];
                value = to_value<T>(factors.sum(single.own.begin, single.own.end, row));

                if(single.cross.end > single.cross.begin)
                {
                    own_sign = factors.factor(single.shared_own_entry, row);
                }
            }

            found.single_packed_entries.push_back(pack_single(static_cast<std::uint32_t>(column),
                                                       static_cast<std::uint32_t>(2 * pair + direction),
                                                       own_sign,
                                                       between_sign(key, low, high)));
            found.single_values.push_back(value);
        }
    }
}

/**
 * Adds the nonzero same-sector double excitations of one string that stay in the store.
 *
 * @param tables The operator tables.
 * @param store The tensor-product store.
 * @param groups The sector group index.
 * @param sector 0 for alpha, 1 for beta.
 * @param num_pairs The number of rank pairs of each sector.
 * @param key The sector string.
 * @param row The sector string embedded in a full key.
 * @param occupied The ranks of the set bits of the string.
 * @param empty The ranks of the clear bits of the string.
 * @param target The buffer for a target string, half_words(sector) words.
 * @param found The connections of the string that receive the columns and values.
 */
template <typename T>
void find_string_doubles(const OperatorTables& tables,
                         const HalfKeyView& store,
                         const SectorGroupIndex& groups,
                         const int sector,
                         const std::array<std::size_t, 2>& num_pairs,
                         const KeyWord* key,
                         const KeyWord* row,
                         const std::vector<std::size_t>& occupied,
                         const std::vector<std::size_t>& empty,
                         std::vector<KeyWord>& target,
                         StringConnections<T>& found)
{
    const FactorList& factors = tables.factors();
    const std::size_t half_words = store.half_words(sector);

    for(std::size_t first = 0; first < occupied.size(); first++)
    {
        for(std::size_t second = first + 1; second < occupied.size(); second++)
        {
            for(std::size_t third = 0; third < empty.size(); third++)
            {
                for(std::size_t fourth = third + 1; fourth < empty.size(); fourth++)
                {
                    std::array<std::size_t, 4> ranks = {occupied[first], occupied[second], empty[third], empty[fourth]};
                    std::sort(ranks.begin(), ranks.end());
                    const std::size_t low = rank_pair_index(ranks[0], ranks[1]);
                    const std::size_t high = rank_pair_index(ranks[2], ranks[3]);
                    const std::uint32_t group = groups.double_excitation[sector].find(low * num_pairs[sector] + high);

                    if(group == NO_GROUP)
                    {
                        continue;
                    }

                    std::copy_n(key, half_words, target.data());
                    unsigned int bits = 0;

                    for(std::size_t flip = 0; flip < 4; flip++)
                    {
                        const std::size_t rank = ranks[flip];
                        target[rank / KEY_WORD_BITS] ^= KeyWord(1) << (rank % KEY_WORD_BITS);
                        bits |= static_cast<unsigned int>(half_bit(key, rank)) << flip;
                    }

                    const std::size_t column = store.find_half_key(sector, target.data());

                    if(column == KEY_NOT_FOUND)
                    {
                        continue;
                    }

                    const EntryRange range = tables.group(group).double_patterns[double_pattern_index(bits)];
                    const T value = to_value<T>(factors.sum(range.begin, range.end, row));

                    if(value != T(0))
                    {
                        found.double_columns.push_back(static_cast<std::uint32_t>(column));
                        found.double_values.push_back(value);
                    }
                }
            }
        }
    }
}

/**
 * Packs the connections of each string into the flat lists of a sector, in string order.
 *
 * @param found The connections of each string of the sector.
 * @return The connections, without slots and cross values.
 */
template <typename T>
SectorConnections<T> pack_sector_connections(const std::vector<StringConnections<T>>& found)
{
    const std::size_t num_strings = found.size();
    SectorConnections<T> connections;
    connections.single_offsets.assign(checked_add(num_strings, 1), 0);
    connections.double_offsets.assign(checked_add(num_strings, 1), 0);

    for(std::size_t string = 0; string < num_strings; string++)
    {
        connections.single_offsets[string + 1] =
            checked_add(connections.single_offsets[string], found[string].single_packed_entries.size());
        connections.double_offsets[string + 1] =
            checked_add(connections.double_offsets[string], found[string].double_columns.size());
    }

    const std::size_t num_singles = connections.single_offsets.back();
    const std::size_t num_doubles = connections.double_offsets.back();
    const std::size_t bytes = checked_add(checked_multiply(num_singles, sizeof(std::uint64_t) + sizeof(T)),
                                             checked_multiply(num_doubles, sizeof(std::uint32_t) + sizeof(T)));
    bytes;
    connections.single_packed_entries.reserve(num_singles);
    connections.single_values.reserve(num_singles);
    connections.double_columns.reserve(num_doubles);
    connections.double_values.reserve(num_doubles);

    for(const StringConnections<T>& string : found)
    {
        connections.single_packed_entries.insert(
            connections.single_packed_entries.end(), string.single_packed_entries.begin(), string.single_packed_entries.end());
        connections.single_values.insert(
            connections.single_values.end(), string.single_values.begin(), string.single_values.end());
        connections.double_columns.insert(
            connections.double_columns.end(), string.double_columns.begin(), string.double_columns.end());
        connections.double_values.insert(
            connections.double_values.end(), string.double_values.begin(), string.double_values.end());
    }

    return connections;
}

/**
 * Builds the connection lists of one sector of a tensor-product store.
 *
 * @param tables The operator tables.
 * @param store The tensor-product store.
 * @param groups The sector group index.
 * @param sector 0 for alpha, 1 for beta.
 * @param num_pairs The number of rank pairs of each sector.
 * @return The connections with slots, without cross values.
 */
template <typename T>
SectorConnections<T> build_sector_connections(const OperatorTables& tables,
                                              const HalfKeyView& store,
                                              const SectorGroupIndex& groups,
                                              const int sector,
                                              const std::array<std::size_t, 2>& num_pairs)
{
    const std::size_t num_strings = store.num_half_keys(sector);
    const std::size_t half_width = sector == 0 ? store.split_bit() : store.width() - store.split_bit();
    std::vector<StringConnections<T>> found(num_strings);

#pragma omp parallel
    {
        std::vector<KeyWord> row(store.num_words());
        std::vector<KeyWord> target(store.half_words(sector));
        std::vector<std::size_t> occupied;
        std::vector<std::size_t> empty;

#pragma omp for schedule(dynamic, 16)
        for(std::size_t string = 0; string < num_strings; string++)
        {
            const KeyWord* key = store.half_key(sector, string);
            embed_half_string(store, sector, string, row.data());
            occupied.clear();
            empty.clear();

            for(std::size_t rank = 0; rank < half_width; rank++)
            {
                (half_bit(key, rank) ? occupied : empty).push_back(rank);
            }

            find_string_singles(tables, store, groups, sector, key, row.data(), occupied, empty, target, found[string]);
            find_string_doubles(tables, store, groups, sector, num_pairs, key, row.data(), occupied, empty, target,
                                found[string]);
        }
    }

    SectorConnections<T> connections = pack_sector_connections(found);
    assign_pair_slots(num_pairs[sector], connections);

    return connections;
}
