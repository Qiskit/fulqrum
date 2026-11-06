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
#include <type_traits>
#include <vector>

#include "../key_words.hpp"
#include "../operator_tables.hpp"
#include "../subspace.hpp"
#include "batches.hpp"
#include "cross.hpp"

/**
 * The tensor-product engine: connection lists of both sectors, the cross-sector
 * coefficient table, and the groups that the generic tier must still evaluate.
 *
 * The cross-sector table has one row per beta slot and two columns per alpha slot.
 * In formed mode the engine keeps a copy of the half strings of both sectors.
 */
template <typename T>
struct TensorProductEngine
{
    bool ready = false;
    bool cross_formed = false;
    std::array<std::size_t, 2> num_strings = {0, 0};
    std::array<std::size_t, 2> num_pairs = {0, 0};
    std::array<std::size_t, 2> num_orbitals = {0, 0};
    std::array<std::size_t, 2> half_words = {0, 0};
    std::array<SectorConnections<T>, 2> sectors;
    std::array<UpperConnections<T>, 2> upper;
    std::array<SectorCrossValues<T>, 2> cross;
    std::array<std::vector<KeyWord>, 2> half_keys;
    std::vector<T> cross_sector_table;
    CrossSlotBatches cross_batches;
    SparseCrossTable<T> sparse_cross;
    std::vector<std::uint8_t> generic_groups;
};

/**
 * Returns the cross value of one slot of a sector on one other-sector string.
 *
 * @param tier The tensor-product engine.
 * @param sector The sector of the single excitation.
 * @param slot The slot.
 * @param other The other-sector string index.
 */
template <typename T>
T cross_value(const TensorProductEngine<T>& tier, const int sector, const std::uint32_t slot, const std::size_t other)
{
    const SectorCrossValues<T>& cross = tier.cross[sector];

    if(!tier.cross_formed)
    {
        return cross.table[slot * tier.num_strings[1 - sector] + other];
    }

    const std::size_t words = tier.half_words[1 - sector];
    const KeyWord* key = tier.half_keys[1 - sector].data() + other * words;
    T total = T(0);

    for(std::uint32_t entry = cross.offsets[slot]; entry < cross.offsets[slot + 1]; entry++)
    {
        const std::uint16_t test = cross.bits[entry];
        total += cross.values[entry] * static_cast<double>(half_bit(key, test / 2) == (test % 2));
    }

    return total;
}

/**
 * Returns the bytes of a vector's elements.
 *
 * @param values The vector.
 */
template <typename V>
std::size_t vector_bytes(const std::vector<V>& values)
{
    return checked_multiply(values.size(), sizeof(V));
}

/**
 * Returns the bytes that a tensor-product engine holds: connection lists, slots, cross
 * values, half-det copies and the cross-sector table.
 *
 * @param tier The tensor-product engine.
 */
template <typename T>
std::size_t tensor_product_engine_bytes(const TensorProductEngine<T>& tier)
{
    std::size_t bytes = 0;
    const auto add = [&bytes](const auto& values) { bytes = checked_add(bytes, vector_bytes(values)); };
    add(tier.cross_sector_table);
    add(tier.generic_groups);
    const CrossSlotBatches& batches = tier.cross_batches;
    add(batches.block_first_batch);
    add(batches.batch_slot);
    add(batches.batch_first);
    add(batches.batch_count);
    add(batches.member_row);
    add(batches.member_source);
    add(batches.member_sign);
    add(batches.tile_block);
    add(batches.tile_first_alpha);
    add(batches.tile_last_alpha);
    const SparseCrossTable<T>& sparse = tier.sparse_cross;
    add(sparse.offsets);
    add(sparse.alpha_slot);
    add(sparse.value);
    add(sparse.single_offsets);
    add(sparse.single_row);
    add(sparse.single_column);
    add(sparse.single_sign);
    for(int sector = 0; sector < 2; sector++)
    {
        const SectorConnections<T>& connections = tier.sectors[sector];
        add(connections.single_offsets);
        add(connections.single_packed_entries);
        add(connections.single_values);
        add(connections.single_cross_index);
        add(connections.double_offsets);
        add(connections.double_columns);
        add(connections.double_values);
        add(connections.pair_slot);
        add(connections.slot_pair);
        add(tier.cross[sector].table);
        add(tier.cross[sector].offsets);
        add(tier.cross[sector].bits);
        add(tier.cross[sector].values);
        add(tier.half_keys[sector]);
        const UpperConnections<T>& upper = tier.upper[sector];
        add(upper.single_offsets);
        add(upper.single_packed_entries);
        add(upper.single_values);
        add(upper.double_offsets);
        add(upper.double_columns);
        add(upper.double_values);
    }
    return bytes;
}

/**
 * Builds the tensor-product engine of an operator on a tensor-product store.
 *
 * The engine requires a tensor-product store with its block layout, at most
 * 256 orbitals (bits) per sector, and at most 2^32 - 1 strings per sector.
 * Otherwise, it returns an engine with ready == false.
 *
 * @param tables The operator tables.
 * @param store The subspace store.
 * @param mode The cross mode.
 * @return The tier.
 */
template <typename T>
TensorProductEngine<T> build_tensor_product_engine(const OperatorTables& tables,
                                                   const HalfKeyView& store,
                                                   const CrossMode mode = CrossMode::automatic)
{
    TensorProductEngine<T> tier;
    const SectorLayout& layout = tables.layout();

    if(!store.is_tensor_product() || !layout.valid ||
       layout.sector_of != block_sectors(store.width(), store.split_bit()))
    {
        return tier;
    }

    for(int sector = 0; sector < 2; sector++)
    {
        const std::size_t orbitals = layout.num_orbitals[sector];
        tier.num_orbitals[sector] = orbitals;
        tier.num_pairs[sector] = orbitals * (orbitals - 1) / 2;
        tier.num_strings[sector] = store.num_half_keys(sector);
        tier.half_words[sector] = store.half_words(sector);

        if(2 * tier.num_pairs[sector] > 0xffffu || tier.num_strings[sector] > 0xffffffffu)
        {
            return tier;
        }
    }

    const SectorGroupIndex groups = index_sector_groups(tables, tier.num_pairs);

    for(int sector = 0; sector < 2; sector++)
    {
        tier.sectors[sector] = build_sector_connections<T>(tables, store, groups, sector, tier.num_pairs);
        tier.upper[sector] = upper_connections(tier.sectors[sector]);
    }

    tier.cross_formed = mode != CrossMode::stored;

    for(int sector = 0; sector < 2 && tier.cross_formed; sector++)
    {
        tier.cross_formed = build_cross_entries<T>(tables, groups, sector, tier.sectors[sector], tier.cross[sector]);
    }

    for(int sector = 0; sector < 2; sector++)
    {
        if(tier.cross_formed)
        {
            const std::size_t words = tier.half_words[sector];
            tier.half_keys[sector].resize(checked_multiply(tier.num_strings[sector], words));

            for(std::size_t string = 0; string < tier.num_strings[sector]; string++)
            {
                std::copy_n(store.half_key(sector, string), words, tier.half_keys[sector].data() + string * words);
            }
        }
        else
        {
            tier.cross[sector] = SectorCrossValues<T>();
            tier.cross[sector].table = build_stored_cross_table<T>(tables, store, groups, sector, tier.sectors[sector]);
        }
    }

    set_cross_arm_columns(tier.sectors[0]);

    if constexpr(std::is_same_v<T, double>)
    {
        tier.sparse_cross = build_sparse_cross_entries<T>(tables, tier.sectors[0], tier.sectors[1]);
        const double cells = double(tier.sectors[0].slot_pair.size()) * double(tier.sectors[1].slot_pair.size());

        if(cells > 0 && tier.sparse_cross.value.size() >= CROSS_BATCH_MIN_DENSITY * cells)
        {
            tier.sparse_cross = SparseCrossTable<T>();
            tier.cross_sector_table = build_slot_cross_sector_table<T>(tables, tier.sectors[0], tier.sectors[1]);
            tier.cross_batches = build_cross_slot_batches(tier.sectors[0], tier.sectors[1], max_threads());
        }
        else
        {
            index_sparse_cross_singles(tier.sectors[0], tier.sparse_cross);
        }
    }
    else
    {
        tier.cross_sector_table = build_slot_cross_sector_table<T>(tables, tier.sectors[0], tier.sectors[1]);
    }

    tier.generic_groups.assign(tables.num_groups(), 0);

    for(std::size_t group = 0; group < tables.num_groups(); group++)
    {
        if(tables.group(group).tier == GroupTier::generic)
        {
            tier.generic_groups[group] = 1;
        }
    }

    tier.ready = true;

    return tier;
}
