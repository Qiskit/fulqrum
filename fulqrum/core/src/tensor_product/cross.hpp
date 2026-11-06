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
#include <utility>
#include <vector>

#include "../checked_size.hpp"
#include "../operator_tables.hpp"
#include "../subspace.hpp"
#include "connections.hpp"
#include "../tier_utils.hpp"

/**
 * How the tensor-product engine gets the cross values of same-sector single excitation.
 */
enum class CrossMode : std::uint8_t
{
    automatic = 0,
    stored = 1,
    formed = 2,
};

/**
 * The cross values of the single excitations of one sector, over its slots.
 */
template <typename T>
struct SectorCrossValues
{
    std::vector<T> table;
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint16_t> bits;
    std::vector<T> values;
};

/**
 * Returns the cross entries of the directed pair of a slot, or an empty range.
 *
 * @param tables The operator tables.
 * @param groups The sector group index.
 * @param sector The sector of the single excitations.
 * @param directed_pair The directed pair.
 */
inline EntryRange slot_cross_entries(const OperatorTables& tables,
                                     const SectorGroupIndex& groups,
                                     const int sector,
                                     const std::uint32_t directed_pair)
{
    const std::uint32_t group = groups.single[sector][directed_pair / 2];

    if(group == NO_GROUP)
    {
        return EntryRange();
    }

    return tables.group(group).single[directed_pair % 2].cross;
}

/**
 * Builds the stored cross table of one sector.
 *
 * @param tables The operator tables.
 * @param store The tensor-product store.
 * @param groups The sector group index.
 * @param sector The sector of the single excitations, 0 or 1.
 * @param connections The connections of the sector, with slots.
 * @return The table, one row of num_strings(other sector) values per slot.
 */
template <typename T>
std::vector<T> build_stored_cross_table(const OperatorTables& tables,
                                        const HalfKeyView& store,
                                        const SectorGroupIndex& groups,
                                        const int sector,
                                        const SectorConnections<T>& connections)
{
    const FactorList& factors = tables.factors();
    const int other = 1 - sector;
    const std::size_t num_other = store.num_half_keys(other);
    const std::size_t num_slots = connections.slot_pair.size();
    checked_multiply(checked_multiply(num_slots, num_other), sizeof(T));
    std::vector<T> table(checked_multiply(num_slots, num_other), T(0));
#pragma omp parallel
    {
        std::vector<KeyWord> row(store.num_words());
#pragma omp for schedule(dynamic, 16)
        for(std::size_t string = 0; string < num_other; string++)
        {
            embed_half_string(store, other, string, row.data());

            for(std::size_t slot = 0; slot < num_slots; slot++)
            {
                const EntryRange cross = slot_cross_entries(tables, groups, sector, connections.slot_pair[slot]);

                if(cross.end > cross.begin)
                {
                    table[slot * num_other + string] = to_value<T>(factors.sum(cross.begin, cross.end, row.data()));
                }
            }
        }
    }

    return table;
}

/**
 * Sets the formed cross entries of one sector from the operator's factor entries.
 *
 * Each entry must test one other-sector qubit with a projector and have no Z qubit.
 *
 * @param tables The operator tables.
 * @param groups The sector group index.
 * @param sector The sector of the single excitations, 0 or 1.
 * @param connections The connections of the sector, with slots.
 * @param cross The cross values of the sector; the function sets the formed entries.
 * @return False, with no entries set, when an entry does not have that form.
 */
template <typename T>
bool build_cross_entries(const OperatorTables& tables,
                         const SectorGroupIndex& groups,
                         const int sector,
                         const SectorConnections<T>& connections,
                         SectorCrossValues<T>& cross)
{
    const FactorList& factors = tables.factors();
    const std::size_t num_words = factors.num_words;
    const std::vector<width_t>& rank = tables.layout().rank_in_sector;
    std::vector<std::uint32_t> offsets = {0};
    std::vector<std::uint16_t> bits;
    std::vector<T> values;

    for(const std::uint32_t directed_pair : connections.slot_pair)
    {
        const EntryRange cross = slot_cross_entries(tables, groups, sector, directed_pair);

        for(std::uint32_t entry = cross.begin; entry < cross.end; entry++)
        {
            const KeyWord* projector = factors.entry_masks(entry);
            const KeyWord* expected = projector + num_words;
            const KeyWord* parity = expected + num_words;
            std::size_t qubit = 0;
            unsigned int count = 0;

            for(std::size_t word = 0; word < num_words; word++)
            {
                if(parity[word] != 0)
                {
                    return false;
                }

                if(projector[word] != 0)
                {
                    count += static_cast<unsigned int>(__builtin_popcountll(projector[word]));
                    qubit = word * KEY_WORD_BITS + static_cast<std::size_t>(__builtin_ctzll(projector[word]));
                }
            }

            if(count != 1)
            {
                return false;
            }

            const unsigned int bit = (expected[qubit / KEY_WORD_BITS] >> (qubit % KEY_WORD_BITS)) & 1u;
            bits.push_back(static_cast<std::uint16_t>(2 * rank[qubit] + bit));
            values.push_back(to_value<T>(factors.values[entry]));
        }

        offsets.push_back(static_cast<std::uint32_t>(bits.size()));
    }

    cross.offsets = std::move(offsets);
    cross.bits = std::move(bits);
    cross.values = std::move(values);

    return true;
}

/**
 * Computes the formed cross values of every slot for a block of other-sector strings.
 *
 * @param cross The formed cross values of the sector.
 * @param keys The half strings of the other sector, words per string each.
 * @param words The number of words of an other-sector string.
 * @param orbitals The number of other-sector orbitals.
 * @param first The first other-sector string of the block.
 * @param count The number of strings in the block, at most R.
 * @param occupancy Scratch of 2 * orbitals * R values.
 * @param block The values, R per slot. Entries past count are zero.
 */
template <int R, typename T>
void form_cross_block(const SectorCrossValues<T>& cross,
                      const KeyWord* keys,
                      const std::size_t words,
                      const std::size_t orbitals,
                      const std::size_t first,
                      const int count,
                      double* occupancy,
                      T* block)
{
    for(std::size_t rank = 0; rank < orbitals; rank++)
    {
        for(int r = 0; r < R; r++)
        {
            const double set = r < count ? static_cast<double>(half_bit(keys + (first + r) * words, rank)) : 0.0;
            occupancy[(2 * rank + 1) * R + r] = set;
            occupancy[2 * rank * R + r] = r < count ? 1.0 - set : 0.0;
        }
    }

    const std::size_t num_slots = cross.offsets.size() - 1;

    for(std::size_t slot = 0; slot < num_slots; slot++)
    {
        T* values = block + slot * R;

        for(int r = 0; r < R; r++)
        {
            values[r] = T(0);
        }

        for(std::uint32_t entry = cross.offsets[slot]; entry < cross.offsets[slot + 1]; entry++)
        {
            const T value = cross.values[entry];
            const double* tested = occupancy + std::size_t(cross.bits[entry]) * R;

            for(int r = 0; r < R; r++)
            {
                values[r] += value * tested[r];
            }
        }
    }
}

/**
 * Packs each alpha single excitation's source column, pair slot and JW sign
 * for cross-sector evaluation.
 *
 * Low 32 bits: source column.
 * High 32 bits: 2 * slot, plus 1 if the JW sign is negative.
 *
 * @param alpha Alpha connections with assigned pair slots; fills single_cross_index.
 */
template <typename T>
void set_cross_arm_columns(SectorConnections<T>& alpha)
{
    const std::size_t num_singles = alpha.single_packed_entries.size();
    alpha.single_cross_index.resize(num_singles);

    for(std::size_t k = 0; k < num_singles; k++)
    {
        const std::uint64_t packed_entry = alpha.single_packed_entries[k];
        const std::uint32_t signed_slot =
            2 * alpha.pair_slot[single_directed_pair(packed_entry)] + (single_jw_sign(packed_entry) < 0 ? 1u : 0u);
        alpha.single_cross_index[k] = std::uint64_t(single_column(packed_entry)) | (std::uint64_t(signed_slot) << 32);
    }
}

/**
 * Builds cross-sector coefficients for the alpha and beta excitation slots in use.
 *
 * @param tables The operator tables.
 * @param alpha The alpha connections, with slots.
 * @param beta The beta connections, with slots.
 */
template <typename T>
std::vector<T> build_slot_cross_sector_table(const OperatorTables& tables,
                                             const SectorConnections<T>& alpha,
                                             const SectorConnections<T>& beta)
{
    const std::size_t alpha_columns = checked_multiply(2, alpha.slot_pair.size());
    const std::size_t cells = checked_multiply(beta.slot_pair.size(), alpha_columns);
    checked_multiply(cells, sizeof(T));
    std::vector<T> table(cells, T(0));
    const std::vector<width_t>& rank = tables.layout().rank_in_sector;
    const std::vector<std::uint8_t>& sector_of = tables.layout().sector_of;

    for(std::size_t group = 0; group < tables.num_groups(); group++)
    {
        const GroupData& plan = tables.group(group);

        if(plan.tier != GroupTier::cross_sector_double)
        {
            continue;
        }

        std::array<std::vector<width_t>, 2> ranks;

        for(const width_t qubit : plan.flips)
        {
            ranks[sector_of[qubit]].push_back(rank[qubit]);
        }

        const std::size_t alpha_pair = rank_pair_index(ranks[0][0], ranks[0][1]);
        const std::size_t beta_pair = rank_pair_index(ranks[1][0], ranks[1][1]);

        for(std::size_t pattern = 0; pattern < 4; pattern++)
        {
            const std::uint32_t alpha_slot = alpha.pair_slot[2 * alpha_pair + (pattern & 1)];
            const std::uint32_t beta_slot = beta.pair_slot[2 * beta_pair + (pattern >> 1)];

            if(alpha_slot == NO_GROUP || beta_slot == NO_GROUP)
            {
                continue;
            }

            const T value = to_value<T>(plan.cross_values[pattern]);
            T* row = table.data() + beta_slot * alpha_columns;
            row[2 * alpha_slot] = value;
            row[2 * alpha_slot + 1] = -value;
        }
    }

    return table;
}

template <typename T>
struct SparseCrossTable
{
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint32_t> alpha_slot;
    std::vector<T> value;
    std::vector<std::uint64_t> single_offsets;
    std::vector<std::uint32_t> single_row;
    std::vector<std::uint32_t> single_column;
    std::vector<double> single_sign;
};

template <typename T>
SparseCrossTable<T> build_sparse_cross_entries(const OperatorTables& tables,
                                               const SectorConnections<T>& alpha,
                                               const SectorConnections<T>& beta)
{
    struct Entry
    {
        std::uint32_t beta_slot;
        std::uint32_t alpha_slot;
        T value;
    };

    const std::vector<width_t>& rank = tables.layout().rank_in_sector;
    const std::vector<std::uint8_t>& sector_of = tables.layout().sector_of;
    std::size_t num_groups = 0;

    for(std::size_t group = 0; group < tables.num_groups(); group++)
    {
        num_groups += tables.group(group).tier == GroupTier::cross_sector_double;
    }

    checked_multiply(checked_multiply(4, num_groups), sizeof(Entry));
    std::vector<Entry> entries;
    entries.reserve(checked_multiply(4, num_groups));

    for(std::size_t group = 0; group < tables.num_groups(); group++)
    {
        const GroupData& plan = tables.group(group);

        if(plan.tier != GroupTier::cross_sector_double)
        {
            continue;
        }

        std::array<std::vector<width_t>, 2> ranks;

        for(const width_t qubit : plan.flips)
        {
            ranks[sector_of[qubit]].push_back(rank[qubit]);
        }

        const std::size_t alpha_pair = rank_pair_index(ranks[0][0], ranks[0][1]);
        const std::size_t beta_pair = rank_pair_index(ranks[1][0], ranks[1][1]);

        for(std::size_t pattern = 0; pattern < 4; pattern++)
        {
            const std::uint32_t alpha_slot = alpha.pair_slot[2 * alpha_pair + (pattern & 1)];
            const std::uint32_t beta_slot = beta.pair_slot[2 * beta_pair + (pattern >> 1)];
            const T value = to_value<T>(plan.cross_values[pattern]);

            if(alpha_slot != NO_GROUP && beta_slot != NO_GROUP && value != T(0))
            {
                entries.push_back({beta_slot, alpha_slot, value});
            }
        }
    }

    std::sort(entries.begin(), entries.end(), [](const Entry& left, const Entry& right) {
        return left.beta_slot != right.beta_slot ? left.beta_slot < right.beta_slot
                                                 : left.alpha_slot < right.alpha_slot;
    });
    SparseCrossTable<T> sparse;
    sparse.offsets.assign(checked_add(beta.slot_pair.size(), 1), 0);
    sparse.alpha_slot.reserve(entries.size());
    sparse.value.reserve(entries.size());

    for(const Entry& entry : entries)
    {
        sparse.offsets[entry.beta_slot + 1]++;
        sparse.alpha_slot.push_back(entry.alpha_slot);
        sparse.value.push_back(entry.value);
    }

    for(std::size_t slot = 0; slot < beta.slot_pair.size(); slot++)
    {
        sparse.offsets[slot + 1] = checked_add(sparse.offsets[slot + 1], sparse.offsets[slot]);
    }

    return sparse;
}

template <typename T>
void index_sparse_cross_singles(const SectorConnections<T>& alpha, SparseCrossTable<T>& sparse)
{
    const std::size_t num_alpha = alpha.single_offsets.size() - 1;
    const std::size_t num_slots = alpha.slot_pair.size();
    const std::size_t num_singles = alpha.single_cross_index.size();
    checked_multiply(num_singles, 2 * sizeof(std::uint32_t) + sizeof(double));
    sparse.single_offsets.assign(checked_add(num_slots, 1), 0);

    for(const std::uint64_t index : alpha.single_cross_index)
    {
        sparse.single_offsets[(index >> 32) / 2 + 1]++;
    }

    for(std::size_t slot = 0; slot < num_slots; slot++)
    {
        sparse.single_offsets[slot + 1] = checked_add(sparse.single_offsets[slot + 1], sparse.single_offsets[slot]);
    }

    sparse.single_row.resize(num_singles);
    sparse.single_column.resize(num_singles);
    sparse.single_sign.resize(num_singles);
    std::vector<std::uint64_t> next(sparse.single_offsets.begin(), sparse.single_offsets.end() - 1);

    for(std::size_t a = 0; a < num_alpha; a++)
    {
        for(std::uint64_t k = alpha.single_offsets[a]; k < alpha.single_offsets[a + 1]; k++)
        {
            const std::uint64_t index = alpha.single_cross_index[k];
            const std::uint64_t at = next[(index >> 32) / 2]++;
            sparse.single_row[at] = static_cast<std::uint32_t>(a);
            sparse.single_column[at] = static_cast<std::uint32_t>(index);
            sparse.single_sign[at] = ((index >> 32) & 1u) ? -1.0 : 1.0;
        }
    }
}

template <typename T>
T sparse_cross_value(const SparseCrossTable<T>& sparse, const std::uint32_t beta_slot, const std::uint32_t signed_slot)
{
    const auto first = sparse.alpha_slot.begin() + sparse.offsets[beta_slot];
    const auto last = sparse.alpha_slot.begin() + sparse.offsets[beta_slot + 1];
    const auto found = std::lower_bound(first, last, signed_slot / 2);

    if(found == last || *found != signed_slot / 2)
    {
        return T(0);
    }

    const T value = sparse.value[found - sparse.alpha_slot.begin()];
    return (signed_slot & 1u) ? -value : value;
}
