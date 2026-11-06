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
#include <cstdint>
#include <utility>
#include <vector>

#include "../checked_size.hpp"
#include "connections.hpp"

inline constexpr int CROSS_BATCH_LANES = 4;
inline constexpr std::size_t CROSS_ALPHA_RANGES = 4;
inline constexpr double CROSS_BATCH_MIN_DENSITY = 0.2;

struct alignas(32) CrossLanes
{
    double value[CROSS_BATCH_LANES];
};

struct CrossSlotBatches
{
    std::vector<std::uint64_t> block_first_batch;
    std::vector<std::uint32_t> batch_slot;
    std::vector<std::uint64_t> batch_first;
    std::vector<std::uint32_t> batch_count;
    std::vector<std::uint32_t> member_row;
    std::vector<std::uint32_t> member_source;
    std::vector<double> member_sign;
    std::vector<std::uint32_t> tile_block;
    std::vector<std::uint32_t> tile_first_alpha;
    std::vector<std::uint32_t> tile_last_alpha;
};

template <typename T>
CrossSlotBatches build_cross_slot_batches(const SectorConnections<T>& alpha,
                                        const SectorConnections<T>& beta,
                                        const std::size_t row_blocks)
{
    struct Member
    {
        std::uint32_t slot;
        std::uint32_t row;
        std::uint32_t source;
        double sign;
    };

    const std::size_t num_alpha = alpha.single_offsets.size() - 1;
    const std::size_t num_beta = beta.single_offsets.size() - 1;
    const std::size_t num_members = beta.single_packed_entries.size();
    const std::size_t block_rows = std::max<std::size_t>(1, ceiling_divide(num_beta, row_blocks));
    checked_multiply(num_members, 3 * sizeof(std::uint32_t) + sizeof(std::uint64_t) + sizeof(double));
    CrossSlotBatches batches;
    batches.member_row.reserve(num_members);
    batches.member_source.reserve(num_members);
    batches.member_sign.reserve(num_members);
    std::vector<Member> members;
    std::vector<std::uint64_t> block_members;

    for(std::size_t first_beta = 0; first_beta < num_beta; first_beta += block_rows)
    {
        const std::size_t last_beta = first_beta + std::min(block_rows, num_beta - first_beta);
        batches.block_first_batch.push_back(batches.batch_slot.size());
        block_members.push_back(beta.single_offsets[last_beta] - beta.single_offsets[first_beta]);
        members.clear();

        for(std::size_t b = first_beta; b < last_beta; b++)
        {
            for(std::uint64_t k = beta.single_offsets[b]; k < beta.single_offsets[b + 1]; k++)
            {
                const std::uint64_t packed_entry = beta.single_packed_entries[k];
                members.push_back({beta.pair_slot[single_directed_pair(packed_entry)],
                                   static_cast<std::uint32_t>(b),
                                   single_column(packed_entry),
                                   single_jw_sign(packed_entry)});
            }
        }

        std::sort(members.begin(), members.end(), [](const Member& left, const Member& right) {
            return left.slot != right.slot ? left.slot < right.slot : left.row < right.row;
        });

        for(std::size_t i = 0; i < members.size(); i++)
        {
            if(i == 0 || members[i].slot != members[i - 1].slot ||
               batches.batch_count.back() == static_cast<std::uint32_t>(CROSS_BATCH_LANES))
            {
                batches.batch_slot.push_back(members[i].slot);
                batches.batch_first.push_back(batches.member_row.size());
                batches.batch_count.push_back(0);
            }

            batches.batch_count.back()++;
            batches.member_row.push_back(members[i].row);
            batches.member_source.push_back(members[i].source);
            batches.member_sign.push_back(members[i].sign);
        }
    }

    batches.block_first_batch.push_back(batches.batch_slot.size());
    const std::vector<std::uint64_t>& offsets = alpha.single_offsets;
    std::vector<std::size_t> cuts = {0};

    for(std::size_t range = 1; range < CROSS_ALPHA_RANGES; range++)
    {
        const std::uint64_t target = offsets.back() / CROSS_ALPHA_RANGES * range + (offsets.back() % CROSS_ALPHA_RANGES) * range / CROSS_ALPHA_RANGES;
        cuts.push_back(std::lower_bound(offsets.begin(), offsets.end(), target) - offsets.begin());
    }

    cuts.push_back(num_alpha);
    std::vector<std::pair<double, std::size_t>> work;

    for(std::size_t block = 0; block < block_members.size(); block++)
    {
        for(std::size_t range = 0; range < CROSS_ALPHA_RANGES; range++)
        {
            if(cuts[range] < cuts[range + 1])
            {
                const double singles = double(offsets[cuts[range + 1]] - offsets[cuts[range]]);
                work.push_back({double(block_members[block]) * singles, checked_add(checked_multiply(block, CROSS_ALPHA_RANGES), range)});
            }
        }
    }

    std::stable_sort(work.begin(), work.end(), [](const auto& left, const auto& right) {
        return left.first > right.first;
    });

    for(const auto& entry : work)
    {
        const std::size_t tile = entry.second;
        const std::size_t range = tile % CROSS_ALPHA_RANGES;
        batches.tile_block.push_back(static_cast<std::uint32_t>(tile / CROSS_ALPHA_RANGES));
        batches.tile_first_alpha.push_back(static_cast<std::uint32_t>(cuts[range]));
        batches.tile_last_alpha.push_back(static_cast<std::uint32_t>(cuts[range + 1]));
    }

    return batches;
}
