/**
 * This code is a part of Fulqrum.
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
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "coefficients.hpp"
#include "groups.hpp"
#include "keys.hpp"
#include "reference.hpp"

namespace fq::full_strs
{


struct ChunkClass
{
    std::vector<std::int32_t> members;
    std::vector<std::uint16_t> sizes;
    std::array<std::int64_t, kSizeBuckets + 1> bucket_start{};


    std::int64_t num_groups() const
    {
        return static_cast<std::int64_t>(sizes.size());
    }
};


struct GroupChunk
{
    GroupSlice slice;
    std::array<ChunkClass, kNumClasses> classes;


    std::int64_t num_members() const
    {
        std::int64_t total = 0;
        for(const ChunkClass& c : classes)
        {
            total += static_cast<std::int64_t>(c.members.size());
        }
        return total;
    }


    std::int64_t num_groups() const
    {
        std::int64_t total = 0;
        for(const ChunkClass& c : classes)
        {
            total += c.num_groups();
        }
        return total;
    }
};


inline void release_chunk(GroupChunk& chunk)
{
    for(ChunkClass& c : chunk.classes)
    {
        std::vector<std::int32_t>().swap(c.members);
        std::vector<std::uint16_t>().swap(c.sizes);
    }
}


inline void pack_chunk_class(const ClassGroups& source, ChunkClass& out)
{
    const std::int64_t num_groups = source.num_groups();
    if(source.members.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    {
        throw std::invalid_argument("A chunk class holds " + std::to_string(source.members.size()) +
                                    " member entries; use more key slices");
    }
    std::vector<std::int64_t> order(static_cast<std::size_t>(num_groups));
    std::array<std::int64_t, kSizeBuckets> counts{};
    for(std::int64_t g = 0; g < num_groups; g++)
    {
        const std::int64_t size = source.member_start[static_cast<std::size_t>(g) + 1] - source.member_start[static_cast<std::size_t>(g)];
        if(size > std::numeric_limits<std::uint16_t>::max())
        {
            throw std::invalid_argument("A group has " + std::to_string(size) + " members; the limit is 65535");
        }
        counts[static_cast<std::size_t>(size_bucket(size))]++;
    }
    out.bucket_start[0] = 0;
    for(int bucket = 0; bucket < kSizeBuckets; bucket++)
    {
        out.bucket_start[static_cast<std::size_t>(bucket) + 1] = out.bucket_start[static_cast<std::size_t>(bucket)] + counts[static_cast<std::size_t>(bucket)];
    }
    std::array<std::int64_t, kSizeBuckets> cursor{};
    for(int bucket = 0; bucket < kSizeBuckets; bucket++)
    {
        cursor[static_cast<std::size_t>(bucket)] = out.bucket_start[static_cast<std::size_t>(bucket)];
    }
    for(std::int64_t g = 0; g < num_groups; g++)
    {
        const std::int64_t size = source.member_start[static_cast<std::size_t>(g) + 1] - source.member_start[static_cast<std::size_t>(g)];
        order[static_cast<std::size_t>(cursor[static_cast<std::size_t>(size_bucket(size))]++)] = g;
    }
    out.sizes.resize(static_cast<std::size_t>(num_groups));
    out.members.resize(source.members.size());
    std::int64_t offset = 0;
    for(std::int64_t k = 0; k < num_groups; k++)
    {
        const std::int64_t g = order[static_cast<std::size_t>(k)];
        const std::int64_t first = source.member_start[static_cast<std::size_t>(g)];
        const std::int64_t size = source.member_start[static_cast<std::size_t>(g) + 1] - first;
        out.sizes[static_cast<std::size_t>(k)] = static_cast<std::uint16_t>(size);
        std::copy_n(source.members.begin() + first, size, out.members.begin() + offset);
        offset += size;
    }
}


template <std::size_t W>
inline GroupChunk build_chunk(const KeyStore<W>& store, const GroupKinds& kinds, const GroupSlice& slice)
{
    const GroupStore groups = build_groups(store, kinds, slice, false);
    GroupChunk chunk;
    chunk.slice = slice;
    for(int cls = 0; cls < kNumClasses; cls++)
    {
        pack_chunk_class(groups.classes[static_cast<std::size_t>(cls)], chunk.classes[static_cast<std::size_t>(cls)]);
    }
    return chunk;
}


template <std::size_t W>
inline std::vector<GroupChunk> build_chunks(const KeyStore<W>& store, const GroupKinds& kinds, const int num_slices)
{
    std::vector<GroupChunk> chunks;
    chunks.reserve(static_cast<std::size_t>(num_slices));
    for(int index = 0; index < num_slices; index++)
    {
        chunks.push_back(build_chunk(store, kinds, GroupSlice{num_slices, index}));
    }
    return chunks;
}


template <std::size_t W, typename T>
inline void chunk_class_partials(const KeyStore<W>& keys,
                                 const ChunkClass& c,
                                 const int cls,
                                 const CoefficientTables<W, T>& tables,
                                 const T* x,
                                 T* partials)
{
    std::vector<std::int64_t> start(static_cast<std::size_t>(c.num_groups()) + 1, 0);
    for(std::int64_t g = 0; g < c.num_groups(); g++)
    {
        start[static_cast<std::size_t>(g) + 1] = start[static_cast<std::size_t>(g)] + c.sizes[static_cast<std::size_t>(g)];
    }
#pragma omp parallel for schedule(dynamic, 256)
    for(std::int64_t g = 0; g < c.num_groups(); g++)
    {
        const std::int64_t first = start[static_cast<std::size_t>(g)];
        const std::int64_t last = start[static_cast<std::size_t>(g) + 1];
        for(std::int64_t k = first; k < last; k++)
        {
            const std::int32_t out = c.members[static_cast<std::size_t>(k)];
            const Word* out_key = keys.alpha(out);
            T acc = T(0);
            for(std::int64_t s = first; s < last; s++)
            {
                const std::int32_t source = c.members[static_cast<std::size_t>(s)];
                const Word* source_key = keys.alpha(source);
                if(source == out || !pair_counts<W>(cls, out_key, source_key))
                {
                    continue;
                }
                acc = fused_multiply_add(tables.element(out_key, source_key), x[source], acc);
            }
            partials[k] = acc;
        }
    }
}


template <std::size_t W, typename T>
inline void chunk_matvec(const KeyStore<W>& keys,
                         const std::vector<GroupChunk>& chunks,
                         const CoefficientTables<W, T>& tables,
                         const std::vector<T>& diagonal,
                         const T* x,
                         T* y)
{
    const std::int64_t num_rows = keys.num_rows();
#pragma omp parallel for schedule(static)
    for(std::int64_t row = 0; row < num_rows; row++)
    {
        y[row] = diagonal[static_cast<std::size_t>(row)] * x[row];
    }
    for(const GroupChunk& chunk : chunks)
    {
        std::vector<std::pair<std::int32_t, T>> entries(static_cast<std::size_t>(chunk.num_members()));
        std::int64_t base = 0;
        for(int cls = 0; cls < kNumClasses; cls++)
        {
            const ChunkClass& c = chunk.classes[static_cast<std::size_t>(cls)];
            std::vector<T> partials(c.members.size());
            chunk_class_partials(keys, c, cls, tables, x, partials.data());
            for(std::size_t k = 0; k < c.members.size(); k++)
            {
                entries[static_cast<std::size_t>(base) + k] = {c.members[k], partials[k]};
            }
            base += static_cast<std::int64_t>(c.members.size());
        }
        std::stable_sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for(const auto& [row, value] : entries)
        {
            y[row] += value;
        }
    }
}

}
