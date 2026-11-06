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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "coefficients.hpp"
#include "keys.hpp"

namespace fq::full_strs
{


enum GroupKind : int
{
    kIntersection = 0,
    kUnion = 1
};


struct GroupKinds
{
    int alpha = kIntersection;
    int beta = kIntersection;
};


inline int default_kind(const int num_orbitals, const int num_electrons)
{
    return num_orbitals - num_electrons < num_electrons ? kUnion : kIntersection;
}


template <std::size_t W>
inline GroupKinds default_kinds(const KeyStore<W>& store)
{
    return {default_kind(store.num_orbitals, store.num_alpha), default_kind(store.num_orbitals, store.num_beta)};
}


struct GroupSlice
{
    int count = 1;
    int index = 0;


    template <std::size_t K>
    bool contains(const std::array<Word, K>& key) const
    {
        if(count == 1)
        {
            return true;
        }
        std::uint64_t hash = 0x5EEDF00DF1A5C0DEULL;
        for(std::size_t word = 0; word < K; word++)
        {
            hash = mix64(hash ^ key[word]);
        }
        return static_cast<int>(hash % static_cast<std::uint64_t>(count)) == index;
    }
};


struct ClassGroups
{
    std::vector<std::int64_t> member_start{0};
    std::vector<std::int32_t> members;
    std::vector<std::int32_t> num_outputs;
    std::vector<std::int64_t> owner;
    std::vector<std::int64_t> slot_start{0};


    std::int64_t num_groups() const
    {
        return static_cast<std::int64_t>(num_outputs.size());
    }


    std::int64_t num_slots() const
    {
        return slot_start.back();
    }
};


struct GroupStore
{
    GroupKinds kinds;
    std::array<ClassGroups, kNumClasses> classes;
    std::array<std::int64_t, kNumClasses + 1> class_slot_start{};
    std::vector<std::int64_t> row_slot_start;
    std::vector<std::int64_t> row_slots;
};


template <std::size_t W>
inline bool pair_counts(const int cls, const Word* out, const Word* source)
{
    int alpha_bits = 0;
    int beta_bits = 0;
    for(std::size_t word = 0; word < W; word++)
    {
        alpha_bits += __builtin_popcountll(out[word] ^ source[word]);
        beta_bits += __builtin_popcountll(out[W + word] ^ source[W + word]);
    }
    switch(cls)
    {
    case kAlphaSingle:
        return alpha_bits != 0;
    case kAlphaDouble:
        return alpha_bits == 4;
    case kBetaSingle:
        return beta_bits != 0;
    case kBetaDouble:
        return beta_bits == 4;
    default:
        return alpha_bits != 0 && beta_bits != 0;
    }
}


template <std::size_t W, typename Visit>
inline void for_each_changed_key(const Word* string, const int num_orbitals, const int kind, const int num_changed, Visit&& visit)
{
    int orbitals[64 * W];
    int count = 0;
    for(int orbital = 0; orbital < num_orbitals; orbital++)
    {
        if(has_orbital<W>(string, orbital) == (kind == kIntersection))
        {
            orbitals[count++] = orbital;
        }
    }
    std::array<Word, W> key;
    std::copy_n(string, W, key.data());
    for(int i = 0; i < count; i++)
    {
        key[static_cast<std::size_t>(orbitals[i] / kWordBits)] ^= Word(1) << (orbitals[i] % kWordBits);
        if(num_changed == 1)
        {
            visit(key, orbitals[i], -1);
        }
        else
        {
            for(int j = i + 1; j < count; j++)
            {
                key[static_cast<std::size_t>(orbitals[j] / kWordBits)] ^= Word(1) << (orbitals[j] % kWordBits);
                visit(key, orbitals[i], orbitals[j]);
                key[static_cast<std::size_t>(orbitals[j] / kWordBits)] ^= Word(1) << (orbitals[j] % kWordBits);
            }
        }
        key[static_cast<std::size_t>(orbitals[i] / kWordBits)] ^= Word(1) << (orbitals[i] % kWordBits);
    }
}


template <std::size_t W, typename Visit>
inline void for_each_key(const Word* string, const int num_orbitals, const int kind, const int num_changed, Visit&& visit)
{
    for_each_changed_key<W>(string, num_orbitals, kind, num_changed,
                            [&](const std::array<Word, W>& key, int, int) { visit(key); });
}


template <std::size_t K>
inline bool key_less(const std::array<Word, K>& a, const std::array<Word, K>& b)
{
    for(std::size_t word = K; word-- > 0;)
    {
        if(a[word] != b[word])
        {
            return a[word] < b[word];
        }
    }
    return false;
}


template <std::size_t K>
struct PendingGroups
{
    struct Descriptor
    {
        std::int64_t owner;
        std::array<Word, K> key;
        std::int64_t offset;
        std::int32_t count;
    };

    std::vector<Descriptor> descriptors;
    std::vector<std::int32_t> members;


    void add(const std::int64_t owner, const std::array<Word, K>& key, const std::vector<std::int32_t>& group_members)
    {
        descriptors.push_back({owner, key, static_cast<std::int64_t>(members.size()), static_cast<std::int32_t>(group_members.size())});
        members.insert(members.end(), group_members.begin(), group_members.end());
    }
};


template <std::size_t K>
inline void merge_groups(const std::vector<PendingGroups<K>>& pending, ClassGroups& out)
{
    struct Ref
    {
        std::int32_t thread;
        std::int64_t index;
    };
    std::vector<Ref> refs;
    for(std::size_t thread = 0; thread < pending.size(); thread++)
    {
        for(std::size_t index = 0; index < pending[thread].descriptors.size(); index++)
        {
            refs.push_back({static_cast<std::int32_t>(thread), static_cast<std::int64_t>(index)});
        }
    }
    parallel_sort(refs, [&pending](const Ref& a, const Ref& b) {
        const auto& da = pending[static_cast<std::size_t>(a.thread)].descriptors[static_cast<std::size_t>(a.index)];
        const auto& db = pending[static_cast<std::size_t>(b.thread)].descriptors[static_cast<std::size_t>(b.index)];
        if(da.owner != db.owner)
        {
            return da.owner < db.owner;
        }
        return key_less(da.key, db.key);
    });
    const std::size_t num_groups = refs.size();
    out.member_start.assign(checked_add(num_groups, 1), 0);
    out.num_outputs.resize(num_groups);
    out.owner.resize(num_groups);
    out.slot_start.assign(checked_add(num_groups, 1), 0);
    for(std::size_t g = 0; g < num_groups; g++)
    {
        const auto& d = pending[static_cast<std::size_t>(refs[g].thread)].descriptors[static_cast<std::size_t>(refs[g].index)];
        out.member_start[g + 1] = checked_index<std::int64_t>(checked_add(static_cast<std::size_t>(out.member_start[g]), static_cast<std::size_t>(d.count)));
        out.num_outputs[g] = d.count;
        out.owner[g] = d.owner;
        out.slot_start[g + 1] = checked_index<std::int64_t>(checked_add(static_cast<std::size_t>(out.slot_start[g]), static_cast<std::size_t>(d.count)));
    }
    out.members.resize(static_cast<std::size_t>(out.member_start.back()));
#pragma omp parallel for schedule(dynamic, 1024)
    for(std::size_t g = 0; g < num_groups; g++)
    {
        const PendingGroups<K>& source = pending[static_cast<std::size_t>(refs[g].thread)];
        const auto& d = source.descriptors[static_cast<std::size_t>(refs[g].index)];
        std::copy_n(source.members.data() + d.offset, d.count, out.members.data() + out.member_start[g]);
    }
}


template <std::size_t W, typename KeyOf>
inline void mark_useful(const int cls, const std::size_t count, KeyOf&& key_of, std::vector<char>& useful)
{
    useful.assign(count, 0);
    if(cls == kAlphaSingle || cls == kBetaSingle)
    {
        useful.assign(count, 1);
        return;
    }
    for(std::size_t i = 0; i < count; i++)
    {
        if(useful[i])
        {
            continue;
        }
        for(std::size_t j = i + 1; j < count; j++)
        {
            if(pair_counts<W>(cls, key_of(i), key_of(j)))
            {
                useful[i] = 1;
                useful[j] = 1;
                break;
            }
        }
        if(!useful[i])
        {
            for(std::size_t j = 0; j < i; j++)
            {
                if(pair_counts<W>(cls, key_of(i), key_of(j)))
                {
                    useful[i] = 1;
                    break;
                }
            }
        }
    }
}


template <std::size_t W>
inline std::size_t keep_useful(const KeyStore<W>& store, const int cls, std::vector<std::int32_t>& candidates)
{
    if(cls == kAlphaSingle || cls == kBetaSingle)
    {
        return candidates.size();
    }
    std::vector<char> useful;
    mark_useful<W>(cls, candidates.size(), [&](const std::size_t i) { return store.alpha(candidates[i]); }, useful);
    std::size_t kept = 0;
    for(std::size_t i = 0; i < candidates.size(); i++)
    {
        if(useful[i])
        {
            candidates[kept++] = candidates[i];
        }
    }
    candidates.resize(kept);
    return kept;
}


template <std::size_t W>
inline void build_sector_groups(const KeyStore<W>& store, const int cls, const int kind, const GroupSlice& slice, ClassGroups& out)
{
    const int sector = (cls == kAlphaSingle || cls == kAlphaDouble) ? 0 : 1;
    const int num_changed = (cls == kAlphaSingle || cls == kBetaSingle) ? 1 : 2;
    const std::int64_t num_owners = sector == 0 ? store.num_lines() : store.num_columns();
    const std::vector<std::int64_t>& start = sector == 0 ? store.line_start : store.column_start;
    std::vector<PendingGroups<W>> pending(static_cast<std::size_t>(max_threads()));
#pragma omp parallel
    {
#ifdef _OPENMP
        PendingGroups<W>& mine = pending[static_cast<std::size_t>(omp_get_thread_num())];
#else
        PendingGroups<W>& mine = pending[0];
#endif
        struct Record
        {
            std::array<Word, W> key;
            std::int32_t row;
        };
        std::vector<Record> records;
        std::vector<std::int32_t> candidates;
#pragma omp for schedule(dynamic, 16)
        for(std::int64_t owner = 0; owner < num_owners; owner++)
        {
            const std::int64_t begin = start[static_cast<std::size_t>(owner)];
            const std::int64_t end = start[static_cast<std::size_t>(owner) + 1];
            if(end - begin < 2)
            {
                continue;
            }
            records.clear();
            for(std::int64_t entry = begin; entry < end; entry++)
            {
                const std::int64_t row = sector == 0 ? entry : store.column_rows[static_cast<std::size_t>(entry)];
                const Word* string = store.alpha(row) + sector * W;
                for_each_key<W>(string, store.num_orbitals, kind, num_changed, [&](const std::array<Word, W>& key) {
                    if(slice.contains(key))
                    {
                        records.push_back({key, static_cast<std::int32_t>(row)});
                    }
                });
            }
            std::sort(records.begin(), records.end(), [](const Record& a, const Record& b) {
                if(a.key != b.key)
                {
                    return key_less(a.key, b.key);
                }
                return a.row < b.row;
            });
            for(std::size_t first = 0; first < records.size();)
            {
                std::size_t last = first + 1;
                while(last < records.size() && records[last].key == records[first].key)
                {
                    last++;
                }
                if(last - first >= 2)
                {
                    candidates.clear();
                    for(std::size_t k = first; k < last; k++)
                    {
                        candidates.push_back(records[k].row);
                    }
                    if(keep_useful(store, cls, candidates) >= 2)
                    {
                        mine.add(owner, records[first].key, candidates);
                    }
                }
                first = last;
            }
        }
    }
    merge_groups(pending, out);
}


template <std::size_t W>
inline void build_mixed_groups(const KeyStore<W>& store, const GroupKinds& kinds, const GroupSlice& slice, ClassGroups& out)
{
    struct LineRecord
    {
        std::array<Word, W> key;
        std::int64_t line;
    };
    const std::int64_t num_lines = store.num_lines();
    std::vector<std::vector<LineRecord>> line_parts(static_cast<std::size_t>(max_threads()));
#pragma omp parallel
    {
#ifdef _OPENMP
        std::vector<LineRecord>& mine = line_parts[static_cast<std::size_t>(omp_get_thread_num())];
#else
        std::vector<LineRecord>& mine = line_parts[0];
#endif
#pragma omp for schedule(static)
        for(std::int64_t line = 0; line < num_lines; line++)
        {
            for_each_key<W>(store.line_beta(line), store.num_orbitals, kinds.beta, 1, [&](const std::array<Word, W>& key) {
                mine.push_back({key, line});
            });
        }
    }
    std::vector<LineRecord> line_records;
    for(auto& part : line_parts)
    {
        line_records.insert(line_records.end(), part.begin(), part.end());
        std::vector<LineRecord>().swap(part);
    }
    parallel_sort(line_records, [](const LineRecord& a, const LineRecord& b) {
        if(a.key != b.key)
        {
            return key_less(a.key, b.key);
        }
        return a.line < b.line;
    });
    std::vector<std::pair<std::int64_t, std::int64_t>> runs;
    for(std::size_t first = 0; first < line_records.size();)
    {
        std::size_t last = first + 1;
        while(last < line_records.size() && line_records[last].key == line_records[first].key)
        {
            last++;
        }
        if(last - first >= 2)
        {
            runs.push_back({static_cast<std::int64_t>(first), static_cast<std::int64_t>(last)});
        }
        first = last;
    }
    std::vector<PendingGroups<2 * W>> pending(static_cast<std::size_t>(max_threads()));
#pragma omp parallel
    {
#ifdef _OPENMP
        PendingGroups<2 * W>& mine = pending[static_cast<std::size_t>(omp_get_thread_num())];
#else
        PendingGroups<2 * W>& mine = pending[0];
#endif
        struct Record
        {
            std::array<Word, W> key;
            std::int32_t row;
        };
        std::vector<Record> records;
        std::vector<std::int32_t> candidates;
#pragma omp for schedule(dynamic, 1)
        for(std::size_t run = 0; run < runs.size(); run++)
        {
            records.clear();
            std::array<Word, 2 * W> group_key;
            std::copy_n(line_records[static_cast<std::size_t>(runs[run].first)].key.data(), W, group_key.data() + W);
            for(std::int64_t k = runs[run].first; k < runs[run].second; k++)
            {
                const std::int64_t line = line_records[static_cast<std::size_t>(k)].line;
                for(std::int64_t row = store.line_start[static_cast<std::size_t>(line)];
                    row < store.line_start[static_cast<std::size_t>(line) + 1];
                    row++)
                {
                    for_each_key<W>(store.alpha(row), store.num_orbitals, kinds.alpha, 1, [&](const std::array<Word, W>& key) {
                        std::copy_n(key.data(), W, group_key.data());
                        if(slice.contains(group_key))
                        {
                            records.push_back({key, static_cast<std::int32_t>(row)});
                        }
                    });
                }
            }
            std::sort(records.begin(), records.end(), [](const Record& a, const Record& b) {
                if(a.key != b.key)
                {
                    return key_less(a.key, b.key);
                }
                return a.row < b.row;
            });
            for(std::size_t first = 0; first < records.size();)
            {
                std::size_t last = first + 1;
                while(last < records.size() && records[last].key == records[first].key)
                {
                    last++;
                }
                const bool two_lines =
                    last - first >= 2 &&
                    store.line_of_row[static_cast<std::size_t>(records[first].row)] !=
                        store.line_of_row[static_cast<std::size_t>(records[last - 1].row)];
                if(two_lines)
                {
                    candidates.clear();
                    for(std::size_t k = first; k < last; k++)
                    {
                        candidates.push_back(records[k].row);
                    }
                    if(keep_useful(store, kMixed, candidates) >= 2)
                    {
                        std::copy_n(records[first].key.data(), W, group_key.data());
                        mine.add(-1, group_key, candidates);
                    }
                }
                first = last;
            }
        }
    }
    merge_groups(pending, out);
}


template <std::size_t W>
inline GroupStore build_groups(const KeyStore<W>& store,
                               const GroupKinds& kinds,
                               const GroupSlice& slice = GroupSlice(),
                               const bool with_row_slots = true)
{
    if(store.num_rows() >= static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()))
    {
        throw std::invalid_argument("A block can hold at most 2^31 - 1 rows, got " + std::to_string(store.num_rows()));
    }
    GroupStore groups;
    groups.kinds = kinds;
    build_sector_groups(store, kAlphaSingle, kinds.alpha, slice, groups.classes[kAlphaSingle]);
    build_sector_groups(store, kAlphaDouble, kinds.alpha, slice, groups.classes[kAlphaDouble]);
    build_sector_groups(store, kBetaSingle, kinds.beta, slice, groups.classes[kBetaSingle]);
    build_sector_groups(store, kBetaDouble, kinds.beta, slice, groups.classes[kBetaDouble]);
    build_mixed_groups(store, kinds, slice, groups.classes[kMixed]);
    groups.class_slot_start[0] = 0;
    for(int cls = 0; cls < kNumClasses; cls++)
    {
        groups.class_slot_start[static_cast<std::size_t>(cls) + 1] =
            groups.class_slot_start[static_cast<std::size_t>(cls)] + groups.classes[static_cast<std::size_t>(cls)].num_slots();
    }
    if(!with_row_slots)
    {
        return groups;
    }
    const std::int64_t num_rows = store.num_rows();
    groups.row_slot_start.assign(static_cast<std::size_t>(num_rows) + 1, 0);
    for(const ClassGroups& c : groups.classes)
    {
        for(std::int64_t g = 0; g < c.num_groups(); g++)
        {
            const std::int64_t first = c.member_start[static_cast<std::size_t>(g)];
            for(std::int64_t k = 0; k < c.num_outputs[static_cast<std::size_t>(g)]; k++)
            {
                groups.row_slot_start[static_cast<std::size_t>(c.members[static_cast<std::size_t>(first + k)]) + 1]++;
            }
        }
    }
    for(std::int64_t row = 0; row < num_rows; row++)
    {
        groups.row_slot_start[static_cast<std::size_t>(row) + 1] = checked_index<std::int64_t>(checked_add(static_cast<std::size_t>(groups.row_slot_start[static_cast<std::size_t>(row) + 1]), static_cast<std::size_t>(groups.row_slot_start[static_cast<std::size_t>(row)])));
    }
    groups.row_slots.resize(static_cast<std::size_t>(groups.row_slot_start.back()));
    std::vector<std::int64_t> cursor(groups.row_slot_start.begin(), groups.row_slot_start.end() - 1);
    for(int cls = 0; cls < kNumClasses; cls++)
    {
        const ClassGroups& c = groups.classes[static_cast<std::size_t>(cls)];
        for(std::int64_t g = 0; g < c.num_groups(); g++)
        {
            const std::int64_t first = c.member_start[static_cast<std::size_t>(g)];
            for(std::int64_t k = 0; k < c.num_outputs[static_cast<std::size_t>(g)]; k++)
            {
                const std::size_t row = static_cast<std::size_t>(c.members[static_cast<std::size_t>(first + k)]);
                groups.row_slots[static_cast<std::size_t>(cursor[row]++)] =
                    groups.class_slot_start[static_cast<std::size_t>(cls)] + c.slot_start[static_cast<std::size_t>(g)] + k;
            }
        }
    }
    return groups;
}


template <std::size_t W>
inline std::vector<std::pair<std::int64_t, std::int64_t>> group_pairs(const KeyStore<W>& store, const GroupStore& groups, const int cls)
{
    const ClassGroups& c = groups.classes[static_cast<std::size_t>(cls)];
    std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
    for(std::int64_t g = 0; g < c.num_groups(); g++)
    {
        const std::int64_t first = c.member_start[static_cast<std::size_t>(g)];
        const std::int64_t last = c.member_start[static_cast<std::size_t>(g) + 1];
        for(std::int64_t k = 0; k < c.num_outputs[static_cast<std::size_t>(g)]; k++)
        {
            const std::int64_t out = c.members[static_cast<std::size_t>(first + k)];
            for(std::int64_t s = first; s < last; s++)
            {
                const std::int64_t source = c.members[static_cast<std::size_t>(s)];
                if(source != out && pair_counts<W>(cls, store.alpha(out), store.alpha(source)))
                {
                    pairs.push_back({out, source});
                }
            }
        }
    }
    parallel_sort(pairs, [](const auto& a, const auto& b) { return a < b; });
    return pairs;
}


template <std::size_t W>
inline int move_class(const Word* a, const Word* b)
{
    int alpha_bits = 0;
    int beta_bits = 0;
    for(std::size_t word = 0; word < W; word++)
    {
        alpha_bits += __builtin_popcountll(a[word] ^ b[word]);
        beta_bits += __builtin_popcountll(a[W + word] ^ b[W + word]);
    }
    if(beta_bits == 0)
    {
        return alpha_bits == 2 ? kAlphaSingle : (alpha_bits == 4 ? kAlphaDouble : -1);
    }
    if(alpha_bits == 0)
    {
        return beta_bits == 2 ? kBetaSingle : (beta_bits == 4 ? kBetaDouble : -1);
    }
    return alpha_bits == 2 && beta_bits == 2 ? kMixed : -1;
}


template <std::size_t W>
inline std::vector<std::pair<std::int64_t, std::int64_t>> brute_force_pairs(const KeyStore<W>& store, const int cls)
{
    const std::int64_t n = store.num_rows();
    std::vector<std::vector<std::pair<std::int64_t, std::int64_t>>> parts(static_cast<std::size_t>(max_threads()));
#pragma omp parallel
    {
#ifdef _OPENMP
        auto& mine = parts[static_cast<std::size_t>(omp_get_thread_num())];
#else
        auto& mine = parts[0];
#endif
#pragma omp for schedule(dynamic, 64)
        for(std::int64_t i = 0; i < n; i++)
        {
            for(std::int64_t j = 0; j < n; j++)
            {
                if(i != j && move_class<W>(store.alpha(i), store.alpha(j)) == cls)
                {
                    mine.push_back({i, j});
                }
            }
        }
    }
    std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
    for(const auto& part : parts)
    {
        pairs.insert(pairs.end(), part.begin(), part.end());
    }
    parallel_sort(pairs, [](const auto& a, const auto& b) { return a < b; });
    return pairs;
}


inline constexpr int kSizeBuckets = 7;


inline int size_bucket(const std::int64_t members)
{
    return std::min(kSizeBuckets - 1, 62 - __builtin_clzll(static_cast<unsigned long long>(members)));
}


struct ClassStats
{
    std::int64_t groups = 0;
    std::int64_t outputs = 0;
    std::int64_t members = 0;
    std::int64_t pairs = 0;
    std::int64_t largest = 0;
    std::array<std::int64_t, kSizeBuckets> size_groups{};
    std::array<std::int64_t, kSizeBuckets> size_members{};
};


template <std::size_t W>
inline ClassStats class_stats(const KeyStore<W>& store, const GroupStore& groups, const int cls)
{
    const ClassGroups& c = groups.classes[static_cast<std::size_t>(cls)];
    ClassStats stats;
    stats.groups = c.num_groups();
    stats.outputs = c.num_slots();
    stats.members = static_cast<std::int64_t>(c.members.size());
    std::int64_t pairs = 0;
    std::int64_t largest = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : pairs) reduction(max : largest)
    for(std::int64_t g = 0; g < c.num_groups(); g++)
    {
        const std::int64_t first = c.member_start[static_cast<std::size_t>(g)];
        const std::int64_t last = c.member_start[static_cast<std::size_t>(g) + 1];
        largest = std::max(largest, last - first);
        for(std::int64_t k = 0; k < c.num_outputs[static_cast<std::size_t>(g)]; k++)
        {
            const Word* out = store.alpha(c.members[static_cast<std::size_t>(first + k)]);
            for(std::int64_t s = first; s < last; s++)
            {
                if(s != first + k && pair_counts<W>(cls, out, store.alpha(c.members[static_cast<std::size_t>(s)])))
                {
                    pairs++;
                }
            }
        }
    }
    stats.pairs = pairs;
    stats.largest = largest;
    for(std::int64_t g = 0; g < c.num_groups(); g++)
    {
        const std::int64_t size = c.member_start[static_cast<std::size_t>(g) + 1] - c.member_start[static_cast<std::size_t>(g)];
        const std::size_t bucket = static_cast<std::size_t>(size_bucket(size));
        stats.size_groups[bucket]++;
        stats.size_members[bucket] += size;
    }
    return stats;
}


template <std::size_t W>
inline std::array<ClassStats, kNumClasses> sliced_stats(const KeyStore<W>& store, const GroupKinds& kinds, const int num_slices)
{
    std::array<ClassStats, kNumClasses> total{};
    for(int index = 0; index < num_slices; index++)
    {
        const GroupStore groups = build_groups(store, kinds, GroupSlice{num_slices, index}, false);
        for(int cls = 0; cls < kNumClasses; cls++)
        {
            const ClassStats part = class_stats(store, groups, cls);
            ClassStats& sum = total[static_cast<std::size_t>(cls)];
            sum.groups += part.groups;
            sum.outputs += part.outputs;
            sum.members += part.members;
            sum.pairs += part.pairs;
            sum.largest = std::max(sum.largest, part.largest);
            for(int bucket = 0; bucket < kSizeBuckets; bucket++)
            {
                sum.size_groups[static_cast<std::size_t>(bucket)] += part.size_groups[static_cast<std::size_t>(bucket)];
                sum.size_members[static_cast<std::size_t>(bucket)] += part.size_members[static_cast<std::size_t>(bucket)];
            }
        }
    }
    return total;
}

}
