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
#include <numeric>
#ifdef __GLIBCXX__
#include <parallel/algorithm>
#endif
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "chunks.hpp"
#include "coefficients.hpp"
#include "groups.hpp"
#include "keys.hpp"

namespace fq::full_strs
{

inline constexpr int kMemberClassShift = 61;
inline constexpr int kMemberFirstShift = 54;
inline constexpr int kMemberSecondShift = 47;
inline constexpr Word kMemberOrbitalMask = 0x7F;
inline constexpr std::int64_t kMaxJoinRows = std::int64_t(1) << kMemberSecondShift;
inline constexpr Word kMemberRowMask = (Word(1) << kMemberSecondShift) - 1;
inline constexpr Word kHeaderCountMask = (Word(1) << kMemberClassShift) - 1;
inline constexpr int kMaxJoinOrbitals = 127;
inline constexpr std::size_t kJoinBuckets = 256;
inline constexpr std::int64_t kMaxReceivedRows = std::int64_t(1) << 32;


inline Word pack_member(const int cls, const int first, const int second, const std::int64_t row)
{
    const Word o1 = first < 0 ? kMemberOrbitalMask : static_cast<Word>(first);
    const Word o2 = second < 0 ? kMemberOrbitalMask : static_cast<Word>(second);
    return (static_cast<Word>(cls) << kMemberClassShift) | (o1 << kMemberFirstShift) | (o2 << kMemberSecondShift) |
           static_cast<Word>(row);
}


inline int member_class(const Word word)
{
    return static_cast<int>(word >> kMemberClassShift);
}


inline int member_orbital(const Word word, const int index)
{
    const int shift = index == 0 ? kMemberFirstShift : kMemberSecondShift;
    const int orbital = static_cast<int>((word >> shift) & kMemberOrbitalMask);
    return orbital == static_cast<int>(kMemberOrbitalMask) ? -1 : orbital;
}


inline std::int64_t member_row(const Word word)
{
    return static_cast<std::int64_t>(word & kMemberRowMask);
}


template <std::size_t W>
inline void flip_orbital(Word* key, const int sector, const int orbital)
{
    key[static_cast<std::size_t>(sector) * W + static_cast<std::size_t>(orbital / kWordBits)] ^= Word(1) << (orbital % kWordBits);
}


template <std::size_t W>
inline void rebuild_member(const int cls, const Word* group_key, const Word member, Word* row)
{
    std::copy_n(group_key, 2 * W, row);
    const int first = member_orbital(member, 0);
    const int second = member_orbital(member, 1);
    if(cls == kMixed)
    {
        flip_orbital<W>(row, 0, first);
        flip_orbital<W>(row, 1, second);
        return;
    }
    const int sector = (cls == kAlphaSingle || cls == kAlphaDouble) ? 0 : 1;
    flip_orbital<W>(row, sector, first);
    if(second >= 0)
    {
        flip_orbital<W>(row, sector, second);
    }
}


template <std::size_t W>
inline int join_rank(const Word* key, const int num_ranks)
{
    std::uint64_t hash = 0xA0761D6478BD642FULL;
    for(std::size_t word = 0; word < 2 * W; word++)
    {
        hash = mix64(hash ^ (key[word] + 0xE7037ED1A0B428DBULL));
    }
    return static_cast<int>(mix64(hash) % static_cast<std::uint64_t>(num_ranks));
}


template <std::size_t W>
inline bool group_key_less(const int cls, const Word* a, const Word* b)
{
    const bool alpha_first = cls == kBetaSingle || cls == kBetaDouble;
    const std::size_t first = alpha_first ? 0 : W;
    const std::size_t second = alpha_first ? W : 0;
    const int high = compare_sector<W>(a + first, b + first);
    if(high != 0)
    {
        return high < 0;
    }
    return compare_sector<W>(a + second, b + second) < 0;
}


inline std::uint64_t pair_digest(const std::int64_t out, const std::int64_t source)
{
    return mix64(static_cast<std::uint64_t>(out) * 0x9E3779B97F4A7C15ULL + static_cast<std::uint64_t>(source));
}


template <std::size_t W>
inline std::uint32_t key_check(const Word* key)
{
    std::uint64_t hash = 0x8EBC6AF09C88C6E3ULL;
    for(std::size_t word = 0; word < 2 * W; word++)
    {
        hash = mix64(hash ^ key[word]);
    }
    return static_cast<std::uint32_t>(hash) | 1u;
}


struct RecordBatch
{
    std::vector<Word> words;
    std::vector<std::int64_t> counts;
};


template <std::size_t W>
inline RecordBatch emit_records(const Word* keys,
                                const std::int64_t num_rows,
                                const std::int64_t row_start,
                                const int num_orbitals,
                                const GroupKinds& kinds,
                                const GroupSlice& slice,
                                const int num_ranks)
{
    constexpr std::size_t R = 1 + 2 * W;
    if(row_start < 0 || row_start + num_rows > kMaxJoinRows)
    {
        throw std::invalid_argument("The join supports global rows below 2^47, got rows up to " + std::to_string(row_start + num_rows));
    }
    if(num_orbitals > kMaxJoinOrbitals)
    {
        throw std::invalid_argument("The join supports at most 127 orbitals per sector, got " + std::to_string(num_orbitals));
    }
    const int num_threads = max_threads();
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(num_threads));
    std::vector<std::vector<std::int32_t>> ranks(static_cast<std::size_t>(num_threads));
#pragma omp parallel
    {
#ifdef _OPENMP
        const std::size_t thread = static_cast<std::size_t>(omp_get_thread_num());
#else
        const std::size_t thread = 0;
#endif
        std::vector<Word>& mine = words[thread];
        std::vector<std::int32_t>& mine_ranks = ranks[thread];
        std::vector<std::pair<std::array<Word, W>, int>> beta_keys;
        std::array<Word, 2 * W> group_key;
        auto emit = [&](const int cls, const int first, const int second, const std::int64_t row) {
            mine.push_back(pack_member(cls, first, second, row));
            mine.insert(mine.end(), group_key.begin(), group_key.end());
            mine_ranks.push_back(join_rank<W>(group_key.data(), num_ranks));
        };
#pragma omp for schedule(static)
        for(std::int64_t local = 0; local < num_rows; local++)
        {
            const Word* key = keys + static_cast<std::size_t>(local) * 2 * W;
            const std::int64_t row = row_start + local;
            for(int changed = 1; changed <= 2; changed++)
            {
                const int alpha_cls = changed == 1 ? kAlphaSingle : kAlphaDouble;
                for_each_changed_key<W>(key, num_orbitals, kinds.alpha, changed, [&](const std::array<Word, W>& part, int first, int second) {
                    if(slice.contains(part))
                    {
                        std::copy_n(part.data(), W, group_key.data());
                        std::copy_n(key + W, W, group_key.data() + W);
                        emit(alpha_cls, first, second, row);
                    }
                });
                const int beta_cls = changed == 1 ? kBetaSingle : kBetaDouble;
                for_each_changed_key<W>(key + W, num_orbitals, kinds.beta, changed, [&](const std::array<Word, W>& part, int first, int second) {
                    if(slice.contains(part))
                    {
                        std::copy_n(key, W, group_key.data());
                        std::copy_n(part.data(), W, group_key.data() + W);
                        emit(beta_cls, first, second, row);
                    }
                });
            }
            beta_keys.clear();
            for_each_changed_key<W>(key + W, num_orbitals, kinds.beta, 1, [&](const std::array<Word, W>& part, int first, int) {
                beta_keys.push_back({part, first});
            });
            for_each_changed_key<W>(key, num_orbitals, kinds.alpha, 1, [&](const std::array<Word, W>& part, int first, int) {
                std::copy_n(part.data(), W, group_key.data());
                for(const auto& [beta_part, beta_orbital] : beta_keys)
                {
                    std::copy_n(beta_part.data(), W, group_key.data() + W);
                    if(slice.contains(group_key))
                    {
                        emit(kMixed, first, beta_orbital, row);
                    }
                }
            });
        }
    }
    RecordBatch out;
    const std::size_t T = words.size();
    const std::size_t P = static_cast<std::size_t>(num_ranks);
    std::vector<std::int64_t> offsets(checked_multiply(T, P), 0);
#pragma omp parallel for schedule(static, 1)
    for(std::size_t thread = 0; thread < T; thread++)
    {
        for(const std::int32_t rank : ranks[thread])
        {
            offsets[thread * P + static_cast<std::size_t>(rank)]++;
        }
    }
    out.counts.assign(P, 0);
    std::int64_t total = 0;
    for(std::size_t rank = 0; rank < P; rank++)
    {
        for(std::size_t thread = 0; thread < T; thread++)
        {
            const std::int64_t count = offsets[thread * P + rank];
            offsets[thread * P + rank] = total;
            total = checked_index<std::int64_t>(checked_add(static_cast<std::size_t>(total), static_cast<std::size_t>(count)));
            out.counts[rank] = checked_index<std::int64_t>(checked_add(static_cast<std::size_t>(out.counts[rank]), static_cast<std::size_t>(count)));
        }
    }
    out.words.resize(checked_multiply(static_cast<std::size_t>(total), R));

#pragma omp parallel for schedule(static, 1)
    for(std::size_t thread = 0; thread < T; thread++)
    {
        for(std::size_t k = 0; k < ranks[thread].size(); k++)
        {
            const std::int64_t slot = offsets[thread * P + static_cast<std::size_t>(ranks[thread][k])]++;
            std::copy_n(words[thread].data() + k * R, R, out.words.data() + static_cast<std::size_t>(slot) * R);
        }
        std::vector<Word>().swap(words[thread]);
    }
    return out;
}


struct ReturnBatch
{
    std::vector<Word> headers;
    std::vector<std::int64_t> header_counts;
    std::vector<Word> members;
    std::vector<std::int64_t> member_counts;
};


template <std::size_t W>
inline ReturnBatch join_runs(const Word* records, const std::int64_t num_records, const std::int64_t* block_starts, const int num_ranks)
{
    constexpr std::size_t R = 1 + 2 * W;
    constexpr std::size_t B = kJoinBuckets;
    using Record = std::array<Word, R>;
    if(num_ranks <= 0) throw std::invalid_argument("Rank count must be positive.");
    checked_multiply(checked_multiply(checked_index<std::size_t>(num_records), R), sizeof(Word));

    auto bucket_of = [records](const std::int64_t k) {
        const Word* r = records + static_cast<std::size_t>(k) * R;
        std::uint64_t hash = mix64(0x2545F4914F6CDD1DULL ^ static_cast<Word>(member_class(r[0])));
        for(std::size_t word = 1; word < R; word++)
        {
            hash = mix64(hash ^ r[word]);
        }
        return static_cast<std::size_t>(hash % B);
    };
    auto less = [](const Record& a, const Record& b) {
        if(member_class(a[0]) != member_class(b[0]))
        {
            return member_class(a[0]) < member_class(b[0]);
        }
        for(std::size_t word = 2 * W; word > 0; word--)
        {
            if(a[word] != b[word])
            {
                return a[word] < b[word];
            }
        }
        return member_row(a[0]) < member_row(b[0]);
    };
    const std::size_t num_threads = static_cast<std::size_t>(max_threads());
    std::vector<std::int64_t> offsets(checked_multiply(num_threads, B), 0);
    std::vector<std::int64_t> bucket_start(B + 1, 0);
    std::vector<std::uint16_t> bucket(static_cast<std::size_t>(num_records));
    std::vector<Record> sorted(static_cast<std::size_t>(num_records));
#pragma omp parallel
    {
#ifdef _OPENMP
        const std::size_t thread = static_cast<std::size_t>(omp_get_thread_num());
#else
        const std::size_t thread = 0;
#endif
        std::int64_t* mine = offsets.data() + thread * B;
#pragma omp for schedule(static)
        for(std::int64_t k = 0; k < num_records; k++)
        {
            const std::size_t b = bucket_of(k);
            bucket[static_cast<std::size_t>(k)] = static_cast<std::uint16_t>(b);
            mine[b]++;
        }
#pragma omp single
        {
            std::int64_t total = 0;
            for(std::size_t b = 0; b < B; b++)
            {
                bucket_start[b] = total;
                for(std::size_t t = 0; t < num_threads; t++)
                {
                    const std::int64_t count = offsets[t * B + b];
                    offsets[t * B + b] = total;
                    total = checked_index<std::int64_t>(checked_add(static_cast<std::size_t>(total), static_cast<std::size_t>(count)));
                }
            }
            bucket_start[B] = total;
        }
#pragma omp for schedule(static)
        for(std::int64_t k = 0; k < num_records; k++)
        {
            const Word* r = records + static_cast<std::size_t>(k) * R;
            std::copy_n(r, R, sorted[static_cast<std::size_t>(mine[bucket[static_cast<std::size_t>(k)]]++)].data());
        }
    }
    std::vector<std::uint16_t>().swap(bucket);
    std::vector<std::vector<std::vector<Word>>> headers(B, std::vector<std::vector<Word>>(static_cast<std::size_t>(num_ranks)));
    std::vector<std::vector<std::vector<Word>>> members(B, std::vector<std::vector<Word>>(static_cast<std::size_t>(num_ranks)));
#pragma omp parallel
    {
        std::vector<Word> rows;
        std::vector<char> useful;
        std::vector<Word> kept;
#pragma omp for schedule(dynamic, 1)
        for(std::size_t b = 0; b < B; b++)
        {
            const auto begin = sorted.begin() + bucket_start[b];
            const auto end = sorted.begin() + bucket_start[b + 1];
            std::sort(begin, end, less);
            for(auto first = begin; first != end;)
            {
                auto last = first + 1;
                while(last != end && member_class((*last)[0]) == member_class((*first)[0]) &&
                      std::equal(first->begin() + 1, first->end(), last->begin() + 1))
                {
                    last++;
                }
                const std::size_t count = static_cast<std::size_t>(last - first);
                if(count < 2)
                {
                    first = last;
                    continue;
                }
                const Word* head = first->data();
                const int cls = member_class(head[0]);
                rows.resize(checked_multiply(count, 2 * W));
                for(std::size_t i = 0; i < count; i++)
                {
                    rebuild_member<W>(cls, head + 1, first[static_cast<std::ptrdiff_t>(i)][0], rows.data() + i * 2 * W);
                }
                mark_useful<W>(cls, count, [&rows](const std::size_t i) { return rows.data() + i * 2 * W; }, useful);
                kept.clear();
                for(std::size_t i = 0; i < count; i++)
                {
                    if(useful[i])
                    {
                        kept.push_back(first[static_cast<std::ptrdiff_t>(i)][0]);
                    }
                }
                if(kept.size() >= 2)
                {
                    int previous = -1;
                    for(const Word member : kept)
                    {
                        const std::int64_t row = member_row(member);
                        const int owner = static_cast<int>(std::upper_bound(block_starts, block_starts + num_ranks + 1, row) - block_starts) - 1;
                        if(owner == previous)
                        {
                            continue;
                        }
                        previous = owner;
                        std::vector<Word>& h = headers[b][static_cast<std::size_t>(owner)];
                        h.push_back((static_cast<Word>(cls) << kMemberClassShift) | static_cast<Word>(kept.size()));
                        h.insert(h.end(), head + 1, head + R);
                        std::vector<Word>& m = members[b][static_cast<std::size_t>(owner)];
                        m.insert(m.end(), kept.begin(), kept.end());
                    }
                }
                first = last;
            }
        }
    }
    std::vector<Record>().swap(sorted);
    ReturnBatch out;
    out.header_counts.assign(static_cast<std::size_t>(num_ranks), 0);
    out.member_counts.assign(static_cast<std::size_t>(num_ranks), 0);
    for(int rank = 0; rank < num_ranks; rank++)
    {
        for(std::size_t b = 0; b < B; b++)
        {
            out.header_counts[static_cast<std::size_t>(rank)] = checked_index<std::int64_t>(checked_add(static_cast<std::size_t>(out.header_counts[static_cast<std::size_t>(rank)]), headers[b][static_cast<std::size_t>(rank)].size() / R));
            out.member_counts[static_cast<std::size_t>(rank)] = checked_index<std::int64_t>(checked_add(static_cast<std::size_t>(out.member_counts[static_cast<std::size_t>(rank)]), members[b][static_cast<std::size_t>(rank)].size()));
        }
    }
    out.headers.reserve(checked_multiply(std::accumulate(out.header_counts.begin(), out.header_counts.end(), std::size_t(0), [](std::size_t sum, std::int64_t count) { return checked_add(sum, static_cast<std::size_t>(count)); }), R));
    out.members.reserve(std::accumulate(out.member_counts.begin(), out.member_counts.end(), std::size_t(0), [](std::size_t sum, std::int64_t count) { return checked_add(sum, static_cast<std::size_t>(count)); }));
    for(int rank = 0; rank < num_ranks; rank++)
    {
        for(std::size_t b = 0; b < B; b++)
        {
            std::vector<Word>& h = headers[b][static_cast<std::size_t>(rank)];
            std::vector<Word>& m = members[b][static_cast<std::size_t>(rank)];
            out.headers.insert(out.headers.end(), h.begin(), h.end());
            out.members.insert(out.members.end(), m.begin(), m.end());
            std::vector<Word>().swap(h);
            std::vector<Word>().swap(m);
        }
    }
    return out;
}


template <std::size_t W>
struct GroupPieces
{
    std::vector<Word> own_keys;
    std::int64_t row_start = 0;
    std::int64_t share_begin = 0;
    std::int64_t share_end = 0;
    std::vector<std::int64_t> halo_rows;
    std::vector<Word> halo_keys;
    std::array<std::vector<GroupChunk>, 2> chunks;


    std::int64_t num_own() const
    {
        return static_cast<std::int64_t>(own_keys.size() / (2 * W));
    }


    void release_chunk(const int kind, const int chunk)
    {
        if(kind < 0 || kind > 1 || chunk < 0 || chunk >= static_cast<int>(chunks[static_cast<std::size_t>(kind)].size()))
        {
            throw std::invalid_argument("Piece chunk " + std::to_string(chunk) + " of kind " + std::to_string(kind) + " is out of range");
        }
        full_strs::release_chunk(chunks[static_cast<std::size_t>(kind)][static_cast<std::size_t>(chunk)]);
    }


    void release_halo()
    {
        std::vector<std::int64_t>().swap(halo_rows);
        std::vector<Word>().swap(halo_keys);
    }


    std::array<std::int64_t, 4> ranges(const int kind) const
    {
        const std::int64_t own = num_own();
        const std::int64_t extended = own + static_cast<std::int64_t>(halo_rows.size());
        if(kind == 0)
        {
            return {share_begin, share_end, 0, own};
        }
        return {0, own, own, extended};
    }


    std::int64_t global_row(const std::int32_t local) const
    {
        return local < num_own() ? row_start + local : halo_rows[static_cast<std::size_t>(local - num_own())];
    }


    const Word* key(const std::int32_t local) const
    {
        const std::int64_t own = num_own();
        return local < own ? own_keys.data() + static_cast<std::size_t>(local) * 2 * W
                           : halo_keys.data() + static_cast<std::size_t>(local - own) * 2 * W;
    }


    template <typename Visit>
    void for_each_pair(const int kind, const int cls, Visit&& visit) const
    {
        const std::array<std::int64_t, 4> range = ranges(kind);
        for(const GroupChunk& chunk : chunks[static_cast<std::size_t>(kind)])
        {
            const ChunkClass& c = chunk.classes[static_cast<std::size_t>(cls)];
            std::int64_t first = 0;
            for(std::int64_t g = 0; g < c.num_groups(); g++)
            {
                const std::int64_t last = first + c.sizes[static_cast<std::size_t>(g)];
                for(std::int64_t k = first; k < last; k++)
                {
                    const std::int32_t out = c.members[static_cast<std::size_t>(k)];
                    if(out < range[0] || out >= range[1])
                    {
                        continue;
                    }
                    for(std::int64_t s = first; s < last; s++)
                    {
                        const std::int32_t source = c.members[static_cast<std::size_t>(s)];
                        if(s != k && source >= range[2] && source < range[3] && pair_counts<W>(cls, key(out), key(source)))
                        {
                            visit(global_row(out), global_row(source));
                        }
                    }
                }
                first = last;
            }
        }
    }


    std::vector<std::int64_t> pair_list(const int kind, const int cls) const
    {
        std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
        for_each_pair(kind, cls, [&pairs](const std::int64_t out, const std::int64_t source) { pairs.push_back({out, source}); });
        std::sort(pairs.begin(), pairs.end());
        std::vector<std::int64_t> flat(checked_multiply(2, pairs.size()));
        for(std::size_t k = 0; k < pairs.size(); k++)
        {
            flat[2 * k] = pairs[k].first;
            flat[2 * k + 1] = pairs[k].second;
        }
        return flat;
    }


    std::pair<std::int64_t, std::uint64_t> pair_digest_sum(const int kind, const int cls) const
    {
        std::int64_t count = 0;
        std::uint64_t digest = 0;
        for_each_pair(kind, cls, [&](const std::int64_t out, const std::int64_t source) {
            count++;
            digest += pair_digest(out, source);
        });
        return {count, digest};
    }
};


struct WorkProfile
{
    std::vector<double> row_total;
    std::vector<double> row_own;
    std::vector<double> block_work;
    std::vector<std::int64_t> row_memberships;
    std::vector<std::int64_t> block_entries;
};


template <std::size_t W>
class GroupReceiver
{
public:
    std::vector<Word> own_keys;
    std::int64_t row_start = 0;
    std::vector<std::int64_t> halo_rows;
    std::vector<Word> halo_keys;
    std::vector<GroupChunk> chunks;


    GroupReceiver(const Word* keys, const std::int64_t num_rows, const std::int64_t first_row, const int num_slices,
                  const std::int64_t total_rows = 0)
        : own_keys(keys, keys + (checked_multiply(checked_multiply(checked_index<std::size_t>(num_rows), 2 * W), sizeof(Word)) / (sizeof(Word)))), row_start(first_row),
          pending_(static_cast<std::size_t>(num_slices)), total_rows_(total_rows)
    {
    }


    std::int64_t num_own() const
    {
        return static_cast<std::int64_t>(own_keys.size() / (2 * W));
    }


    void add_slice(const int slice, const Word* headers, const std::int64_t num_headers, const Word* members, const std::int64_t num_members)
    {
        constexpr std::size_t R = 1 + 2 * W;
        if(slice < 0 || slice >= static_cast<int>(pending_.size()) || finished_)
        {
            throw std::invalid_argument("Slice " + std::to_string(slice) + " is out of range or the receiver is finished");
        }
        std::array<Received, kNumClasses>& classes = pending_[static_cast<std::size_t>(slice)];
        const std::int64_t own = num_own();
        std::vector<std::int64_t> cursor(static_cast<std::size_t>(num_headers));
        std::vector<std::int64_t> slot(static_cast<std::size_t>(num_headers));
        std::vector<std::int64_t> offset(static_cast<std::size_t>(num_headers));
        std::array<std::int64_t, kNumClasses> class_headers{};
        std::array<std::int64_t, kNumClasses> class_rows{};
        std::int64_t next = 0;
        for(std::int64_t h = 0; h < num_headers; h++)
        {
            const Word* header = headers + static_cast<std::size_t>(h) * R;
            const int cls = member_class(header[0]);
            const std::int64_t count = static_cast<std::int64_t>(header[0] & kHeaderCountMask);
            if(cls < 0 || cls >= kNumClasses || next + count > num_members)
            {
                throw std::invalid_argument("A run header does not match the received member words");
            }
            cursor[static_cast<std::size_t>(h)] = next;
            slot[static_cast<std::size_t>(h)] = class_headers[static_cast<std::size_t>(cls)]++;
            offset[static_cast<std::size_t>(h)] = class_rows[static_cast<std::size_t>(cls)];
            class_rows[static_cast<std::size_t>(cls)] = checked_index<std::int64_t>(checked_add(static_cast<std::size_t>(class_rows[static_cast<std::size_t>(cls)]), static_cast<std::size_t>(count)));
            next = checked_index<std::int64_t>(checked_add(static_cast<std::size_t>(next), static_cast<std::size_t>(count)));
        }
        if(next != num_members)
        {
            throw std::invalid_argument("The received member words do not match the run headers");
        }
        std::array<std::size_t, kNumClasses> key_base{};
        std::array<std::size_t, kNumClasses> start_base{};
        std::array<std::size_t, kNumClasses> row_base{};
        for(std::size_t cls = 0; cls < kNumClasses; cls++)
        {
            Received& c = classes[cls];
            key_base[cls] = c.keys.size();
            start_base[cls] = c.start.size();
            row_base[cls] = c.rows.size();
            c.keys.resize(checked_add(key_base[cls], checked_multiply(static_cast<std::size_t>(class_headers[cls]), 2 * W)));
            c.start.resize(checked_add(start_base[cls], static_cast<std::size_t>(class_headers[cls])));
            c.rows.resize(checked_add(row_base[cls], static_cast<std::size_t>(class_rows[cls])));
        }
        std::int64_t max_row = -1;
#pragma omp parallel for schedule(static) reduction(max : max_row)
        for(std::int64_t k = 0; k < num_members; k++)
        {
            max_row = std::max(max_row, member_row(members[static_cast<std::size_t>(k)]));
        }
        if(max_row >= kMaxReceivedRows)
        {
            throw std::invalid_argument("The receiver stores global rows below 2^32, got row " + std::to_string(max_row));
        }
        if(max_row >= 0 && static_cast<std::size_t>(max_row) >= halo_seen_.size() * kWordBits)
        {

            const std::size_t rows = std::max(static_cast<std::size_t>(max_row) + 1, static_cast<std::size_t>(total_rows_));
            const std::size_t words = rows / kWordBits + 1;
            halo_seen_.reserve(words);
            halo_seen_.resize(words, 0);
            halo_check_.reserve(checked_multiply(words, kWordBits));
            halo_check_.resize(checked_multiply(words, kWordBits), 0);
        }
        std::string error;


        for(int pass = 0; pass < 2 && error.empty(); pass++)
        {
#pragma omp parallel for schedule(dynamic, 256)
            for(std::int64_t h = 0; h < num_headers; h++)
            {
                const Word* header = headers + static_cast<std::size_t>(h) * R;
                const int cls = member_class(header[0]);
                const std::int64_t count = static_cast<std::int64_t>(header[0] & kHeaderCountMask);
                const std::size_t c_index = static_cast<std::size_t>(cls);
                Received& c = classes[c_index];
                const std::size_t row_at = row_base[c_index] + static_cast<std::size_t>(offset[static_cast<std::size_t>(h)]);
                if(pass == 0)
                {
                    const std::size_t at = static_cast<std::size_t>(slot[static_cast<std::size_t>(h)]);
                    std::copy(header + 1, header + R, c.keys.begin() + static_cast<std::ptrdiff_t>(key_base[c_index] + at * 2 * W));
                    c.start[start_base[c_index] + at] = static_cast<std::int64_t>(row_at) + count;
                }
                std::array<Word, 2 * W> row;
                for(std::int64_t k = 0; k < count; k++)
                {
                    const Word member = members[static_cast<std::size_t>(cursor[static_cast<std::size_t>(h)] + k)];
                    const std::int64_t global = member_row(member);
                    const std::int64_t local = global - row_start;
                    const bool is_own = local >= 0 && local < own;
                    if(pass == 0)
                    {
                        if(member_class(member) != cls)
                        {
#pragma omp critical(fq_add_slice_error)
                            error = "A member word has another class than its run";
                            continue;
                        }
                        c.rows[row_at + static_cast<std::size_t>(k)] = static_cast<std::uint32_t>(global);
                    }
                    else if(is_own)
                    {
                        continue;
                    }
                    rebuild_member<W>(cls, header + 1, member, row.data());
                    std::uint32_t* stored = halo_check_.data() + static_cast<std::size_t>(global);
                    if(is_own)
                    {
                        if(!std::equal(row.begin(), row.end(), own_keys.data() + static_cast<std::size_t>(local) * 2 * W))
                        {
#pragma omp critical(fq_add_slice_error)
                            error = "The rebuilt key of own row " + std::to_string(global) + " differs from its key";
                        }
                    }
                    else if(pass == 0)
                    {
                        const Word bit = Word(1) << (static_cast<std::size_t>(global) % kWordBits);
                        if(!(__atomic_fetch_or(&halo_seen_[static_cast<std::size_t>(global) / kWordBits], bit, __ATOMIC_RELAXED) & bit))
                        {
                            *stored = key_check<W>(row.data());
                        }
                    }
                    else if(key_check<W>(row.data()) != *stored)
                    {
#pragma omp critical(fq_add_slice_error)
                        error = "Two runs rebuild different keys for halo row " + std::to_string(global);
                    }
                }
            }
        }
        if(!error.empty())
        {
            throw std::invalid_argument(error);
        }
    }


    void finish()
    {
        const std::size_t num_words = halo_seen_.size();
        halo_prefix_.assign(num_words + 1, 0);
        for(std::size_t w = 0; w < num_words; w++)
        {
            halo_prefix_[w + 1] = halo_prefix_[w] + __builtin_popcountll(halo_seen_[w]);
        }
        const std::size_t num_halo = static_cast<std::size_t>(halo_prefix_[num_words]);
        halo_rows.resize(num_halo);
#pragma omp parallel for schedule(static)
        for(std::size_t w = 0; w < num_words; w++)
        {
            std::size_t at = static_cast<std::size_t>(halo_prefix_[w]);
            for(Word bits = halo_seen_[w]; bits != 0; bits &= bits - 1)
            {
                halo_rows[at++] = static_cast<std::int64_t>(w * kWordBits + static_cast<std::size_t>(__builtin_ctzll(bits)));
            }
        }


        std::uint32_t* check = halo_check_.data();
        for(std::size_t at = 0; at < num_halo; at++)
        {
            check[at] = check[static_cast<std::size_t>(halo_rows[at])];
        }
        halo_check_.resize(num_halo);
        if(halo_check_.capacity() > halo_check_.size() + halo_check_.size() / 4)
        {
            halo_check_.shrink_to_fit();
        }
        std::vector<Word>().swap(halo_keys);
        const std::int64_t own = num_own();
        if(own + static_cast<std::int64_t>(halo_rows.size()) >= static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()))
        {
            throw std::invalid_argument("The own and halo rows reach 2^31 (" + std::to_string(own) + " + " +
                                        std::to_string(halo_rows.size()) + "); use more ranks");
        }
        chunks.resize(pending_.size());
        for(std::size_t slice = 0; slice < pending_.size(); slice++)
        {
            chunks[slice].slice = GroupSlice{static_cast<int>(pending_.size()), static_cast<int>(slice)};
            for(int cls = 0; cls < kNumClasses; cls++)
            {
                Received& c = pending_[slice][static_cast<std::size_t>(cls)];
                pack_chunk_class(ordered_groups(c, cls), chunks[slice].classes[static_cast<std::size_t>(cls)]);
                c = Received();
            }
        }
        std::vector<Word>().swap(halo_seen_);
        std::vector<std::int64_t>().swap(halo_prefix_);
        finished_ = true;
    }


    void finish_global(const std::int64_t* block_starts, const int num_blocks, const std::int64_t stride)
    {
        if(num_blocks <= 0 || stride < 0)
            throw std::invalid_argument("Padded block count and stride must be valid.");
        if(checked_multiply(static_cast<std::size_t>(num_blocks), static_cast<std::size_t>(stride)) >= static_cast<std::size_t>(kMaxReceivedRows))
        {
            throw std::invalid_argument("The padded rows reach 2^32 (" + std::to_string(num_blocks) + " x " +
                                        std::to_string(stride) + ")");
        }
        auto padded = [block_starts, num_blocks, stride](const std::int64_t global) {
            const std::int64_t owner = std::upper_bound(block_starts, block_starts + num_blocks + 1, global) - block_starts - 1;
            return static_cast<std::int32_t>(static_cast<std::uint32_t>(owner * stride + (global - block_starts[owner])));
        };
        chunks.resize(pending_.size());
        for(std::size_t slice = 0; slice < pending_.size(); slice++)
        {
            chunks[slice].slice = GroupSlice{static_cast<int>(pending_.size()), static_cast<int>(slice)};
            for(int cls = 0; cls < kNumClasses; cls++)
            {
                Received& c = pending_[slice][static_cast<std::size_t>(cls)];
                pack_chunk_class(ordered_groups_by(c, cls, padded, true), chunks[slice].classes[static_cast<std::size_t>(cls)]);
                c = Received();
            }
        }
        std::vector<Word>().swap(halo_seen_);
        std::vector<std::uint32_t>().swap(halo_check_);
        std::vector<std::int64_t>().swap(halo_prefix_);
        finished_ = true;
    }


    std::int64_t global_row(const std::int32_t local) const
    {
        return local < num_own() ? row_start + local : halo_rows[static_cast<std::size_t>(local - num_own())];
    }


    const Word* key(const std::int32_t local) const
    {
        const std::int64_t own = num_own();
        return local < own ? own_keys.data() + static_cast<std::size_t>(local) * 2 * W
                           : halo_keys.data() + static_cast<std::size_t>(local - own) * 2 * W;
    }


    template <typename Visit>
    void for_each_pair(const int cls, Visit&& visit) const
    {
        const std::int64_t own = num_own();
        for(const GroupChunk& chunk : chunks)
        {
            const ChunkClass& c = chunk.classes[static_cast<std::size_t>(cls)];
            std::int64_t first = 0;
            for(std::int64_t g = 0; g < c.num_groups(); g++)
            {
                const std::int64_t last = first + c.sizes[static_cast<std::size_t>(g)];
                for(std::int64_t k = first; k < last && c.members[static_cast<std::size_t>(k)] < own; k++)
                {
                    const std::int32_t out = c.members[static_cast<std::size_t>(k)];
                    for(std::int64_t s = first; s < last; s++)
                    {
                        const std::int32_t source = c.members[static_cast<std::size_t>(s)];
                        if(source != out && pair_counts<W>(cls, key(out), key(source)))
                        {
                            visit(global_row(out), global_row(source));
                        }
                    }
                }
                first = last;
            }
        }
    }


    std::vector<std::int64_t> output_memberships() const
    {
        const std::int64_t own = num_own();
        std::vector<std::int64_t> counts(static_cast<std::size_t>(own), 0);
        for(const GroupChunk& chunk : chunks)
        {
            for(const ChunkClass& c : chunk.classes)
            {
                const std::int64_t n = static_cast<std::int64_t>(c.members.size());
#pragma omp parallel for schedule(static)
                for(std::int64_t k = 0; k < n; k++)
                {
                    const std::int32_t member = c.members[static_cast<std::size_t>(k)];
                    if(member < own)
                    {
                        __atomic_fetch_add(&counts[static_cast<std::size_t>(member)], 1, __ATOMIC_RELAXED);
                    }
                }
            }
        }
        return counts;
    }


    std::pair<std::int64_t, std::uint64_t> pair_digest_sum(const int cls) const
    {
        std::int64_t count = 0;
        std::uint64_t digest = 0;
        for_each_pair(cls, [&](const std::int64_t out, const std::int64_t source) {
            count++;
            digest += pair_digest(out, source);
        });
        return {count, digest};
    }


    std::vector<std::int64_t> pair_list(const int cls) const
    {
        std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
        for_each_pair(cls, [&pairs](const std::int64_t out, const std::int64_t source) { pairs.push_back({out, source}); });
        std::sort(pairs.begin(), pairs.end());
        std::vector<std::int64_t> flat(checked_multiply(2, pairs.size()));
        for(std::size_t k = 0; k < pairs.size(); k++)
        {
            flat[2 * k] = pairs[k].first;
            flat[2 * k + 1] = pairs[k].second;
        }
        return flat;
    }


    void release_chunk(const int chunk)
    {
        if(!finished_ || chunk < 0 || chunk >= static_cast<int>(chunks.size()))
        {
            throw std::invalid_argument("Chunk " + std::to_string(chunk) + " is out of range or the receiver is not finished");
        }
        full_strs::release_chunk(chunks[static_cast<std::size_t>(chunk)]);
    }


    void release_halo()
    {
        std::vector<std::int64_t>().swap(halo_rows);
        std::vector<Word>().swap(halo_keys);
    }


    Word* halo_key_buffer()
    {
        halo_keys.assign(checked_multiply(halo_rows.size(), 2 * W), 0);
        return halo_keys.data();
    }


    void verify_halo_keys()
    {
        if(halo_keys.size() != halo_rows.size() * 2 * W)
        {
            throw std::invalid_argument("The halo keys are not written (call halo_key_buffer first)");
        }
        if(halo_check_.size() == halo_rows.size())
        {
            const std::int64_t count = static_cast<std::int64_t>(halo_rows.size());
            std::int64_t bad = 0;
            std::int64_t first = -1;
#pragma omp parallel for schedule(static) reduction(+ : bad) reduction(max : first)
            for(std::int64_t h = 0; h < count; h++)
            {
                const std::uint32_t check = halo_check_[static_cast<std::size_t>(h)];
                if(check != 0 && key_check<W>(halo_keys.data() + static_cast<std::size_t>(h) * 2 * W) != check)
                {
                    bad++;
                    first = std::max(first, halo_rows[static_cast<std::size_t>(h)]);
                }
            }
            if(bad != 0)
            {
                throw std::invalid_argument(std::to_string(bad) + " halo keys differ from their rebuilt keys (row " +
                                            std::to_string(first) + ")");
            }
        }
        std::vector<std::uint32_t>().swap(halo_check_);
    }


    std::vector<Word> halo_bitmap(const std::int64_t total_rows) const
    {
        if(!halo_rows.empty() && halo_rows.back() >= total_rows)
        {
            throw std::invalid_argument("A halo row is at or above the row count " + std::to_string(total_rows));
        }
        std::vector<Word> bits(static_cast<std::size_t>(total_rows) / kWordBits + 1, 0);
        const std::int64_t count = static_cast<std::int64_t>(halo_rows.size());
#pragma omp parallel for schedule(static)
        for(std::int64_t h = 0; h < count; h++)
        {
            const std::size_t row = static_cast<std::size_t>(halo_rows[static_cast<std::size_t>(h)]);
            __atomic_fetch_or(&bits[row / kWordBits], Word(1) << (row % kWordBits), __ATOMIC_RELAXED);
        }
        return bits;
    }


    void extend_halo(const Word* bits, const std::size_t num_words)
    {
        if(!finished_)
        {
            throw std::invalid_argument("The receiver must be finished before its halo is extended");
        }
        const std::int64_t own = num_own();
        auto has = [bits, num_words](const std::int64_t row) {
            const std::size_t w = static_cast<std::size_t>(row) / kWordBits;
            return w < num_words && ((bits[w] >> (static_cast<std::size_t>(row) % kWordBits)) & 1) != 0;
        };
        const std::int64_t old_count = static_cast<std::int64_t>(halo_rows.size());
        std::int64_t missing = 0;
        std::int64_t own_set = 0;
#pragma omp parallel for schedule(static) reduction(+ : missing)
        for(std::int64_t h = 0; h < old_count; h++)
        {
            missing += has(halo_rows[static_cast<std::size_t>(h)]) ? 0 : 1;
        }
#pragma omp parallel for schedule(static) reduction(+ : own_set)
        for(std::int64_t local = 0; local < own; local++)
        {
            own_set += has(row_start + local) ? 1 : 0;
        }
        if(missing != 0 || own_set != 0)
        {
            throw std::invalid_argument("The halo bitmap misses " + std::to_string(missing) + " halo rows and holds " +
                                        std::to_string(own_set) + " own rows");
        }
        std::vector<std::int64_t> prefix(num_words + 1, 0);
        for(std::size_t w = 0; w < num_words; w++)
        {
            prefix[w + 1] = prefix[w] + __builtin_popcountll(bits[w]);
        }
        const std::int64_t new_count = prefix[num_words];
        if(checked_add(static_cast<std::size_t>(own), static_cast<std::size_t>(new_count)) >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        {
            throw std::invalid_argument("The own and halo rows reach 2^31 (" + std::to_string(own) + " + " +
                                        std::to_string(new_count) + "); use more ranks");
        }

        auto position = [bits, &prefix](const std::int64_t row) {
            const std::size_t w = static_cast<std::size_t>(row) / kWordBits;
            const Word below = (Word(1) << (static_cast<std::size_t>(row) % kWordBits)) - 1;
            return prefix[w] + __builtin_popcountll(bits[w] & below);
        };
        for(GroupChunk& chunk : chunks)
        {
            for(ChunkClass& c : chunk.classes)
            {
                const std::int64_t n = static_cast<std::int64_t>(c.members.size());
#pragma omp parallel for schedule(static)
                for(std::int64_t k = 0; k < n; k++)
                {
                    const std::int32_t member = c.members[static_cast<std::size_t>(k)];
                    if(member >= own)
                    {
                        c.members[static_cast<std::size_t>(k)] = static_cast<std::int32_t>(
                            own + position(halo_rows[static_cast<std::size_t>(member - own)]));
                    }
                }
            }
        }


        auto spread = [&](auto& values, const std::size_t width) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            values.resize(checked_multiply(static_cast<std::size_t>(new_count), width));
            std::int64_t old = old_count - 1;
            for(std::int64_t at = new_count - 1; at >= 0; at--)
            {
                Value* target = values.data() + static_cast<std::size_t>(at) * width;
                if(old >= 0 && position(halo_rows[static_cast<std::size_t>(old)]) == at)
                {
                    if(old != at)
                    {
                        std::copy_n(values.data() + static_cast<std::size_t>(old) * width, width, target);
                    }
                    old--;
                }
                else
                {
                    std::fill_n(target, width, Value(0));
                }
            }
        };
        if(!halo_keys.empty())
        {
            spread(halo_keys, 2 * W);
        }
        if(!halo_check_.empty())
        {
            spread(halo_check_, 1);
        }
        std::vector<std::int64_t>().swap(halo_rows);
        halo_rows.resize(static_cast<std::size_t>(new_count));
#pragma omp parallel for schedule(static)
        for(std::size_t w = 0; w < num_words; w++)
        {
            std::size_t at = static_cast<std::size_t>(prefix[w]);
            for(Word word = bits[w]; word != 0; word &= word - 1)
            {
                halo_rows[at++] = static_cast<std::int64_t>(w * kWordBits + static_cast<std::size_t>(__builtin_ctzll(word)));
            }
        }
    }


    std::vector<std::int32_t> halo_chunk_counts() const
    {
        const std::int64_t own = num_own();
        std::vector<std::int32_t> counts(halo_rows.size(), 0);
        std::vector<std::int32_t> seen(halo_rows.size(), -1);
        for(std::size_t index = 0; index < chunks.size(); index++)
        {
            for(const ChunkClass& c : chunks[index].classes)
            {
                for(const std::int32_t member : c.members)
                {
                    if(member >= own && seen[static_cast<std::size_t>(member - own)] != static_cast<std::int32_t>(index))
                    {
                        seen[static_cast<std::size_t>(member - own)] = static_cast<std::int32_t>(index);
                        counts[static_cast<std::size_t>(member - own)]++;
                    }
                }
            }
        }
        return counts;
    }


    WorkProfile work_profile(const double* coefficients, const std::int32_t* halo_block, const int num_blocks,
                             const int own_block) const
    {
        const std::int64_t own = num_own();
        WorkProfile out;
        out.row_total.assign(static_cast<std::size_t>(own), 0.0);
        out.row_own.assign(static_cast<std::size_t>(own), 0.0);
        out.block_work.assign(static_cast<std::size_t>(num_blocks), 0.0);
        out.row_memberships.assign(static_cast<std::size_t>(own), 0);
        out.block_entries.assign(static_cast<std::size_t>(num_blocks), 0);
        std::vector<std::int64_t> group_members(static_cast<std::size_t>(num_blocks), 0);
        for(const GroupChunk& chunk : chunks)
        {
            for(int cls = 0; cls < kNumClasses; cls++)
            {
                const ChunkClass& c = chunk.classes[static_cast<std::size_t>(cls)];
                std::int64_t first = 0;
                for(std::int64_t g = 0; g < c.num_groups(); g++)
                {
                    const std::int64_t size = c.sizes[static_cast<std::size_t>(g)];
                    const std::int64_t last = first + size;
                    const double coefficient = coefficients[cls * kSizeBuckets + size_bucket(size)];
                    std::int64_t block_members = 0;
                    while(first + block_members < last && c.members[static_cast<std::size_t>(first + block_members)] < own)
                    {
                        block_members++;
                    }
                    for(std::int64_t k = first; k < first + block_members; k++)
                    {
                        const std::size_t row = static_cast<std::size_t>(c.members[static_cast<std::size_t>(k)]);
                        out.row_total[row] += coefficient * static_cast<double>(size - 1);
                        out.row_own[row] += coefficient * static_cast<double>(block_members - 1);
                        out.row_memberships[row]++;
                    }
                    out.block_work[static_cast<std::size_t>(own_block)] +=
                        coefficient * static_cast<double>(block_members) * static_cast<double>(block_members - 1);
                    if(block_members >= 2)
                    {
                        out.block_entries[static_cast<std::size_t>(own_block)] += block_members;
                    }
                    for(std::int64_t k = first + block_members; k < last; k++)
                    {
                        const std::int32_t source = halo_block[c.members[static_cast<std::size_t>(k)] - own];
                        out.block_work[static_cast<std::size_t>(source)] += coefficient * static_cast<double>(block_members);
                        group_members[static_cast<std::size_t>(source)]++;
                    }
                    for(int block = 0; block < num_blocks; block++)
                    {
                        if(group_members[static_cast<std::size_t>(block)] > 0)
                        {
                            out.block_entries[static_cast<std::size_t>(block)] += block_members + group_members[static_cast<std::size_t>(block)];
                            group_members[static_cast<std::size_t>(block)] = 0;
                        }
                    }
                    first = last;
                }
            }
        }
        return out;
    }


    GroupPieces<W> pieces(const std::int64_t share_begin, const std::int64_t share_end, const std::uint8_t* assigned) const
    {
        if(!finished_)
        {
            throw std::invalid_argument("The receiver must be finished before its pieces are built");
        }
        const std::int64_t own = num_own();
        GroupPieces<W> out;
        out.own_keys = own_keys;
        out.row_start = row_start;
        out.share_begin = share_begin;
        out.share_end = share_end;
        std::vector<std::int32_t> new_index(halo_rows.size(), -1);
        for(std::size_t h = 0; h < halo_rows.size(); h++)
        {
            if(assigned[h])
            {
                new_index[h] = static_cast<std::int32_t>(own + static_cast<std::int64_t>(out.halo_rows.size()));
                out.halo_rows.push_back(halo_rows[h]);
                out.halo_keys.insert(out.halo_keys.end(), halo_keys.begin() + static_cast<std::ptrdiff_t>(h * 2 * W),
                                     halo_keys.begin() + static_cast<std::ptrdiff_t>((h + 1) * 2 * W));
            }
        }
        for(int kind = 0; kind < 2; kind++)
        {
            out.chunks[static_cast<std::size_t>(kind)].resize(chunks.size());
        }
        for(std::size_t slice = 0; slice < chunks.size(); slice++)
        {
            for(int kind = 0; kind < 2; kind++)
            {
                out.chunks[static_cast<std::size_t>(kind)][slice].slice = chunks[slice].slice;
            }
            for(int cls = 0; cls < kNumClasses; cls++)
            {
                const ChunkClass& source = chunks[slice].classes[static_cast<std::size_t>(cls)];
                std::array<ClassGroups, 2> kinds;
                std::vector<std::int32_t> list;
                std::int64_t first = 0;
                for(std::int64_t g = 0; g < source.num_groups(); g++)
                {
                    const std::int64_t last = first + source.sizes[static_cast<std::size_t>(g)];
                    std::int64_t block_members = 0;
                    bool in_share = false;
                    list.clear();
                    for(std::int64_t k = first; k < last; k++)
                    {
                        const std::int32_t member = source.members[static_cast<std::size_t>(k)];
                        if(member < own)
                        {
                            block_members++;
                            in_share = in_share || (member >= share_begin && member < share_end);
                        }
                    }
                    const auto block_begin = source.members.begin() + first;
                    if(block_members >= 2 && in_share)
                    {
                        add_group(kinds[0], block_begin, block_begin + block_members);
                    }
                    for(std::int64_t k = first + block_members; k < last; k++)
                    {
                        const std::int32_t mapped = new_index[static_cast<std::size_t>(source.members[static_cast<std::size_t>(k)] - own)];
                        if(mapped >= 0)
                        {
                            list.push_back(mapped);
                        }
                    }
                    if(block_members >= 1 && !list.empty())
                    {
                        list.insert(list.begin(), block_begin, block_begin + block_members);
                        add_group(kinds[1], list.begin(), list.end());
                    }
                    first = last;
                }
                for(int kind = 0; kind < 2; kind++)
                {
                    pack_chunk_class(kinds[static_cast<std::size_t>(kind)],
                                     out.chunks[static_cast<std::size_t>(kind)][slice].classes[static_cast<std::size_t>(cls)]);
                }
            }
        }
        return out;
    }

private:


    template <typename Iterator>
    static void add_group(ClassGroups& groups, Iterator begin, Iterator end)
    {
        groups.members.insert(groups.members.end(), begin, end);
        groups.member_start.push_back(static_cast<std::int64_t>(groups.members.size()));
        groups.num_outputs.push_back(static_cast<std::int32_t>(end - begin));
    }

    struct Received
    {
        std::vector<Word> keys;
        std::vector<std::int64_t> start{0};
        std::vector<std::uint32_t> rows;
    };

    std::vector<std::array<Received, kNumClasses>> pending_;
    std::int64_t total_rows_ = 0;
    std::vector<Word> halo_seen_;
    std::vector<std::uint32_t> halo_check_;
    std::vector<std::int64_t> halo_prefix_;
    bool finished_ = false;


    std::int32_t local_index(const std::int64_t global) const
    {
        const std::int64_t local = global - row_start;
        if(local >= 0 && local < num_own())
        {
            return static_cast<std::int32_t>(local);
        }
        const std::size_t w = static_cast<std::size_t>(global) / kWordBits;
        const Word below = (Word(1) << (static_cast<std::size_t>(global) % kWordBits)) - 1;
        return static_cast<std::int32_t>(num_own() + halo_prefix_[w] + __builtin_popcountll(halo_seen_[w] & below));
    }


    ClassGroups ordered_groups(const Received& c, const int cls) const
    {
        return ordered_groups_by(c, cls, [this](const std::int64_t global) { return local_index(global); }, false);
    }


    template <typename Map>
    ClassGroups ordered_groups_by(const Received& c, const int cls, Map&& map, const bool unsigned_rows) const
    {
        const std::int64_t num_groups = static_cast<std::int64_t>(c.start.size()) - 1;

        const std::size_t first = (cls == kBetaSingle || cls == kBetaDouble) ? 0 : W;
        const std::size_t second = W - first;
        std::vector<std::pair<std::array<Word, 2 * W>, std::int64_t>> entries(static_cast<std::size_t>(num_groups));
#pragma omp parallel for schedule(static)
        for(std::int64_t g = 0; g < num_groups; g++)
        {
            const Word* key = c.keys.data() + static_cast<std::size_t>(g) * 2 * W;
            auto& entry = entries[static_cast<std::size_t>(g)];
            for(std::size_t i = 0; i < W; i++)
            {
                entry.first[i] = key[first + W - 1 - i];
                entry.first[W + i] = key[second + W - 1 - i];
            }
            entry.second = g;
        }
#ifdef __GLIBCXX__
        __gnu_parallel::sort(entries.begin(), entries.end());
#else
        parallel_sort(entries, std::less<>());
#endif
        std::vector<std::int64_t> order(static_cast<std::size_t>(num_groups));
#pragma omp parallel for schedule(static)
        for(std::int64_t g = 0; g < num_groups; g++)
        {
            order[static_cast<std::size_t>(g)] = entries[static_cast<std::size_t>(g)].second;
        }
        std::vector<std::pair<std::array<Word, 2 * W>, std::int64_t>>().swap(entries);
        ClassGroups out;
        out.member_start.assign(static_cast<std::size_t>(num_groups) + 1, 0);
        out.num_outputs.resize(static_cast<std::size_t>(num_groups));
        out.owner.assign(static_cast<std::size_t>(num_groups), -1);
        for(std::int64_t g = 0; g < num_groups; g++)
        {
            const std::int64_t source = order[static_cast<std::size_t>(g)];
            out.member_start[static_cast<std::size_t>(g) + 1] = out.member_start[static_cast<std::size_t>(g)] +
                                                                 c.start[static_cast<std::size_t>(source) + 1] -
                                                                 c.start[static_cast<std::size_t>(source)];
        }
        out.members.resize(static_cast<std::size_t>(out.member_start.back()));
        const std::int64_t own = num_own();
#pragma omp parallel for schedule(dynamic, 1024)
        for(std::int64_t g = 0; g < num_groups; g++)
        {
            const std::int64_t source = order[static_cast<std::size_t>(g)];
            const std::int64_t begin = out.member_start[static_cast<std::size_t>(g)];
            const std::int64_t end = out.member_start[static_cast<std::size_t>(g) + 1];
            std::int64_t from = c.start[static_cast<std::size_t>(source)];
            for(std::int64_t k = begin; k < end; k++)
            {
                out.members[static_cast<std::size_t>(k)] = map(static_cast<std::int64_t>(c.rows[static_cast<std::size_t>(from++)]));
            }
            if(unsigned_rows)
            {
                std::sort(out.members.begin() + begin, out.members.begin() + end, [](const std::int32_t a, const std::int32_t b) {
                    return static_cast<std::uint32_t>(a) < static_cast<std::uint32_t>(b);
                });
                out.num_outputs[static_cast<std::size_t>(g)] = static_cast<std::int32_t>(end - begin);
                continue;
            }
            std::sort(out.members.begin() + begin, out.members.begin() + end);
            out.num_outputs[static_cast<std::size_t>(g)] =
                static_cast<std::int32_t>(std::lower_bound(out.members.begin() + begin, out.members.begin() + end, own) -
                                          (out.members.begin() + begin));
        }
        out.slot_start.assign(static_cast<std::size_t>(num_groups) + 1, 0);
        for(std::int64_t g = 0; g < num_groups; g++)
        {
            out.slot_start[static_cast<std::size_t>(g) + 1] = out.slot_start[static_cast<std::size_t>(g)] + out.num_outputs[static_cast<std::size_t>(g)];
        }
        return out;
    }
};

}
