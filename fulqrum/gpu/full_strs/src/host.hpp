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

#include <array>
#include <cstdint>
#include <vector>

#include "chunks.hpp"
#include "coefficients.hpp"
#include "groups.hpp"
#include "join.hpp"
#include "keys.hpp"
#include "rank.hpp"
#include "reference.hpp"

namespace fq::full_strs
{


template <std::size_t W>
class HostLayout
{
public:
    KeyStore<W> keys;
    GroupStore groups;


    HostLayout(const Word* input, const std::int64_t num_rows, const int num_orbitals, const int alpha_kind, const int beta_kind)
        : keys(build_key_store<W>(input, num_rows, num_orbitals))
    {
        GroupKinds kinds = default_kinds(keys);
        if(alpha_kind >= 0)
        {
            kinds.alpha = alpha_kind;
        }
        if(beta_kind >= 0)
        {
            kinds.beta = beta_kind;
        }
        groups = build_groups(keys, kinds);
    }


    ClassStats stats(const int cls) const
    {
        return class_stats(keys, groups, cls);
    }


    std::vector<std::int64_t> stored_pairs(const int cls) const
    {
        return flatten(group_pairs(keys, groups, cls));
    }


    std::vector<std::int64_t> scanned_pairs(const int cls) const
    {
        return flatten(brute_force_pairs(keys, cls));
    }


    ChunkClass chunk_class(const int num_slices, const int slice, const int cls) const
    {
        return build_chunk(keys, groups.kinds, GroupSlice{num_slices, slice}).classes[static_cast<std::size_t>(cls)];
    }


    std::pair<std::int64_t, std::uint64_t> pair_digest_sum(const int cls) const
    {
        const ClassGroups& c = groups.classes[static_cast<std::size_t>(cls)];
        std::int64_t count = 0;
        std::uint64_t digest = 0;
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : count, digest)
        for(std::int64_t g = 0; g < c.num_groups(); g++)
        {
            const std::int64_t first = c.member_start[static_cast<std::size_t>(g)];
            const std::int64_t last = c.member_start[static_cast<std::size_t>(g) + 1];
            for(std::int64_t k = first; k < first + c.num_outputs[static_cast<std::size_t>(g)]; k++)
            {
                const std::int64_t out = c.members[static_cast<std::size_t>(k)];
                for(std::int64_t s = first; s < last; s++)
                {
                    const std::int64_t source = c.members[static_cast<std::size_t>(s)];
                    if(source != out && pair_counts<W>(cls, keys.alpha(out), keys.alpha(source)))
                    {
                        count++;
                        digest += pair_digest(out, source);
                    }
                }
            }
        }
        return {count, digest};
    }

private:


    static std::vector<std::int64_t> flatten(const std::vector<std::pair<std::int64_t, std::int64_t>>& pairs)
    {
        std::vector<std::int64_t> out(2 * pairs.size());
        for(std::size_t k = 0; k < pairs.size(); k++)
        {
            out[2 * k] = pairs[k].first;
            out[2 * k + 1] = pairs[k].second;
        }
        return out;
    }
};


struct HostCounts
{
    std::int64_t num_rows = 0;
    GroupKinds kinds;
    std::array<ClassStats, kNumClasses> classes{};
    double key_seconds = 0.0;
    double group_seconds = 0.0;
};


template <std::size_t W>
inline HostCounts count_groups(const Word* input,
                               const std::int64_t num_rows,
                               const int num_orbitals,
                               const int alpha_kind,
                               const int beta_kind,
                               const int num_slices)
{
    HostCounts out;
    const double start = omp_get_wtime();
    const KeyStore<W> keys = build_key_store<W>(input, num_rows, num_orbitals);
    out.key_seconds = omp_get_wtime() - start;
    out.num_rows = keys.num_rows();
    out.kinds = default_kinds(keys);
    if(alpha_kind >= 0)
    {
        out.kinds.alpha = alpha_kind;
    }
    if(beta_kind >= 0)
    {
        out.kinds.beta = beta_kind;
    }
    out.classes = sliced_stats(keys, out.kinds, num_slices);
    out.group_seconds = omp_get_wtime() - start - out.key_seconds;
    return out;
}


template <std::size_t W, typename T>
class HostOperator
{
public:
    KeyStore<W> keys;
    OperatorTerms<W> terms;
    CoefficientTables<W, T> tables;
    std::vector<T> diagonal;
    GroupKinds kinds;
    GroupStore groups;
    std::vector<GroupChunk> chunks;


    HostOperator(const QubitOperator& op,
                 const Word* input,
                 const std::int64_t num_rows,
                 const int alpha_kind,
                 const int beta_kind,
                 const int num_slices)
    {
        const int n = orbitals_for_width(op.width, kWordBits * static_cast<int>(W));
        keys = build_key_store<W>(input, num_rows, n);
        terms = split_operator<W>(op, std::is_same_v<T, double>);
        tables = build_coefficients<W, T>(terms, store_spec(keys));
        diagonal = diagonal_values<W, T>(terms, keys);
        kinds = default_kinds(keys);
        if(alpha_kind >= 0)
        {
            kinds.alpha = alpha_kind;
        }
        if(beta_kind >= 0)
        {
            kinds.beta = beta_kind;
        }
        if(num_slices <= 0)
        {
            groups = build_groups(keys, kinds);
        }
        else
        {
            chunks = build_chunks(keys, kinds, num_slices);
        }
    }


    void matvec(const T* x, T* y) const
    {
        if(!chunks.empty())
        {
            throw std::logic_error("This build holds chunks; use chunk_matvec");
        }
        reference_matvec(keys, groups, tables, diagonal, x, y);
    }


    void chunk_partials(const T* x, const int chunk, const int cls, T* partials) const
    {
        chunk_class_partials(keys, chunks.at(static_cast<std::size_t>(chunk)).classes.at(static_cast<std::size_t>(cls)), cls, tables, x, partials);
    }


    void chunk_matvec(const T* x, T* y) const
    {
        if(chunks.empty())
        {
            throw std::logic_error("This build holds the unsliced group store; use matvec");
        }
        fq::full_strs::chunk_matvec(keys, chunks, tables, diagonal, x, y);
    }


    T table_element(const Word* out, const Word* source) const
    {
        return tables.element(out, source);
    }


    T term_element(const Word* out, const Word* source) const
    {
        std::array<Word, 2 * W> flips;
        for(std::size_t word = 0; word < 2 * W; word++)
        {
            flips[word] = out[word] ^ source[word];
        }
        const auto [cls, index] = classify_flips<W>(flips.data(), keys.num_orbitals);
        if(cls < 0)
        {
            return T(0);
        }
        const std::int64_t g = terms.find_group(cls, index);
        return g < 0 ? T(0) : terms.template group_value<T>(g, source);
    }
};

}
