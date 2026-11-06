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

#include <cmath>
#include <cstdint>
#include <vector>

#include "coefficients.hpp"
#include "groups.hpp"
#include "keys.hpp"

namespace fq::full_strs
{


template <typename T>
inline T fused_multiply_add(const T& a, const T& b, const T& c)
{
    if constexpr(std::is_same_v<T, double>)
    {
        return std::fma(a, b, c);
    }
    else
    {
        return a * b + c;
    }
}


template <std::size_t W, typename T>
inline void group_partials(const KeyStore<W>& keys,
                           const GroupStore& groups,
                           const CoefficientTables<W, T>& tables,
                           const T* x,
                           T* partials)
{
    for(int cls = 0; cls < kNumClasses; cls++)
    {
        const ClassGroups& c = groups.classes[static_cast<std::size_t>(cls)];
        const std::int64_t slot_offset = groups.class_slot_start[static_cast<std::size_t>(cls)];
#pragma omp parallel for schedule(dynamic, 256)
        for(std::int64_t g = 0; g < c.num_groups(); g++)
        {
            const std::int64_t first = c.member_start[static_cast<std::size_t>(g)];
            const std::int64_t last = c.member_start[static_cast<std::size_t>(g) + 1];
            for(std::int64_t k = 0; k < c.num_outputs[static_cast<std::size_t>(g)]; k++)
            {
                const std::int32_t out = c.members[static_cast<std::size_t>(first + k)];
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
                partials[slot_offset + c.slot_start[static_cast<std::size_t>(g)] + k] = acc;
            }
        }
    }
}


template <std::size_t W, typename T>
inline void reference_matvec(const KeyStore<W>& keys,
                             const GroupStore& groups,
                             const CoefficientTables<W, T>& tables,
                             const std::vector<T>& diagonal,
                             const T* x,
                             T* y)
{
    std::vector<T> partials(static_cast<std::size_t>(groups.class_slot_start[kNumClasses]));
    group_partials(keys, groups, tables, x, partials.data());
    const std::int64_t num_rows = keys.num_rows();
#pragma omp parallel for schedule(static)
    for(std::int64_t row = 0; row < num_rows; row++)
    {
        T acc = diagonal[static_cast<std::size_t>(row)] * x[row];
        for(std::int64_t s = groups.row_slot_start[static_cast<std::size_t>(row)];
            s < groups.row_slot_start[static_cast<std::size_t>(row) + 1];
            s++)
        {
            acc += partials[static_cast<std::size_t>(groups.row_slots[static_cast<std::size_t>(s)])];
        }
        y[row] = acc;
    }
}

}
