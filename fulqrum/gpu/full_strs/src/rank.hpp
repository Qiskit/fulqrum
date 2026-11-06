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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "coefficients.hpp"
#include "keys.hpp"

namespace fq::full_strs
{


template <std::size_t W>
inline std::array<Word, 2 * W> block_reference(const Word* input, const std::int64_t num_rows, const int num_orbitals)
{
    return store_spec(build_key_store<W>(input, num_rows, num_orbitals)).reference;
}


template <std::size_t W, typename T>
class RankOperator
{
public:
    KeyStore<W> own;
    OperatorTerms<W> terms;
    CoefficientTables<W, T> tables;
    std::vector<T> diagonal;


    RankOperator(const QubitOperator& op,
                 const Word* own_keys,
                 const std::int64_t num_own,
                 const Word* reference)
    {
        const int n = orbitals_for_width(op.width, kWordBits * static_cast<int>(W));
        own = build_key_store<W>(own_keys, num_own, n);
        if(own.num_rows() != num_own || !std::equal(own.keys.begin(), own.keys.end(), own_keys))
        {
            throw std::invalid_argument("The own rows must be sorted by (beta, alpha) and unique");
        }
        terms = split_operator<W>(op, std::is_same_v<T, double>);
        TableSpec<W> spec;
        spec.num_alpha = own.num_alpha;
        spec.num_beta = own.num_beta;
        std::copy_n(reference, 2 * W, spec.reference.data());
        tables = build_coefficients<W, T>(terms, spec);
        diagonal = diagonal_values<W, T>(terms, own);
    }
};

}
