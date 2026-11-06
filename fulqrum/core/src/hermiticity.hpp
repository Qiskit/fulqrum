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
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "base.hpp"
#include "constants.hpp"
#include "elements.hpp"

inline constexpr std::size_t HERMITICITY_TEST_POINTS = 2;
inline constexpr double HERMITICITY_TOLERANCE = 1e-10;
inline constexpr std::uint64_t HERMITICITY_SEED = 0x5eed5eed5eedULL;

/**
 * Linearly interpolates between the operator's matrix entries for row bits 0 and 1.
 *
 * @param value The operator code (OPER_VALUE_*).
 * @param row_bit The row bit, any real value.
 * @param transposed When true, returns the entry <col|op|row> instead.
 * @return The entry.
 */
inline std::complex<double>
operator_entry(const unsigned char value, const double row_bit, const bool transposed)
{
    const std::complex<double>* entries = OPER_ELEMENTS + 4 * value;
    if(value < OPER_VALUE_X)
    {
        return (1.0 - row_bit) * entries[0] + row_bit * entries[3];
    }
    const std::complex<double> row_zero = transposed ? entries[2] : entries[1];
    const std::complex<double> row_one = transposed ? entries[1] : entries[2];
    return (1.0 - row_bit) * row_zero + row_bit * row_one;
}

/**
 * Tells whether the terms of one flip set equal their adjoint.
 *
 * @param terms The operator terms.
 * @param begin The first term of the flip set.
 * @param end One past the last term of the flip set.
 * @param points The test points, one real value per qubit each.
 * @return True when the terms are Hermitian at every test point.
 */
inline bool terms_are_hermitian(const std::vector<OperatorTerm_t>& terms,
                                const std::size_t begin,
                                const std::size_t end,
                                const std::vector<std::vector<double>>& points)
{
    for(const std::vector<double>& point : points)
    {
        std::complex<double> value = 0;
        std::complex<double> adjoint = 0;
        double scale = 0;
        for(std::size_t index = begin; index < end; index++)
        {
            const OperatorTerm_t& term = terms[index];
            std::complex<double> forward = term.coeff;
            std::complex<double> backward = std::conj(term.coeff);
            for(std::size_t factor = 0; factor < term.indices.size(); factor++)
            {
                const double row_bit = point[term.indices[factor]];
                forward *= operator_entry(term.values[factor], row_bit, false);
                backward *= std::conj(operator_entry(term.values[factor], row_bit, true));
            }
            value += forward;
            adjoint += backward;
            scale += std::abs(forward) + std::abs(backward);
        }
        if(std::abs(value - adjoint) > HERMITICITY_TOLERANCE * scale)
        {
            return false;
        }
    }
    return true;
}

/**
 * Makes the random test points of the Hermiticity check.
 *
 * @param width The number of qubits.
 * @return HERMITICITY_TEST_POINTS points with values in (0, 1), from a fixed seed.
 */
inline std::vector<std::vector<double>> hermiticity_test_points(const width_t width)
{
    std::mt19937_64 engine(HERMITICITY_SEED);
    std::uniform_real_distribution<double> distribution(0.05, 0.95);
    std::vector<std::vector<double>> points(HERMITICITY_TEST_POINTS, std::vector<double>(width));
    for(std::vector<double>& point : points)
    {
        for(double& value : point)
        {
            value = distribution(engine);
        }
    }
    return points;
}

/**
 * Returns the first group of an operator that is not Hermitian.
 *
 * Each group must hold the terms of one flip set, as group_sort() makes them.
 *
 * @param oper The operator.
 * @param group_ptrs The group boundaries, num_groups + 1 values.
 * @param num_groups The number of groups.
 * @return The group number, or num_groups when every group is Hermitian.
 */
inline std::size_t first_non_hermitian_group(const QubitOperator& oper,
                                             const std::size_t* group_ptrs,
                                             const std::size_t num_groups)
{
    const std::vector<std::vector<double>> points = hermiticity_test_points(oper.width);
    std::size_t first_failure = num_groups;
#pragma omp parallel for schedule(dynamic, 64) reduction(min : first_failure)
    for(std::size_t group = 0; group < num_groups; group++)
    {
        if(!terms_are_hermitian(oper.terms, group_ptrs[group], group_ptrs[group + 1], points))
        {
            first_failure = std::min(first_failure, group);
        }
    }
    return first_failure;
}
