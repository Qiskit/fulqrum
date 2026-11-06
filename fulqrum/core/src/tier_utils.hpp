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

#include <complex>
#include <cstddef>
#include <type_traits>

/**
 * Returns the complex conjugate of a value, or the value itself when it is real.
 *
 * @param value The value.
 */
template <typename T>
inline T conjugate(const T& value)
{
    if constexpr(std::is_same_v<T, double>)
    {
        return value;
    }
    else
    {
        return std::conj(value);
    }
}

/**
 * Converts a complex table value to the value type of a tier.
 *
 * @param value The complex value.
 */
template <typename T>
inline T to_value(const std::complex<double>& value)
{
    if constexpr(std::is_same_v<T, double>)
    {
        return value.real();
    }
    else
    {
        return value;
    }
}

/**
 * Returns the index of an unordered pair of sector ranks.
 *
 * @param low The lower rank.
 * @param high The higher rank.
 */
inline std::size_t rank_pair_index(const std::size_t low, const std::size_t high)
{
    return high * (high - 1) / 2 + low;
}
