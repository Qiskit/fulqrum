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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

inline std::size_t checked_multiply(std::size_t a, std::size_t b)
{
    std::size_t result = 0;
    if(__builtin_mul_overflow(a, b, &result))
        throw std::overflow_error("Size multiplication overflows size_t.");
    return result;
}

inline std::size_t checked_add(std::size_t a, std::size_t b)
{
    if(a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("Size addition overflows size_t.");
    return a + b;
}

inline std::size_t ceiling_divide(std::size_t value, std::size_t divisor)
{
    if(divisor == 0)
        throw std::invalid_argument("Size divisor must be positive.");
    return value / divisor + (value % divisor != 0);
}

template <typename I, typename V>
I checked_index(V value)
{
    static_assert(std::is_integral_v<I> && std::is_integral_v<V>);
    if constexpr(std::is_signed_v<V>)
        if(value < 0)
            throw std::invalid_argument("Size or index must be nonnegative.");
    if(static_cast<std::uintmax_t>(value) > static_cast<std::uintmax_t>(std::numeric_limits<I>::max()))
        throw std::overflow_error("Size or index exceeds the destination type.");
    return static_cast<I>(value);
}
