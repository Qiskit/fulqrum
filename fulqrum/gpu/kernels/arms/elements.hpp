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

#include <cstdint>

namespace fq::arms
{


inline constexpr double kElementScreen = 1e-14;


constexpr double pair_element(const double c0, const double sign, const double factor)
{
    return sign * factor + c0;
}


constexpr double cross_element(const double coefficient, const double alpha_sign, const double beta_sign)
{
    return coefficient * alpha_sign * beta_sign;
}


constexpr bool is_applied(const double element)
{
    return element > kElementScreen || element < -kElementScreen;
}

#if defined(__CUDACC__)


inline constexpr unsigned long long kElementScreenBits = 0x3D06849B86A12B9Bull;
inline constexpr unsigned long long kInfinityBits = 0x7FF0000000000000ull;


__device__ __forceinline__ bool is_applied_bits(const double element)
{
    const unsigned long long magnitude =
        static_cast<unsigned long long>(__double_as_longlong(element)) & 0x7fffffffffffffffull;
    return magnitude > kElementScreenBits && magnitude <= kInfinityBits;
}
#endif

}
