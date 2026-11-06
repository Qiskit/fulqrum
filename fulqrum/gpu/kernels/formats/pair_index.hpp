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

namespace fq::pair_index
{


constexpr int gpu_pair(const int low, const int high, const int orbitals)
{
    return low * (2 * orbitals - low - 1) / 2 + (high - low - 1);
}


constexpr int core_pair(const int low, const int high)
{
    return high * (high - 1) / 2 + low;
}


constexpr int num_pairs(const int orbitals)
{
    return orbitals * (orbitals - 1) / 2;
}


constexpr std::uint64_t between_mask(const int low, const int high)
{
    return ((std::uint64_t(1) << high) - 1) ^ ((std::uint64_t(1) << (low + 1)) - 1);
}

}
