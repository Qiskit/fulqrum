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

namespace fq::sell
{


inline constexpr int kSliceRows = 32;


inline constexpr std::int64_t kMaxSlots = 9223372036854775807;


constexpr int slice_of(const int row)
{
    return row / kSliceRows;
}


constexpr int lane_of(const int row)
{
    return row % kSliceRows;
}


constexpr std::int64_t slot(const std::int64_t slice_offset, const int row, const int k)
{
    return slice_offset + std::int64_t(k) * kSliceRows + lane_of(row);
}

}
