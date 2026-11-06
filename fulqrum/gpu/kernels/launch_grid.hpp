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

#include <cuda_runtime.h>
#include <type_traits>

#include "formats/clique_word.hpp"
#include "formats/params.hpp"

namespace fq::launch
{


inline constexpr int kArmThreads = 256;


inline constexpr int kCliqueThreads = fq::clique_word::kWarpsPerBlock * 32;


inline dim3 beta_grid(const unsigned int x, const int n_blocks)
{
    const unsigned int y = n_blocks <= 0 ? 1u : static_cast<unsigned int>(n_blocks < 65535 ? n_blocks : 65535);
    const unsigned int z = n_blocks <= 0 ? 1u : static_cast<unsigned int>((static_cast<long long>(n_blocks) + y - 1) / y);
    return dim3(x, y, z);
}


inline int rows_per_thread(const int requested)
{
    if(requested >= 8)
    {
        return 8;
    }
    if(requested >= 4)
    {
        return 4;
    }
    if(requested >= 2)
    {
        return 2;
    }
    return 1;
}


template <int K>
dim3 alpha_column_grid(const FqTileGeometry& geometry)
{
    const long long per_block = static_cast<long long>(geometry.threads_per_block) * K;
    const long long n_a = geometry.a_stop - geometry.a_start;
    return beta_grid(static_cast<unsigned int>((n_a + per_block - 1) / per_block), geometry.b_stop - geometry.b_start);
}


template <int K>
long long alpha_column_blocks(const FqTileGeometry& geometry)
{
    const long long per_block = static_cast<long long>(geometry.threads_per_block) * K;
    const long long n_a = geometry.a_stop - geometry.a_start;
    return ((n_a + per_block - 1) / per_block) * (static_cast<long long>(geometry.b_stop) - geometry.b_start);
}


template <int K>
dim3 beta_row_block_grid(const FqTileGeometry& geometry)
{
    const int threads = geometry.threads_per_block;
    return beta_grid(static_cast<unsigned int>((geometry.a_stop - geometry.a_start + threads - 1) / threads),
                     static_cast<int>((static_cast<long long>(geometry.b_stop) - geometry.b_start + K - 1) / K));
}


inline dim3 linear_grid(const long long blocks)
{
    const long long x = blocks < 65535 ? (blocks > 0 ? blocks : 1) : 65535;
    const long long rest = (blocks + x - 1) / x;
    const long long y = rest < 65535 ? (rest > 0 ? rest : 1) : 65535;
    const long long z = (rest + y - 1) / y;
    return dim3(static_cast<unsigned int>(x), static_cast<unsigned int>(y), static_cast<unsigned int>(z > 0 ? z : 1));
}


inline bool empty_tile(const FqTileGeometry& geometry)
{
    return geometry.a_stop <= geometry.a_start || geometry.b_stop <= geometry.b_start;
}


template <typename Launch>
int with_rows_per_thread(const FqTileGeometry& geometry, const int requested, Launch launch)
{
    if(empty_tile(geometry))
    {
        return int(cudaSuccess);
    }
    switch(rows_per_thread(requested))
    {
    case 8:
        launch(std::integral_constant<int, 8>{});
        break;
    case 4:
        launch(std::integral_constant<int, 4>{});
        break;
    case 2:
        launch(std::integral_constant<int, 2>{});
        break;
    default:
        launch(std::integral_constant<int, 1>{});
        break;
    }
    return int(cudaGetLastError());
}

}
