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
#include <cstdint>
#include <cuda_runtime.h>

#include "launch.hpp"

namespace
{

constexpr int kDiagonalThreads = 256;
constexpr std::int64_t kMaxDiagonalBlocks = 65535;


__global__ void __launch_bounds__(kDiagonalThreads) diagonal_kernel(const FsVectors vectors)
{
    const std::int64_t stride = std::int64_t(gridDim.x) * blockDim.x;
    for(std::int64_t row = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; row < vectors.num_rows; row += stride)
    {
        vectors.y[row] = vectors.diagonal[row] * vectors.x[row];
    }
}

}

extern "C"
{

    int fs_launch_diagonal(const FsVectors* vectors, void* stream)
    {
        if(vectors->num_rows <= 0)
        {
            return int(cudaSuccess);
        }
        const std::int64_t blocks = (vectors->num_rows + kDiagonalThreads - 1) / kDiagonalThreads;
        const unsigned int grid = static_cast<unsigned int>(blocks < kMaxDiagonalBlocks ? blocks : kMaxDiagonalBlocks);
        diagonal_kernel<<<grid, kDiagonalThreads, 0, static_cast<cudaStream_t>(stream)>>>(*vectors);
        return int(cudaGetLastError());
    }
}
