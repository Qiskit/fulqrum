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
#include <cstddef>
#include <cuda_runtime.h>

#include "launch.hpp"

extern "C"
{

    const char* rdm_error_name(const int status)
    {
        return cudaGetErrorName(static_cast<cudaError_t>(status));
    }

    std::size_t rdm_param_struct_size(const int which)
    {
        switch(which)
        {
        case kRdmSlabPair:
            return sizeof(RdmSlabPair);
        case kRdmPairs:
            return sizeof(RdmPairs);
        case kRdmDiagonal:
            return sizeof(RdmDiagonal);
        case kRdmSingles:
            return sizeof(RdmSingles);
        case kRdmTensor:
            return sizeof(RdmTensor);
        case kRdmAbSlabs:
            return sizeof(RdmAbSlabs);
        case kRdmAbLists:
            return sizeof(RdmAbLists);
        case kRdmWeighted:
            return sizeof(RdmWeighted);
        case kRdmRows:
            return sizeof(RdmRows);
        default:
            return 0;
        }
    }
}
