# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import ctypes

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
primme_mpi = pytest.importorskip("fulqrum.gpu.eigensolver._primme_mpi")


def offload_library():
    library = ctypes.CDLL(primme_mpi.__file__)
    if not hasattr(library, "fq_primme_dense_update_f64"):
        pytest.skip("the PRIMME extension is built without the GPU offload")
    long_, int_, ptr = ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p
    update = library.fq_primme_dense_update_f64
    update.argtypes = (
        [ptr, ptr, ptr, long_, int_, long_, ptr, int_, ptr]
        + [ptr, int_, int_, long_] * 5
        + [ptr, ptr, int_, int_, ptr, int_, int_]
        + [ptr, int_, int_] * 2
        + [long_]
    )
    update.restype = int_
    library.fq_primme_gpu_last_error.restype = ctypes.c_char_p
    return library


@pytest.mark.cuda_error_path
def test_a_failed_dense_update_reports_the_call_and_frees_its_workspace():
    library = offload_library()
    rows, columns = 1 << 20, 2
    h = np.eye(columns)
    h_values = np.zeros(columns)
    x0 = np.zeros((rows, columns), order="F")
    arguments = (
        (
            None,
            None,
            None,
            rows,
            columns,
            rows,
            h.ctypes.data,
            columns,
            h_values.ctypes.data,
            x0.ctypes.data,
            0,
            columns,
            rows,
        )
        + (None, 0, 0, 0) * 4
        + (None, None, 0, 0, None, 0, 0)
        + (None, 0, 0) * 2
        + (rows,)
    )
    assert library.fq_primme_dense_update_f64(*arguments) == -1
    cupy.cuda.Device().synchronize()
    free_before = cupy.cuda.runtime.memGetInfo()[0]
    library.fq_primme_gpu_last_error_clear()
    assert library.fq_primme_dense_update_f64(*arguments) == -1
    message = library.fq_primme_gpu_last_error().decode()
    assert "cudaMemcpy2D" in message and "cudaErrorInvalidValue" in message, message
    assert cupy.cuda.runtime.memGetInfo()[0] == free_before
    library.fq_primme_gpu_last_error_clear()
    assert library.fq_primme_gpu_last_error() == b""
