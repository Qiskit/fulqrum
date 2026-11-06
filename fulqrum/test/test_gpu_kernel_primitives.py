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
from pathlib import Path

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

LIBRARY = (
    Path(__file__).resolve().parents[1]
    / "gpu"
    / "kernels"
    / "tests"
    / "libfq_kernel_tests.so"
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not LIBRARY.exists(),
        reason="GPU kernel test library not built (FQ_BUILD_GPU=1)",
    ),
]


@pytest.fixture(scope="module")
def library():
    lib = ctypes.CDLL(str(LIBRARY))
    pointer, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.fq_test_sell_stream.argtypes = [pointer, pointer, pointer, i32, i32, pointer]
    lib.fq_test_decode_cross.argtypes = [pointer, i64, pointer, pointer, pointer]
    lib.fq_test_pair_element.argtypes = [
        pointer,
        pointer,
        pointer,
        i64,
        pointer,
        pointer,
    ]
    lib.fq_test_row_accumulator.argtypes = [pointer, pointer, i32, i32, pointer]
    lib.fq_test_warp.argtypes = [pointer, i32, pointer, pointer]
    lib.fq_test_tile_geometry_size.restype = ctypes.c_size_t
    return lib


def address(array):
    return ctypes.c_void_p(int(array.data.ptr))


def run(status):
    cupy.cuda.Device().synchronize()
    assert status == 0, f"CUDA error {status}"


def test_the_edge_stream_walks_each_sell_row_in_edge_order(library):
    from fulqrum.gpu.distributed.tables import _sell_layout

    rng = np.random.default_rng(5)
    lengths = rng.integers(0, 6, size=70)
    ptr = np.concatenate([[0], np.cumsum(lengths)])
    words = rng.integers(1, 2**62, size=int(ptr[-1]), dtype=np.uint64)
    offsets, row_lengths, slots = _sell_layout(ptr)
    sell = np.zeros(int(offsets[-1]), dtype=np.uint64)
    sell[slots] = words
    max_length = int(lengths.max())
    out = cupy.zeros(len(lengths) * max_length, dtype=cupy.uint64)
    inputs = [
        cupy.asarray(offsets.astype(np.int64)),
        cupy.asarray(row_lengths.astype(np.int32)),
        cupy.asarray(sell),
    ]
    run(
        library.fq_test_sell_stream(
            *(address(a) for a in inputs), len(lengths), max_length, address(out)
        )
    )
    result = cupy.asnumpy(out).reshape(len(lengths), max_length)
    for row, length in enumerate(lengths):
        np.testing.assert_array_equal(
            result[row, :length], words[ptr[row] : ptr[row + 1]]
        )
        assert not result[row, length:].any()


def test_cross_words_decode_on_the_device_as_encoded_on_the_host(library):
    from fulqrum.gpu.distributed.tile_tables import cross_words

    rng = np.random.default_rng(7)
    columns = rng.integers(0, 2**31 - 1, size=500)
    pairs = rng.integers(0, 65536, size=500)
    signs = rng.choice([-1, 1], size=500)
    words = cupy.asarray(cross_words(columns, pairs, signs))
    out_columns = cupy.zeros(500, dtype=cupy.int32)
    out_pairs = cupy.zeros(500, dtype=cupy.int32)
    out_signs = cupy.zeros(500, dtype=cupy.float64)
    run(
        library.fq_test_decode_cross(
            address(words),
            500,
            address(out_columns),
            address(out_pairs),
            address(out_signs),
        )
    )
    np.testing.assert_array_equal(cupy.asnumpy(out_columns), columns)
    np.testing.assert_array_equal(cupy.asnumpy(out_pairs), pairs)
    np.testing.assert_array_equal(cupy.asnumpy(out_signs), signs)


def test_the_pair_element_compiles_to_one_fused_multiply_add(library):
    rng = np.random.default_rng(11)
    c0 = cupy.asarray(rng.standard_normal(4096))
    sign = cupy.asarray(rng.choice([-1.0, 1.0], size=4096))
    factor = cupy.asarray(rng.standard_normal(4096) * 1e3)
    element = cupy.zeros(4096)
    fused = cupy.zeros(4096)
    run(
        library.fq_test_pair_element(
            address(c0),
            address(sign),
            address(factor),
            4096,
            address(element),
            address(fused),
        )
    )
    np.testing.assert_array_equal(cupy.asnumpy(element), cupy.asnumpy(fused))


def test_the_row_accumulator_sums_each_row(library):
    rng = np.random.default_rng(13)
    n_edges, n_rows = 9, 1030
    elements = rng.standard_normal(n_edges)
    x = rng.standard_normal((n_rows, n_edges))
    out = cupy.zeros(n_rows)
    inputs = [cupy.asarray(elements), cupy.asarray(x)]
    run(
        library.fq_test_row_accumulator(
            *(address(a) for a in inputs), n_edges, n_rows, address(out)
        )
    )
    np.testing.assert_allclose(cupy.asnumpy(out), x @ elements, rtol=1e-14, atol=1e-14)


def test_warp_any_and_broadcast_see_the_whole_warp(library):
    n = 200
    flags = np.zeros(n, dtype=np.int32)
    flags[[5, 70, 199]] = [3, 7, 9]
    flags[64] = 4
    any_out = cupy.zeros(n, dtype=cupy.int32)
    lane0 = cupy.zeros(n, dtype=cupy.int32)
    device_flags = cupy.asarray(flags)
    run(
        library.fq_test_warp(address(device_flags), n, address(any_out), address(lane0))
    )
    warps = np.arange(n) // 32
    expected_any = np.array([flags[warps == w].any() for w in warps], dtype=np.int32)
    np.testing.assert_array_equal(cupy.asnumpy(any_out), expected_any)
    np.testing.assert_array_equal(cupy.asnumpy(lane0), flags[warps * 32])


def test_the_tile_geometry_has_the_layout_of_the_header(library):
    assert library.fq_test_tile_geometry_size() == 56
