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


import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
kernel_launch = pytest.importorskip("fulqrum.gpu.distributed.kernel_launch")

pytestmark = pytest.mark.gpu


def address(array):
    return 0 if array is None else int(array.data.ptr)


def diagonal_reference(diag, x, alpha_slot, beta_slot, a_range, b_range, n_compact):
    out = np.zeros((b_range[1] - b_range[0], a_range[1] - a_range[0]))
    for b in range(*b_range):
        for a in range(*a_range):
            if alpha_slot[a] >= 0 and beta_slot[b] >= 0:
                out[b - b_range[0], a - a_range[0]] = (
                    diag[b, a] * x[beta_slot[b] * n_compact + alpha_slot[a]]
                )
    return out.ravel()


@pytest.mark.parametrize("form", ["stored", "recomputed"])
@pytest.mark.parametrize("a_range,b_range", [((0, 37), (0, 29)), ((5, 33), (3, 17))])
def test_diagonal_kernel_matches_the_reference(form, a_range, b_range):
    rng = np.random.default_rng(3)
    n_a, n_b, orbitals = 37, 29, 9
    a0, b0 = 4, 2
    alpha_strings = np.sort(
        rng.choice(2**orbitals, size=n_a + a0, replace=False).astype(np.uint64)
    )
    alpha_slot = np.arange(n_a, dtype=np.int32)
    alpha_slot[[3, 11]] = -1
    beta_slot = np.arange(n_b, dtype=np.int32)
    beta_slot[7] = -1
    x = rng.standard_normal(n_a * n_b)
    out0 = rng.standard_normal((b_range[1] - b_range[0]) * (a_range[1] - a_range[0]))
    if form == "stored":
        diag = rng.standard_normal((n_b, n_a))
        tables = dict(local=cupy.asarray(diag), alpha=None, beta=None, mb=None)
    else:
        d_alpha, d_beta = rng.standard_normal(n_a), rng.standard_normal(n_b)
        mb = rng.standard_normal((orbitals, n_b))
        bits = (
            (alpha_strings[a0:, None] >> np.arange(orbitals, dtype=np.uint64))
            & np.uint64(1)
        ).astype(float)
        diag = d_alpha[None, :] + d_beta[:, None] + ((1.0 - 2.0 * bits) @ mb).T
        tables = dict(
            local=None,
            alpha=cupy.asarray(d_alpha),
            beta=cupy.asarray(d_beta),
            mb=cupy.asarray(mb),
        )
    device = dict(
        x=cupy.asarray(x),
        out=cupy.asarray(out0),
        alpha_slot=cupy.asarray(alpha_slot),
        beta_slot=cupy.asarray(beta_slot),
        strings=cupy.asarray(alpha_strings),
    )
    geometry = (
        a0,
        n_a,
        b0,
        n_b,
        a0,
        n_a,
        b0,
        n_b,
        a_range[0],
        a_range[1],
        b_range[0],
        b_range[1],
        n_a,
    )
    kernel_launch.launch_diagonal(
        geometry,
        address(device["x"]),
        address(device["out"]),
        address(device["alpha_slot"]),
        address(device["beta_slot"]),
        address(tables["local"]),
        address(tables["alpha"]),
        address(tables["beta"]),
        address(tables["mb"]),
        address(device["strings"]),
        orbitals,
        0,
    )
    cupy.cuda.Device().synchronize()
    expected = out0 + diagonal_reference(
        diag, x, alpha_slot, beta_slot, a_range, b_range, n_a
    )
    np.testing.assert_allclose(
        cupy.asnumpy(device["out"]), expected, rtol=1e-13, atol=1e-13
    )


def test_an_empty_range_launches_nothing():
    geometry = (0, 4, 0, 4, 0, 4, 0, 4, 2, 2, 0, 4, 4)
    kernel_launch.launch_diagonal(geometry, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0)


def random_alpha_singles(rng, n_a, n_b, n_src, orbitals):
    lengths = rng.integers(0, 5, size=n_a)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    columns = rng.integers(0, n_src, size=int(offsets[-1]))
    pairs = rng.integers(0, orbitals * (orbitals - 1) // 2, size=int(offsets[-1]))
    signs = rng.choice([-1, 1], size=int(offsets[-1]))
    return offsets, columns, pairs, signs


@pytest.mark.parametrize("rows_per_thread", [1, 2, 4, 8])
@pytest.mark.parametrize("b_range", [(0, 23), (4, 19)])
def test_alpha_singles_pair_kernel_matches_the_reference(rows_per_thread, b_range):
    from fulqrum.gpu.distributed.tables import _sell_layout
    from fulqrum.gpu.distributed.tile_tables import cross_words

    rng = np.random.default_rng(rows_per_thread)
    n_a, n_b, n_src, orbitals = 70, 23, 41, 6
    offsets, columns, pairs, signs = random_alpha_singles(
        rng, n_a, n_b, n_src, orbitals
    )
    c0 = rng.standard_normal(columns.size)
    c0[::7] = 0.0
    table = rng.standard_normal((n_b, orbitals * (orbitals - 1) // 2))
    beta_slot = np.arange(n_b, dtype=np.int32)
    beta_slot[5] = -1
    x = rng.standard_normal(n_b * n_src)
    sell_offsets, sell_lengths, slots = _sell_layout(offsets)
    words = np.zeros(int(sell_offsets[-1]), dtype=np.uint64)
    words[slots] = cross_words(columns, pairs, signs)
    c0_sell = np.zeros(words.size)
    c0_sell[slots] = c0
    n_rows = b_range[1] - b_range[0]
    out0 = rng.standard_normal(n_rows * n_a)
    device = [
        cupy.asarray(a)
        for a in (
            x,
            out0,
            np.arange(n_a, dtype=np.int32),
            beta_slot,
            sell_offsets.astype(np.int64),
            sell_lengths.astype(np.int32),
            words,
            c0_sell,
            table,
        )
    ]
    geometry = (0, n_a, 0, n_b, 0, n_src, 0, n_b, 0, n_a, b_range[0], b_range[1], n_src)
    kernel_launch.launch_alpha_singles_pair(
        geometry, *(address(a) for a in device), table.shape[1], rows_per_thread, 0
    )
    cupy.cuda.Device().synchronize()
    expected = out0.reshape(n_rows, n_a).copy()
    for b in range(*b_range):
        if beta_slot[b] < 0:
            continue
        for a in range(n_a):
            for e in range(offsets[a], offsets[a + 1]):
                element = signs[e] * table[b, pairs[e]] + c0[e]
                if abs(element) > 1e-14:
                    expected[b - b_range[0], a] += (
                        element * x[beta_slot[b] * n_src + columns[e]]
                    )
    np.testing.assert_allclose(
        cupy.asnumpy(device[1]), expected.ravel(), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("rows_per_thread", [1, 4])
def test_alpha_singles_delta_kernel_matches_the_reference(rows_per_thread):
    rng = np.random.default_rng(10 + rows_per_thread)
    n_a, n_b, n_src, orbitals, b0 = 45, 17, 30, 7, 3
    offsets, columns, _, _ = random_alpha_singles(rng, n_a, n_b, n_src, orbitals)
    c0 = rng.standard_normal(columns.size)
    delta = rng.standard_normal((columns.size, orbitals))
    beta_strings = np.sort(
        rng.choice(2**orbitals, size=n_b + b0, replace=False).astype(np.uint64)
    )
    alpha_slot = np.arange(n_src, dtype=np.int32)
    alpha_slot[2] = -1
    beta_slot = np.arange(n_b, dtype=np.int32)
    x = rng.standard_normal(n_b * n_src)
    out0 = np.zeros(n_b * n_a)
    device = [
        cupy.asarray(a)
        for a in (
            x,
            out0,
            alpha_slot,
            beta_slot,
            offsets.astype(np.int64),
            columns.astype(np.int32),
            c0,
            delta,
            beta_strings,
        )
    ]
    geometry = (0, n_a, b0, n_b, 0, n_src, b0, n_b, 0, n_a, 0, n_b, n_src)
    kernel_launch.launch_alpha_singles_delta(
        geometry, *(address(a) for a in device), orbitals, rows_per_thread, 0
    )
    cupy.cuda.Device().synchronize()
    expected = np.zeros((n_b, n_a))
    bits = (
        (beta_strings[b0:, None] >> np.arange(orbitals, dtype=np.uint64)) & np.uint64(1)
    ).astype(float)
    for b in range(n_b):
        for a in range(n_a):
            for e in range(offsets[a], offsets[a + 1]):
                if alpha_slot[columns[e]] < 0:
                    continue
                element = c0[e] + bits[b] @ delta[e]
                if abs(element) > 1e-14:
                    expected[b, a] += (
                        element * x[beta_slot[b] * n_src + alpha_slot[columns[e]]]
                    )
    np.testing.assert_allclose(
        cupy.asnumpy(device[1]), expected.ravel(), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("alphas_per_thread", [1, 2, 4, 8])
@pytest.mark.parametrize("a_range", [(0, 300), (40, 290)])
def test_beta_singles_pair_kernel_matches_the_reference(alphas_per_thread, a_range):
    from fulqrum.gpu.distributed.tile_tables import SIGNED_PAIR_NEGATIVE

    rng = np.random.default_rng(20 + alphas_per_thread)
    n_a, n_b, n_src, orbitals = 300, 9, 13, 5
    offsets, columns, pairs, signs = random_alpha_singles(
        rng, n_b, n_a, n_src, orbitals
    )
    c0 = rng.standard_normal(columns.size)
    c0[::5] = 0.0
    words = pairs.astype(np.uint32) | np.where(
        signs < 0, np.uint32(SIGNED_PAIR_NEGATIVE), np.uint32(0)
    )
    table = rng.standard_normal((orbitals * (orbitals - 1) // 2, n_a))
    alpha_slot = np.arange(n_a, dtype=np.int32)
    alpha_slot[[7, 250]] = -1
    beta_slot = np.arange(n_src, dtype=np.int32)
    beta_slot[3] = -1
    x = rng.standard_normal(n_src * n_a)
    n_cols = a_range[1] - a_range[0]
    out0 = rng.standard_normal(n_b * n_cols)
    device = [
        cupy.asarray(a)
        for a in (
            x,
            out0,
            alpha_slot,
            beta_slot,
            offsets.astype(np.int64),
            columns.astype(np.int32),
            c0,
            words,
            table,
        )
    ]
    geometry = (0, n_a, 0, n_b, 0, n_a, 0, n_src, a_range[0], a_range[1], 0, n_b, n_a)
    kernel_launch.launch_beta_singles_pair(
        geometry, *(address(a) for a in device), n_a, alphas_per_thread, 0
    )
    cupy.cuda.Device().synchronize()
    expected = out0.reshape(n_b, n_cols).copy()
    for b in range(n_b):
        for e in range(offsets[b], offsets[b + 1]):
            if beta_slot[columns[e]] < 0:
                continue
            for a in range(*a_range):
                if alpha_slot[a] < 0:
                    continue
                element = signs[e] * table[pairs[e], a] + c0[e]
                if abs(element) > 1e-14:
                    expected[b, a - a_range[0]] += (
                        element * x[beta_slot[columns[e]] * n_a + alpha_slot[a]]
                    )
    np.testing.assert_allclose(
        cupy.asnumpy(device[1]), expected.ravel(), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("block_group", [0, 1])
@pytest.mark.parametrize("alphas_per_thread", [1, 2, 4, 8])
@pytest.mark.parametrize("a_range", [(0, 300), (40, 290)])
def test_beta_singles_pair_hoist_kernel_equals_the_screen_first_kernel(
    alphas_per_thread, a_range, block_group
):
    from fulqrum.gpu.distributed.tile_tables import SIGNED_PAIR_NEGATIVE

    rng = np.random.default_rng(40 + alphas_per_thread)
    n_a, n_b, n_src, orbitals = 300, 9, 13, 5
    offsets, columns, pairs, signs = random_alpha_singles(
        rng, n_b, n_a, n_src, orbitals
    )
    c0 = rng.standard_normal(columns.size)
    c0[::5] = 0.0
    words = pairs.astype(np.uint32) | np.where(
        signs < 0, np.uint32(SIGNED_PAIR_NEGATIVE), np.uint32(0)
    )
    table = rng.standard_normal((orbitals * (orbitals - 1) // 2, n_a))
    table[:, ::7] = 0.0
    alpha_slot = np.arange(n_a, dtype=np.int32)
    alpha_slot[[7, 250]] = -1
    beta_slot = np.arange(n_src, dtype=np.int32)
    beta_slot[3] = -1
    x = rng.standard_normal(n_src * n_a)
    n_cols = a_range[1] - a_range[0]
    out0 = rng.standard_normal(n_b * n_cols)
    geometry = (0, n_a, 0, n_b, 0, n_a, 0, n_src, a_range[0], a_range[1], 0, n_b, n_a)
    results = []
    for hoist in (False, True):
        device = [
            cupy.asarray(a)
            for a in (
                x,
                out0,
                alpha_slot,
                beta_slot,
                offsets.astype(np.int64),
                columns.astype(np.int32),
                c0,
                words,
                table,
            )
        ]
        kernel_launch.launch_beta_singles_pair(
            geometry,
            *(address(a) for a in device),
            n_a,
            alphas_per_thread,
            0,
            block_group,
            hoist,
        )
        cupy.cuda.Device().synchronize()
        results.append(cupy.asnumpy(device[1]))
    assert np.array_equal(results[0], results[1])


@pytest.mark.parametrize("alphas_per_thread", [1, 4])
def test_beta_singles_delta_kernel_matches_the_reference(alphas_per_thread):
    rng = np.random.default_rng(30 + alphas_per_thread)
    n_a, n_b, n_src, orbitals, a0 = 280, 7, 11, 9, 5
    offsets, columns, _, _ = random_alpha_singles(rng, n_b, n_a, n_src, orbitals)
    c0 = rng.standard_normal(columns.size)
    delta = rng.standard_normal((columns.size, orbitals))
    alpha_strings = np.sort(
        rng.choice(2**orbitals, size=n_a + a0, replace=False).astype(np.uint64)
    )
    alpha_slot = np.arange(n_a, dtype=np.int32)
    beta_slot = np.arange(n_src, dtype=np.int32)
    beta_slot[1] = -1
    x = rng.standard_normal(n_src * n_a)
    out0 = np.zeros(n_b * n_a)
    device = [
        cupy.asarray(a)
        for a in (
            x,
            out0,
            alpha_slot,
            beta_slot,
            offsets.astype(np.int64),
            columns.astype(np.int32),
            c0,
            delta,
            alpha_strings,
        )
    ]
    geometry = (a0, n_a, 0, n_b, a0, n_a, 0, n_src, 0, n_a, 0, n_b, n_a)
    kernel_launch.launch_beta_singles_delta(
        geometry, *(address(a) for a in device), orbitals, alphas_per_thread, 0
    )
    cupy.cuda.Device().synchronize()
    bits = (
        (alpha_strings[a0:, None] >> np.arange(orbitals, dtype=np.uint64))
        & np.uint64(1)
    ).astype(float)
    expected = np.zeros((n_b, n_a))
    for b in range(n_b):
        for e in range(offsets[b], offsets[b + 1]):
            if beta_slot[columns[e]] < 0:
                continue
            for a in range(n_a):
                element = c0[e] + bits[a] @ delta[e]
                if abs(element) > 1e-14:
                    expected[b, a] += (
                        element * x[beta_slot[columns[e]] * n_a + alpha_slot[a]]
                    )
    np.testing.assert_allclose(
        cupy.asnumpy(device[1]), expected.ravel(), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    "rows_per_thread,min_blocks,identity_slot",
    [(1, 4, False), (2, 5, False), (4, 6, True), (8, 6, False), (4, 4, True)],
)
def test_alpha_doubles_kernel_matches_the_reference(
    rows_per_thread, min_blocks, identity_slot
):
    from fulqrum.gpu.distributed.tables import _sell_layout

    rng = np.random.default_rng(40 + rows_per_thread + min_blocks)
    n_a, n_b, n_src = 75, 21, 33
    offsets, columns, _, _ = random_alpha_singles(rng, n_a, n_b, n_src, 4)
    values = rng.standard_normal(columns.size)
    values[::6] = 1e-15
    alpha_slot = np.arange(n_src, dtype=np.int32)
    if not identity_slot:
        alpha_slot = rng.permutation(n_src).astype(np.int32)
        alpha_slot[4] = -1
    beta_slot = np.arange(n_b, dtype=np.int32)
    beta_slot[2] = -1
    x = rng.standard_normal(n_b * n_src)
    sell_offsets, sell_lengths, slots = _sell_layout(offsets)
    sell_columns = np.zeros(int(sell_offsets[-1]), dtype=np.int32)
    sell_columns[slots] = columns
    sell_values = np.zeros(sell_columns.size)
    sell_values[slots] = values
    out0 = rng.standard_normal(n_b * n_a)
    device = [
        cupy.asarray(a)
        for a in (
            x,
            out0,
            alpha_slot,
            beta_slot,
            sell_offsets.astype(np.int64),
            sell_lengths.astype(np.int32),
            sell_columns,
            sell_values,
        )
    ]
    geometry = (0, n_a, 0, n_b, 0, n_src, 0, n_b, 0, n_a, 0, n_b, n_src)
    kernel_launch.launch_alpha_doubles(
        geometry,
        *(address(a) for a in device),
        rows_per_thread,
        min_blocks,
        identity_slot,
        0,
    )
    cupy.cuda.Device().synchronize()
    expected = out0.reshape(n_b, n_a).copy()
    for b in range(n_b):
        if beta_slot[b] < 0:
            continue
        for a in range(n_a):
            for e in range(offsets[a], offsets[a + 1]):
                slot = columns[e] if identity_slot else alpha_slot[columns[e]]
                if slot >= 0 and abs(values[e]) > 1e-14:
                    expected[b, a] += values[e] * x[beta_slot[b] * n_src + slot]
    np.testing.assert_allclose(
        cupy.asnumpy(device[1]), expected.ravel(), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("alphas_per_thread", [1, 2, 4, 8])
def test_beta_doubles_kernel_matches_the_reference(alphas_per_thread):
    rng = np.random.default_rng(50 + alphas_per_thread)
    n_a, n_b, n_src = 290, 8, 12
    offsets, columns, _, _ = random_alpha_singles(rng, n_b, n_a, n_src, 4)
    values = rng.standard_normal(columns.size)
    values[::4] = -1e-15
    alpha_slot = rng.permutation(n_a).astype(np.int32)
    alpha_slot[9] = -1
    beta_slot = np.arange(n_src, dtype=np.int32)
    beta_slot[5] = -1
    x = rng.standard_normal(n_src * n_a)
    a_range = (30, 260)
    n_cols = a_range[1] - a_range[0]
    out0 = rng.standard_normal(n_b * n_cols)
    device = [
        cupy.asarray(a)
        for a in (
            x,
            out0,
            alpha_slot,
            beta_slot,
            offsets.astype(np.int64),
            columns.astype(np.int32),
            values,
        )
    ]
    geometry = (0, n_a, 0, n_b, 0, n_a, 0, n_src, a_range[0], a_range[1], 0, n_b, n_a)
    kernel_launch.launch_beta_doubles(
        geometry, *(address(a) for a in device), alphas_per_thread, 0
    )
    cupy.cuda.Device().synchronize()
    expected = out0.reshape(n_b, n_cols).copy()
    for b in range(n_b):
        for e in range(offsets[b], offsets[b + 1]):
            if beta_slot[columns[e]] < 0 or abs(values[e]) <= 1e-14:
                continue
            for a in range(*a_range):
                if alpha_slot[a] >= 0:
                    expected[b, a - a_range[0]] += (
                        values[e] * x[beta_slot[columns[e]] * n_a + alpha_slot[a]]
                    )
    np.testing.assert_allclose(
        cupy.asnumpy(device[1]), expected.ravel(), rtol=1e-12, atol=1e-12
    )


def double_class_case(rng, nan=None):
    from itertools import combinations

    strings = np.array(
        sorted(sum(1 << o for o in c) for c in combinations(range(12), 6)),
        dtype=np.uint64,
    )
    out_strings, src_strings = strings[:700], strings[150:]
    diff = np.bitwise_count(out_strings[:, None] ^ src_strings[None, :]) == 4
    rows, cols = np.nonzero(diff)
    values = rng.standard_normal(rows.size)
    values[::9] = 1e-15
    offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=len(out_strings)))].astype(
        np.int64
    )
    n_src, stride = len(src_strings), 160
    x = rng.standard_normal(n_src * stride)
    if nan is not None:
        e = np.flatnonzero(np.abs(values) > 1e-14)[7]
        column = cols[e]
        if nan == "hidden":
            values[cols == column] = (
                1e-15  # every edge that reads this source row is screened
            )
            x.reshape(n_src, stride)[column] = np.nan
        else:
            x.reshape(n_src, stride)[column, 20] = np.nan
    return dict(
        offsets=offsets,
        columns=cols.astype(np.int32),
        values=values,
        out_strings=out_strings,
        src_strings=src_strings,
        x=x,
        n_src=n_src,
        stride=stride,
    )


def run_double_class_case(case, min_size, a_range, beta_slot):
    from fulqrum.gpu.distributed.doubles_classes import double_class_tables

    n_b, n_src, stride = len(case["out_strings"]), case["n_src"], case["stride"]
    tables = double_class_tables(
        case["offsets"],
        case["columns"],
        case["values"],
        case["out_strings"],
        case["src_strings"],
        min_size,
    )
    n_cols = a_range[1] - a_range[0]
    geometry = (
        0,
        150,
        0,
        n_b,
        0,
        150,
        0,
        n_src,
        a_range[0],
        a_range[1],
        0,
        n_b,
        stride,
    )
    alpha_slot = cupy.arange(stride, dtype=cupy.int32)
    d_beta_slot, x = cupy.asarray(beta_slot), cupy.asarray(case["x"])
    outs = []
    for form in ("class", "csr"):
        out = cupy.zeros(n_b * n_cols)
        common = (
            geometry,
            address(x),
            address(out),
            address(alpha_slot),
            address(d_beta_slot),
        )
        if form == "csr":
            d = [
                cupy.asarray(a)
                for a in (case["offsets"], case["columns"], case["values"])
            ]
            kernel_launch.launch_beta_doubles(*common, *(address(a) for a in d), 4, 0)
        else:
            d = [
                cupy.asarray(getattr(tables, k))
                for k in (
                    "blocks",
                    "block_offsets",
                    "tile_info",
                    "row_members",
                    "col_members",
                )
            ]
            flag = cupy.zeros(1, dtype=cupy.int32)
            d.append(flag)
            kernel_launch.launch_beta_doubles_class(
                *common, *(address(a) for a in d), tables.stats["tiles"], 2, 0
            )
            if tables.stats["residual_edges"]:
                r = [
                    cupy.asarray(getattr(tables, k))
                    for k in ("residual_offsets", "residual_columns", "residual_values")
                ]
                kernel_launch.launch_beta_doubles(
                    *common, *(address(a) for a in r), 4, 0
                )
        cupy.cuda.Device().synchronize()
        outs.append(cupy.asnumpy(out))
    return outs[0], outs[1], tables.stats


@pytest.mark.parametrize("min_size", [1, 6, 16])
@pytest.mark.parametrize("a_range", [(0, 150), (7, 133)])
def test_beta_doubles_class_kernel_matches_the_csr_kernel(min_size, a_range):
    rng = np.random.default_rng(60 + min_size)
    case = double_class_case(rng)
    beta_slot = rng.permutation(case["n_src"]).astype(np.int32)
    beta_slot[11] = -1
    got, want, stats = run_double_class_case(case, min_size, a_range, beta_slot)
    assert stats["classes"] > 0 and stats["class_edges"] > 0
    assert (stats["residual_edges"] == 0) == (min_size == 1)
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13 * np.abs(want).max())


@pytest.mark.parametrize("nan", ["hidden", "visible"])
def test_beta_doubles_class_kernel_keeps_a_hidden_nan_out_of_the_output(nan):
    case = double_class_case(np.random.default_rng(61), nan=nan)
    beta_slot = np.arange(case["n_src"], dtype=np.int32)
    got, want, _ = run_double_class_case(case, 1, (0, 150), beta_slot)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    assert np.isnan(want).any() == (nan == "visible")
    finite = ~np.isnan(want)
    np.testing.assert_allclose(
        got[finite], want[finite], rtol=1e-13, atol=1e-13 * np.abs(want[finite]).max()
    )


def test_beta_doubles_class_kernel_ignores_a_nan_outside_the_output_alphas():
    case = double_class_case(np.random.default_rng(63))
    case["x"].reshape(case["n_src"], case["stride"])[:, 140] = np.nan
    beta_slot = np.arange(case["n_src"], dtype=np.int32)
    got, want, _ = run_double_class_case(case, 1, (7, 133), beta_slot)
    assert np.isfinite(want).all()
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13 * np.abs(want).max())


@pytest.mark.parametrize("min_size", [1, 8, 40])
def test_double_class_tables_built_on_the_device_equal_the_host_tables(min_size):
    from fulqrum.gpu.distributed.doubles_classes import double_class_tables

    case = double_class_case(np.random.default_rng(64))
    case["values"][3] = np.inf
    case["values"][4] = np.nan
    offsets = case["offsets"].copy()
    offsets[5:9] = offsets[
        5
    ]  # rows 5..7 empty; row 8 takes their edges (some (row, column) pairs twice)
    host = double_class_tables(
        offsets,
        case["columns"],
        case["values"],
        case["out_strings"],
        case["src_strings"],
        min_size,
    )
    device = double_class_tables(
        *(
            cupy.asarray(a)
            for a in (
                offsets,
                case["columns"],
                case["values"],
                case["out_strings"],
                case["src_strings"],
            )
        ),
        min_size,
        xp=cupy,
    )
    assert device.stats == host.stats
    # a block entry that sums repeated (row, column) pairs may differ in the last bit (device atomic order)
    np.testing.assert_allclose(
        cupy.asnumpy(device.blocks), host.blocks, rtol=1e-15, atol=0
    )
    for name in (
        "block_offsets",
        "tile_info",
        "row_members",
        "col_members",
        "residual_offsets",
        "residual_columns",
        "residual_values",
    ):
        np.testing.assert_array_equal(
            cupy.asnumpy(getattr(device, name)), getattr(host, name), err_msg=name
        )


def test_double_class_tables_without_residual_edges_build_on_the_device():
    from fulqrum.gpu.distributed.doubles_classes import double_class_tables

    case = double_class_case(np.random.default_rng(64))
    args = (
        case["offsets"],
        case["columns"],
        case["values"],
        case["out_strings"],
        case["src_strings"],
    )
    host = double_class_tables(*args, 1)
    assert host.stats["residual_edges"] == 0
    device = double_class_tables(*(cupy.asarray(a) for a in args), 1, xp=cupy)
    assert device.stats == host.stats
    np.testing.assert_array_equal(
        cupy.asnumpy(device.residual_offsets), host.residual_offsets
    )


@pytest.mark.parametrize("slab", [64, 100, 512])
@pytest.mark.parametrize("nan", [None, "hidden", "visible"])
@pytest.mark.parametrize("leftover", ["sell", "xt"])
def test_alpha_doubles_class_launch_matches_the_sell_kernel(slab, nan, leftover):
    from itertools import combinations

    from fulqrum.gpu.distributed.doubles_classes import (
        double_class_tables,
        sell_from_csr,
    )
    from fulqrum.gpu.distributed.tables import _sell_layout

    rng = np.random.default_rng(70 + slab)
    strings = np.array(
        sorted(sum(1 << o for o in c) for c in combinations(range(12), 6)),
        dtype=np.uint64,
    )[:220]
    n_a = n_src = len(strings)
    n_b = 150
    rows, cols = np.nonzero(np.bitwise_count(strings[:, None] ^ strings[None, :]) == 4)
    values = rng.standard_normal(rows.size)
    values[::9] = 1e-15
    offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=n_a))].astype(np.int64)
    x = rng.standard_normal(n_b * n_src)
    if nan == "visible":
        x[17 * n_src + cols[5]] = np.nan
    elif nan == "hidden":
        column = cols[np.flatnonzero(np.abs(values) > 1e-14)[3]]
        values[cols == column] = 1e-15
        x.reshape(n_b, n_src)[:, column] = np.nan
    sell_offsets, sell_lengths, slots = _sell_layout(offsets)
    sell_columns = np.zeros(int(sell_offsets[-1]), dtype=np.int32)
    sell_columns[slots] = cols
    sell_values = np.zeros(sell_columns.size)
    sell_values[slots] = values
    tables = double_class_tables(offsets, cols, values, strings, strings, 6)
    assert tables.stats["tiles"] > 0 and tables.stats["residual_edges"] > 0
    residual = [
        cupy.asarray(a)
        for a in sell_from_csr(
            tables.residual_offsets, tables.residual_columns, tables.residual_values
        )
    ]
    alpha_slot = cupy.arange(n_src, dtype=cupy.int32)
    beta_slot = cupy.arange(n_b, dtype=cupy.int32)
    geometry = (0, n_a, 0, n_b, 0, n_src, 0, n_b, 0, n_a, 0, n_b, n_src)
    d_x = cupy.asarray(x)
    sell = [
        cupy.asarray(a)
        for a in (
            sell_offsets.astype(np.int64),
            sell_lengths.astype(np.int32),
            sell_columns,
        )
    ]
    outs = []
    for form in ("class", "sell"):
        out = cupy.zeros(n_b * n_a)
        common = (
            geometry,
            address(d_x),
            address(out),
            address(alpha_slot),
            address(beta_slot),
        )
        if form == "sell":
            kernel_launch.launch_alpha_doubles(
                *common,
                *(address(a) for a in sell),
                address(cupy.asarray(sell_values)),
                4,
                0,
                True,
                0,
            )
        else:
            d = [
                cupy.asarray(getattr(tables, k))
                for k in (
                    "blocks",
                    "block_offsets",
                    "tile_info",
                    "row_members",
                    "col_members",
                )
            ]
            flag = cupy.zeros(1, dtype=cupy.int32)
            xt, yt = cupy.empty(slab * n_src), cupy.empty(slab * n_a)
            left = [
                cupy.asarray(tables.residual_offsets),
                cupy.asarray(tables.residual_columns.astype(np.int32)),
                cupy.asarray(tables.residual_values),
                cupy.arange(slab, dtype=cupy.int32),
            ]
            extra = tuple(address(a) for a in left) if leftover == "xt" else ()
            kernel_launch.launch_alpha_doubles_class(
                *common,
                *(address(a) for a in d),
                address(flag),
                tables.stats["tiles"],
                2,
                address(xt),
                address(yt),
                slab,
                n_src,
                0,
                *extra,
            )
            if leftover == "sell":
                kernel_launch.launch_alpha_doubles(
                    *common, *(address(a) for a in residual), 4, 0, True, 0
                )
        cupy.cuda.Device().synchronize()
        outs.append(cupy.asnumpy(out))
    got, want = outs
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    assert np.isnan(want).any() == (nan == "visible")
    finite = ~np.isnan(want)
    np.testing.assert_allclose(
        got[finite], want[finite], rtol=1e-13, atol=1e-13 * np.abs(want[finite]).max()
    )


@pytest.mark.parametrize("xp_name", ["numpy", "cupy"])
def test_sell_from_csr_matches_the_sell_layout_of_the_tables(xp_name):
    from fulqrum.gpu.distributed.doubles_classes import sell_from_csr
    from fulqrum.gpu.distributed.tables import _sell_layout

    xp = np if xp_name == "numpy" else cupy
    rng = np.random.default_rng(80)
    lengths = rng.integers(0, 7, size=75)
    lengths[[3, 40, 41]] = 0
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    columns = rng.integers(0, 1000, size=int(offsets[-1])).astype(np.int32)
    values = rng.standard_normal(columns.size)
    off, ln, sell_cols, sell_vals = (
        np.asarray(a) if xp is np else cupy.asnumpy(a)
        for a in sell_from_csr(
            xp.asarray(offsets), xp.asarray(columns), xp.asarray(values), xp=xp
        )
    )
    want_off, want_len, slots = _sell_layout(offsets)
    np.testing.assert_array_equal(off, want_off)
    np.testing.assert_array_equal(ln, want_len)
    np.testing.assert_array_equal(sell_cols[slots], columns)
    np.testing.assert_array_equal(sell_vals[slots], values)
    padding = np.setdiff1d(np.arange(len(sell_vals)), slots)
    assert not sell_vals[padding].any()


def test_alpha_doubles_class_launch_rejects_a_leftover_without_a_slab_identity():
    geometry = (0, 64, 0, 8, 0, 64, 0, 8, 0, 64, 0, 8, 64)
    with pytest.raises(RuntimeError, match="cudaErrorInvalidValue"):
        kernel_launch.launch_alpha_doubles_class(
            geometry, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 64, 64, 0, 1, 1, 1, 0
        )


def test_alpha_doubles_class_launch_rejects_a_partial_output_range():
    geometry = (0, 64, 0, 8, 0, 64, 0, 8, 4, 60, 0, 8, 64)
    with pytest.raises(RuntimeError, match="cudaErrorInvalidValue"):
        kernel_launch.launch_alpha_doubles_class(
            geometry, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 64, 64, 0
        )


def test_double_class_tables_keep_every_applied_edge_once():
    from fulqrum.gpu.distributed.doubles_classes import (
        CHUNK,
        CHUNK_DOUBLES,
        double_class_tables,
    )

    case = double_class_case(np.random.default_rng(62))
    case["values"][3] = (
        np.inf
    )  # applied (|inf| > screen) but not finite: the residual takes it; NaN is not applied
    case["values"][4] = np.nan
    for min_size in (1, 8, 40):
        t = double_class_tables(
            case["offsets"],
            case["columns"],
            case["values"],
            case["out_strings"],
            case["src_strings"],
            min_size,
        )
        applied = int((np.abs(case["values"]) > 1e-14).sum())
        assert t.stats["class_edges"] + t.stats["residual_edges"] == applied
        assert int(np.count_nonzero(t.blocks)) == t.stats["class_edges"]
        assert (
            np.isinf(t.residual_values).sum() == 1
            and not np.isnan(t.residual_values).any()
        )
        dense = {}
        for tile, (r0, nr, c0, nc) in enumerate(t.tile_info):
            for r in range(nr):
                for k in range(nc):
                    pos = (
                        t.block_offsets[tile]
                        + (k // CHUNK) * CHUNK_DOUBLES
                        + ((k % CHUNK) // 4) * 256
                        + (r // 8) * 32
                        + (r % 8) * 4
                        + k % 4
                    )
                    if t.blocks[pos] != 0:
                        dense[
                            (int(t.row_members[r0 + r]), int(t.col_members[c0 + k]))
                        ] = t.blocks[pos]
        rows = np.repeat(np.arange(len(case["offsets"]) - 1), np.diff(case["offsets"]))
        for r, c, v in zip(rows, case["columns"], case["values"]):
            if (int(r), int(c)) in dense:
                assert dense[(int(r), int(c))] == v


@pytest.mark.parametrize("a_start", [5, 32])
def test_sell_alpha_kernels_read_the_absolute_bra_row(a_start):
    from fulqrum.gpu.distributed.tables import _sell_layout
    from fulqrum.gpu.distributed.tile_tables import cross_words

    rng = np.random.default_rng(60 + a_start)
    n_a, n_b, n_src, orbitals = 70, 9, 41, 6
    offsets, columns, pairs, signs = random_alpha_singles(
        rng, n_a, n_b, n_src, orbitals
    )
    c0, values = rng.standard_normal(columns.size), rng.standard_normal(columns.size)
    table = rng.standard_normal((n_b, orbitals * (orbitals - 1) // 2))
    x = rng.standard_normal(n_b * n_src)
    sell_offsets, sell_lengths, slots = _sell_layout(offsets)
    words = np.zeros(int(sell_offsets[-1]), dtype=np.uint64)
    words[slots] = cross_words(columns, pairs, signs)
    sell_c0, sell_values, sell_columns = (
        np.zeros(words.size),
        np.zeros(words.size),
        np.zeros(words.size, np.int32),
    )
    sell_c0[slots], sell_values[slots], sell_columns[slots] = c0, values, columns
    slots_a = np.arange(n_src, dtype=np.int32)
    slots_b = np.arange(n_b, dtype=np.int32)
    geometry = (0, n_a, 0, n_b, 0, n_src, 0, n_b, a_start, n_a, 0, n_b, n_src)
    shared = [
        cupy.asarray(a)
        for a in (
            x,
            slots_a,
            slots_b,
            sell_offsets.astype(np.int64),
            sell_lengths.astype(np.int32),
        )
    ]
    out_pair = cupy.zeros(n_b * (n_a - a_start))
    out_doubles = cupy.zeros(n_b * (n_a - a_start))
    pair_tables = [cupy.asarray(a) for a in (words, sell_c0, table)]
    double_tables = [cupy.asarray(a) for a in (sell_columns, sell_values)]
    x_d, slot_a_d, slot_b_d, off_d, len_d = shared
    kernel_launch.launch_alpha_singles_pair(
        geometry,
        address(x_d),
        address(out_pair),
        address(slot_a_d),
        address(slot_b_d),
        address(off_d),
        address(len_d),
        *(address(a) for a in pair_tables),
        table.shape[1],
        4,
        0,
    )
    kernel_launch.launch_alpha_doubles(
        geometry,
        address(x_d),
        address(out_doubles),
        address(slot_a_d),
        address(slot_b_d),
        address(off_d),
        address(len_d),
        *(address(a) for a in double_tables),
        4,
        6,
        True,
        0,
    )
    cupy.cuda.Device().synchronize()
    expected_pair = np.zeros((n_b, n_a - a_start))
    expected_doubles = np.zeros((n_b, n_a - a_start))
    for b in range(n_b):
        for a in range(a_start, n_a):
            for e in range(offsets[a], offsets[a + 1]):
                element = signs[e] * table[b, pairs[e]] + c0[e]
                if abs(element) > 1e-14:
                    expected_pair[b, a - a_start] += element * x[b * n_src + columns[e]]
                if abs(values[e]) > 1e-14:
                    expected_doubles[b, a - a_start] += (
                        values[e] * x[b * n_src + columns[e]]
                    )
    np.testing.assert_allclose(
        cupy.asnumpy(out_pair), expected_pair.ravel(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        cupy.asnumpy(out_doubles), expected_doubles.ravel(), rtol=1e-12, atol=1e-12
    )


def gpu_pair(low, high, orbitals):
    return low * (2 * orbitals - low - 1) // 2 + (high - low - 1)


@pytest.mark.parametrize("a_range", [(0, 70), (5, 70), (32, 61)])
@pytest.mark.parametrize("b_range", [(0, 19), (3, 17)])
def test_cross_sell_kernel_matches_the_reference(a_range, b_range):
    from fulqrum.gpu.distributed.tables import _sell_layout
    from fulqrum.gpu.distributed.tile_tables import cross_words

    rng = np.random.default_rng(70 + a_range[0] + b_range[0])
    n_a, n_b, n_src, n_b_src, num_alpha, num_beta, b0 = 70, 19, 41, 23, 6, 7, 3
    offsets, columns, pairs, signs = random_alpha_singles(
        rng, n_a, n_b, n_src, num_alpha
    )
    columns = columns.astype(np.int64)
    columns[::9] = -1
    sell_offsets, sell_lengths, slots = _sell_layout(offsets)
    words = np.zeros(int(sell_offsets[-1]), dtype=np.uint64)
    words[slots] = cross_words(np.where(columns < 0, 0xFFFFFFFF, columns), pairs, signs)
    beta_lengths = rng.integers(0, 4, size=n_b)
    beta_offsets = np.concatenate([[0], np.cumsum(beta_lengths)]).astype(np.int64)
    beta_columns = rng.integers(0, n_b_src, size=int(beta_offsets[-1])).astype(np.int32)
    beta_low = rng.integers(0, num_beta - 1, size=beta_columns.size)
    beta_high = np.array(
        [rng.integers(low + 1, num_beta) for low in beta_low], dtype=np.int64
    )
    beta_strings = np.sort(
        rng.choice(2**num_beta, size=b0 + n_b, replace=False)
    ).astype(np.uint64)
    beta_slot = rng.permutation(n_b_src).astype(np.int32)
    beta_slot[4] = -1
    g = rng.standard_normal(
        (num_beta * (num_beta - 1) // 2, num_alpha * (num_alpha - 1) // 2)
    )
    g[:, 2] = 0.0
    x = rng.standard_normal(n_b_src * n_src)
    n_rows, n_cols = b_range[1] - b_range[0], a_range[1] - a_range[0]
    out0 = rng.standard_normal(n_rows * n_cols)
    device = [
        cupy.asarray(a)
        for a in (
            x,
            out0,
            beta_slot,
            beta_offsets,
            beta_columns,
            beta_low.astype(np.uint8),
            beta_high.astype(np.uint8),
            beta_strings,
            g,
            sell_offsets.astype(np.int64),
            sell_lengths.astype(np.int32),
            words,
        )
    ]
    geometry = (
        0,
        n_a,
        b0,
        n_b,
        0,
        n_src,
        0,
        n_b_src,
        a_range[0],
        a_range[1],
        b_range[0],
        b_range[1],
        n_src,
    )
    kernel_launch.launch_cross_sell(
        geometry, *(address(a) for a in device), num_alpha, num_beta, 0
    )
    cupy.cuda.Device().synchronize()
    expected = out0.reshape(n_rows, n_cols).copy()
    for b in range(*b_range):
        beta = int(beta_strings[b0 + b])
        for edge in range(beta_offsets[b], beta_offsets[b + 1]):
            slot = beta_slot[beta_columns[edge]]
            if slot < 0:
                continue
            low, high = int(beta_low[edge]), int(beta_high[edge])
            between = ((1 << high) - 1) ^ ((1 << (low + 1)) - 1)
            beta_sign = -1.0 if bin(beta & between).count("1") % 2 else 1.0
            row = g[gpu_pair(low, high, num_beta)]
            for a in range(*a_range):
                for e in range(offsets[a], offsets[a + 1]):
                    if columns[e] < 0 or row[pairs[e]] == 0.0:
                        continue
                    expected[b - b_range[0], a - a_range[0]] += (
                        row[pairs[e]]
                        * signs[e]
                        * beta_sign
                        * x[slot * n_src + columns[e]]
                    )
    np.testing.assert_allclose(
        cupy.asnumpy(device[1]), expected.ravel(), rtol=1e-12, atol=1e-12
    )


def test_cross_direct_kernel_matches_the_reference():
    from fulqrum.gpu.distributed.tables import _sell_layout
    from fulqrum.gpu.distributed.tile_tables import cross_words

    rng = np.random.default_rng(81)
    n_a, n_b, n_src, n_b_src, num_alpha, num_beta = 70, 19, 41, 23, 6, 7
    offsets, columns, pairs, signs = random_alpha_singles(
        rng, n_a, n_b, n_src, num_alpha
    )
    columns = columns.astype(np.int64)
    columns[::9] = -1
    sell_offsets, sell_lengths, slots = _sell_layout(offsets)
    words = np.zeros(int(sell_offsets[-1]), dtype=np.uint64)
    words[slots] = cross_words(np.where(columns < 0, 0xFFFFFFFF, columns), pairs, signs)
    counts = rng.integers(0, 4, size=n_b)
    counts[6] = 0
    starts = np.concatenate([[0], np.cumsum(counts)])
    order = rng.permutation(n_b)
    tasks = np.array([[b, starts[b], starts[b + 1]] for b in order], dtype=np.int32)
    n_words = int(starts[-1])
    beta_columns = rng.integers(0, n_b_src, size=n_words)
    beta_pairs = rng.integers(0, num_beta * (num_beta - 1) // 2, size=n_words)
    beta_signs = rng.choice([-1, 1], size=n_words)
    beta_words = cross_words(beta_columns, beta_pairs, beta_signs)
    g = rng.standard_normal(
        (num_beta * (num_beta - 1) // 2, num_alpha * (num_alpha - 1) // 2)
    )
    g[:, 2] = 0.0
    x = rng.standard_normal(n_b_src * n_src)
    out0 = rng.standard_normal(n_b * n_a)
    device = [
        cupy.asarray(a)
        for a in (
            x,
            out0,
            sell_offsets.astype(np.int64),
            sell_lengths.astype(np.int32),
            words,
            beta_words,
            tasks,
            g,
        )
    ]
    geometry = (0, n_a, 0, n_b, 0, n_src, 0, n_b_src, 0, n_a, 0, n_b, n_src)
    kernel_launch.launch_cross_direct(
        geometry, *(address(a) for a in device), g.shape[1], n_b, 0
    )
    cupy.cuda.Device().synchronize()
    expected = out0.reshape(n_b, n_a).copy()
    for b, first, last in tasks:
        for w in range(first, last):
            row = g[beta_pairs[w]]
            for a in range(n_a):
                for e in range(offsets[a], offsets[a + 1]):
                    if columns[e] < 0 or row[pairs[e]] == 0.0:
                        continue
                    expected[b, a] += (
                        row[pairs[e]]
                        * signs[e]
                        * beta_signs[w]
                        * x[beta_columns[w] * n_src + columns[e]]
                    )
    assert (expected.ravel() != out0).sum() > out0.size // 4
    np.testing.assert_allclose(
        cupy.asnumpy(device[1]), expected.ravel(), rtol=1e-12, atol=1e-12
    )


CLIQUE_ORBITALS = 34
CLIQUE_FAMILIES = (
    ((0, 1), range(2, 34)),
    ((2, 3), range(4, 13)),
    ((13, 14), range(15, 27)),
    ((27, 28), range(29, 31)),
    ((4, 5), (6, 7, 8, 9, 10, 11, 12, 31)),
)


def clique_strings():
    strings = {
        (1 << p) | (1 << q) | (1 << h)
        for (p, q), holes in CLIQUE_FAMILIES
        for h in holes
    }
    return np.array(sorted(strings), dtype=np.uint64)


def clique_singles(strings, orbitals):
    from fulqrum.gpu.distributed.tables import _sell_layout
    from fulqrum.gpu.distributed.tile_tables import cross_words

    index = {int(s): i for i, s in enumerate(strings)}
    edges = {}
    rows = []
    for r, s in enumerate(int(x) for x in strings):
        row = []
        for p in range(orbitals):
            for q in range(orbitals):
                if not (s >> p) & 1 or (s >> q) & 1:
                    continue
                j = index.get(s ^ (1 << p) ^ (1 << q))
                if j is None:
                    continue
                low, high = min(p, q), max(p, q)
                between = ((1 << high) - 1) ^ ((1 << (low + 1)) - 1)
                sign = -1 if bin(s & between).count("1") % 2 else 1
                edges[(r, j)] = (gpu_pair(low, high, orbitals), sign)
                row.append((j, gpu_pair(low, high, orbitals), sign))
        rows.append(row)
    offsets, lengths, slots = _sell_layout(np.r_[0, np.cumsum([len(r) for r in rows])])
    flat = [e for row in rows for e in row]
    words = np.zeros(int(offsets[-1]), dtype=np.uint64)
    words[slots] = cross_words(
        [e[0] for e in flat], [e[1] for e in flat], [e[2] for e in flat]
    )
    return offsets.astype(np.int64), lengths.astype(np.int32), words, edges


def clique_case(rng, transposed, tile_rows, nan=None, distinct=False):
    from fulqrum.gpu.distributed.aabb_parents import (
        orbital_g,
        parent_plan,
        clique_size_buckets,
        clique_edge_group_plan,
    )
    from fulqrum.gpu.distributed.tile_tables import cross_words

    strings = clique_strings()
    offsets, lengths, sell, edges = clique_singles(strings, CLIQUE_ORBITALS)
    plan = parent_plan(
        strings, offsets, lengths, sell, CLIQUE_ORBITALS, 2, kind="intersection"
    )
    assert plan["stats"]["residual_edges"] == 0
    parent_ptr, members = plan["parent_ptr"], plan["members"]
    assert sorted(np.diff(parent_ptr)) == [2, 8, 9, 12, 32]
    ids, start = clique_size_buckets(parent_ptr)
    n_clique, n_rows, word_pairs = len(strings), 19, 21
    n_src_rows = n_clique if transposed else 23
    n_columns = 29 if transposed else n_src_rows
    n_words = 160
    pairs = np.where(
        rng.random(n_words) < 0.8,
        rng.integers(0, 3, n_words),
        rng.integers(0, word_pairs, n_words),
    )
    columns = rng.integers(0, n_columns - 1, n_words)
    signs = rng.choice([-1, 1], n_words)
    rows = rng.integers(0, n_rows, n_words)
    if distinct:
        pairs = rng.integers(0, word_pairs, n_words)
        _, keep = np.unique(rows * word_pairs + pairs, return_index=True)
        keep = np.sort(keep)
        pairs, columns, signs, rows = (
            pairs[keep],
            columns[keep],
            signs[keep],
            rows[keep],
        )
        n_words = len(keep)
    g = rng.standard_normal((word_pairs, CLIQUE_ORBITALS * (CLIQUE_ORBITALS - 1) // 2))
    g[:, ::17] = 0.0
    if nan is not None:
        pairs[5], columns[5] = word_pairs - 1, n_columns - 1
        if nan == "hidden":
            g[word_pairs - 1] = 0.0
    groups = clique_edge_group_plan(
        cross_words(columns, pairs, signs), rows, n_rows, tile_rows
    )
    stride = n_columns if transposed else n_clique
    x = rng.standard_normal(n_src_rows * stride)
    if nan is not None and transposed:
        x.reshape(-1, stride)[:, n_columns - 1] = np.nan
    if nan is not None and not transposed:
        x.reshape(-1, stride)[n_columns - 1] = np.nan
    out_rows, out_columns = (n_clique, n_rows) if transposed else (n_rows, n_clique)
    out0 = rng.standard_normal(out_rows * out_columns)
    expected = out0.reshape(out_rows, out_columns).copy()
    member_ids = (members & np.uint64(0xFFFFFFFF)).astype(np.int64)
    for word in range(n_words):
        for parent in range(len(parent_ptr) - 1):
            group = member_ids[parent_ptr[parent] : parent_ptr[parent + 1]]
            for k in group:
                for l in group:
                    if k == l:
                        continue
                    pair, sign = edges[(int(k), int(l))]
                    coefficient = g[pairs[word], pair]
                    if coefficient == 0.0:
                        continue
                    source = (
                        x[l * stride + columns[word]]
                        if transposed
                        else x[columns[word] * stride + l]
                    )
                    term = coefficient * sign * signs[word] * source
                    if transposed:
                        expected[k, rows[word]] += term
                    else:
                        expected[rows[word], k] += term
    return dict(
        groups=groups,
        parent_ptr=parent_ptr,
        members=members,
        ids=ids,
        start=start,
        coefficients=orbital_g(g, CLIQUE_ORBITALS),
        x=x,
        out0=out0,
        expected=expected.ravel(),
        n_rows=n_rows,
        n_clique=n_clique,
        n_src_rows=n_src_rows,
        stride=stride,
    )


def run_clique_case(
    case,
    transposed,
    tile_rows,
    edges_per_batch,
    mma=False,
    distinct_rows=False,
    panel=False,
):
    shared_limit = min(
        int(cupy.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]), 227 * 1024
    )
    kernel_launch.prepare_cross_clique(shared_limit)
    groups = case["groups"]
    device = {
        name: cupy.asarray(value)
        for name, value in (
            ("x", case["x"]),
            ("out", case["out0"]),
            ("words", groups["words"]),
            ("group_ptr", groups["group_ptr"]),
            ("group_bpid", groups["group_bpid"]),
            ("tile_group_ptr", groups["tile_group_ptr"]),
            ("parent_ptr", case["parent_ptr"]),
            ("members", case["members"]),
            ("ids", case["ids"]),
            ("coefficients", case["coefficients"]),
        )
    }
    if transposed:
        geometry = (
            0,
            case["n_rows"],
            0,
            case["n_clique"],
            0,
            case["stride"],
            0,
            case["n_src_rows"],
            0,
            case["n_rows"],
            0,
            case["n_clique"],
            case["stride"],
        )
    else:
        geometry = (
            0,
            case["n_clique"],
            0,
            case["n_rows"],
            0,
            case["n_clique"],
            0,
            case["n_src_rows"],
            0,
            case["n_clique"],
            0,
            case["n_rows"],
            case["stride"],
        )
    start = case["start"]
    columns = int((groups["words"] & np.uint64(0xFFFFFFFF)).max()) + 1
    for bucket in range(len(start) - 1):
        count = int(start[bucket + 1] - start[bucket])
        if count == 0:
            continue
        scratch = {}
        if panel:
            scratch = dict(
                panel=cupy.zeros(
                    count * columns * kernel_launch.clique_panel_row(bucket)
                ),
                nonfinite=cupy.zeros(1, dtype=cupy.int32),
            )
        kernel_launch.launch_cross_clique(
            geometry,
            address(device["x"]),
            address(device["out"]),
            address(device["words"]),
            address(device["group_ptr"]),
            address(device["group_bpid"]),
            address(device["tile_group_ptr"]),
            address(device["parent_ptr"]),
            address(device["members"]),
            address(device["ids"]) + 4 * int(start[bucket]),
            address(device["coefficients"]),
            count,
            groups["ntiles"],
            tile_rows,
            CLIQUE_ORBITALS,
            bucket,
            edges_per_batch,
            transposed,
            0,
            mma=mma,
            distinct_rows=distinct_rows,
            panel=address(scratch.get("panel")),
            nonfinite=address(scratch.get("nonfinite")),
            panel_columns=columns if panel else 0,
        )
    cupy.cuda.Device().synchronize()
    return cupy.asnumpy(device["out"])


@pytest.mark.parametrize("mma", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("edges_per_batch", [4, 8])
@pytest.mark.parametrize("tile_rows", [4, 8, 64])
def test_cross_clique_kernel_matches_the_cross_edge_formula(
    transposed, edges_per_batch, tile_rows, mma
):
    case = clique_case(np.random.default_rng(91), transposed, tile_rows)
    assert (case["expected"] != case["out0"]).sum() > case["out0"].size // 4
    scale = np.abs(case["expected"]).max()
    for _ in range(2):
        out = run_clique_case(case, transposed, tile_rows, edges_per_batch, mma)
        np.testing.assert_allclose(
            out, case["expected"], rtol=1e-13, atol=1e-13 * scale
        )


@pytest.mark.parametrize("mma", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("nan", ["hidden", "visible"])
def test_cross_clique_kernel_keeps_a_hidden_nan_out_of_the_output(transposed, nan, mma):
    case = clique_case(np.random.default_rng(92), transposed, 8, nan=nan)
    out = run_clique_case(case, transposed, 8, 4, mma)
    expected = case["expected"]
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    assert np.isnan(expected).any() == (nan == "visible")
    finite = ~np.isnan(expected)
    np.testing.assert_allclose(
        out[finite],
        expected[finite],
        rtol=1e-13,
        atol=1e-13 * np.abs(expected[finite]).max(),
    )


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("edges_per_batch", [4, 8])
@pytest.mark.parametrize("tile_rows", [4, 8, 64])
def test_cross_clique_mma_kernel_with_distinct_rows_matches_the_cross_edge_formula(
    transposed, edges_per_batch, tile_rows
):
    case = clique_case(np.random.default_rng(93), transposed, tile_rows, distinct=True)
    assert case["groups"]["distinct_rows"]
    assert (case["expected"] != case["out0"]).sum() > case["out0"].size // 8
    scale = np.abs(case["expected"]).max()
    out = run_clique_case(
        case, transposed, tile_rows, edges_per_batch, mma=True, distinct_rows=True
    )
    np.testing.assert_allclose(out, case["expected"], rtol=1e-13, atol=1e-13 * scale)


@pytest.mark.parametrize("edges_per_batch", [4, 8])
@pytest.mark.parametrize("tile_rows", [4, 8, 64])
def test_cross_clique_panel_kernel_matches_the_cross_edge_formula(
    edges_per_batch, tile_rows
):
    case = clique_case(np.random.default_rng(94), False, tile_rows, distinct=True)
    assert case["groups"]["distinct_rows"]
    scale = np.abs(case["expected"]).max()
    for _ in range(2):
        out = run_clique_case(
            case,
            False,
            tile_rows,
            edges_per_batch,
            mma=True,
            distinct_rows=True,
            panel=True,
        )
        np.testing.assert_allclose(
            out, case["expected"], rtol=1e-13, atol=1e-13 * scale
        )


@pytest.mark.parametrize("nan", ["hidden", "visible"])
def test_cross_clique_panel_kernel_falls_back_on_a_non_finite_source(nan):
    case = clique_case(np.random.default_rng(95), False, 8, nan=nan, distinct=True)
    out = run_clique_case(
        case,
        False,
        8,
        4,
        mma=True,
        distinct_rows=case["groups"]["distinct_rows"],
        panel=case["groups"]["distinct_rows"],
    )
    expected = case["expected"]
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    assert np.isnan(expected).any() == (nan == "visible")
    finite = ~np.isnan(expected)
    np.testing.assert_allclose(
        out[finite],
        expected[finite],
        rtol=1e-13,
        atol=1e-13 * np.abs(expected[finite]).max(),
    )


def run_lbg_case(case, tile_rows, per_launch=None):
    from fulqrum.gpu.distributed import aabb_lbg

    shared_limit = min(
        int(cupy.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]), 227 * 1024
    )
    kernel_launch.prepare_cross_clique(shared_limit)
    groups = case["groups"]
    words, goff, gpair = aabb_lbg.retile(
        groups["words"],
        groups["group_ptr"],
        groups["group_bpid"],
        groups["tile_group_ptr"],
        tile_rows,
    )
    begin, end, pair = aabb_lbg.segments(goff, gpair)
    device = {
        name: cupy.asarray(value)
        for name, value in (
            ("x", case["x"]),
            ("out", case["out0"]),
            ("words", groups["words"]),
            ("group_ptr", groups["group_ptr"]),
            ("group_bpid", groups["group_bpid"]),
            ("tile_group_ptr", groups["tile_group_ptr"]),
            ("parent_ptr", case["parent_ptr"]),
            ("members", case["members"]),
            ("ids", case["ids"]),
            ("coefficients", case["coefficients"]),
            ("lbg_words", words),
            ("begin", begin),
            ("end", end),
            ("pair", pair),
        )
    }
    flag = cupy.zeros(1, dtype=cupy.int32)
    geometry = (
        0,
        case["n_clique"],
        0,
        case["n_rows"],
        0,
        case["n_clique"],
        0,
        case["n_src_rows"],
        0,
        case["n_clique"],
        0,
        case["n_rows"],
        case["stride"],
    )
    columns = int((words & np.uint64(0xFFFFFFFF)).max()) + 1
    kernel_launch.launch_lbg_finite(
        address(device["x"]), case["n_src_rows"] * case["stride"], address(flag), 0
    )
    start = case["start"]
    for bucket in range(len(start) - 1):
        count = int(start[bucket + 1] - start[bucket])
        if count == 0:
            continue
        row = kernel_launch.clique_panel_row(bucket)
        stride = 8 * ((8 + 4 * bucket + 7) // 8)
        per = count if per_launch is None else per_launch
        scratch = cupy.zeros(per * (columns * row + case["n_rows"] * stride))

        def tables(first, n):
            return (
                address(device["words"]),
                address(device["group_ptr"]),
                address(device["group_bpid"]),
                address(device["tile_group_ptr"]),
                address(device["parent_ptr"]),
                address(device["members"]),
                address(device["ids"]) + 4 * (int(start[bucket]) + first),
                address(device["coefficients"]),
                n,
                groups["ntiles"],
                tile_rows,
                CLIQUE_ORBITALS,
                bucket,
                4,
                True,
                address(scratch),
                address(flag),
                columns,
            )

        for first in range(0, count, per):
            n = min(per, count - first)
            kernel_launch.launch_cross_clique_pack(
                geometry, address(device["x"]), tables(first, n), 0
            )
            kernel_launch.launch_cross_lbg(
                geometry,
                address(device["x"]),
                address(device["out"]),
                tables(first, n),
                address(device["lbg_words"]),
                address(device["begin"]),
                address(device["end"]),
                address(device["pair"]),
                len(begin),
                address(scratch) + 8 * n * columns * row,
                stride,
                address(flag),
                0,
            )
        kernel_launch.launch_cross_clique_gated(
            geometry, address(device["x"]), address(device["out"]), tables(0, count), 0
        )
    cupy.cuda.Device().synchronize()
    return cupy.asnumpy(device["out"]), int(flag.get()[0])


@pytest.mark.parametrize("per_launch", [None, 1, 2])
@pytest.mark.parametrize("tile_rows", [4, 8, 64])
def test_cross_lbg_matches_the_cross_edge_formula(tile_rows, per_launch):
    case = clique_case(np.random.default_rng(97), False, tile_rows, distinct=True)
    assert case["groups"]["distinct_rows"]
    scale = np.abs(case["expected"]).max()
    for _ in range(2):
        out, flag = run_lbg_case(case, tile_rows, per_launch)
        assert flag == 0
        np.testing.assert_allclose(
            out, case["expected"], rtol=1e-13, atol=1e-13 * scale
        )


@pytest.mark.parametrize("nan", ["hidden", "visible"])
def test_cross_lbg_falls_back_to_rc_on_a_non_finite_source(nan):
    case = clique_case(np.random.default_rng(98), False, 8, nan=nan, distinct=True)
    if not case["groups"]["distinct_rows"]:
        pytest.skip("case without distinct rows")
    out, flag = run_lbg_case(case, 8, 2)
    assert flag == 1
    expected = case["expected"]
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    assert np.isnan(expected).any() == (nan == "visible")
    finite = ~np.isnan(expected)
    np.testing.assert_allclose(
        out[finite],
        expected[finite],
        rtol=1e-13,
        atol=1e-13 * np.abs(expected[finite]).max(),
    )


def test_cross_clique_gated_launch_adds_only_when_the_flag_is_set():
    case = clique_case(np.random.default_rng(99), False, 8, distinct=True)
    shared_limit = min(
        int(cupy.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]), 227 * 1024
    )
    kernel_launch.prepare_cross_clique(shared_limit)
    groups = case["groups"]
    device = {
        name: cupy.asarray(value)
        for name, value in (
            ("x", case["x"]),
            ("words", groups["words"]),
            ("group_ptr", groups["group_ptr"]),
            ("group_bpid", groups["group_bpid"]),
            ("tile_group_ptr", groups["tile_group_ptr"]),
            ("parent_ptr", case["parent_ptr"]),
            ("members", case["members"]),
            ("ids", case["ids"]),
            ("coefficients", case["coefficients"]),
        )
    }
    geometry = (
        0,
        case["n_clique"],
        0,
        case["n_rows"],
        0,
        case["n_clique"],
        0,
        case["n_src_rows"],
        0,
        case["n_clique"],
        0,
        case["n_rows"],
        case["stride"],
    )
    sentinel = cupy.zeros(1)
    start = case["start"]
    for value in (0, 1):
        flag = cupy.full(1, value, dtype=cupy.int32)
        out = cupy.asarray(case["out0"])
        for bucket in range(len(start) - 1):
            count = int(start[bucket + 1] - start[bucket])
            if count == 0:
                continue
            tables = (
                address(device["words"]),
                address(device["group_ptr"]),
                address(device["group_bpid"]),
                address(device["tile_group_ptr"]),
                address(device["parent_ptr"]),
                address(device["members"]),
                address(device["ids"]) + 4 * int(start[bucket]),
                address(device["coefficients"]),
                count,
                groups["ntiles"],
                8,
                CLIQUE_ORBITALS,
                bucket,
                4,
                True,
                address(sentinel),
                address(flag),
                0,
            )
            kernel_launch.launch_cross_clique_gated(
                geometry, address(device["x"]), address(out), tables, 0
            )
        cupy.cuda.Device().synchronize()
        if value == 0:
            np.testing.assert_array_equal(cupy.asnumpy(out), case["out0"])
        else:
            np.testing.assert_allclose(
                cupy.asnumpy(out),
                case["expected"],
                rtol=1e-13,
                atol=1e-13 * np.abs(case["expected"]).max(),
            )


SMALL_FAMILIES = tuple(
    ((p, p + 1), tuple(range(p + 2, p + 2 + m)))
    for p, m in ((0, 2), (4, 3), (9, 4), (15, 5))
)


def small_clique_case(rng, nan=None):
    from fulqrum.gpu.distributed.aabb_parents import orbital_g
    from fulqrum.gpu.distributed.tile_tables import cross_words

    orbitals = 22
    strings = np.array(
        sorted(
            {
                (1 << p) | (1 << q) | (1 << h)
                for (p, q), holes in SMALL_FAMILIES
                for h in holes
            }
        ),
        dtype=np.uint64,
    )
    offsets, lengths, sell, _ = clique_singles(strings, orbitals)
    n_a, n_b, n_src_rows = len(strings), 13, 17
    word_pairs = 9
    counts = rng.integers(1, 5, size=n_b)
    starts = np.concatenate([[0], np.cumsum(counts)])
    tasks = np.array(
        [[b, starts[b], starts[b + 1]] for b in range(n_b)], dtype=np.int32
    )
    n_words = int(starts[-1])
    beta_columns = rng.integers(0, n_src_rows - 1, size=n_words)
    beta_pairs = rng.integers(0, word_pairs - 1, size=n_words)
    beta_words = cross_words(
        beta_columns, beta_pairs, rng.choice([-1, 1], size=n_words)
    )
    g = rng.standard_normal((word_pairs, orbitals * (orbitals - 1) // 2))
    g[:, ::11] = 0.0
    x = rng.standard_normal(n_src_rows * n_a)
    if nan is not None:
        beta_words[0] = cross_words(
            np.array([n_src_rows - 1]), np.array([word_pairs - 1]), np.array([1])
        )[0]
        x.reshape(n_src_rows, n_a)[n_src_rows - 1] = np.nan
        if nan == "hidden":
            g[word_pairs - 1] = 0.0
    return dict(
        strings=strings,
        offsets=offsets,
        lengths=lengths,
        sell=sell,
        tasks=tasks,
        beta_words=beta_words,
        g=g,
        coefficients=orbital_g(g, orbitals),
        x=x,
        out0=rng.standard_normal(n_b * n_a),
        n_a=n_a,
        n_b=n_b,
        n_src_rows=n_src_rows,
        orbitals=orbitals,
    )


def run_small_clique_case(case, form, flag_value=None):
    from fulqrum.gpu.distributed import aabb_lbg

    device = {
        name: cupy.asarray(value)
        for name, value in (
            ("x", case["x"]),
            ("out", case["out0"]),
            ("offsets", case["offsets"].astype(np.int64)),
            ("lengths", case["lengths"]),
            ("sell", case["sell"]),
            ("tasks", case["tasks"]),
            ("beta_words", case["beta_words"]),
            ("g", case["g"]),
            ("coefficients", case["coefficients"]),
        )
    }
    n_a, n_b = case["n_a"], case["n_b"]
    geometry = (0, n_a, 0, n_b, 0, n_a, 0, case["n_src_rows"], 0, n_a, 0, n_b, n_a)
    flag = cupy.full(1, 0 if flag_value is None else flag_value, dtype=cupy.int32)
    if form == "rsl":
        kernel_launch.launch_lbg_finite(
            address(device["x"]), case["n_src_rows"] * n_a, address(flag), 0
        )
        groups = aabb_lbg.small_cliques(
            case["strings"],
            case["offsets"],
            case["lengths"],
            case["sell"],
            None,
            case["orbitals"],
        )
        gorb = case["coefficients"].reshape(-1, case["orbitals"] ** 2)
        keep = []
        for (m, mixed), (bf, bc, mem) in sorted(groups.items()):
            d = [cupy.asarray(bf), cupy.asarray(bc), cupy.asarray(mem)]
            table = (
                cupy.asarray(
                    aabb_lbg.mixed_table(bf, bc, mem, m, gorb, case["orbitals"])
                )
                if mixed
                else None
            )
            panel = cupy.zeros(len(bf) * case["n_src_rows"] * m * 32)
            keep += d + [table, panel]
            kernel_launch.launch_cross_rsl(
                geometry,
                address(device["x"]),
                address(device["out"]),
                address(d[0]),
                address(d[1]),
                address(d[2]),
                address(panel),
                address(device["tasks"]),
                address(device["beta_words"]),
                address(device["coefficients"]),
                address(table),
                address(flag),
                len(bf),
                aabb_lbg.RSL_ROWS_PER_JOB,
                case["n_src_rows"],
                gorb.shape[0],
                case["orbitals"],
                m,
                0,
            )
    gate = address(flag) if form in ("rsl", "gated") else 0
    kernel_launch.launch_cross_direct(
        geometry,
        address(device["x"]),
        address(device["out"]),
        address(device["offsets"]),
        address(device["lengths"]),
        address(device["sell"]),
        address(device["beta_words"]),
        address(device["tasks"]),
        address(device["g"]),
        case["g"].shape[1],
        n_b,
        0,
        gate=gate,
    )
    cupy.cuda.Device().synchronize()
    return cupy.asnumpy(device["out"]), int(flag.get()[0])


def test_small_clique_case_is_complete_cliques_of_two_to_five():
    from fulqrum.gpu.distributed import aabb_lbg

    case = small_clique_case(np.random.default_rng(101))
    groups = aabb_lbg.small_cliques(
        case["strings"],
        case["offsets"],
        case["lengths"],
        case["sell"],
        None,
        case["orbitals"],
    )
    assert sorted({m for m, _ in groups}) == [2, 3, 4, 5]


def test_cross_rsl_matches_the_direct_form():
    case = small_clique_case(np.random.default_rng(102))
    reference, _ = run_small_clique_case(case, "direct")
    assert (reference != case["out0"]).sum() > case["out0"].size // 4
    out, flag = run_small_clique_case(case, "rsl")
    assert flag == 0
    np.testing.assert_allclose(
        out, reference, rtol=1e-13, atol=1e-13 * np.abs(reference).max()
    )


@pytest.mark.parametrize("nan", ["hidden", "visible"])
def test_cross_rsl_falls_back_to_the_direct_form_on_a_non_finite_source(nan):
    case = small_clique_case(np.random.default_rng(103), nan=nan)
    reference, _ = run_small_clique_case(case, "direct")
    out, flag = run_small_clique_case(case, "rsl")
    assert flag == 1
    assert np.isnan(reference).any() == (nan == "visible")
    np.testing.assert_array_equal(np.isnan(out), np.isnan(reference))
    finite = ~np.isnan(reference)
    np.testing.assert_allclose(
        out[finite],
        reference[finite],
        rtol=1e-13,
        atol=1e-13 * np.abs(reference[finite]).max(),
    )


def test_cross_direct_gated_launch_adds_only_when_the_flag_is_set():
    case = small_clique_case(np.random.default_rng(104))
    reference, _ = run_small_clique_case(case, "direct")
    off, _ = run_small_clique_case(case, "gated", 0)
    on, _ = run_small_clique_case(case, "gated", 1)
    np.testing.assert_array_equal(off, case["out0"])
    np.testing.assert_allclose(
        on, reference, rtol=1e-13, atol=1e-13 * np.abs(reference).max()
    )


def test_cross_clique_panel_launch_rejects_a_transposed_source():
    case = clique_case(np.random.default_rng(96), True, 8, distinct=True)
    with pytest.raises(RuntimeError):
        run_clique_case(case, True, 8, 4, mma=True, distinct_rows=True, panel=True)


def test_clique_edge_group_plan_reports_whether_group_rows_are_distinct():
    from fulqrum.gpu.distributed.aabb_parents import clique_edge_group_plan
    from fulqrum.gpu.distributed.tile_tables import cross_words

    columns = np.array([0, 1, 2, 3])
    signs = np.array([1, -1, 1, 1])
    # pair 0 rows 0 and 1, pair 1 rows 0 and 1: every group has distinct rows
    plan = clique_edge_group_plan(
        cross_words(columns, np.array([0, 0, 1, 1]), signs),
        np.array([0, 1, 0, 1]),
        2,
        2,
    )
    assert plan["distinct_rows"]
    # pair 0 twice on row 1
    plan = clique_edge_group_plan(
        cross_words(columns, np.array([0, 0, 0, 1]), signs),
        np.array([0, 1, 1, 0]),
        2,
        2,
    )
    assert not plan["distinct_rows"]
    # the same row in two different row tiles is a different group
    plan = clique_edge_group_plan(
        cross_words(columns[:2], np.array([0, 0]), signs[:2]), np.array([0, 2]), 4, 2
    )
    assert plan["distinct_rows"]
    assert clique_edge_group_plan(np.zeros(0, np.uint64), np.zeros(0, np.int64), 2, 2)[
        "distinct_rows"
    ]


@pytest.mark.cuda_error_path
@pytest.mark.parametrize("arm", ["alpha_singles_pair", "cross_sell"])
def test_launch_errors_name_the_cuda_error_and_empty_tiles_launch_nothing(arm):

    def launch(a_stop):
        geometry = (0, 4, 0, 4, 0, 4, 0, 4, 0, a_stop, 0, 4, 4, 2048)
        if arm == "alpha_singles_pair":
            kernel_launch.launch_alpha_singles_pair(
                geometry, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0
            )
        else:
            kernel_launch.launch_cross_sell(
                geometry, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0
            )

    with pytest.raises(
        RuntimeError, match=r"kernel launch failed: cudaError\w+ \(\d+\)$"
    ):
        launch(4)
    launch(0)
    cupy.cuda.Device().synchronize()


@pytest.mark.parametrize("arm", ["alpha_singles_delta", "alpha_doubles"])
def test_edge_offsets_beyond_int32_address_the_tables(arm):
    from fulqrum.gpu.distributed.tables import _sell_layout

    rng = np.random.default_rng(91)
    n_a, n_b, n_src, orbitals = 40, 9, 25, 6
    offsets, columns, _, _ = random_alpha_singles(rng, n_a, n_b, n_src, orbitals)
    base = 2**32 + 5
    x = cupy.asarray(rng.standard_normal(n_b * n_src))
    alpha_slot = cupy.arange(n_src, dtype=cupy.int32)
    beta_slot = cupy.arange(n_b, dtype=cupy.int32)
    geometry = (0, n_a, 0, n_b, 0, n_src, 0, n_b, 0, n_a, 0, n_b, n_src)
    if arm == "alpha_singles_delta":
        beta_strings = cupy.asarray(
            np.sort(rng.choice(2**orbitals, size=n_b, replace=False)).astype(np.uint64)
        )
        tables = [
            cupy.asarray(columns.astype(np.int32)),
            cupy.asarray(rng.standard_normal(columns.size)),
            cupy.asarray(rng.standard_normal((columns.size, orbitals))),
        ]
        per_edge = [4, 8, 8 * orbitals]
        row_offsets = offsets.astype(np.int64)
    else:
        sell_offsets, sell_lengths, slots = _sell_layout(offsets)
        sell_columns = np.zeros(int(sell_offsets[-1]), dtype=np.int32)
        sell_columns[slots] = columns
        sell_values = np.zeros(sell_columns.size)
        sell_values[slots] = rng.standard_normal(columns.size)
        tables = [cupy.asarray(sell_columns), cupy.asarray(sell_values)]
        per_edge = [4, 8]
        row_offsets = sell_offsets.astype(np.int64)
        lengths = cupy.asarray(sell_lengths.astype(np.int32))

    def run(shift):
        out = cupy.zeros(n_b * n_a)
        off = cupy.asarray(row_offsets + shift)
        moved = [address(table) - shift * size for table, size in zip(tables, per_edge)]
        if arm == "alpha_singles_delta":
            kernel_launch.launch_alpha_singles_delta(
                geometry,
                address(x),
                address(out),
                address(alpha_slot),
                address(beta_slot),
                address(off),
                *moved,
                address(beta_strings),
                orbitals,
                4,
                0,
            )
        else:
            kernel_launch.launch_alpha_doubles(
                geometry,
                address(x),
                address(out),
                address(alpha_slot),
                address(beta_slot),
                address(off),
                address(lengths),
                *moved,
                4,
                0,
                True,
                0,
            )
        cupy.cuda.Device().synchronize()
        return cupy.asnumpy(out)

    reference = run(0)
    assert np.any(reference != 0.0)
    np.testing.assert_array_equal(run(base), reference)
