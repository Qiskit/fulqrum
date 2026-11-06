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

from fulqrum.gpu.distributed.tables import (
    ArmCacheLengths,
    SourceTables,
    TableOptions,
    TileTables,
    _alpha2_apid_asign,
    _alpha2_sell,
    _sell_layout,
    alpha_tables,
    arm_cache_sizes,
    array_bytes,
    beta_tables,
    hbm_fit_error,
    pair_table,
    values_on_edges,
)

LAUNCH_NAMES = {
    "d_alpha2_ptr",
    "d_alpha2_j",
    "d_alpha2_p0",
    "d_alpha2_p1",
    "d_alpha2_apid",
    "d_alpha2_asign",
    "d_alpha2_sell_off",
    "d_alpha2_sell_len",
    "d_alpha2_sell_j",
    "d_alpha2_sell_apid",
    "d_alpha2_sell_asign",
    "d_alpha2_sell_pack",
    "d_alpha4_ptr",
    "d_alpha4_j",
    "d_alpha4_p0",
    "d_alpha4_p1",
    "d_alpha4_p2",
    "d_alpha4_p3",
    "d_alpha4_sell_off",
    "d_alpha4_sell_len",
    "d_alpha4_sell_j",
    "d_alpha4_sell_p0",
    "d_alpha4_sell_p1",
    "d_alpha4_sell_p2",
    "d_alpha4_sell_p3",
    "d_beta2_ptr",
    "d_beta2_j",
    "d_beta2_p0",
    "d_beta2_p1",
    "d_beta4_ptr",
    "d_beta4_j",
    "d_beta4_p0",
    "d_beta4_p1",
    "d_beta4_p2",
    "d_beta4_p3",
}


def _singles():
    ptr = np.array([0, 3, 7, 10, 13, 16])
    j = np.array([1, 2, 3, 0, 2, 3, 4, 0, 1, 4, 0, 1, 4, 1, 2, 3])
    p0 = np.array([1, 0, 1, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0, 0, 1, 0], dtype=np.uint8)
    p1 = np.array([2, 2, 3, 2, 1, 3, 3, 2, 1, 3, 3, 3, 2, 3, 3, 2], dtype=np.uint8)
    return ptr, j, p0, p1, p0, p1


def _doubles(rows=5):
    ptr = np.array([0] + [1] * rows)
    position = np.array([0], dtype=np.uint8)
    return ptr, np.array([4]), position, position + 1, position + 2, position + 3


ROWS = np.array([0b0011, 0b0101, 0b0110, 0b1001, 0b1100], dtype=np.uint64)


def _options(factored=True):
    return TableOptions(4, factored)


def test_pair_ids_and_signs_match_the_documented_example():
    ptr, _, p0, p1, _, _ = _singles()
    apid, asign = _alpha2_apid_asign(ptr, p0, p1, ROWS, 4)
    assert apid.dtype == np.int32 and asign.dtype == np.int8
    np.testing.assert_array_equal(apid[:3], [3, 1, 4])
    np.testing.assert_array_equal(asign[:3], [1, -1, 1])


def test_sell_layout_puts_edge_k_of_row_a_at_its_slot():
    ptr = np.array([0, 2, 2, 5])
    off, row_len, dst = _sell_layout(ptr)
    np.testing.assert_array_equal(off, [0, 96])
    np.testing.assert_array_equal(row_len, [2, 0, 3])
    np.testing.assert_array_equal(dst, [0, 32, 2, 34, 66])


def test_alpha2_sell_pack_holds_column_pair_id_and_sign():
    ptr, j, p0, p1, _, _ = _singles()
    apid, asign = _alpha2_apid_asign(ptr, p0, p1, ROWS, 4)
    off, length, sj, sapid, sasign, pack = _alpha2_sell(ptr, j, apid, asign)
    _, _, dst = _sell_layout(ptr)
    np.testing.assert_array_equal(pack[dst] & 0xFFFFFFFF, j)
    np.testing.assert_array_equal((pack[dst] >> 32) & 0xFFFF, apid)
    np.testing.assert_array_equal(
        ((pack[dst] >> 48) & 0xFFFF).astype(np.uint16).view(np.int16), asign
    )
    assert int(off[-1]) == pack.size


@pytest.mark.parametrize("factored", [True, False])
def test_launch_view_has_every_launcher_name(factored):
    tables = SourceTables(
        alpha_tables(_singles(), _doubles(), ROWS, _options(factored)),
        beta_tables(_singles(), _doubles()),
    )
    view = tables.launch_view
    assert set(view) == LAUNCH_NAMES
    assert (view["d_alpha2_apid"] is not None) == factored
    assert (view["d_alpha2_sell_pack"] is not None) == factored
    assert view["d_alpha4_sell_j"] is not None
    assert (
        view["d_alpha2_ptr"].dtype == np.int64 and view["d_alpha2_p0"].dtype == np.uint8
    )
    offsets = (
        "d_alpha2_ptr",
        "d_alpha4_ptr",
        "d_beta2_ptr",
        "d_beta4_ptr",
        "d_alpha2_sell_off",
        "d_alpha4_sell_off",
    )
    for name in offsets:
        if view[name] is not None:
            assert view[name].dtype == np.int64, name
    for name in ("d_alpha2_j", "d_beta2_j", "d_alpha4_sell_j", "d_alpha4_sell_len"):
        assert view[name].dtype == np.int32, name


def test_arm_cache_lengths_skip_caches_the_kernel_does_not_read():
    tables = SourceTables(
        alpha_tables(_singles(), _doubles(), ROWS, _options()),
        beta_tables(_singles(), _doubles()),
    )
    full = tables.arm_cache_lengths(True, True, False, True)
    assert full == ArmCacheLengths(1, 1, 16, 16)
    assert tables.arm_cache_lengths(True, False, False, True) == ArmCacheLengths(
        0, 1, 0, 16
    )
    assert tables.arm_cache_lengths(False, True, False, True) == ArmCacheLengths(
        1, 0, 16, 0
    )
    assert tables.arm_cache_lengths(False, False, False, False) == full
    assert tables.arm_cache_lengths(True, True, True, True).aaaa == 32


def test_arm_cache_sizes_hold_one_delta_per_edge_and_other_sector_orbital():
    lengths = ArmCacheLengths(3, 5, 7, 11)
    sizes = arm_cache_sizes(lengths, 4, 6)
    assert sizes == {
        "aaaa": 3,
        "bbbb": 5,
        "aa_c0": 7,
        "aa_cpb": 42,
        "bb_c0": 11,
        "bb_cpb": 44,
    }
    assert arm_cache_sizes(ArmCacheLengths(0, 0, 0, 0), 4, 6)["aa_cpb"] == 0


def test_tile_bytes_split_into_connection_and_aabb_only_tables():
    alpha = alpha_tables(_singles(), _doubles(), ROWS, _options())
    beta = beta_tables(_singles(), _doubles())
    tile = TileTables(
        {0: alpha},
        {0: beta},
        {0: SourceTables(alpha, beta), 1: SourceTables(alpha, beta)},
    )
    cross = array_bytes((alpha.apid, alpha.asign, alpha.singles_sell))
    assert tile.cross_sector_bytes() == cross > 0
    assert tile.connection_bytes() + cross == array_bytes(alpha) + array_bytes(beta)


def test_hbm_fit_error_reports_only_when_the_prediction_exceeds_the_limit():
    assert hbm_fit_error({"a": 40, "b": 45}, 100, 200, 0.85) is None
    message = hbm_fit_error({"a": 40, "b": 46}, 100, 200, 0.85)
    assert "predicted=86 B" in message and "'b': 46" in message


def test_values_on_edges_follow_the_gpu_edge_order():
    gpu_ptr, gpu_columns = np.array([0, 3, 4]), np.array([2, 0, 1, 1])
    host_offsets, host_columns, host_values = (
        np.array([0, 2, 3]),
        np.array([0, 2, 1]),
        np.array([5.0, 7.0, 9.0]),
    )
    values = values_on_edges(
        gpu_ptr, gpu_columns, host_offsets, host_columns, host_values
    )
    np.testing.assert_array_equal(values, [7.0, 5.0, 0.0, 9.0])


def test_values_on_edges_refuses_a_host_edge_without_a_gpu_edge():
    with pytest.raises(ValueError, match="no GPU connection"):
        values_on_edges(
            np.array([0, 1]),
            np.array([0]),
            np.array([0, 2]),
            np.array([0, 1]),
            np.array([1.0, 2.0]),
        )


def test_pair_table_maps_core_pairs_to_gpu_pair_ids():
    orbitals = 4
    core = [(p, q) for q in range(orbitals) for p in range(q)]
    factors = np.repeat(np.arange(len(core), dtype=float), 2)[None, :] * np.array(
        [[1.0], [2.0]]
    )
    table = pair_table(factors, orbitals, 1e-12)
    for index, (p, q) in enumerate(core):
        gpu = p * (2 * orbitals - p - 1) // 2 + (q - p - 1)
        np.testing.assert_array_equal(table[:, gpu], [index, 2 * index])


def test_pair_table_refuses_direction_dependent_factors():
    factors = np.array([[1.0, 1.0 + 1e-6]])
    with pytest.raises(ValueError, match="direction"):
        pair_table(factors, 2, 1e-12)
    np.testing.assert_array_equal(
        pair_table(np.array([[1.0, 1.0 + 1e-15]]), 2, 1e-12), [[1.0]]
    )


def test_alpha2_sell_pack_words_equal_the_kernel_format():
    from fulqrum.gpu.distributed.tile_tables import SELL_SLICE_ROWS, cross_words

    ptr, j, p0, p1, _, _ = _singles()
    apid, asign = _alpha2_apid_asign(ptr, p0, p1, ROWS, 4)
    _, _, sj, sapid, sasign, pack = _alpha2_sell(ptr, j, apid, asign)
    _, _, slots = _sell_layout(ptr)
    assert SELL_SLICE_ROWS == 32
    np.testing.assert_array_equal(
        pack[slots], cross_words(sj[slots], sapid[slots], sasign[slots])
    )


def test_csr_row_pointers_keep_values_beyond_int32():
    from fulqrum.gpu.distributed.tables import _csr

    ptr = np.array([0, 2**31 - 1, 2**31 + 7, 2**33], dtype=np.int64)
    table = _csr((ptr, np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.uint8)))
    assert table.ptr.dtype == np.int64
    np.testing.assert_array_equal(table.ptr, ptr)


def test_connection_builders_give_int64_row_pointers():
    tile_build = pytest.importorskip("fulqrum.gpu.distributed.tile_build")
    from fulqrum.gpu.distributed.tile_discovery import (
        _build_tile_flip_csr_hybrid,
        _empty_csr_table,
    )

    bra = np.array([0b0011, 0b0101, 0b0110], dtype=np.uint64)
    src = np.array([0b0011, 0b1001, 0b1010], dtype=np.uint64)
    dense = tile_build.build_tile_flip_csr_dense(bra, src, 2, 1)
    masks = tile_build.build_tile_flip_csr_masks(
        bra,
        src,
        np.array([0b0110], dtype=np.uint64),
        np.zeros((1, 4), dtype=np.uint8),
        1,
    )
    hybrid = _build_tile_flip_csr_hybrid(
        bra, src, [(np.uint64(0b0110), (np.uint8(1), np.uint8(2)))], 2
    )
    for name, result in (
        ("dense", dense),
        ("masks", masks),
        ("hybrid", hybrid),
        ("empty", _empty_csr_table(3)),
    ):
        assert result[0].dtype == np.int64, name
    assert int(dense[0][-1]) > 0


def test_clique_offsets_that_do_not_fit_int32_raise():
    from fulqrum.gpu.distributed.aabb_parents import _int32

    np.testing.assert_array_equal(
        _int32([0, 5, 2**31 - 1], "group offsets"), [0, 5, 2**31 - 1]
    )
    with pytest.raises(OverflowError, match="group offsets do not fit int32"):
        _int32([0, 2**31], "group offsets")


def test_native_mask_positions_are_validated_before_scanning():
    native = pytest.importorskip("fulqrum.gpu.distributed.tile_build")
    strings = np.array([1], dtype=np.uint64)
    masks = np.array([3], dtype=np.uint64)
    positions = np.array([[0, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="Mask positions"):
        native.build_tile_flip_csr_masks(strings, strings, masks, positions, 1)
