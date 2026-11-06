# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fulqrum as fq
from fulqrum.core.spmv import FulqrumSpMV
from fulqrum.exceptions import FulqrumError

_DATA = Path(__file__).parent / "data/lih.json"
_ROOT = Path(__file__).resolve().parents[2]


def _lih_operator():
    op = fq.FermionicOperator.from_json(_DATA).extended_jw_transformation()
    op.set_type(2)
    return op


def _half_strings(width, weight):
    strings = [format(value, f"0{width}b") for value in range(2**width)]
    strings = [text for text in strings if text.count("1") == weight]
    rng = np.random.default_rng(3)
    rng.shuffle(strings)
    return strings


def test_half_string_subspace_contract():
    op = _lih_operator()
    half = op.width // 2
    alpha = _half_strings(half, 2)
    beta = _half_strings(half, 1)
    with pytest.raises(ValueError):
        fq.Subspace([alpha + alpha[:3], beta])
    subspace = fq.Subspace([alpha, beta])

    assert subspace.half_strs_mode == 1
    assert subspace.num_alpha_qubits == half
    assert subspace.width == op.width
    assert list(subspace.alpha_strs) == sorted(set(alpha))
    assert list(subspace.beta_strs) == sorted(set(beta))
    assert all(len(text) == half for text in subspace.alpha_strs)
    num_alpha = len(subspace.alpha_strs)
    assert subspace.size() == len(subspace) == num_alpha * len(subspace.beta_strs)

    alpha_arr, beta_arr = subspace.to_alpha_beta_arrays()
    assert alpha_arr.dtype == np.uint64 and beta_arr.dtype == np.uint64
    assert np.all(np.diff(alpha_arr.astype(np.int64)) > 0)
    assert np.all(np.diff(beta_arr.astype(np.int64)) > 0)
    for index in (0, 5, len(subspace) - 1):
        beta_index, alpha_index = divmod(index, num_alpha)
        expected = subspace.beta_strs[beta_index] + subspace.alpha_strs[alpha_index]
        assert subspace.get_n_th_bitstring(index) == expected


def test_full_string_subspace_is_not_half_string():
    subspace = fq.Subspace([["0011", "0101"]])
    assert subspace.half_strs_mode == 0
    assert subspace.alpha_strs is None and subspace.beta_strs is None
    with pytest.raises(FulqrumError):
        subspace.to_alpha_beta_arrays()


def test_spmv_contract_and_diagonal_slice():
    op = _lih_operator()
    half = op.width // 2
    subspace = fq.Subspace([_half_strings(half, 2), _half_strings(half, 2)])
    hsub = fq.SubspaceHamiltonian(op, subspace)

    assert isinstance(hsub.spmv, FulqrumSpMV)
    assert hsub.spmv.subspace is subspace
    assert hsub.off_H.type == 2
    assert hsub.diag_H.type in (1, 2)

    alpha_arr, beta_arr = subspace.to_alpha_beta_arrays()
    sliced = hsub.spmv.diagonal_slice_product_space(
        alpha_arr, beta_arr, 0, len(alpha_arr), subspace.num_alpha_qubits
    )
    full = hsub.diagonal_vector() - hsub.spmv.const_energy
    assert np.allclose(np.asarray(sliced), full, atol=1e-12)


def test_operator_arrays_cythonizes(tmp_path):
    pytest.importorskip("Cython")
    source = _ROOT / "fulqrum/gpu/common/operator_arrays.pyx"
    if not source.exists():
        pytest.skip("GPU sources are not present")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cython",
            "-3",
            "--cplus",
            "-I",
            str(_ROOT),
            "-I",
            np.get_include(),
            str(source),
            "-o",
            str(tmp_path / "operator_arrays.cpp"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_alpha_beta_arrays_refuse_halves_wider_than_64_qubits():
    narrow = "0" * 63 + "1"
    wide = "0" * 64 + "1"
    with pytest.raises(FulqrumError):
        fq.Subspace([[wide], [narrow]]).to_alpha_beta_arrays()
    with pytest.raises(FulqrumError):
        fq.Subspace([[narrow], [wide]]).to_alpha_beta_arrays()
    alpha_arr, beta_arr = fq.Subspace([[narrow], [narrow]]).to_alpha_beta_arrays()
    assert alpha_arr[0] == 1 and beta_arr[0] == 1


def test_gpu_term_arrays_refuse_widths_above_256():
    operator_arrays = pytest.importorskip("fulqrum.gpu.common.operator_arrays")
    op = fq.QubitOperator(300, [("XX", [0, 299], 1.0)])
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([["0" * 300, "1" + "0" * 298 + "1"]]))
    with pytest.raises(ValueError):
        operator_arrays.to_gpu_arrays(hsub.spmv)


def test_tile_diagonal_refuses_a_missing_diagonal_method():
    module = pytest.importorskip("fulqrum.gpu.distributed.dist_matvec_tile_resident")
    hamiltonian = module.TileGPUResidentHamiltonian.__new__(
        module.TileGPUResidentHamiltonian
    )
    hamiltonian._has_diag = 1
    hamiltonian._n_local = 4
    hamiltonian._const_energy = 0.0
    with pytest.raises(RuntimeError):
        hamiltonian._host_diag_local(object(), 2)


def test_num_groups_keeps_the_ladder_order():
    op = _lih_operator()
    _, off = op.split_diagonal()
    unsorted_count = off.num_groups
    off.group_sort()
    assert off.num_groups == unsorted_count
    off.group_term_sort_by_ladder_int(2)
    order = [off[k].coeff for k in range(off.size())]
    assert off.num_groups == unsorted_count
    assert [off[k].coeff for k in range(off.size())] == order


def test_ladder_bins_are_checked_before_use():
    op = _lih_operator()
    half = op.width // 2
    subspace = fq.Subspace([_half_strings(half, 2), _half_strings(half, 2)])
    hsub = fq.SubspaceHamiltonian(op, subspace)
    hsub.spmv.check_ladder_bins()
    stale = FulqrumSpMV(
        hsub.diag_H,
        hsub.const_energy,
        hsub.off_H,
        subspace,
        hsub.group_ptrs,
        np.zeros(1, dtype=np.uintp),
    )
    with pytest.raises(ValueError, match="ladder bins"):
        stale.check_ladder_bins()
    hsub.spmv = stale
    operator_arrays = pytest.importorskip("fulqrum.gpu.common.operator_arrays")
    with pytest.raises(ValueError, match="ladder bins"):
        operator_arrays.to_gpu_arrays(stale, skip_diag=True)


def test_gpu_preparation_stays_map_free():
    from fulqrum.gpu.common.operator_arrays import to_gpu_arrays
    from fulqrum.gpu.distributed.operator_check import unsupported_groups
    from fulqrum.gpu.distributed.tile_tables import (
        cross_sector_table,
        cross_table,
        sector_tiles,
        tile_input_error,
    )

    op = _lih_operator()
    half = op.width // 2
    subspace = fq.Subspace([_half_strings(half, 2), _half_strings(half, 2)])
    hsub = fq.SubspaceHamiltonian(op, subspace)
    alpha, beta = subspace.to_alpha_beta_arrays()
    assert to_gpu_arrays(hsub.spmv, skip_diag=True)["diag_vec"] is None
    assert unsupported_groups(hsub.spmv, half) == []
    assert tile_input_error(hsub.spmv, half, alpha, beta) is None
    for sector in (0, 1):
        count = len(alpha) if sector == 0 else len(beta)
        sector_tiles(hsub.spmv, sector, 1, count - 1, [0, count], threads=1)
        cross_table(hsub.spmv, sector, 0, count, threads=1)
    cross_sector_table(hsub.spmv)
    hsub.spmv.diagonal_slice_product_space(alpha, beta, 1, len(alpha) - 1, half)
    hsub.spmv.diag_separable_pieces(alpha, beta, 1, len(alpha) - 1, half)
    assert not hsub.spmv._cpu_diagnostics()["map_materialized"]


@pytest.mark.parametrize("constant", [0.0, 3.25])
def test_local_diagonal_and_separable_pieces(constant):
    op = fq.QubitOperator(
        8,
        [
            ("Z", [0], 0.7),
            ("ZZ", [1, 6], 0.3),
            ("11", [2, 7], -0.9),
        ],
    )
    if constant:
        op += fq.QubitOperator(8, [("", [], constant)])
    op.set_type(2)
    strings = _half_strings(4, 2)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([strings, strings]))
    alpha, beta = hsub.spmv.subspace.to_alpha_beta_arrays()
    a0, a1 = 1, len(alpha) - 1
    local = hsub.spmv.diagonal_slice_product_space(alpha, beta, a0, a1, 4)
    da, db, coupling, ok = hsub.spmv.diag_separable_pieces(alpha, beta, a0, a1, 4)
    assert ok
    za = 1.0 - 2.0 * (
        (alpha[a0:a1, None] >> np.arange(4, dtype=np.uint64)) & np.uint64(1)
    )
    zb = 1.0 - 2.0 * ((beta[:, None] >> np.arange(4, dtype=np.uint64)) & np.uint64(1))
    np.testing.assert_allclose(
        local, (db[:, None] + da[None, :] + zb @ coupling.T @ za.T).ravel(), atol=1e-13
    )
    expected = (
        hsub.diagonal_vector().reshape(len(beta), len(alpha))[:, a0:a1].ravel()
        - constant
    )
    np.testing.assert_allclose(local, expected, atol=1e-13)
    assert hsub.spmv.diagonal_slice_product_space(alpha, beta, a0, a0, 4).size == 0
    with pytest.raises(ValueError):
        hsub.spmv.diagonal_slice_product_space(alpha, beta, a1, a0, 4)
    with pytest.raises(ValueError):
        hsub.spmv.diag_separable_pieces(alpha, beta, 0, len(alpha) + 1, 4)


def test_separable_diagonal_refuses_multiple_other_orbitals():
    op = fq.QubitOperator(8, [("ZZZ", [0, 4, 5], 0.5)])
    op.set_type(2)
    strings = _half_strings(4, 2)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([strings, strings]))
    alpha, beta = hsub.spmv.subspace.to_alpha_beta_arrays()
    assert not hsub.spmv.diag_separable_pieces(alpha, beta, 0, len(alpha), 4)[3]


def test_metadata_export_does_not_allocate_product_diagonal(monkeypatch):
    from fulqrum.gpu.common.operator_arrays import to_gpu_arrays

    strings = [format(value, "016b") for value in range(2000)]
    op = fq.QubitOperator(32, [("Z", [0], 0.5)])
    op.set_type(2)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([strings, strings]))
    original_zeros = np.zeros

    def reject_full_diagonal(shape, *args, **kwargs):
        count = int(np.prod(shape)) if isinstance(shape, tuple) else int(shape)
        if count >= len(hsub.spmv.subspace):
            raise MemoryError("injected full diagonal allocation failure")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", reject_full_diagonal)
    assert to_gpu_arrays(hsub.spmv, skip_diag=True)["diag_vec"] is None
    with pytest.raises(MemoryError):
        to_gpu_arrays(hsub.spmv)


def test_full_string_host_inputs_stay_map_free():
    from fulqrum.gpu.full_strs.host import HostOperator

    op = _lih_operator()
    half = op.width // 2
    strings = sorted(_half_strings(half, 2))[:4]
    subspace = fq.Subspace([[b + a for b in strings for a in strings]])
    labels = [subspace.get_n_th_bitstring(row) for row in range(len(subspace))]
    keys = np.array(
        [(int(label[half:], 2), int(label[:half], 2)) for label in labels],
        dtype=np.uint64,
    )
    host = HostOperator(op, keys)
    np.testing.assert_array_equal(host.sorted_keys(), keys)
    assert np.isfinite(host.matvec(np.ones(host.num_rows))).all()
    assert subspace._storage_state()["map_build_count"] == 0


@pytest.mark.parametrize(
    "sector,begin,end,bounds",
    [
        (2, 0, 1, [0, 2]),
        (0, 3, 2, [0, 2]),
        (0, 0, 100, [0, 2]),
        (0, 0, 1, []),
        (0, 0, 1, [2, 1]),
        (0, 0, 1, [0, 100]),
    ],
)
def test_invalid_tile_ranges_raise_before_parallel_entry(sector, begin, end, bounds):
    from fulqrum.gpu.distributed.tile_tables import sector_tiles

    op = _lih_operator()
    strings = _half_strings(op.width // 2, 2)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([strings, strings]))
    with pytest.raises(ValueError):
        sector_tiles(hsub.spmv, sector, begin, end, bounds)


def test_stale_ladder_bins_with_reordered_boundaries_raise():
    op = _lih_operator()
    strings = _half_strings(op.width // 2, 2)
    subspace = fq.Subspace([strings, strings])
    hsub = fq.SubspaceHamiltonian(op, subspace)
    bins = hsub.group_ladder_ptrs.copy()
    bins[1] = bins[-1]
    bins[2] = 0
    stale = FulqrumSpMV(
        hsub.diag_H, hsub.const_energy, hsub.off_H, subspace, hsub.group_ptrs, bins
    )
    with pytest.raises(ValueError, match="ladder bins"):
        stale.check_ladder_bins()
