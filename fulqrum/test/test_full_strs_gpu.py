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

"""GPU kernels of the full-string path against the host reference."""

import importlib

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def full_strs_gpu_available():
    """Return True when the full-string kernel launch module imports and cupy sees a device."""
    try:
        importlib.import_module("fulqrum.gpu.full_strs.kernel_launch")
        import cupy  # noqa: PLC0415

        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001
        return False


requires_gpu = pytest.mark.skipif(
    not full_strs_gpu_available(),
    reason="needs the full-string GPU build and a CUDA device",
)


@requires_gpu
def test_diagonal_launch_is_exact():
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.full_strs import kernel_launch  # noqa: PLC0415

    rng = np.random.default_rng(1)
    rows = 1_000_003
    x = rng.normal(size=rows)
    diagonal = rng.normal(size=rows)
    d_x = cupy.asarray(x)
    d_diagonal = cupy.asarray(diagonal)
    d_y = cupy.full(rows, np.nan)
    cupy.cuda.runtime.deviceSynchronize()
    kernel_launch.launch_diagonal(d_x.data.ptr, d_y.data.ptr, d_diagonal.data.ptr, rows)
    cupy.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cupy.asnumpy(d_y), diagonal * x)


@requires_gpu
def test_empty_diagonal_launch_is_a_no_op():
    from fulqrum.gpu.full_strs import kernel_launch  # noqa: PLC0415

    kernel_launch.launch_diagonal(0, 0, 0, 0)


@requires_gpu
def test_device_layout_round_trip():
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.full_strs.device import TABLE_ARRAYS, DeviceLayout  # noqa: PLC0415
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415
    from fulqrum.test.test_full_strs_coefficients import random_operator  # noqa: PLC0415
    from fulqrum.test.test_full_strs_matvec import clustered_keys  # noqa: PLC0415

    rng = np.random.default_rng(21)
    host = HostOperator(
        random_operator(10, 22), clustered_keys(rng, 10, 5, 4, 3000, 60), num_slices=3
    )
    layout = DeviceLayout(host)
    stream = cupy.cuda.Stream(non_blocking=True)
    layout.wait(stream)
    stream.synchronize()
    assert np.array_equal(cupy.asnumpy(layout.keys), host.key_words())
    assert np.array_equal(cupy.asnumpy(layout.diagonal), host.diagonal_view())
    tables = host.table_views()
    for name in TABLE_ARRAYS:
        assert np.array_equal(cupy.asnumpy(layout.tables[name]), tables[name])
    for chunk in range(host.num_chunks):
        for cls in range(5):
            members, sizes, bucket_start = host.chunk_views(chunk, cls)
            d_members, d_sizes, d_offsets = layout.chunks[chunk].arrays[cls]
            assert np.array_equal(cupy.asnumpy(d_members[: members.size]), members)
            assert np.array_equal(cupy.asnumpy(d_sizes[: sizes.size]), sizes)
            assert np.array_equal(
                cupy.asnumpy(d_offsets),
                np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)]),
            )
            assert layout.chunks[chunk].classes[cls].bucket_start == bucket_start
    assert layout.device_bytes() > 0


def small_host(
    seed,
    alpha_kind=None,
    beta_kind=None,
    num_slices=3,
    norb=10,
    num_alpha=5,
    num_beta=4,
    rows=3000,
):
    """Return a host build with chunks of a random operator on clustered random rows."""
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415
    from fulqrum.test.test_full_strs_coefficients import random_operator  # noqa: PLC0415
    from fulqrum.test.test_full_strs_matvec import clustered_keys  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    keys = clustered_keys(rng, norb, num_alpha, num_beta, rows, 60)
    return HostOperator(
        random_operator(norb, seed + 1),
        keys,
        alpha_kind,
        beta_kind,
        num_slices=num_slices,
    )


def dense_host(norb, alpha_kind=None, beta_kind=None):
    """Return a host build with large groups: all 4-electron alpha strings on norb orbitals, 3 beta lines."""
    import itertools  # noqa: PLC0415

    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415
    from fulqrum.test.test_full_strs_coefficients import random_operator  # noqa: PLC0415

    alphas = [
        sum(1 << k for k in combo) for combo in itertools.combinations(range(norb), 4)
    ]
    betas = [0b11, 0b101, 0b110]
    keys = np.array([(a, b) for a in alphas for b in betas], dtype=np.uint64)
    return HostOperator(
        random_operator(norb, 41), keys, alpha_kind, beta_kind, num_slices=2
    )


def check_partials(host, binned):
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.full_strs.device import DeviceLayout  # noqa: PLC0415

    layout = DeviceLayout(host, "resident")
    stream = cupy.cuda.Stream(non_blocking=True)
    layout.wait(stream)
    x = np.random.default_rng(32).normal(size=host.num_rows)
    d_x = cupy.asarray(x)
    cupy.cuda.runtime.deviceSynchronize()
    buckets = set()
    for chunk in range(host.num_chunks):
        for cls in range(5):
            expected = host.chunk_partials(x, chunk, cls)
            if expected.size == 0:
                continue
            bucket_start = host.chunk_views(chunk, cls)[2]
            buckets.update(b for b in range(7) if bucket_start[b + 1] > bucket_start[b])
            partials = layout.class_partials(d_x, chunk, cls, stream, binned)
            stream.synchronize()
            first = cupy.asnumpy(partials)
            partials = layout.class_partials(d_x, chunk, cls, stream, binned)
            stream.synchronize()
            second = cupy.asnumpy(partials)
            assert np.array_equal(first, second)
            scale = max(np.max(np.abs(expected)), 1e-300)
            assert np.max(np.abs(first - expected)) / scale <= 1e-14, (chunk, cls)
    return buckets


KIND_PAIRS = [
    ("intersection", "intersection"),
    ("intersection", "union"),
    ("union", "intersection"),
    ("union", "union"),
]


@requires_gpu
@pytest.mark.parametrize("binned", [False, True])
@pytest.mark.parametrize("alpha_kind, beta_kind", KIND_PAIRS)
@pytest.mark.parametrize("shape", [(10, 5, 4), (10, 7, 3), (9, 2, 6)])
def test_group_partials_equal_the_cpu(alpha_kind, beta_kind, shape, binned):
    norb, num_alpha, num_beta = shape
    check_partials(
        small_host(31, alpha_kind, beta_kind, 2, norb, num_alpha, num_beta), binned
    )


@requires_gpu
@pytest.mark.parametrize("binned", [False, True])
@pytest.mark.parametrize("alpha_kind, beta_kind", KIND_PAIRS)
@pytest.mark.parametrize("norb, bucket", [(18, 5), (20, 6)])
def test_group_partials_of_large_groups_equal_the_cpu(
    norb, bucket, alpha_kind, beta_kind, binned
):
    buckets = check_partials(dense_host(norb, alpha_kind, beta_kind), binned)
    if alpha_kind == "intersection":
        assert bucket in buckets


def gpu_matvec(host, x, binned, layout=None, residency="resident"):
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.full_strs.device import DeviceLayout  # noqa: PLC0415

    layout = layout or DeviceLayout(host, residency)
    stream = cupy.cuda.Stream(non_blocking=True)
    layout.wait(stream)
    d_x = cupy.empty(host.num_rows, dtype=np.float64)
    d_x.set(x, stream=stream)
    d_y = cupy.full(host.num_rows, np.nan)
    cupy.cuda.runtime.deviceSynchronize()
    layout.matvec(d_x, d_y, stream, binned)
    stream.synchronize()
    return cupy.asnumpy(d_y)


@requires_gpu
@pytest.mark.parametrize("binned", [False, True])
@pytest.mark.parametrize("alpha_kind, beta_kind", KIND_PAIRS)
@pytest.mark.parametrize("num_slices", [1, 3])
def test_gpu_matvec_equals_the_cpu(alpha_kind, beta_kind, num_slices, binned):
    from fulqrum.gpu.full_strs.device import DeviceLayout  # noqa: PLC0415

    host = small_host(51, alpha_kind, beta_kind, num_slices)
    x = np.random.default_rng(52).normal(size=host.num_rows)
    layout = DeviceLayout(host)
    y = gpu_matvec(host, x, binned, layout)
    expected = host.chunk_matvec(x)
    assert np.max(np.abs(y - expected)) / np.max(np.abs(expected)) <= 1e-14
    assert np.array_equal(y, gpu_matvec(host, x, binned, layout))


@requires_gpu
@pytest.mark.parametrize("norb", [18, 20])
def test_gpu_matvec_of_large_groups_equals_the_cpu(norb):
    host = dense_host(norb)
    x = np.random.default_rng(53).normal(size=host.num_rows)
    expected = host.chunk_matvec(x)
    for binned in (False, True):
        y = gpu_matvec(host, x, binned)
        assert np.max(np.abs(y - expected)) / np.max(np.abs(expected)) <= 1e-14


@requires_gpu
def test_hamiltonian_contract_and_eigensolve():
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.eigensolver.primme_distributed import distributed_primme_eigsh  # noqa: PLC0415
    from fulqrum.gpu.full_strs import FullStrsHamiltonian  # noqa: PLC0415
    from fulqrum.test.test_full_strs_coefficients import random_operator  # noqa: PLC0415
    from fulqrum.test.test_full_strs_matvec import clustered_keys, cpu_hamiltonian  # noqa: PLC0415

    rng = np.random.default_rng(61)
    op = random_operator(10, 62)
    keys = clustered_keys(rng, 10, 5, 4, 3000, 60)
    H = FullStrsHamiltonian(op, keys)
    assert H.shape == (H.n_local, H.n_local)
    assert H.supports_device_matvec and H._is_primary
    cpu = cpu_hamiltonian(op, H.keys, 10)
    x = rng.normal(size=H.n_local)
    expected = cpu.matvec(x)
    assert np.max(np.abs(H.matvec(x) - expected)) / np.max(np.abs(expected)) <= 1e-12
    d_y = H.matvec(cupy.asarray(x))
    assert isinstance(d_y, cupy.ndarray)
    np.testing.assert_allclose(
        H.diag_local, np.asarray(cpu.diagonal_vector()).real, rtol=0, atol=1e-12
    )
    evals, _, stats = distributed_primme_eigsh(
        H,
        1,
        tol=1e-8,
        method="PRIMME_GD_plusK",
        ncv=4,
        minRestartSize=2,
        maxPrevRetain=1,
        precond_diag_local=H.diag_local,
        gpu=False,
        return_stats=True,
    )
    dense = cpu.to_csr_linearoperator()
    matrix = dense.matrix if hasattr(dense, "matrix") else dense
    from scipy.sparse.linalg import eigsh  # noqa: PLC0415

    reference = eigsh(matrix, k=1, which="SA", tol=1e-12)[0][0]
    assert abs(evals[0] - reference) <= 1e-8
    assert stats["numMatvecs"] > 0


@requires_gpu
@pytest.mark.parametrize("binned", [False, True])
def test_streamed_and_resident_chunks_give_the_same_bits(binned):
    from fulqrum.gpu.full_strs.device import DeviceLayout  # noqa: PLC0415

    host = small_host(71, num_slices=5)
    x = np.random.default_rng(72).normal(size=host.num_rows)
    resident = DeviceLayout(host, "resident")
    streamed = DeviceLayout(host, "streamed")
    assert resident.num_streamed == 0 and streamed.num_streamed == 5
    y_resident = gpu_matvec(host, x, binned, resident)
    y_streamed = gpu_matvec(host, x, binned, streamed)
    assert np.array_equal(y_resident, y_streamed)
    assert np.array_equal(y_streamed, gpu_matvec(host, x, binned, streamed))
    chunk_bytes = resident.planned_chunk_bytes
    budget = (
        resident.planned_fixed_bytes
        + chunk_bytes[0]
        + chunk_bytes[1]
        + resident.planned_stream_bytes
    )
    mixed = DeviceLayout(host, "auto", device_budget_bytes=budget)
    assert 0 < mixed.num_streamed < 5
    assert np.array_equal(gpu_matvec(host, x, binned, mixed), y_resident)
    expected = host.chunk_matvec(x)
    assert np.max(np.abs(y_resident - expected)) / np.max(np.abs(expected)) <= 1e-14


@requires_gpu
def test_too_small_device_budget_raises():
    from fulqrum.gpu.full_strs.device import DeviceLayout  # noqa: PLC0415

    with pytest.raises(ValueError, match="more key slices"):
        DeviceLayout(small_host(73, num_slices=2), "auto", device_budget_bytes=1000)


class KindsHost:
    """A host build that serves the chunks of a HostOperator as several kinds with given row ranges."""

    def __init__(self, host, kinds):
        """Wrap the host build; inputs: the HostOperator and the kinds (6 row bounds each)."""
        self.host = host
        self.chosen = kinds
        self.num_rows = host.num_rows
        self.num_orbitals = host.num_orbitals
        self.num_chunks = host.num_chunks

    def chunk_kinds(self):
        """Return the kinds."""
        return self.chosen

    def chunk_views(self, chunk, cls, kind):
        """Return the chunk views of the wrapped host for every kind."""
        return self.host.chunk_views(chunk, cls)

    def key_words(self):
        """Return the keys."""
        return self.host.key_words()

    def diagonal_view(self):
        """Return the diagonal."""
        return self.host.diagonal_view()

    def table_views(self):
        """Return the tables."""
        return self.host.table_views()


@requires_gpu
@pytest.mark.parametrize("binned", [False, True])
@pytest.mark.parametrize("seed", [21, 22])
def test_split_output_kinds_give_the_same_bits(seed, binned):
    """Two kinds that split the output rows at mid-range, full sources, give the single-kind bits."""
    host = small_host(seed, num_slices=3)
    n = host.num_rows
    half = n // 2 + 7
    x = np.random.default_rng(seed).normal(size=n)
    kinds = [(0, half, 0, n, 0, half), (half, n, 0, n, half, n)]
    np.testing.assert_array_equal(
        gpu_matvec(KindsHost(host, kinds), x, binned), gpu_matvec(host, x, binned)
    )


@requires_gpu
@pytest.mark.parametrize("binned", [False, True])
@pytest.mark.parametrize("seed", [23, 24])
def test_split_source_kinds_equal_the_cpu(seed, binned):
    """Two kinds that split the source rows (the own and remote shape) equal the CPU matvec to 1e-14."""
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415
    from fulqrum.test.test_full_strs_coefficients import random_operator  # noqa: PLC0415
    from fulqrum.test.test_full_strs_matvec import clustered_keys  # noqa: PLC0415

    host = small_host(seed, num_slices=3)
    keys = clustered_keys(np.random.default_rng(seed), 10, 5, 4, 3000, 60)
    reference = HostOperator(random_operator(10, seed + 1), keys)
    n = host.num_rows
    split = n // 3
    x = np.random.default_rng(seed).normal(size=n)
    kinds = [(0, n, 0, split, 0, n), (0, n, split, n, 0, n)]
    first = gpu_matvec(KindsHost(host, kinds), x, binned)
    expected = reference.matvec(x)
    assert np.abs(first - expected).max() <= 1e-14 * np.abs(expected).max()
    np.testing.assert_array_equal(first, gpu_matvec(KindsHost(host, kinds), x, binned))
    assert not np.array_equal(first, gpu_matvec(host, x, binned)) or n < 10


@requires_gpu
@pytest.mark.parametrize("binned", [False, True])
@pytest.mark.parametrize("residency, budget_chunks", [("resident", None), ("auto", 2)])
def test_fixed_reduce_gives_the_sorting_bits(residency, budget_chunks, binned):
    from fulqrum.gpu.full_strs.device import DeviceLayout  # noqa: PLC0415

    host = small_host(41, num_slices=5)
    x = np.random.default_rng(41).normal(size=host.num_rows)
    budget = None
    if budget_chunks is not None:
        probe = DeviceLayout(host, "resident", fixed_reduce=True)
        chunk_bytes = probe.planned_chunk_bytes
        budget = (
            probe.planned_fixed_bytes
            + chunk_bytes[0]
            + chunk_bytes[1]
            + probe.planned_stream_bytes
        )
    fixed = DeviceLayout(host, residency, budget, fixed_reduce=True)
    sorting = DeviceLayout(
        host, residency if budget is None else "streamed", fixed_reduce=False
    )
    resident = sum(1 for chunk in fixed.chunks if chunk is not None)
    assert fixed.num_fixed_reduce == 5 and resident > 0
    assert residency == "resident" or 0 < fixed.num_streamed < 5
    assert sorting.num_fixed_reduce == 0
    assert fixed.entry_rows.size == 1  # no unit sorts, so no sort buffers
    np.testing.assert_array_equal(
        gpu_matvec(host, x, binned, layout=fixed),
        gpu_matvec(host, x, binned, layout=sorting),
    )


@requires_gpu
def test_fixed_reduce_with_split_kinds():
    """Two kinds with output windows"""
    from fulqrum.gpu.full_strs.device import DeviceLayout  # noqa: PLC0415

    host = small_host(43, num_slices=3)
    n = host.num_rows
    kinds = [(0, n // 2, 0, n, 0, n // 2), (n // 2, n, 0, n, n // 2, n)]
    x = np.random.default_rng(43).normal(size=n)
    fixed = DeviceLayout(KindsHost(host, kinds), "resident", fixed_reduce=True)
    sorting = DeviceLayout(KindsHost(host, kinds), "resident", fixed_reduce=False)
    assert fixed.num_fixed_reduce == 6
    np.testing.assert_array_equal(
        gpu_matvec(host, x, True, layout=fixed),
        gpu_matvec(host, x, True, layout=sorting),
    )
    np.testing.assert_array_equal(
        gpu_matvec(host, x, True, layout=fixed), gpu_matvec(host, x, True)
    )
