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

from fulqrum.test.test_full_strs_gpu import gpu_matvec, requires_gpu

pytestmark = pytest.mark.gpu


def problem(seed, norb=10, num_alpha=5, num_beta=4, rows=3000):
    from fulqrum.gpu.full_strs.distribute import sort_unique  # noqa: PLC0415
    from fulqrum.test.test_full_strs_coefficients import random_operator  # noqa: PLC0415
    from fulqrum.test.test_full_strs_matvec import clustered_keys  # noqa: PLC0415

    keys = clustered_keys(
        np.random.default_rng(seed), norb, num_alpha, num_beta, rows, 60
    )
    return random_operator(norb, seed + 1), sort_unique(
        np.asarray(keys, dtype=np.uint64)
    )


def rank_layout(comm, operator, part, num_slices, kinds=(None, None)):
    from fulqrum.gpu.full_strs.device import DeviceLayout  # noqa: PLC0415
    from fulqrum.gpu.full_strs.distribute import distribute_keys, join_groups  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import _RankHost  # noqa: PLC0415
    from fulqrum.gpu.full_strs.host import RankOperator, block_reference  # noqa: PLC0415

    norb = operator.width // 2
    block = distribute_keys(comm, part, norb)
    receiver = join_groups(comm, block, num_slices, kinds[0], kinds[1])
    reference = comm.bcast(
        block_reference(block.keys, norb) if comm.Get_rank() == 0 else None, root=0
    )
    host = _RankHost(RankOperator(operator, block.keys, reference), receiver, norb)
    return block, host, DeviceLayout(host, "resident")


@requires_gpu
@pytest.mark.parametrize("binned", [False, True])
@pytest.mark.parametrize("kinds", [(None, None), ("intersection", "union")])
def test_one_rank_adapter_gives_the_single_block_bits(kinds, binned):
    from fulqrum.gpu.distributed.local_comm import LocalComm  # noqa: PLC0415
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415

    operator, keys = problem(3)
    x = np.random.default_rng(7).normal(size=keys.shape[0])
    single = HostOperator(operator, keys, kinds[0], kinds[1], num_slices=3)
    _, host, layout = rank_layout(LocalComm.create(1)[0], operator, keys, 3, kinds)
    np.testing.assert_array_equal(
        gpu_matvec(host, x, binned, layout=layout), gpu_matvec(single, x, binned)
    )


def run_hamiltonian(
    operator,
    keys,
    size,
    x,
    recut=True,
    repeats=2,
    exchange="stepped",
    copies=1,
    plan=None,
):
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415

    parts = np.array_split(
        keys[np.random.default_rng(size).permutation(keys.shape[0])], size
    )

    def body(comm):
        cupy.cuda.Device(0).use()
        h = FullStrsHamiltonian(
            operator,
            parts[comm.Get_rank()],
            comm=comm,
            num_slices=3,
            recut=recut if copies == 1 else False,
            exchange=exchange,
            copies=copies,
            plan=plan,
        )
        assert h.shape == (keys.shape[0], keys.shape[0])
        assert h._is_primary
        x_local = cupy.asarray(x[h.row_start : h.row_start + h.n_local])
        outputs = []
        for _ in range(repeats):
            out = cupy.full(h.n_local, np.nan)
            h._matvec(x_local, out=out)
            cupy.cuda.get_current_stream().synchronize()
            outputs.append(cupy.asnumpy(out))
        np.testing.assert_array_equal(
            h.keys, keys[h.row_start : h.row_start + h.n_local]
        )
        return h.row_start, outputs, h.num_halo

    return run_ranks(LocalComm.create(size, timeout=300.0), body)


@requires_gpu
@pytest.mark.parametrize("recut", [False, True])
@pytest.mark.parametrize("size", [2, 3, 4])
def test_rank_matvec_equals_the_cpu(size, recut):
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415

    operator, keys = problem(11)
    x = np.random.default_rng(5).normal(size=keys.shape[0])
    reference = HostOperator(operator, keys).matvec(x)
    scale = np.abs(reference).max()
    results = run_hamiltonian(operator, keys, size, x, recut)
    covered = 0
    for row_start, outputs, num_halo in results:
        part = reference[row_start : row_start + outputs[0].size]
        assert np.abs(outputs[0] - part).max() <= 1e-14 * scale
        np.testing.assert_array_equal(outputs[0], outputs[1])
        assert num_halo > 0
        covered += outputs[0].size
    assert covered == keys.shape[0]


@requires_gpu
def test_ranks_without_halo():
    from fulqrum.gpu.full_strs.distribute import sort_unique  # noqa: PLC0415
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415
    from fulqrum.test.test_full_strs_coefficients import random_operator  # noqa: PLC0415
    from fulqrum.test.test_full_strs_distribute import random_keys  # noqa: PLC0415

    low = random_keys(np.random.default_rng(4), 200, 5, 3, 3)
    keys = sort_unique(np.vstack((low, low << np.uint64(5))))
    operator = random_operator(10, 9)
    x = np.random.default_rng(6).normal(size=keys.shape[0])
    reference = HostOperator(operator, keys).matvec(x)
    results = run_hamiltonian(operator, keys, 2, x, recut=False)
    assert [num_halo for _, _, num_halo in results] == [0, 0]
    for row_start, outputs, _ in results:
        part = reference[row_start : row_start + outputs[0].size]
        assert np.abs(outputs[0] - part).max() <= 1e-14 * np.abs(reference).max()


@requires_gpu
@pytest.mark.parametrize("size, copies", [(3, 1), (4, 2)])
def test_released_host_copies_give_the_same_bits(size, copies):
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415

    operator, keys = problem(11)
    x = np.random.default_rng(8).normal(size=keys.shape[0])
    parts = np.array_split(keys, size)

    def body(comm):
        cupy.cuda.Device(0).use()
        ys = []
        for release in (False, True):
            h = FullStrsHamiltonian(
                operator,
                parts[comm.Get_rank()],
                comm=comm,
                num_slices=3,
                recut=False,
                copies=copies,
                release_host_copies=release,
            )
            out = h._matvec(cupy.asarray(x[h.row_start : h.row_start + h.n_local]))
            cupy.cuda.get_current_stream().synchronize()
            ys.append(cupy.asnumpy(out))
            assert (h._host.halo_key_words().size == 0) == (release or h.num_halo == 0)
            layout = h._layout
            kinds = len(layout.kinds)
            resident = [
                unit for unit in range(len(layout.views)) if layout.resident[unit]
            ]
            assert resident
            for unit in resident:
                views = [
                    h._host.chunk_views(unit, cls)
                    if kinds == 1
                    else h._host.chunk_views(unit // kinds, cls, unit % kinds)
                    for cls in range(5)
                ]
                if release:
                    assert layout.views[unit] is None
                    assert (
                        sum(int(view[0].size) + int(view[1].size) for view in views)
                        == 0
                    )
                else:
                    assert layout.views[unit] is not None
        return ys

    for keep, released in run_ranks(LocalComm.create(size, timeout=300.0), body):
        np.testing.assert_array_equal(keep, released)


def run_two_level(operator, keys, size, x, domain_size, recut):
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415

    parts = np.array_split(
        keys[np.random.default_rng(size).permutation(keys.shape[0])], size
    )

    def body(comm):
        cupy.cuda.Device(0).use()
        ys, halos = [], []
        for exchange in ("stepped", "two_level"):
            h = FullStrsHamiltonian(
                operator,
                parts[comm.Get_rank()],
                comm=comm,
                num_slices=3,
                recut=recut,
                exchange=exchange,
                domain_size=domain_size if exchange == "two_level" else None,
            )
            out = h._matvec(cupy.asarray(x[h.row_start : h.row_start + h.n_local]))
            cupy.cuda.get_current_stream().synchronize()
            ys.append(cupy.asnumpy(out))
            halos.append(h.num_halo)
        return ys[0], ys[1], halos, h._halo.traffic

    return run_ranks(LocalComm.create(size, timeout=300.0), body)


@requires_gpu
@pytest.mark.parametrize("size, domain_size", [(4, 2), (6, 3), (5, 2), (4, 1), (3, 2)])
@pytest.mark.parametrize("recut", [False, True])
def test_two_level_exchange_gives_the_stepped_bits(size, domain_size, recut):
    operator, keys = problem(11)
    x = np.random.default_rng(9).normal(size=keys.shape[0])
    results = run_two_level(operator, keys, size, x, domain_size, recut)
    for stepped, two_level, (halo_stepped, halo_two_level), traffic in results:
        np.testing.assert_array_equal(stepped, two_level)
        assert halo_two_level >= halo_stepped
        assert traffic["peers"] <= size - 1


@requires_gpu
def test_two_level_exchange_needs_two_domains():
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415

    operator, keys = problem(11)
    parts = np.array_split(keys, 2)

    def body(comm):
        cupy.cuda.Device(0).use()
        with pytest.raises(ValueError, match="at least two shared-memory domains"):
            FullStrsHamiltonian(
                operator,
                parts[comm.Get_rank()],
                comm=comm,
                num_slices=3,
                exchange="two_level",
                domain_size=2,
            )
        return True

    assert all(run_ranks(LocalComm.create(2, timeout=300.0), body))


def run_by_key(operator, keys, size, x, exchange, domain_size=None):
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415

    parts = np.array_split(
        keys[np.random.default_rng(size).permutation(keys.shape[0])], size
    )

    def body(comm):
        cupy.cuda.Device(0).use()
        entries = {}
        for groups in ("by_rows", "by_key"):
            h = FullStrsHamiltonian(
                operator,
                parts[comm.Get_rank()],
                comm=comm,
                num_slices=3,
                recut=False,
                exchange=exchange,
                domain_size=domain_size,
                groups=groups,
            )
            entries[groups] = h._layout.num_entries
        ys = []
        for _ in range(2):
            out = h._matvec(cupy.asarray(x[h.row_start : h.row_start + h.n_local]))
            cupy.cuda.get_current_stream().synchronize()
            ys.append(cupy.asnumpy(out))
        return h.row_start, ys, entries

    return run_ranks(LocalComm.create(size, timeout=300.0), body)


@requires_gpu
@pytest.mark.parametrize(
    "size, exchange, domain_size",
    [
        (2, "stepped", None),
        (3, "stepped", None),
        (4, "grouped", None),
        (4, "two_level", 2),
        (5, "two_level", 2),
        (3, "two_level", 2),
        (6, "two_level", 3),
    ],
)
def test_groups_by_key_equal_the_cpu(size, exchange, domain_size):
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415

    operator, keys = problem(11)
    x = np.random.default_rng(13).normal(size=keys.shape[0])
    reference = HostOperator(operator, keys).matvec(x)
    scale = np.abs(reference).max()
    results = run_by_key(operator, keys, size, x, exchange, domain_size)
    covered = 0
    for row_start, (first, second), entries in results:
        part = reference[row_start : row_start + first.size]
        assert np.abs(first - part).max() <= 1e-13 * scale
        np.testing.assert_array_equal(first, second)
        covered += first.size
    assert covered == keys.shape[0]
    assert sum(e["by_key"] for _, _, e in results) <= sum(
        e["by_rows"] for _, _, e in results
    )


@requires_gpu
def test_groups_by_key_need_one_copy():
    from fulqrum.gpu.distributed.local_comm import LocalComm  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415

    operator, keys = problem(11)
    with pytest.raises(ValueError, match="one copy per block"):
        FullStrsHamiltonian(
            operator, keys, comm=LocalComm.create(1)[0], copies=2, groups="by_key"
        )


@requires_gpu
@pytest.mark.parametrize(
    "size, groups, exchange, domain_size",
    [
        (1, "by_rows", "stepped", None),
        (3, "by_rows", "stepped", None),
        (3, "by_key", "stepped", None),
        (4, "by_key", "two_level", 2),
    ],
)
def test_atomic_reduce_equals_the_cpu(size, groups, exchange, domain_size):
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415

    operator, keys = problem(11)
    x = np.random.default_rng(17).normal(size=keys.shape[0])
    reference = HostOperator(operator, keys).matvec(x)
    scale = np.abs(reference).max()
    parts = np.array_split(
        keys[np.random.default_rng(size).permutation(keys.shape[0])], size
    )

    def body(comm):
        cupy.cuda.Device(0).use()
        sizes = {}
        for atomic in (False, True):
            h = FullStrsHamiltonian(
                operator,
                parts[comm.Get_rank()] if size > 1 else keys,
                comm=comm,
                num_slices=3,
                recut=False,
                exchange=exchange,
                domain_size=domain_size,
                groups=groups,
                atomic_reduce=atomic,
            )
            sizes[atomic] = h.device_bytes
        out = h._matvec(cupy.asarray(x[h.row_start : h.row_start + h.n_local]))
        cupy.cuda.get_current_stream().synchronize()
        return h.row_start, cupy.asnumpy(out), sizes

    if size == 1:
        from mpi4py import MPI  # noqa: PLC0415

        results = [body(MPI.COMM_SELF)]
    else:
        results = run_ranks(LocalComm.create(size, timeout=300.0), body)
    covered = 0
    for row_start, y, sizes in results:
        assert (
            np.abs(y - reference[row_start : row_start + y.size]).max() <= 1e-13 * scale
        )
        assert sizes[True] < sizes[False]
        covered += y.size
    assert covered == keys.shape[0]


@requires_gpu
@pytest.mark.parametrize("size", [2, 3, 5])
def test_full_vectors_equal_the_cpu(size):
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415

    operator, keys = problem(11)
    x = np.random.default_rng(19).normal(size=keys.shape[0])
    reference = HostOperator(operator, keys).matvec(x)
    scale = np.abs(reference).max()
    parts = np.array_split(
        keys[np.random.default_rng(size).permutation(keys.shape[0])], size
    )

    def body(comm):
        cupy.cuda.Device(0).use()
        h = FullStrsHamiltonian(
            operator,
            parts[comm.Get_rank()],
            comm=comm,
            num_slices=3,
            groups="by_key",
            vectors="full",
        )
        assert (
            h._atomic_reduce and h.num_halo == comm.Get_size() * h._full[1] - h.n_local
        )
        out = h._matvec(cupy.asarray(x[h.row_start : h.row_start + h.n_local]))
        cupy.cuda.get_current_stream().synchronize()
        return h.row_start, cupy.asnumpy(out)

    covered = 0
    for row_start, y in run_ranks(LocalComm.create(size, timeout=300.0), body):
        assert (
            np.abs(y - reference[row_start : row_start + y.size]).max() <= 1e-13 * scale
        )
        covered += y.size
    assert covered == keys.shape[0]


@requires_gpu
@pytest.mark.parametrize("size", [2, 3, 4])
def test_stepped_and_grouped_exchange_give_the_same_bits(size):
    operator, keys = problem(17)
    x = np.random.default_rng(8).normal(size=keys.shape[0])
    stepped = run_hamiltonian(operator, keys, size, x, repeats=1, exchange="stepped")
    grouped = run_hamiltonian(operator, keys, size, x, repeats=1, exchange="grouped")
    for (start_s, out_s, _), (start_g, out_g, _) in zip(stepped, grouped):
        assert start_s == start_g
        np.testing.assert_array_equal(out_s[0], out_g[0])


@requires_gpu
@pytest.mark.parametrize("binned", [False, True])
def test_one_block_of_four_copies_gives_the_single_rank_bits(binned):
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415

    operator, keys = problem(29)
    x = np.random.default_rng(3).normal(size=keys.shape[0])
    single = gpu_matvec(HostOperator(operator, keys, num_slices=3), x, binned)
    parts = np.array_split(keys, 4)

    def body(comm):
        cupy.cuda.Device(0).use()
        h = FullStrsHamiltonian(
            operator,
            parts[comm.Get_rank()],
            comm=comm,
            num_slices=3,
            binned=binned,
            copies=4,
        )
        out = cupy.full(h.n_local, np.nan)
        h._matvec(cupy.asarray(x[h.row_start : h.row_start + h.n_local]), out=out)
        cupy.cuda.get_current_stream().synchronize()
        return h.row_start, cupy.asnumpy(out), h.num_halo

    results = sorted(
        run_ranks(LocalComm.create(4, timeout=300.0), body), key=lambda r: r[0]
    )
    assert [r[2] for r in results] == [0, 0, 0, 0]
    np.testing.assert_array_equal(np.concatenate([r[1] for r in results]), single)


@requires_gpu
@pytest.mark.parametrize("size, copies", [(2, 2), (4, 2), (4, 4)])
def test_copies_equal_the_cpu(size, copies):
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415

    operator, keys = problem(31)
    x = np.random.default_rng(9).normal(size=keys.shape[0])
    reference = HostOperator(operator, keys).matvec(x)
    scale = np.abs(reference).max()
    stepped = run_hamiltonian(operator, keys, size, x, copies=copies)
    grouped = run_hamiltonian(
        operator, keys, size, x, repeats=1, exchange="grouped", copies=copies
    )
    covered = 0
    for (row_start, outputs, _), (start_g, out_g, _) in zip(stepped, grouped):
        part = reference[row_start : row_start + outputs[0].size]
        assert np.abs(outputs[0] - part).max() <= 1e-14 * scale
        np.testing.assert_array_equal(outputs[0], outputs[1])
        assert row_start == start_g
        np.testing.assert_array_equal(outputs[0], out_g[0])
        covered += outputs[0].size
    assert covered == keys.shape[0]


@requires_gpu
@pytest.mark.parametrize("size, copies", [(4, 1), (4, 2), (2, 2)])
def test_plan_round_trip_gives_the_same_bits(size, copies):
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415

    operator, keys = problem(37)
    x = np.random.default_rng(4).normal(size=keys.shape[0])
    parts = np.array_split(
        keys[np.random.default_rng(size).permutation(keys.shape[0])], size
    )

    def body(comm, plan):
        cupy.cuda.Device(0).use()
        rank = comm.Get_rank()
        h = FullStrsHamiltonian(
            operator,
            parts[(rank + 1) % size] if plan else parts[rank],
            comm=comm,
            num_slices=3,
            copies=copies,
            recut=copies == 1,
            plan=plan,
        )
        out = cupy.full(h.n_local, np.nan)
        h._matvec(cupy.asarray(x[h.row_start : h.row_start + h.n_local]), out=out)
        cupy.cuda.get_current_stream().synchronize()
        return h.row_start, cupy.asnumpy(out), h.plan.to_dict()

    first = run_ranks(
        LocalComm.create(size, timeout=300.0), lambda comm: body(comm, None)
    )
    plan = first[0][2]
    assert all(r[2]["digest"] == plan["digest"] for r in first)
    second = run_ranks(
        LocalComm.create(size, timeout=300.0), lambda comm: body(comm, plan)
    )
    for (start_a, out_a, _), (start_b, out_b, plan_b) in zip(first, second):
        assert start_a == start_b
        assert plan_b["digest"] == plan["digest"]
        np.testing.assert_array_equal(out_a, out_b)


@requires_gpu
def test_different_plans_are_refused():
    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415
    from fulqrum.gpu.full_strs.plan import LayoutPlan  # noqa: PLC0415

    operator, keys = problem(3)
    parts = np.array_split(keys, 2)

    def body(comm):
        plan = LayoutPlan(num_slices=2 + comm.Get_rank())
        return FullStrsHamiltonian(
            operator, parts[comm.Get_rank()], comm=comm, plan=plan
        )

    with pytest.raises(ValueError, match="layout plan"):
        run_ranks(LocalComm.create(2, timeout=60.0), body)


@requires_gpu
@pytest.mark.parametrize(
    "block_cut, shares",
    [
        ("membership", "equal"),
        ("fitted", "equal"),
        ("row", "balanced"),
        ("fitted", "balanced"),
    ],
)
@pytest.mark.parametrize("size, copies", [(4, 2), (4, 1), (6, 2)])
def test_planned_layouts_equal_the_cpu(size, copies, block_cut, shares):
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415
    from fulqrum.gpu.full_strs.plan import LayoutPlan  # noqa: PLC0415

    operator, keys = problem(47)
    x = np.random.default_rng(2).normal(size=keys.shape[0])
    reference = HostOperator(operator, keys).matvec(x)
    plan = LayoutPlan(copies=copies, num_slices=3, block_cut=block_cut, shares=shares)
    results = run_hamiltonian(operator, keys, size, x, plan=plan)
    covered = 0
    for row_start, outputs, _ in results:
        part = reference[row_start : row_start + outputs[0].size]
        assert np.abs(outputs[0] - part).max() <= 1e-14 * np.abs(reference).max()
        np.testing.assert_array_equal(outputs[0], outputs[1])
        covered += outputs[0].size
    assert covered == keys.shape[0]


@requires_gpu
@pytest.mark.parametrize("size, copies", [(4, 1), (4, 2)])
def test_default_layout_is_pairs_with_balanced_shares(size, copies):
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415

    operator, keys = problem(53)
    x = np.random.default_rng(1).normal(size=keys.shape[0])
    reference = HostOperator(operator, keys).matvec(x)
    parts = np.array_split(keys, size)

    def body(comm):
        cupy.cuda.Device(0).use()
        h = FullStrsHamiltonian(
            operator, parts[comm.Get_rank()], comm=comm, num_slices=3, copies=copies
        )
        out = cupy.empty(h.n_local, dtype=np.float64)
        h._matvec(cupy.asarray(x[h.row_start : h.row_start + h.n_local]), out=out)
        cupy.cuda.get_current_stream().synchronize()
        return h.row_start, cupy.asnumpy(out), h.plan.to_dict()

    for row_start, out, plan in run_ranks(LocalComm.create(size, timeout=300.0), body):
        assert (
            plan["block_cut"] == "fitted"
            and plan["coefficients"] == [1.0] * 35
            and plan["shares"] == "balanced"
        )
        assert (
            np.abs(out - reference[row_start : row_start + out.size]).max()
            <= 1e-14 * np.abs(reference).max()
        )


@requires_gpu
def test_automatic_copies():
    import cupy  # noqa: PLC0415

    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415
    from fulqrum.gpu.full_strs.host import HostOperator  # noqa: PLC0415

    operator, keys = problem(59)
    x = np.random.default_rng(8).normal(size=keys.shape[0])
    reference = HostOperator(operator, keys).matvec(x)
    parts = np.array_split(keys, 4)

    def body(comm):
        cupy.cuda.Device(0).use()
        h = FullStrsHamiltonian(
            operator, parts[comm.Get_rank()], comm=comm, num_slices=3, copies="auto"
        )
        out = cupy.empty(h.n_local, dtype=np.float64)
        h._matvec(cupy.asarray(x[h.row_start : h.row_start + h.n_local]), out=out)
        cupy.cuda.get_current_stream().synchronize()
        return h.row_start, cupy.asnumpy(out), h.plan.to_dict()

    results = run_ranks(LocalComm.create(4, timeout=300.0), body)
    plan = results[0][2]
    assert set(plan["measured"]["copies_candidates"]) == {"1", "2", "4"}
    assert plan["copies"] in (1, 2, 4)
    for row_start, out, rank_plan in results:
        assert rank_plan["digest"] == plan["digest"]
        assert (
            np.abs(out - reference[row_start : row_start + out.size]).max()
            <= 1e-14 * np.abs(reference).max()
        )
