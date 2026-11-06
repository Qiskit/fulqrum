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


import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks
from fulqrum.gpu.full_strs.distribute import (
    balanced_bounds,
    build_copy_groups,
    plan_copy_groups,
    build_distributed_groups,
    distribute_keys,
    join_groups,
    output_memberships,
    redistribute,
    sort_unique,
)
from fulqrum.gpu.full_strs.host import CLASSES, KINDS, HostLayout
from fulqrum.test.test_full_strs_distribute import random_keys

SETS = Path(os.environ.get("FQ_FULL_STRS_SETS", "/nonexistent"))
SMALL_SETS = sorted((SETS / "small").glob("*.bin")) if SETS.exists() else []
SUBSAMPLES = [SETS / "sub_head.bin", SETS / "sub_uniform.bin"] if SETS.exists() else []
KIND_PAIRS = list(itertools.product(KINDS, KINDS))


def read_set(path):
    raw = np.fromfile(path, dtype=np.uint64)
    count, norb = int(raw[0]), int(raw[1])
    return raw[2 : 2 + 2 * count].reshape(count, 2), norb


def clustered_keys(rng, norb, alpha, beta, rows, pool):
    alphas = random_keys(rng, pool, norb, alpha, beta)[:, 0]
    betas = random_keys(rng, pool, norb, alpha, beta)[:, 1]
    keys = np.stack(
        (alphas[rng.integers(0, pool, rows)], betas[rng.integers(0, pool, rows)]),
        axis=1,
    )
    return sort_unique(keys.astype(np.uint64))


def run_join(keys, norb, size, num_slices, kinds, starts=None):
    parts = np.array_split(
        keys[np.random.default_rng(size).permutation(keys.shape[0])], size
    )

    def body(comm):
        block = distribute_keys(comm, parts[comm.Get_rank()], norb)
        if starts is not None:
            block = redistribute(comm, block, starts)
        return block, join_groups(comm, block, num_slices, kinds[0], kinds[1])

    results = run_ranks(LocalComm.create(size, timeout=120.0), body)
    return [r[0] for r in results], [r[1] for r in results]


def rank_outputs(receiver, cls):
    own = receiver.num_own
    return sum(
        int(np.count_nonzero(receiver.chunk_class(c, cls)[0] < own))
        for c in range(receiver.num_chunks())
    )


def check_join(
    keys, norb, size, num_slices, kinds, brute=True, starts=None, results=None
):
    blocks, receivers = (
        results
        if results is not None
        else run_join(keys, norb, size, num_slices, kinds, starts)
    )
    layout = HostLayout(keys, norb, kinds[0], kinds[1])
    stats = layout.stats()
    unique = sort_unique(keys)
    for block, receiver in zip(blocks, receivers):
        halo = receiver.halo_rows()
        np.testing.assert_array_equal(receiver.halo_keys(), unique[halo])
        assert not np.any(
            (halo >= block.row_start) & (halo < block.row_start + block.n_local)
        )
    for cls in range(len(CLASSES)):
        union = np.concatenate([receiver.pairs(cls) for receiver in receivers])
        union = union[np.lexsort((union[:, 1], union[:, 0]))]
        np.testing.assert_array_equal(
            union, layout.stored_pairs(cls), err_msg=CLASSES[cls]
        )
        if brute:
            np.testing.assert_array_equal(
                union, layout.scanned_pairs(cls), err_msg=CLASSES[cls]
            )
        count = sum(receiver.pair_digest(cls)[0] for receiver in receivers)
        digest = sum(receiver.pair_digest(cls)[1] for receiver in receivers) % (1 << 64)
        assert (count, digest) == layout.pair_digest(cls)
        assert (
            sum(rank_outputs(receiver, cls) for receiver in receivers)
            == stats[CLASSES[cls]]["outputs"]
        )
    return blocks, receivers


@pytest.mark.parametrize("num_slices", [1, 3])
@pytest.mark.parametrize("kinds", KIND_PAIRS)
@pytest.mark.parametrize("shape", [(10, 5, 4), (10, 7, 3), (9, 2, 6)])
def test_one_rank_reproduces_the_single_block_chunks(shape, kinds, num_slices):
    norb, alpha, beta = shape
    keys = clustered_keys(
        np.random.default_rng(norb + alpha), norb, alpha, beta, 1500, 40
    )
    _, receivers = run_join(keys, norb, 1, num_slices, kinds)
    receiver = receivers[0]
    layout = HostLayout(keys, norb, kinds[0], kinds[1])
    assert receiver.halo_rows().size == 0
    for chunk, cls in itertools.product(range(num_slices), range(len(CLASSES))):
        members, sizes, buckets = receiver.chunk_class(chunk, cls)
        ref_members, ref_sizes, ref_buckets = layout.chunk_class(num_slices, chunk, cls)
        np.testing.assert_array_equal(members, ref_members)
        np.testing.assert_array_equal(sizes, ref_sizes)
        assert buckets == ref_buckets


@pytest.mark.parametrize("size", [1, 2, 3, 4])
@pytest.mark.parametrize("kinds", KIND_PAIRS)
@pytest.mark.parametrize("shape", [(10, 5, 4), (9, 2, 6), (12, 9, 8)])
def test_pairs_over_ranks_equal_the_single_block(shape, kinds, size):
    norb, alpha, beta = shape
    keys = clustered_keys(
        np.random.default_rng(size * 7 + norb), norb, alpha, beta, 900, 30
    )
    check_join(keys, norb, size, 2, kinds)


@pytest.mark.parametrize("size", [2, 3])
def test_line_split_over_ranks(size):
    rng = np.random.default_rng(21)
    norb = 10
    alphas = sort_unique(random_keys(rng, 400, norb, 5, 5))[:, 0]
    line = np.stack((alphas, np.full(alphas.size, 0b11111, dtype=np.uint64)), axis=1)
    rest = clustered_keys(rng, norb, 5, 5, 300, 20)
    keys = sort_unique(np.vstack((line, rest)))
    blocks, _ = check_join(keys, norb, size, 2, (None, None))
    betas = keys[:, 1]
    starts = blocks[0].block_starts
    split = [r for r in range(1, size) if betas[starts[r] - 1] == betas[starts[r]]]
    assert split, "no line spans a block boundary"


def test_rank_without_linked_source():
    rng = np.random.default_rng(4)
    low = random_keys(rng, 200, 5, 3, 3)
    high = low << np.uint64(5)
    keys = sort_unique(np.vstack((low, high)))
    assert keys.shape[0] % 2 == 0
    _, receivers = check_join(keys, 10, 2, 2, (None, None))
    assert [r.halo_rows().size for r in receivers] == [0, 0]


@pytest.mark.parametrize("cut", [(0, 7, 400, 403), (0, 1, 2, 3), (0, 600, 601, 602)])
def test_skewed_block_cuts(cut):
    keys = clustered_keys(np.random.default_rng(13), 10, 5, 4, 1500, 40)
    starts = np.array(list(cut) + [keys.shape[0]], dtype=np.int64)
    blocks, _ = check_join(keys, 10, 4, 2, ("union", "intersection"), starts=starts)
    assert [b.n_local for b in blocks] == np.diff(starts).tolist()


@pytest.mark.parametrize("size", [2, 3, 4])
def test_recut_by_memberships(size):
    keys = np.vstack(
        (
            clustered_keys(np.random.default_rng(31), 10, 5, 5, 1500, 25),
            clustered_keys(np.random.default_rng(32), 10, 5, 5, 300, 200),
        )
    )
    keys = sort_unique(keys)
    parts = np.array_split(keys, size)

    def body(comm):
        return build_distributed_groups(comm, parts[comm.Get_rank()], 10, 2)

    results = run_ranks(LocalComm.create(size, timeout=120.0), body)
    blocks = [r[0] for r in results]
    receivers = [r[1] for r in results]
    for block in blocks:
        np.testing.assert_array_equal(block.block_starts, blocks[0].block_starts)
    assert all(block.n_local >= 1 for block in blocks)
    check_join(keys, 10, size, 2, (None, None), results=(blocks, receivers))
    info = results[0][2]
    work = [int(output_memberships(r).sum()) for r in receivers]
    assert info["imbalance_after"] == pytest.approx(max(work) / np.mean(work))
    assert info["imbalance_after"] <= info["imbalance_before"]


@pytest.mark.parametrize("size", [2, 3])
def test_recut_on_a_probe_slice(size):
    keys = sort_unique(
        np.vstack(
            (
                clustered_keys(np.random.default_rng(31), 10, 5, 5, 1500, 25),
                clustered_keys(np.random.default_rng(32), 10, 5, 5, 300, 200),
            )
        )
    )
    parts = np.array_split(keys, size)

    def body(comm):
        return build_distributed_groups(
            comm, parts[comm.Get_rank()], 10, 2, probe_slices=4
        )

    results = run_ranks(LocalComm.create(size, timeout=120.0), body)
    blocks = [r[0] for r in results]
    for block in blocks:
        np.testing.assert_array_equal(block.block_starts, blocks[0].block_starts)
    check_join(
        keys, 10, size, 2, (None, None), results=(blocks, [r[1] for r in results])
    )


def test_recut_without_work_keeps_the_row_cut():
    keys = np.array(
        [[0b111 << (3 * k), 0b111 << (3 * k)] for k in range(4)], dtype=np.uint64
    )
    parts = [keys[k : k + 1] for k in range(4)]

    def body(comm):
        return build_distributed_groups(comm, parts[comm.Get_rank()], 12, 1)

    results = run_ranks(LocalComm.create(4, timeout=60.0), body)
    np.testing.assert_array_equal(results[0][0].block_starts, [0, 1, 2, 3, 4])
    assert results[0][2]["imbalance_before"] == 1.0


@pytest.mark.parametrize("size", [3, 4])
def test_many_slices_on_few_rows(size):
    keys = clustered_keys(np.random.default_rng(size), 8, 4, 3, 90, 9)
    check_join(keys, 8, size, 16, (None, None))


def test_receiver_sized_by_the_received_rows(monkeypatch):
    from fulqrum.gpu.full_strs import host  # noqa: PLC0415

    keys = clustered_keys(np.random.default_rng(81), 10, 5, 4, 1500, 40)
    kinds = ("union", "intersection")
    _, sized = run_join(keys, 10, 3, 3, kinds)
    original = host.GroupReceiver
    monkeypatch.setattr(
        host,
        "GroupReceiver",
        lambda rows, row_start, num_slices, total_rows=0: original(
            rows, row_start, num_slices
        ),
    )
    blocks, unsized = run_join(keys, 10, 3, 3, kinds)
    check_join(keys, 10, 3, 3, kinds, results=(blocks, unsized))
    for a, b in zip(sized, unsized):
        np.testing.assert_array_equal(a.halo_rows(), b.halo_rows())
        np.testing.assert_array_equal(a.halo_keys(), b.halo_keys())
        for chunk in range(a.num_chunks()):
            for cls in range(len(CLASSES)):
                for left, right in zip(
                    a.chunk_class(chunk, cls)[:2], b.chunk_class(chunk, cls)[:2]
                ):
                    np.testing.assert_array_equal(left, right)


def test_extended_halo_keeps_every_pair():
    keys = clustered_keys(np.random.default_rng(91), 10, 5, 4, 1500, 40)
    blocks, receivers = run_join(keys, 10, 3, 3, ("union", "intersection"))
    before = [
        [receiver.pairs(cls) for cls in range(len(CLASSES))] for receiver in receivers
    ]
    unique = sort_unique(keys)
    rng = np.random.default_rng(92)
    for block, receiver in zip(blocks, receivers):
        halo = receiver.halo_rows()
        bits = receiver.halo_bitmap(block.num_rows)
        others = np.setdiff1d(
            np.arange(block.num_rows),
            np.concatenate(
                (halo, np.arange(block.row_start, block.row_start + block.n_local))
            ),
        )
        extra = rng.choice(others, size=min(7, others.size), replace=False)
        for row in extra:
            bits[row // 64] |= np.uint64(1) << np.uint64(row % 64)
        receiver.extend_halo(bits)
        grown = receiver.halo_rows()
        np.testing.assert_array_equal(grown, np.union1d(halo, extra))
        added = np.isin(grown, extra)
        np.testing.assert_array_equal(receiver.halo_keys()[~added], unique[halo])
        assert not receiver.halo_keys()[added].any()
    after = [
        [receiver.pairs(cls) for cls in range(len(CLASSES))] for receiver in receivers
    ]
    for old, new in zip(before, after):
        for left, right in zip(old, new):
            np.testing.assert_array_equal(left, right)


def test_extended_halo_refuses_a_bad_bitmap():
    keys = clustered_keys(np.random.default_rng(93), 10, 5, 4, 600, 30)
    blocks, receivers = run_join(keys, 10, 2, 2, ("union", "intersection"))
    block, receiver = blocks[0], receivers[0]
    bits = receiver.halo_bitmap(block.num_rows)
    missing = bits.copy()
    row = int(receiver.halo_rows()[0])
    missing[row // 64] &= ~(np.uint64(1) << np.uint64(row % 64))
    with pytest.raises(ValueError, match="misses 1 halo rows"):
        receiver.extend_halo(missing)
    with_own = bits.copy()
    own = block.row_start
    with_own[own // 64] |= np.uint64(1) << np.uint64(own % 64)
    with pytest.raises(ValueError, match="holds 1 own rows"):
        receiver.extend_halo(with_own)


def test_wrong_halo_keys_are_refused():
    keys = clustered_keys(np.random.default_rng(95), 10, 5, 4, 600, 30)
    parts = np.array_split(keys, 2)

    def body(comm):
        block = distribute_keys(comm, parts[comm.Get_rank()], 10)
        receiver = join_groups(
            comm, block, 2, "union", "intersection", fetch_keys=False
        )
        buffer = receiver.halo_key_buffer()
        buffer[:] = sort_unique(keys)[receiver.halo_rows()]
        buffer[0, 0] ^= np.uint64(1)
        with pytest.raises(ValueError, match="differ from their rebuilt keys"):
            receiver.verify_halo_keys()
        return True

    assert all(run_ranks(LocalComm.create(2, timeout=120.0), body))


def test_full_vector_receiver_holds_the_by_key_groups():
    keys = clustered_keys(np.random.default_rng(97), 10, 5, 4, 1500, 40)

    def body(vectors):
        def run(comm):
            block = distribute_keys(comm, np.array_split(keys, 3)[comm.Get_rank()], 10)
            receiver = join_groups(
                comm,
                block,
                3,
                "union",
                "intersection",
                groups="by_key",
                vectors=vectors,
            )
            return block, receiver

        return run_ranks(LocalComm.create(3, timeout=120.0), run)

    def groups_of(block, receiver, vectors, stride):
        out = []
        halo = receiver.halo_rows()
        for chunk in range(receiver.num_chunks()):
            for cls in range(len(CLASSES)):
                members, sizes, _ = receiver.chunk_class(chunk, cls)
                rows = members.view(np.uint32).astype(np.int64)
                if vectors == "full":
                    owner = rows // stride
                    rows = block.block_starts[owner] + rows - owner * stride
                else:
                    own = rows < block.n_local
                    rows = np.where(
                        own,
                        block.row_start + rows,
                        halo[np.clip(rows - block.n_local, 0, None)],
                    )
                for group in (
                    np.split(rows, np.cumsum(sizes.astype(np.int64))[:-1])
                    if sizes.size
                    else []
                ):
                    out.append(
                        (chunk, cls, tuple(sorted(group.tolist())))
                    )  # the halo form orders own rows first
        return sorted(out)

    halo_form = body("halo")
    full_form = body("full")
    for (block, halo_receiver), (_, full_receiver) in zip(halo_form, full_form):
        stride = int(np.diff(block.block_starts).max())
        assert full_receiver.halo_rows().size == 0
        assert groups_of(block, full_receiver, "full", stride) == groups_of(
            block, halo_receiver, "halo", stride
        )


LAYOUT_SCRIPT = """
import hashlib, numpy as np
from fulqrum.test.test_full_strs_join import clustered_keys, run_join
keys = clustered_keys(np.random.default_rng(77), 10, 5, 4, 1500, 40)
_, receivers = run_join(keys, 10, 3, 3, ("union", "intersection"))
digest = hashlib.sha256()
for r in receivers:
    digest.update(r.halo_rows().tobytes())
    digest.update(r.halo_keys().tobytes())
    for c in range(r.num_chunks()):
        for cls in range(5):
            members, sizes, buckets = r.chunk_class(c, cls)
            digest.update(members.tobytes())
            digest.update(sizes.tobytes())
            digest.update(np.array(buckets, dtype=np.int64).tobytes())
print(digest.hexdigest())
"""


def test_layout_does_not_depend_on_threads():
    digests = []
    for threads in ("1", "4"):
        env = dict(os.environ, OMP_NUM_THREADS=threads)
        result = subprocess.run(
            [sys.executable, "-c", LAYOUT_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        digests.append(result.stdout.strip())
    assert digests[0] == digests[1]


def test_redistribute_refuses_an_empty_block():
    keys = clustered_keys(np.random.default_rng(2), 8, 4, 4, 200, 20)
    parts = np.array_split(keys, 3)

    def body(comm):
        block = distribute_keys(comm, parts[comm.Get_rank()], 8)
        return redistribute(comm, block, [0, 5, 5, block.num_rows])

    with pytest.raises(ValueError, match="at least one row per rank"):
        run_ranks(LocalComm.create(3, timeout=60.0), body)


def test_refusals():
    keys = clustered_keys(np.random.default_rng(1), 8, 4, 4, 300, 20)
    parts = np.array_split(keys, 2)

    def body(comm):
        block = distribute_keys(comm, parts[comm.Get_rank()], 8)
        return join_groups(comm, block, 2 + comm.Get_rank())

    with pytest.raises(ValueError, match="different \\(slices, kinds\\)"):
        run_ranks(LocalComm.create(2, timeout=60.0), body)


@pytest.mark.long
@pytest.mark.skipif(
    not SMALL_SETS, reason="FQ_FULL_STRS_SETS does not hold the generated sets"
)
@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("kinds", KIND_PAIRS)
def test_generated_sets(kinds, size):
    paths = SMALL_SETS + [p for p in SUBSAMPLES if p.exists()]
    for path in paths:
        keys, norb = read_set(path)
        keys = sort_unique(keys)
        if keys.shape[0] < size:
            continue
        check_join(keys, norb, size, 3, kinds, brute=keys.shape[0] <= 4000)


def run_copies(keys, norb, size, copies, num_slices, kinds=(None, None)):
    parts = np.array_split(
        keys[np.random.default_rng(size + copies).permutation(keys.shape[0])], size
    )

    def body(comm):
        return build_copy_groups(
            comm, parts[comm.Get_rank()], norb, num_slices, copies, kinds[0], kinds[1]
        )

    return run_ranks(LocalComm.create(size, timeout=120.0), body)


def check_coverage(
    keys, norb, size, copies, num_slices, kinds=(None, None), layouts=None
):
    layouts = (
        layouts
        if layouts is not None
        else run_copies(keys, norb, size, copies, num_slices, kinds)
    )
    layout = HostLayout(keys, norb, kinds[0], kinds[1])
    unique = sort_unique(keys)
    for item in layouts:
        np.testing.assert_array_equal(item.plan["work"], layouts[0].plan["work"])
        np.testing.assert_array_equal(
            item.plan["assignment"], layouts[0].plan["assignment"]
        )
        pieces = item.pieces
        assert pieces.ranges(1)[2] == pieces.num_own, (
            "remote sources must exclude the block members"
        )
        assert pieces.ranges(0)[:2] == (
            item.share.row_start - item.block_start,
            item.share.row_start - item.block_start + item.share.n_local,
        )
        halo = pieces.halo_rows()
        owner = np.searchsorted(item.block_starts, halo, side="right") - 1
        assert np.all(item.plan["assignment"][item.block][owner] == item.copy)
        np.testing.assert_array_equal(
            pieces.halo_key_words().reshape(-1, 2), unique[halo]
        )
        np.testing.assert_array_equal(
            item.block_keys,
            unique[item.block_start : item.block_start + pieces.num_own],
        )
    for cls in range(len(CLASSES)):
        union = np.concatenate(
            [item.pieces.pairs(kind, cls) for item in layouts for kind in (0, 1)]
        )
        union = union[np.lexsort((union[:, 1], union[:, 0]))]
        np.testing.assert_array_equal(
            union, layout.stored_pairs(cls), err_msg=CLASSES[cls]
        )
        count = sum(
            item.pieces.pair_digest(kind, cls)[0] for item in layouts for kind in (0, 1)
        )
        digest = sum(
            item.pieces.pair_digest(kind, cls)[1] for item in layouts for kind in (0, 1)
        ) % (1 << 64)
        assert (count, digest) == layout.pair_digest(cls)
    return layouts


@pytest.mark.parametrize(
    "size, copies", [(2, 2), (4, 2), (4, 4), (3, 3), (6, 2), (6, 3)]
)
@pytest.mark.parametrize("shape", [(10, 5, 4), (9, 2, 6)])
def test_copy_pieces_cover_every_pair_once(shape, size, copies):
    norb, alpha, beta = shape
    keys = clustered_keys(
        np.random.default_rng(size * 10 + copies + norb), norb, alpha, beta, 1200, 30
    )
    layouts = check_coverage(keys, norb, size, copies, 2)
    assert any(item.pieces.halo_rows().size for item in layouts) or size == copies


@pytest.mark.parametrize("kinds", KIND_PAIRS)
def test_copy_pieces_cover_every_pair_for_every_kind_pair(kinds):
    keys = clustered_keys(np.random.default_rng(41), 10, 5, 4, 1200, 30)
    check_coverage(keys, 10, 6, 2, 2, kinds)


def test_copy_pieces_with_own_and_two_remote_blocks():
    keys = sort_unique(clustered_keys(np.random.default_rng(43), 10, 5, 5, 1500, 20))
    starts = np.array([keys.shape[0] * 2 * r // 6 for r in range(4)])
    _, receivers = run_join(keys, 10, 3, 2, (None, None), starts=starts)
    found = False
    for rank, receiver in enumerate(receivers):
        own = receiver.num_own
        owners = np.concatenate(
            (
                np.full(own, rank),
                np.searchsorted(starts, receiver.halo_rows(), side="right") - 1,
            )
        )
        for chunk in range(receiver.num_chunks()):
            for cls in range(len(CLASSES)):
                members, sizes, _ = receiver.chunk_class(chunk, cls)
                first = 0
                for size in sizes:
                    if len(set(owners[members[first : first + size]].tolist())) == 3:
                        found = True
                    first += int(size)
    assert found, "no group spans the own block and two remote blocks"
    check_coverage(keys, 10, 6, 2, 2)


@pytest.mark.parametrize(
    "block_cut, shares",
    [
        ("membership", "equal"),
        ("fitted", "equal"),
        ("row", "balanced"),
        ("fitted", "balanced"),
        ("fitted", "own"),
    ],
)
@pytest.mark.parametrize("size, copies", [(4, 2), (6, 2), (6, 3)])
def test_planned_copy_layouts_cover_every_pair_once(size, copies, block_cut, shares):
    from fulqrum.gpu.full_strs.plan import LayoutPlan  # noqa: PLC0415

    keys = sort_unique(
        np.vstack(
            (
                clustered_keys(np.random.default_rng(51), 10, 5, 5, 1500, 25),
                clustered_keys(np.random.default_rng(52), 10, 5, 5, 300, 200),
            )
        )
    )
    plan = LayoutPlan(copies=copies, num_slices=2, block_cut=block_cut, shares=shares)
    parts = np.array_split(keys, size)
    results = run_ranks(
        LocalComm.create(size, timeout=120.0),
        lambda comm: plan_copy_groups(comm, parts[comm.Get_rank()], 10, plan),
    )
    layouts = [r[0] for r in results]
    check_coverage(keys, 10, size, copies, 2, layouts=layouts)
    measured = results[0][1]
    loads = np.array(measured["copy_loads"])
    first = np.array(measured["copy_loads_row_cut"])
    assert loads.size == size and abs(loads.sum() - first.sum()) <= 1e-9 * first.sum()
    if shares == "balanced":
        assert loads.max() / loads.mean() <= first.max() / first.mean() + 1e-12


def test_balanced_bounds():
    row_own = np.ones(100)
    assert balanced_bounds(row_own, [40.0, 0.0]) == [0, 30, 100]
    assert balanced_bounds(row_own, [0.0, 0.0, 0.0, 0.0]) == [0, 25, 50, 75, 100]
    assert balanced_bounds(row_own, [1000.0, 0.0]) == [0, 1, 100]


@pytest.mark.parametrize("size", [1, 3, 4])
def test_work_profile_matches_the_counts(size):
    from fulqrum.gpu.full_strs.distribute import block_work  # noqa: PLC0415

    keys = clustered_keys(np.random.default_rng(61 + size), 10, 5, 4, 1200, 30)
    blocks, receivers = run_join(keys, 10, size, 3, (None, None))
    for rank, (block, receiver) in enumerate(zip(blocks, receivers)):
        halo_block = (
            np.searchsorted(block.block_starts, receiver.halo_rows(), side="right") - 1
        ).astype(np.int32)
        total, own, memberships, work, entries = receiver.work_profile(
            np.ones(35), halo_block, size, rank
        )
        assert entries[rank] <= memberships.sum() and np.all(entries[work == 0] == 0)
        np.testing.assert_array_equal(memberships, output_memberships(receiver))
        np.testing.assert_array_equal(
            work, block_work(receiver, block.block_starts, rank).astype(np.float64)
        )
        assert total.sum() == work.sum()
        assert own.sum() == work[rank]
        assert np.all(own <= total)


def test_choose_copies():
    from fulqrum.gpu.full_strs.plan import choose_copies  # noqa: PLC0415

    candidates = {
        1: {"volume_max": 0.9, "work_imbalance": 1.0},
        2: {"volume_max": 0.6, "work_imbalance": 1.1},
        4: {"volume_max": 0.5, "work_imbalance": 1.6},
    }
    assert choose_copies(candidates) == 2
    candidates[4]["work_imbalance"] = 1.2
    assert choose_copies(candidates) == 4
    candidates[2]["volume_max"] = 0.5
    assert choose_copies(candidates) == 2
    assert (
        choose_copies(
            {
                1: {"volume_max": 1, "work_imbalance": 2.0},
                2: {"volume_max": 1, "work_imbalance": 1.5},
            }
        )
        == 2
    )


@pytest.mark.parametrize("size", [4, 6])
def test_plan_copies_agrees_with_the_metrics(size):
    from fulqrum.gpu.full_strs.distribute import layout_metrics, plan_copies  # noqa: PLC0415
    from fulqrum.gpu.full_strs.plan import choose_copies  # noqa: PLC0415

    keys = clustered_keys(np.random.default_rng(71), 10, 5, 4, 1200, 30)
    parts = np.array_split(np.vstack((keys, keys[:100])), size)
    results = run_ranks(
        LocalComm.create(size, timeout=300.0),
        lambda comm: plan_copies(comm, parts[comm.Get_rank()], 10, 2),
    )
    chosen, candidates = results[0]
    assert all(r == results[0] for r in results)
    assert set(candidates) == {t for t in (1, 2, 4) if size % t == 0}
    assert chosen == choose_copies(candidates)
    unique = sort_unique(keys).shape[0]
    for copies, metrics in candidates.items():
        direct = run_ranks(
            LocalComm.create(size, timeout=300.0),
            lambda comm: layout_metrics(comm, parts[comm.Get_rank()], 10, 2, copies),
        )
        assert all(d[2] == unique for d in direct)
        assert metrics["volume_max"] == max(d[0] for d in direct) / unique
        works = np.array([d[1] for d in direct])
        assert metrics["work_imbalance"] == pytest.approx(works.max() / works.mean())


@pytest.mark.parametrize("blocks", [2, 3])
def test_padded_global_rows_do_not_wrap(blocks):
    from fulqrum.gpu.full_strs.host import GroupReceiver

    receiver = GroupReceiver(np.array([[1, 1]], dtype=np.uint64), 0, 1)
    starts = np.arange(blocks + 1, dtype=np.int64)
    with pytest.raises((ValueError, OverflowError)):
        receiver.finish_global(starts, int(np.iinfo(np.int64).max))
