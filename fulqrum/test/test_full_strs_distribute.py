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

"""Distribution of full determinant strings over in-process ranks."""

import numpy as np
import pytest

from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks
from fulqrum.gpu.full_strs.distribute import (
    distribute_keys,
    exchange_rows,
    lower_bound,
    sort_columns,
    sort_unique,
)


def random_strings(rng, count, num_orbitals, electrons, num_words):
    """Return random sector strings with a fixed electron count."""
    out = np.zeros((count, num_words), dtype=np.uint64)
    for row in range(count):
        for orbital in rng.choice(num_orbitals, size=electrons, replace=False):
            out[row, orbital // 64] |= np.uint64(1) << np.uint64(orbital % 64)
    return out


def random_keys(rng, count, num_orbitals, alpha, beta, num_words=1):
    """Return random rows (alpha words, then beta words) with fixed electron counts."""
    return np.hstack(
        (
            random_strings(rng, count, num_orbitals, alpha, num_words),
            random_strings(rng, count, num_orbitals, beta, num_words),
        )
    )


def reference(parts):
    """Return the sorted unique rows of all parts, by (beta, alpha), from Python tuples."""
    rows = {tuple(int(v) for v in row) for part in parts for row in part}
    ordered = sorted(rows, key=lambda row: row[::-1])
    width = parts[0].shape[1]
    return np.array(ordered, dtype=np.uint64).reshape(-1, width)


def run_distribution(parts, num_orbitals):
    """Run distribute_keys on one in-process rank per part and return the KeyBlock of every rank."""
    comms = LocalComm.create(len(parts), timeout=60.0)
    return run_ranks(
        comms, lambda comm: distribute_keys(comm, parts[comm.Get_rank()], num_orbitals)
    )


def check_blocks(blocks, parts):
    """Check that the blocks are contiguous, balanced and together equal the sorted unique input."""
    expected = reference(parts)
    size = len(blocks)
    total = expected.shape[0]
    starts = np.array([total * r // size for r in range(size + 1)], dtype=np.int64)
    for rank, block in enumerate(blocks):
        assert block.num_rows == total
        np.testing.assert_array_equal(block.block_starts, starts)
        assert block.row_start == starts[rank]
        assert block.n_local == starts[rank + 1] - starts[rank]
        assert block.keys.dtype == np.uint64 and block.keys.flags.c_contiguous
        np.testing.assert_array_equal(
            block.keys, expected[starts[rank] : starts[rank + 1]]
        )
        np.testing.assert_array_equal(
            block.owner_of(np.arange(total)),
            np.searchsorted(starts, np.arange(total), side="right") - 1,
        )


@pytest.mark.parametrize("num_words", [1, 2])
def test_sort_unique_and_lower_bound(num_words):
    """Local sort and unique follow the (beta, alpha) order; lower_bound equals a linear search."""
    rng = np.random.default_rng(3)
    norb = 70 if num_words == 2 else 9
    keys = random_keys(rng, 400, norb, 3, 2, num_words)
    keys = np.vstack((keys, keys[:150]))
    rng.shuffle(keys)
    unique = sort_unique(keys)
    np.testing.assert_array_equal(unique, reference([keys]))
    columns = np.ascontiguousarray(sort_columns(unique))
    tuples = [tuple(row) for row in columns.T]
    probes = [tuples[0], tuples[-1], tuples[len(tuples) // 2]]
    probes += [
        tuple(row)
        for row in sort_columns(random_keys(rng, 30, norb, 3, 2, num_words)).T
    ]
    probes += [
        tuple(np.uint64(0) for _ in tuples[0]),
        tuple(~np.uint64(0) for _ in tuples[0]),
    ]
    for probe in probes:
        expected = next(
            (i for i, row in enumerate(tuples) if row >= probe), len(tuples)
        )
        assert lower_bound(columns, probe) == expected


@pytest.mark.parametrize("size", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint64, np.int64, np.int32])
def test_exchange_rows(size, dtype):
    """Every rank receives the row ranges that the other ranks address to it, in source-rank order."""
    rng = np.random.default_rng(size)
    width = 3
    plans = [rng.integers(0, 5, size=size) for _ in range(size)]
    for plan in plans:
        plan[rng.integers(0, size)] = 0
    sends = [
        np.arange(int(plan.sum()) * width, dtype=dtype).reshape(-1, width) + 1000 * rank
        for rank, plan in enumerate(plans)
    ]

    def body(comm):
        """Exchange the rows of this rank."""
        rank = comm.Get_rank()
        return exchange_rows(comm, sends[rank], plans[rank])

    results = run_ranks(LocalComm.create(size, timeout=60.0), body)
    for rank, (recv, counts) in enumerate(results):
        pieces = []
        for source in range(size):
            begin = int(plans[source][:rank].sum())
            pieces.append(sends[source][begin : begin + int(plans[source][rank])])
        np.testing.assert_array_equal(counts, [plans[s][rank] for s in range(size)])
        assert recv.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(recv, np.concatenate(pieces).reshape(-1, width))


def test_exchange_rows_rejects_bad_counts():
    """A count sum that differs from the row count raises on every rank."""

    def body(comm):
        """Give rank 1 a wrong count."""
        counts = [1, 1] if comm.Get_rank() == 0 else [1, 2]
        return exchange_rows(comm, np.zeros((2, 2), dtype=np.uint64), counts)

    with pytest.raises(ValueError, match="rank 1: send counts"):
        run_ranks(LocalComm.create(2, timeout=60.0), body)


@pytest.mark.parametrize("size", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(10, 5, 4), (12, 3, 6), (64, 2, 3)])
def test_distribute_random(size, shape):
    """Random rows with duplicates inside and between ranks."""
    rng = np.random.default_rng(size * 100 + shape[0])
    norb, alpha, beta = shape
    common = random_keys(rng, 200, norb, alpha, beta)
    parts = []
    for _ in range(size):
        own = random_keys(rng, int(rng.integers(300, 900)), norb, alpha, beta)
        part = np.vstack((own, own[:50], common))
        rng.shuffle(part)
        parts.append(part)
    check_blocks(run_distribution(parts, norb), parts)


@pytest.mark.parametrize("size", [2, 3, 4])
def test_distribute_uneven_inputs(size):
    """All rows on one rank, the other ranks empty; then a single empty rank."""
    rng = np.random.default_rng(11)
    keys = random_keys(rng, 1000, 10, 5, 5)
    parts = [np.zeros((0, 2), dtype=np.uint64) for _ in range(size)]
    parts[size - 1] = keys
    check_blocks(run_distribution(parts, 10), parts)
    parts = [random_keys(rng, 500, 10, 5, 5) for _ in range(size)]
    parts[0] = np.zeros((0, 2), dtype=np.uint64)
    check_blocks(run_distribution(parts, 10), parts)


@pytest.mark.parametrize("size", [2, 3, 4])
def test_distribute_heavy_duplicates(size):
    """Few distinct rows repeated many times on every rank: equal splitters, one owner per row."""
    rng = np.random.default_rng(5)
    distinct = sort_unique(random_keys(rng, 40, 8, 4, 4))[: size + 2]
    parts = [
        distinct[rng.integers(0, distinct.shape[0], size=3000)] for _ in range(size)
    ]
    check_blocks(run_distribution(parts, 8), parts)


def test_distribute_word_boundary_and_two_words():
    """Orbital 63 is used with W = 1; W = 2 rows on 100 orbitals."""
    rng = np.random.default_rng(9)
    keys = random_keys(rng, 600, 63, 3, 3)
    keys |= np.uint64(1) << np.uint64(63)
    parts = np.array_split(keys, 3)
    check_blocks(run_distribution(parts, 64), parts)
    wide = random_keys(rng, 900, 100, 4, 4, num_words=2)
    parts = np.array_split(np.vstack((wide, wide[:100])), 4)
    check_blocks(run_distribution(parts, 100), parts)


def test_distribute_fewer_rows_than_ranks():
    """Two unique rows on three ranks raise on every rank."""
    keys = np.array([[0b011, 0b001], [0b101, 0b001]], dtype=np.uint64)
    parts = [keys, keys, keys]
    with pytest.raises(ValueError, match="2 unique rows cannot fill 3 ranks"):
        run_distribution(parts, 3)


@pytest.mark.parametrize(
    "case",
    [
        "outside",
        "electrons_local",
        "electrons_ranks",
        "width",
        "orbitals",
        "empty",
        "shape",
    ],
)
def test_distribute_refusals(case):
    """Bad input on one rank raises ValueError on every rank, with the reason."""
    good = np.array(
        [[0b0011, 0b0101], [0b0110, 0b0101], [0b1001, 0b0011]], dtype=np.uint64
    )
    parts = [good.copy(), good.copy(), good.copy()]
    orbitals = [4, 4, 4]
    match = {
        "outside": "rank 1: row 2 has a bit at or above orbital 4",
        "electrons_local": "rank 2: row 1 has 3 alpha",
        "electrons_ranks": "different \\(alpha, beta\\) electron counts",
        "width": "different widths",
        "orbitals": "different orbital counts",
        "empty": "no rows on any of the 3 ranks",
        "shape": "rank 0: keys must have shape",
    }[case]
    if case == "outside":
        parts[1][2, 1] = np.uint64(0b10001)
    elif case == "electrons_local":
        parts[2][1, 0] = np.uint64(0b0111)
    elif case == "electrons_ranks":
        parts[1] = np.array([[0b0111, 0b0101]], dtype=np.uint64)
    elif case == "width":
        zeros = np.zeros((3, 1), dtype=np.uint64)
        parts[0] = np.hstack((good[:, :1], zeros, good[:, 1:], zeros))
    elif case == "orbitals":
        orbitals[2] = 5
    elif case == "empty":
        parts = [np.zeros((0, 2), dtype=np.uint64)] * 3
    elif case == "shape":
        parts[0] = good.reshape(-1)

    def body(comm):
        """Distribute the part of this rank."""
        rank = comm.Get_rank()
        return distribute_keys(comm, parts[rank], orbitals[rank])

    with pytest.raises(ValueError, match=match):
        run_ranks(LocalComm.create(3, timeout=60.0), body)
