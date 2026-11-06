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

"""Stored groups of the full-string GPU path against the O(N^2) pair sets."""

import itertools
import os
from pathlib import Path

import numpy as np
import pytest

from fulqrum.gpu.full_strs.host import CLASSES, KINDS, HostLayout

SETS = Path(os.environ.get("FQ_FULL_STRS_SETS", "/nonexistent"))
SMALL_SETS = sorted((SETS / "small").glob("*.bin")) if SETS.exists() else []
SUBSAMPLES = [SETS / "sub_head.bin", SETS / "sub_uniform.bin"] if SETS.exists() else []


def read_set(path):
    """Return the keys and the number of orbitals of a set file."""
    raw = np.fromfile(path, dtype=np.uint64)
    count, norb = int(raw[0]), int(raw[1])
    return raw[2:].reshape(count, 2), norb


def check_pair_sets(keys, norb, alpha_kind, beta_kind):
    """Assert that the stored groups give every connected ordered pair exactly once."""
    layout = HostLayout(keys, norb, alpha_kind, beta_kind)
    assert layout.kinds == (alpha_kind, beta_kind)
    total = 0
    for cls in range(len(CLASSES)):
        stored = layout.stored_pairs(cls)
        scanned = layout.scanned_pairs(cls)
        assert stored.shape == scanned.shape, CLASSES[cls]
        assert np.array_equal(stored, scanned), CLASSES[cls]
        total += scanned.shape[0]
    stats = layout.stats()
    assert sum(entry["pairs"] for entry in stats.values()) == total
    return total


def test_default_kinds_follow_holes_and_electrons():
    rng = np.random.default_rng(1)
    alpha = np.array(
        [
            sum(1 << int(k) for k in rng.choice(36, 27, replace=False))
            for _ in range(50)
        ],
        dtype=np.uint64,
    )
    beta = np.array(
        [sum(1 << int(k) for k in rng.choice(36, 9, replace=False)) for _ in range(50)],
        dtype=np.uint64,
    )
    assert HostLayout(np.stack([alpha, beta], axis=1), 36).kinds == (
        "union",
        "intersection",
    )


@pytest.mark.long
@pytest.mark.skipif(
    not SMALL_SETS, reason="set FQ_FULL_STRS_SETS to the generated test sets"
)
@pytest.mark.parametrize(
    "path", SMALL_SETS + SUBSAMPLES, ids=[p.stem for p in SMALL_SETS + SUBSAMPLES]
)
@pytest.mark.parametrize("alpha_kind, beta_kind", list(itertools.product(KINDS, KINDS)))
def test_every_connected_pair_exactly_once(path, alpha_kind, beta_kind):
    keys, norb = read_set(path)
    assert check_pair_sets(keys, norb, alpha_kind, beta_kind) > 0
