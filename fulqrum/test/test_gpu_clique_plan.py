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


import pytest

from fulqrum.gpu.distributed.aabb_parents import (
    CLIQUE_MAX_SHARED_BYTES,
    clique_shared_bytes,
    validate_clique_plan,
    clique_size_buckets,
)


def valid_plan(transposed):
    import numpy as np

    sizes = np.array([3, 8, 9, 20, 32])
    _, start = clique_size_buckets(np.r_[0, np.cumsum(sizes)].astype(np.int32))
    return dict(
        transposed=transposed,
        nparents=len(sizes),
        residual_edges=7 if transposed else 4,
        n_beta_edges=7 if transposed else 50,
        n_rows=130,
        orbitals=12,
        n_b_src=40,
        n_b_out=40,
        tile_rows=64,
        eb=4,
        max_shared_bytes=CLIQUE_MAX_SHARED_BYTES,
        grouped_edges=dict(
            bucket_start=[int(x) for x in start],
            ntiles=3,
            ngroups=9,
            nwords=7 if transposed else 50,
        ),
    )


@pytest.mark.parametrize("transposed", [False, True])
def test_a_fitting_plan_passes(transposed):
    validate_clique_plan(**valid_plan(transposed))
    validate_clique_plan(**dict(valid_plan(transposed), grouped_edges=None))


BROKEN = [
    ("no parents", False, dict(nparents=0)),
    ("negative residual", False, dict(residual_edges=-1)),
    ("orbitals", False, dict(orbitals=65)),
    ("orbitals", True, dict(orbitals=1)),
    ("residual edges", True, dict(residual_edges=6)),
    ("source beta count", True, dict(n_b_src=39)),
    ("no beta edges", False, dict(n_beta_edges=0, grouped_edges=None)),
    ("row tile of", False, dict(tile_rows=0)),
    ("edges per batch", False, dict(eb=6)),
    ("do not cover", False, dict(n_rows=200)),
    ("do not cover", True, dict(n_rows=128)),
    ("no edge groups", False, dict(grouped_edges="ngroups=0")),
    ("no tile words", True, dict(grouped_edges="nwords=0")),
    ("tile words for", False, dict(grouped_edges="nwords=49")),
    ("do not span", False, dict(grouped_edges="bucket_start=short")),
    ("do not span", False, dict(grouped_edges="bucket_start=total")),
    ("decrease", False, dict(grouped_edges="bucket_start=decrease")),
    ("shared memory", False, dict(tile_rows=300, n_rows=600, grouped_edges="ntiles=2")),
    ("more than the 65536 bytes of the device", False, dict(max_shared_bytes=65536)),
]


@pytest.mark.parametrize("message,transposed,change", BROKEN)
def test_each_broken_invariant_raises(message, transposed, change):
    args = valid_plan(transposed)
    clique_change = (
        change.pop("grouped_edges", None)
        if isinstance(change.get("grouped_edges"), str)
        else None
    )
    args.update(change)
    if clique_change is not None:
        key, value = clique_change.split("=")
        grouped_edges = dict(args["grouped_edges"])
        start = list(grouped_edges["bucket_start"])
        if key == "bucket_start":
            grouped_edges["bucket_start"] = {
                "short": start[:-1],
                "total": start[:-1] + [start[-1] + 1],
                "decrease": [0, 3, 2] + start[3:],
            }[value]
        else:
            grouped_edges[key] = int(value)
        args["grouped_edges"] = grouped_edges
    with pytest.raises(ValueError, match=message):
        validate_clique_plan(**args)


def test_shared_memory_rule_matches_the_tile_size_limit():
    assert clique_shared_bytes(32, 8, 219) == 232448 <= CLIQUE_MAX_SHARED_BYTES
    assert clique_shared_bytes(32, 8, 220) > CLIQUE_MAX_SHARED_BYTES
