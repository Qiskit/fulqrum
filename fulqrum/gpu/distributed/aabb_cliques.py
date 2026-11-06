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


def _tasks_from_rows(rows, n_rows):

    import numpy as np

    rows = np.asarray(rows, dtype=np.int64)
    counts = (
        np.bincount(rows, minlength=n_rows) if len(rows) else np.zeros(n_rows, np.int64)
    )
    ptr = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    if ptr[-1] > np.iinfo(np.int32).max:
        raise OverflowError("beta words exceed int32")
    return np.column_stack((np.arange(n_rows), ptr[:-1], ptr[1:])).astype(np.int32)


def beta_clique_plan(beta_strings, graph, num_beta, min_size, kind="auto"):


    import numpy as np

    from .aabb_parents import _residual_sell, choose_parent_plan, unpack_sell

    n_rows = int(graph["n_rows"])
    if int(graph["n_source"]) != n_rows or len(beta_strings) != n_rows:
        raise ValueError("beta clique plan needs a square self beta graph")
    rows = np.asarray(graph["rows"], dtype=np.int64)
    words = np.asarray(graph["words"], dtype=np.uint64)
    off, lengths, pack = _residual_sell(rows, words, n_rows)
    plan = choose_parent_plan(
        np.asarray(beta_strings, dtype=np.uint64),
        off,
        lengths,
        pack,
        int(num_beta),
        min_parent=int(min_size),
        kind=kind,
    )
    vptr = plan["parent_ptr"]
    sizes = np.diff(vptr.astype(np.int64))
    order = np.argsort(sizes, kind="stable")
    nv_small = int(np.count_nonzero(sizes <= 16))
    rrows, rwords = unpack_sell(
        plan["residual_off"], plan["residual_lengths"], plan["residual_pack"]
    )
    return dict(
        vptr=vptr,
        vmembers=plan["members"],
        vids=order.astype(np.int32),
        nv_small=nv_small,
        nv_large=int(len(sizes) - nv_small),
        vres_words=np.ascontiguousarray(rwords, dtype=np.uint64),
        vres_tasks=_tasks_from_rows(rrows, n_rows),
        stats=plan["stats"],
    )
