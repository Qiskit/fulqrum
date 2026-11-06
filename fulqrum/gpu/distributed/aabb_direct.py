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


def full_source_plan(graph, n_alpha):


    import numpy as np

    limit = np.iinfo(np.int32).max
    n_beta, n_source = int(graph["n_rows"]), int(graph["n_source"])
    if any(n < 0 or n > limit for n in (n_alpha, n_beta, n_source)):
        raise ValueError("full-source dimensions exceed nonnegative int32")
    rows = np.asarray(graph["rows"])
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or len(rows) > limit
        or len(rows) != len(graph["words"])
        or np.any(rows < 0)
        or np.any(rows >= n_beta)
        or np.any(rows[1:] < rows[:-1])
    ):
        raise ValueError("invalid sorted beta graph")
    if not n_alpha or not n_beta or not len(rows):
        return np.empty((0, 8), dtype=np.int32), np.empty((0, 3), dtype=np.int32)
    if not n_source:
        raise ValueError("nonempty beta graph has no source rows")
    counts = np.bincount(rows.astype(np.int64), minlength=n_beta)
    ptr = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    tasks = np.column_stack((np.arange(n_beta), ptr[:-1], ptr[1:])).astype(np.int32)
    jobs = np.array([[0, n_alpha, 0, 0, 0, n_source, 0, n_beta]], dtype=np.int32)
    return jobs, np.ascontiguousarray(tasks)


def _int32_array(values, name):


    import numpy as np

    a = np.asarray(values)
    if a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if a.size and (np.any(a < 0) or np.any(a > np.iinfo(np.int32).max)):
        raise ValueError(f"{name} exceeds nonnegative int32")
    return np.ascontiguousarray(a, dtype=np.int32)


def sorted_beta_graph(
    ptr, columns, p0, p1, bra_beta, num_beta, n_source, source_slots=None
):


    import numpy as np

    ptr = _int32_array(ptr, "beta pointers")
    if not ptr.size or ptr[0] != 0 or np.any(np.diff(ptr.astype(np.int64)) < 0):
        raise ValueError("invalid beta CSR pointers")
    if not 0 <= n_source <= np.iinfo(np.int32).max or not 1 <= num_beta <= 64:
        raise ValueError("invalid beta dimensions")
    nnz, n_rows = int(ptr[-1]), len(ptr) - 1
    payload = [np.asarray(a) for a in (columns, p0, p1)]
    if any(a.ndim != 1 or len(a) < nnz for a in payload):
        raise ValueError("beta payload shorter than CSR pointers")
    j, p, q = [_int32_array(a[:nnz], "beta payload") for a in payload]
    bra = np.asarray(bra_beta, dtype=np.uint64)
    if bra.ndim != 1 or len(bra) != n_rows:
        raise ValueError("beta bra shape does not match pointers")
    if np.any(p >= q) or np.any(q >= num_beta):
        raise ValueError("beta pairs must satisfy 0 <= p < q < num_beta")
    rows = np.repeat(np.arange(n_rows, dtype=np.int32), np.diff(ptr))
    if source_slots is not None:
        slots = np.asarray(source_slots)
        if slots.ndim != 1 or slots.dtype.kind not in "iu" or np.any(j >= len(slots)):
            raise ValueError("invalid beta source slot map")
        mapped = slots[j]
        if np.any(mapped < -1) or np.any(mapped >= n_source):
            raise ValueError("beta source slot outside input")
        keep = mapped >= 0
        j = mapped[keep].astype(np.int32)
        rows, p, q = rows[keep], p[keep], q[keep]
    elif np.any(j >= n_source):
        raise ValueError("beta column outside input")
    order = np.lexsort((j, rows))
    rows, j, p, q = (a[order] for a in (rows, j, p, q))
    pid = p * (2 * int(num_beta) - p - 1) // 2 + q - p - 1
    mask = ((np.uint64(1) << q.astype(np.uint64)) - np.uint64(1)) ^ (
        (np.uint64(1) << (p + 1).astype(np.uint64)) - np.uint64(1)
    )
    parity = bra[rows] & mask
    for shift in (32, 16, 8, 4, 2, 1):
        parity ^= parity >> np.uint64(shift)
    sign = np.where((parity & np.uint64(1)) != 0, 65535, 1).astype(np.uint64)
    words = (
        j.astype(np.uint64)
        | (pid.astype(np.uint64) << np.uint64(32))
        | (sign << np.uint64(48))
    )
    return {
        "rows": rows,
        "columns": j,
        "words": words,
        "n_rows": n_rows,
        "n_source": int(n_source),
    }
