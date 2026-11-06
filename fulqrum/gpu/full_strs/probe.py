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


import time

import cupy
import numpy as np

from . import kernel_launch
from .device import NUM_CLASSES, DeviceLayout
from .host import HostOperator

NUM_BUCKETS = 7


def kernel_coefficients(operator, keys, num_slices=1, repeats=3):


    host = HostOperator(operator, keys, num_slices=int(num_slices))
    layout = DeviceLayout(host, "resident")
    stream = cupy.cuda.Stream(non_blocking=True)
    layout.wait(stream)
    x = cupy.asarray(np.random.default_rng(0).normal(size=host.num_rows))
    pointers = layout.table_pointers()
    seconds = np.zeros((NUM_CLASSES, NUM_BUCKETS))
    reads = np.zeros((NUM_CLASSES, NUM_BUCKETS))
    launches = np.zeros((NUM_CLASSES, NUM_BUCKETS))
    overhead = _launch_overhead(layout, x, stream, pointers, repeats)
    entry_seconds, entries = 0.0, 0
    for chunk in range(len(layout.views)):
        entry_seconds += _time_entries(layout, chunk, stream, repeats)
        entries += sum(view.num_members for view in layout.chunks[chunk].classes)
        classes = layout.chunks[chunk].classes
        for cls in range(NUM_CLASSES):
            view = classes[cls]
            sizes = layout.views[chunk][cls][1].astype(np.float64)
            for bucket in range(NUM_BUCKETS):
                first, last = view.bucket_start[bucket], view.bucket_start[bucket + 1]
                if last == first:
                    continue
                reads[cls, bucket] += float(
                    (sizes[first:last] * (sizes[first:last] - 1)).sum()
                )
                launches[cls, bucket] += 1
                seconds[cls, bucket] += _time_launch(
                    layout,
                    view,
                    cls,
                    bucket,
                    first,
                    last - first,
                    x,
                    stream,
                    pointers,
                    repeats,
                )
    measured = reads > 0
    coefficients = np.zeros((NUM_CLASSES, NUM_BUCKETS))
    coefficients[measured] = (
        seconds[measured] - overhead * launches[measured]
    ) / reads[measured]
    positive = coefficients[measured & (coefficients > 0)]
    floor = float(positive.min()) if positive.size else 1e-12
    coefficients[measured] = np.maximum(coefficients[measured], floor)
    fallback = float(coefficients[measured].mean()) if measured.any() else 1.0
    for cls in range(NUM_CLASSES):
        known = np.flatnonzero(measured[cls])
        for bucket in range(NUM_BUCKETS):
            if not measured[cls, bucket]:
                coefficients[cls, bucket] = (
                    coefficients[cls, known[np.argmin(np.abs(known - bucket))]]
                    if known.size
                    else fallback
                )
    return {
        "coefficients": coefficients.tolist(),
        "entry_cost": entry_seconds / max(entries, 1),
        "launch_overhead": overhead,
        "seconds": seconds.tolist(),
        "reads": reads.tolist(),
        "launches": launches.tolist(),
        "rows": int(host.num_rows),
    }


def _launch_overhead(layout, x, stream, pointers, repeats):

    for chunk in range(len(layout.views)):
        for cls, view in enumerate(layout.chunks[chunk].classes):
            if view.bucket_start[1] > view.bucket_start[0]:
                return _time_launch(
                    layout,
                    view,
                    cls,
                    0,
                    view.bucket_start[0],
                    1,
                    x,
                    stream,
                    pointers,
                    repeats,
                )
    return 0.0


def _time_entries(layout, chunk, stream, repeats):

    classes = layout.chunks[chunk].classes
    y = cupy.zeros(layout.num_own, dtype=np.float64)
    best = None
    for repeat in range(repeats + 1):
        start, end = cupy.cuda.Event(), cupy.cuda.Event()
        start.record(stream)
        offset = 0
        for view in classes:
            if view.num_members:
                cupy.cuda.runtime.memcpyAsync(
                    layout.entry_rows.data.ptr + 4 * offset,
                    view.members,
                    4 * view.num_members,
                    cupy.cuda.runtime.memcpyDeviceToDevice,
                    stream.ptr,
                )
                offset += view.num_members
        kernel_launch.launch_reduce(
            layout.entry_rows.data.ptr,
            layout.entry_rows_alt.data.ptr,
            layout.entry_values.data.ptr,
            layout.entry_values_alt.data.ptr,
            layout.reduce_scratch.data.ptr,
            layout.reduce_scratch.nbytes,
            offset,
            layout.end_bit,
            0,
            layout.num_own,
            y.data.ptr,
            stream.ptr,
        )
        end.record(stream)
        end.synchronize()
        elapsed = cupy.cuda.get_elapsed_time(start, end) / 1000.0
        if repeat > 0:
            best = elapsed if best is None else min(best, elapsed)
    return best


def _time_launch(layout, view, cls, bucket, first, count, x, stream, pointers, repeats):

    halo = layout.halo_keys.data.ptr if layout.halo_keys.size else 0
    args = (
        layout.keys.data.ptr,
        halo,
        layout.num_own,
        pointers,
        view.members,
        view.offsets,
        layout.partials.data.ptr,
    )
    ranges = (0, layout.num_own, 0, layout.num_rows)
    best = None
    for repeat in range(repeats + 1):
        start, end = cupy.cuda.Event(), cupy.cuda.Event()
        start.record(stream)
        kernel_launch.launch_groups_binned(
            *args, first, count, cls, bucket, ranges, x.data.ptr, stream.ptr
        )
        end.record(stream)
        end.synchronize()
        elapsed = cupy.cuda.get_elapsed_time(start, end) / 1000.0
        if repeat > 0:
            best = elapsed if best is None else min(best, elapsed)
    return best


def machine(comm):


    free, _ = cupy.cuda.runtime.memGetInfo()
    data = np.ones(8 << 20)
    device = cupy.empty_like(data)
    device.set(data)
    cupy.cuda.runtime.deviceSynchronize()
    begin = time.perf_counter()
    device.set(data)
    cupy.cuda.runtime.deviceSynchronize()
    bandwidth = data.nbytes / max(time.perf_counter() - begin, 1e-9)
    available = 0
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except OSError:
        available = 0
    values = comm.allgather((int(free), float(bandwidth), int(available)))
    return {
        "device_free_bytes": min(v[0] for v in values),
        "host_to_device_bytes_per_s": min(v[1] for v in values),
        "host_available_bytes": min(v[2] for v in values),
    }
