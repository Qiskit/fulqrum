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


import cupy
import numpy as np

from . import kernel_launch

TABLE_ARRAYS = (
    "single_value",
    "single_delta",
    "single_reference",
    "double_value",
    "mixed_value",
)
NUM_CLASSES = 5
RESIDENCIES = ("auto", "resident", "streamed")
DEVICE_RESERVE_BYTES = 512 << 20
NUM_SLOTS = 2
MAX_UNIT_ENTRIES = (1 << 31) - 1


COUNT_PIECE = 1 << 26


def _count_in_range(members, low, high):

    count = 0
    for first in range(0, int(members.size), COUNT_PIECE):
        part = members[first : first + COUNT_PIECE]
        count += (
            int(np.count_nonzero(part < high))
            if low <= 0
            else int(np.count_nonzero((part >= low) & (part < high)))
        )
    return count


class ChunkClassView:


    def __init__(self, members, offsets, num_members, bucket_start):
        self.members = members
        self.offsets = offsets
        self.num_members = int(num_members)
        self.bucket_start = [int(value) for value in bucket_start]

    @property
    def num_groups(self):

        return self.bucket_start[-1]


class ResidentChunk:


    def __init__(self, views, stream, scratch):
        self.arrays = []
        self.classes = []
        for members, sizes, bucket_start in views:
            d_members = cupy.empty(max(1, members.size), dtype=np.int32)
            d_sizes = cupy.empty(max(1, sizes.size), dtype=np.uint16)
            d_offsets = cupy.empty(sizes.size + 1, dtype=np.int32)
            if members.size:
                d_members[: members.size].set(members, stream=stream)
            if sizes.size:
                d_sizes[: sizes.size].set(sizes, stream=stream)
            kernel_launch.launch_group_offsets(
                d_sizes.data.ptr,
                sizes.size,
                d_offsets.data.ptr,
                scratch.data.ptr,
                scratch.nbytes,
                stream.ptr,
            )
            self.arrays.append((d_members, d_sizes, d_offsets))
            self.classes.append(
                ChunkClassView(
                    d_members.data.ptr, d_offsets.data.ptr, members.size, bucket_start
                )
            )

    def nbytes(self):

        return sum(array.nbytes for arrays in self.arrays for array in arrays)


class StreamSlot:


    def __init__(self, max_members, max_groups, scratch_bytes):
        self.members = cupy.empty(max(1, max_members), dtype=np.int32)
        self.sizes = cupy.empty(max(1, max_groups), dtype=np.uint16)
        self.offsets = cupy.empty(max_groups + NUM_CLASSES, dtype=np.int32)
        self.scratch = cupy.empty(max(1, scratch_bytes), dtype=np.uint8)
        self.released = None

    def load(self, views, stream):


        classes = []
        member_offset = 0
        group_offset = 0
        for cls, (members, sizes, bucket_start) in enumerate(views):
            members_ptr = self.members.data.ptr + 4 * member_offset
            sizes_ptr = self.sizes.data.ptr + 2 * group_offset
            offsets_ptr = self.offsets.data.ptr + 4 * (group_offset + cls)
            if members.size:
                cupy.cuda.runtime.memcpyAsync(
                    members_ptr,
                    members.ctypes.data,
                    members.nbytes,
                    cupy.cuda.runtime.memcpyHostToDevice,
                    stream.ptr,
                )
            if sizes.size:
                cupy.cuda.runtime.memcpyAsync(
                    sizes_ptr,
                    sizes.ctypes.data,
                    sizes.nbytes,
                    cupy.cuda.runtime.memcpyHostToDevice,
                    stream.ptr,
                )
            kernel_launch.launch_group_offsets(
                sizes_ptr,
                sizes.size,
                offsets_ptr,
                self.scratch.data.ptr,
                self.scratch.nbytes,
                stream.ptr,
            )
            classes.append(
                ChunkClassView(members_ptr, offsets_ptr, members.size, bucket_start)
            )
            member_offset += members.size
            group_offset += sizes.size
        return classes

    def nbytes(self):

        return (
            self.members.nbytes
            + self.sizes.nbytes
            + self.offsets.nbytes
            + self.scratch.nbytes
        )


class DeviceLayout:


    def __init__(
        self,
        host,
        residency="auto",
        device_budget_bytes=None,
        fixed_reduce=True,
        reserve_elements=0,
        atomic_reduce=False,
        device_keys=None,
    ):
        if host.num_chunks == 0:
            raise ValueError(
                "The host build has no chunks; build it with num_slices >= 1"
            )
        if residency not in RESIDENCIES:
            raise ValueError(
                f"residency must be one of {RESIDENCIES}; got {residency!r}"
            )
        self.host = host
        self.num_rows = host.num_rows
        self.num_own = int(getattr(host, "num_own", host.num_rows))
        self.num_orbitals = host.num_orbitals
        self.atomic_reduce = bool(atomic_reduce)
        if self.atomic_reduce:
            fixed_reduce = False
        self.upload_stream = cupy.cuda.Stream(non_blocking=True)
        self.copy_stream = cupy.cuda.Stream(non_blocking=True)
        if device_budget_bytes is None:
            free, _ = cupy.cuda.runtime.memGetInfo()
            device_budget_bytes = (
                free
                - DEVICE_RESERVE_BYTES
                - 8 * self.num_rows
                - 8 * self.num_own
                - 8 * int(reserve_elements)
            )
        if hasattr(host, "chunk_kinds"):
            self.kinds = [tuple(int(v) for v in kind) for kind in host.chunk_kinds()]
            self.views = [
                [host.chunk_views(chunk, cls, kind) for cls in range(NUM_CLASSES)]
                for chunk in range(host.num_chunks)
                for kind in range(len(self.kinds))
            ]
        else:
            self.kinds = [(0, self.num_own, 0, self.num_rows, 0, self.num_own)]
            self.views = [
                [host.chunk_views(chunk, cls) for cls in range(NUM_CLASSES)]
                for chunk in range(host.num_chunks)
            ]
        self.unit_kind = [unit % len(self.kinds) for unit in range(len(self.views))]
        num_units = len(self.views)
        chunk_members = [sum(view[0].size for view in chunk) for chunk in self.views]
        chunk_groups = [sum(view[1].size for view in chunk) for chunk in self.views]
        self.num_entries = int(
            sum(chunk_members)
        )
        largest_class_groups = max(
            view[1].size for chunk in self.views for view in chunk
        )
        self.largest_class_members = max(
            view[0].size for chunk in self.views for view in chunk
        )
        largest_chunk = max(chunk_members)
        if largest_chunk >= MAX_UNIT_ENTRIES:
            raise ValueError(
                f"A chunk holds {largest_chunk} member entries, the reduce supports fewer than 2^31; "
                f"use more key slices"
            )
        self.end_bit = max(1, int(self.num_rows - 1).bit_length())
        scan_bytes = kernel_launch.group_offsets_scratch_bytes(largest_class_groups)
        keys = host.key_words() if device_keys is None else device_keys
        halo_keys = (
            host.halo_key_words()
            if hasattr(host, "halo_key_words") and device_keys is None
            else np.zeros(0, dtype=np.uint64)
        )

        self.key_rows = (
            self.num_own if device_keys is None else int(device_keys.size) // 2
        )
        diagonal = host.diagonal_view()
        tables = host.table_views()
        fixed = (
            keys.nbytes
            + halo_keys.nbytes
            + diagonal.nbytes
            + sum(tables[name].nbytes for name in TABLE_ARRAYS)
        )
        fixed += scan_bytes
        if not self.atomic_reduce:
            fixed += 8 * largest_chunk
        if not fixed_reduce and not self.atomic_reduce:
            fixed += self._sort_bytes(largest_chunk)
        resident_bytes = [
            4 * members + 6 * groups + 4 * NUM_CLASSES
            for members, groups in zip(chunk_members, chunk_groups)
        ]

        plans = sum(self._planned_plan_bytes()) if fixed_reduce else 0
        self.planned_fixed_bytes = fixed + plans
        self.planned_chunk_bytes = resident_bytes
        self.planned_scan_bytes = scan_bytes
        self.planned_stream_bytes = self._stream_bytes(
            max(chunk_members), max(chunk_groups), scan_bytes
        )
        self.resident = [False] * num_units
        if residency == "resident":
            self.resident = [True] * num_units
        elif residency == "auto":
            budget = device_budget_bytes - fixed - plans
            for chunk in range(num_units):
                later = [c for c in range(chunk + 1, num_units)]
                reserve = (
                    self._stream_bytes(
                        max(chunk_members[c] for c in later),
                        max(chunk_groups[c] for c in later),
                        scan_bytes,
                    )
                    if later
                    else 0
                )
                if resident_bytes[chunk] + reserve <= budget:
                    self.resident[chunk] = True
                    budget -= resident_bytes[chunk]
                else:
                    break
        streamed = [chunk for chunk in range(num_units) if not self.resident[chunk]]
        slot_members = max([chunk_members[chunk] for chunk in streamed] or [0])
        slot_groups = max([chunk_groups[chunk] for chunk in streamed] or [0])
        needed = fixed + sum(
            resident_bytes[c] for c in range(num_units) if self.resident[c]
        )
        if streamed:
            needed += NUM_SLOTS * (4 * slot_members + 6 * slot_groups + scan_bytes)
        if residency == "auto" and needed + plans > device_budget_bytes:
            raise ValueError(
                f"The full-string layout needs {needed + plans} device bytes, the budget is "
                f"{device_budget_bytes}; use more key slices"
            )
        self._plan_budget = device_budget_bytes - needed
        if device_keys is None:
            self.keys = cupy.empty(keys.size, dtype=np.uint64)
            self.keys.set(keys, stream=self.upload_stream)
        else:
            self.keys = device_keys
        self.halo_keys = cupy.empty(halo_keys.size, dtype=np.uint64)
        self.diagonal = cupy.empty(diagonal.size, dtype=np.float64)
        if halo_keys.size:
            self.halo_keys.set(halo_keys, stream=self.upload_stream)
        self.diagonal.set(diagonal, stream=self.upload_stream)
        self.tables = {}
        for name in TABLE_ARRAYS:
            self.tables[name] = cupy.empty(max(1, tables[name].size), dtype=np.float64)
            if tables[name].size:
                self.tables[name][: tables[name].size].set(
                    tables[name], stream=self.upload_stream
                )
        self.reference = cupy.empty(2, dtype=np.uint64)
        self.reference.set(
            np.array(tables["reference"], dtype=np.uint64), stream=self.upload_stream
        )
        self.num_pairs = tables["num_pairs"]
        self.num_quads = tables["num_quads"]
        scratch = cupy.empty(max(1, scan_bytes), dtype=np.uint8)
        self.chunks = [
            ResidentChunk(views, self.upload_stream, scratch)
            if self.resident[chunk]
            else None
            for chunk, views in enumerate(self.views)
        ]
        self.slots = (
            [
                StreamSlot(slot_members, slot_groups, scan_bytes)
                for _ in range(NUM_SLOTS)
            ]
            if streamed
            else []
        )
        self.partials = None
        self.entry_values = cupy.empty(
            1 if self.atomic_reduce else max(1, largest_chunk), dtype=np.float64
        )
        self.uploaded = self.upload_stream.record()
        self.upload_stream.synchronize()
        self.reduce_plans = [None] * len(self.views)
        if fixed_reduce:
            self._build_reduce_plans()

            cupy.get_default_memory_pool().free_all_blocks()

        sorting = (
            0
            if self.atomic_reduce
            else max(
                [
                    chunk_members[unit]
                    for unit in range(num_units)
                    if self.reduce_plans[unit] is None
                ]
                or [0]
            )
        )
        self.entry_rows = cupy.empty(max(1, sorting), dtype=np.int32)
        self.entry_rows_alt = cupy.empty(max(1, sorting), dtype=np.int32)
        self.entry_values_alt = cupy.empty(max(1, sorting), dtype=np.float64)
        self.reduce_scratch = cupy.empty(
            max(
                1,
                kernel_launch.reduce_scratch_bytes(sorting, self.end_bit)
                if sorting
                else 1,
            ),
            dtype=np.uint8,
        )

    def _sort_bytes(self, members):

        return (
            16 * members + kernel_launch.reduce_scratch_bytes(members, self.end_bit)
            if members
            else 0
        )

    @staticmethod
    def _stream_bytes(members, groups, scan_bytes):


        return NUM_SLOTS * (4 * members + 6 * groups + scan_bytes)

    def _planned_plan_bytes(self):


        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        jobs = [
            (
                unit,
                members,
                self.kinds[self.unit_kind[unit]][4],
                self.kinds[self.unit_kind[unit]][5],
            )
            for unit, views in enumerate(self.views)
            for members, _, _ in views
        ]
        counts = [0] * len(self.views)
        with ThreadPoolExecutor(max_workers=max(1, min(16, len(jobs)))) as pool:
            for unit, count in zip(
                (job[0] for job in jobs),
                pool.map(lambda job: _count_in_range(*job[1:]), jobs),
            ):
                counts[unit] += count
        return [
            4 * count + 8 * min(count, kind[5] - kind[4]) + 4
            for count, kind in zip(
                counts,
                (self.kinds[self.unit_kind[unit]] for unit in range(len(self.views))),
            )
        ]

    def _unit_members(self, unit):

        if self.resident[unit]:
            return [
                d_members[: view.num_members]
                for view, (d_members, _, _) in zip(
                    self.chunks[unit].classes, self.chunks[unit].arrays
                )
            ]
        slot = self.slots[0]
        slot.load(self.views[unit], self.upload_stream)
        arrays, offset = [], 0
        for members, _, _ in self.views[unit]:
            arrays.append(slot.members[offset : offset + members.size])
            offset += members.size
        return arrays

    @staticmethod
    def _reduce_entries(member_arrays, low, high):


        rows, positions = [], []
        base = 0
        for members in member_arrays:
            if members.size:
                keep = cupy.flatnonzero((members >= low) & (members < high))
                rows.append(members[keep])
                positions.append(keep.astype(np.int32) + np.int32(base))
                del keep
            base += members.size
        if not rows:
            return cupy.zeros(0, dtype=np.int32), cupy.zeros(0, dtype=np.int32)
        return cupy.concatenate(rows), cupy.concatenate(positions)

    def _build_reduce_plans(self):


        stream = self.upload_stream
        with stream:
            num_runs = cupy.zeros(1, dtype=np.int32)
        for unit in range(len(self.views)):
            kind = self.kinds[self.unit_kind[unit]]
            with stream:
                rows, positions = self._reduce_entries(
                    self._unit_members(unit), kind[4], kind[5]
                )
                count = int(rows.size)
                rows_alt, order, order_alt, sorted_order, run_rows, run_counts = (
                    cupy.empty(max(1, count), dtype=np.int32) for _ in range(6)
                )
                scratch_bytes = (
                    kernel_launch.reduce_plan_scratch_bytes(count, self.end_bit)
                    if count
                    else 1
                )
                scratch = cupy.empty(max(1, scratch_bytes), dtype=np.uint8)
            kernel_launch.build_reduce_plan(
                rows.data.ptr,
                rows_alt.data.ptr,
                order.data.ptr,
                order_alt.data.ptr,
                scratch.data.ptr,
                scratch.nbytes,
                count,
                self.end_bit,
                sorted_order.data.ptr,
                run_rows.data.ptr,
                run_counts.data.ptr,
                num_runs.data.ptr,
                stream.ptr,
            )
            stream.synchronize()
            runs = int(num_runs.get()[0])
            del rows, rows_alt, order, order_alt, scratch
            need = 4 * count + 8 * runs + 4
            free, _ = cupy.cuda.runtime.memGetInfo()
            if need <= min(free - DEVICE_RESERVE_BYTES, self._plan_budget):
                self._plan_budget -= need
                with stream:
                    perm = positions[sorted_order[:count]]
                    plan_rows = cupy.array(run_rows[:runs], copy=True)
                    plan_starts = cupy.concatenate(
                        (
                            cupy.zeros(1, dtype=np.int64),
                            cupy.cumsum(run_counts[:runs], dtype=np.int64),
                        )
                    ).astype(np.int32)
                stream.synchronize()
                self.reduce_plans[unit] = (perm, plan_rows, plan_starts, runs)
            del positions, sorted_order, run_rows, run_counts

    def release_host_chunks(self):


        release = getattr(self.host, "release_units", None)
        units = [
            unit
            for unit in range(len(self.views))
            if self.resident[unit] and self.views[unit] is not None
        ]
        if release is None or not units:
            return []
        for unit in units:
            self.views[unit] = None
        release(units, len(self.kinds))
        return units

    @property
    def num_fixed_reduce(self):

        return sum(1 for plan in self.reduce_plans if plan is not None)

    @property
    def num_streamed(self):

        return sum(1 for resident in self.resident if not resident)

    def wait(self, stream):

        stream.wait_event(self.uploaded)

    def table_pointers(self):

        pointers = {name: self.tables[name].data.ptr for name in TABLE_ARRAYS}
        pointers["reference"] = self.reference.data.ptr
        pointers["num_pairs"] = self.num_pairs
        pointers["num_quads"] = self.num_quads
        pointers["num_orbitals"] = self.num_orbitals
        return pointers

    def _classes(self, chunk, stream, slot_index):


        if self.resident[chunk]:
            return self.chunks[chunk].classes, None
        slot = self.slots[slot_index]
        if slot.released is not None:
            self.copy_stream.wait_event(slot.released)
        classes = slot.load(self.views[chunk], self.copy_stream)
        stream.wait_event(self.copy_stream.record())
        return classes, slot

    def matvec(self, x, y, stream, binned=True, diagonal=True):


        if diagonal:
            kernel_launch.launch_diagonal(
                x.data.ptr, y.data.ptr, self.diagonal.data.ptr, self.num_own, stream.ptr
            )
        pointers = self.table_pointers()
        streamed = 0
        for chunk in range(len(self.views)):
            classes, slot = self._classes(chunk, stream, streamed % NUM_SLOTS)
            streamed += 0 if slot is None else 1
            kind = self.kinds[self.unit_kind[chunk]]
            plan = self.reduce_plans[chunk]
            if self.atomic_reduce:
                for cls, view in enumerate(classes):
                    if view.num_members:
                        self._launch_groups(
                            view,
                            cls,
                            x,
                            stream,
                            binned,
                            0,
                            pointers,
                            kind[:4],
                            y.data.ptr,
                        )
                if slot is not None:
                    slot.released = stream.record()
                continue
            offset = 0
            for cls, view in enumerate(classes):
                if view.num_members == 0:
                    continue
                self._launch_groups(
                    view,
                    cls,
                    x,
                    stream,
                    binned,
                    self.entry_values.data.ptr + 8 * offset,
                    pointers,
                    kind[:4],
                )
                if plan is None:
                    cupy.cuda.runtime.memcpyAsync(
                        self.entry_rows.data.ptr + 4 * offset,
                        view.members,
                        4 * view.num_members,
                        cupy.cuda.runtime.memcpyDeviceToDevice,
                        stream.ptr,
                    )
                offset += view.num_members
            if plan is not None:
                perm, plan_rows, plan_starts, count = plan
                kernel_launch.launch_gather_runs(
                    perm.data.ptr,
                    plan_rows.data.ptr,
                    plan_starts.data.ptr,
                    count,
                    self.entry_values.data.ptr,
                    y.data.ptr,
                    stream.ptr,
                )
                if slot is not None:
                    slot.released = stream.record()
                continue
            kernel_launch.launch_reduce(
                self.entry_rows.data.ptr,
                self.entry_rows_alt.data.ptr,
                self.entry_values.data.ptr,
                self.entry_values_alt.data.ptr,
                self.reduce_scratch.data.ptr,
                self.reduce_scratch.nbytes,
                offset,
                self.end_bit,
                kind[4],
                kind[5],
                y.data.ptr,
                stream.ptr,
            )
            if slot is not None:
                slot.released = stream.record()

    def _launch_groups(
        self, view, cls, x, stream, binned, partials, pointers, ranges, y=0
    ):


        halo = self.halo_keys.data.ptr if self.halo_keys.size else 0
        args = (
            self.keys.data.ptr,
            halo,
            self.key_rows,
            pointers,
            view.members,
            view.offsets,
            partials,
        )
        if not binned:
            kernel_launch.launch_groups_uniform(
                *args, 0, view.num_groups, cls, ranges, x.data.ptr, stream.ptr, y
            )
            return
        for bucket in range(len(view.bucket_start) - 1):
            first = view.bucket_start[bucket]
            count = view.bucket_start[bucket + 1] - first
            if count > 0:
                kernel_launch.launch_groups_binned(
                    *args, first, count, cls, bucket, ranges, x.data.ptr, stream.ptr, y
                )

    def class_partials(self, x, chunk, cls, stream, binned=True):


        if self.partials is None:
            self.partials = cupy.empty(
                max(1, self.largest_class_members), dtype=np.float64
            )
        classes, slot = self._classes(chunk, stream, 0)
        view = classes[cls]
        self._launch_groups(
            view,
            cls,
            x,
            stream,
            binned,
            self.partials.data.ptr,
            self.table_pointers(),
            self.kinds[self.unit_kind[chunk]][:4],
        )
        if slot is not None:
            slot.released = stream.record()
        return self.partials[: view.num_members]

    def device_bytes(self):

        total = self.keys.nbytes + self.halo_keys.nbytes + self.diagonal.nbytes
        total += sum(array.nbytes for array in self.tables.values())
        total += sum(chunk.nbytes() for chunk in self.chunks if chunk is not None)
        total += sum(slot.nbytes() for slot in self.slots)
        total += (
            0 if self.partials is None else self.partials.nbytes
        ) + self.reduce_scratch.nbytes
        total += (
            self.entry_rows.nbytes
            + self.entry_rows_alt.nbytes
            + self.entry_values.nbytes
            + self.entry_values_alt.nbytes
        )
        total += sum(
            sum(array.nbytes for array in plan[:3])
            for plan in self.reduce_plans
            if plan is not None
        )
        return total
