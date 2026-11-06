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


from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .exchange_plan import FORWARD


@dataclass
class ExchangeExecutor:


    backend: object
    comm: object
    copy_stream: object
    compute_stream: object
    begin: Callable
    end: Callable
    slot_events: list

    def wait_slot(self, slot: int) -> None:

        event = self.slot_events[slot]
        if event is not None:
            self.copy_stream.wait_event(event)

    def mark_slot(self, slot: int, event) -> None:

        self.slot_events[slot] = event

    def run_ring(
        self,
        steps: list,
        slots: list,
        local,
        receive_count: Callable,
        tile_view: Callable,
        launch: Callable,
        send_pinned=None,
        receive_pinned=None,
        after_receive: Optional[Callable] = None,
    ) -> None:


        for index, step in enumerate(steps, start=1):
            self.wait_slot(step.slot)
            start = self.begin()
            if step.transfer == FORWARD:
                received = slots[step.slot]
                with self.copy_stream:
                    self.backend.ring_shift_device(
                        slots[(index - 1) & 1].ravel(),
                        received.ravel(),
                        step.send_to,
                        step.receive_from,
                        self.comm,
                        stream=self.copy_stream,
                        send_pinned=send_pinned,
                        recv_pinned=None
                        if receive_pinned is None
                        else receive_pinned[step.slot],
                    )
            else:
                count = int(receive_count(step))
                received = slots[0].ravel()[:count]
                zero = np.zeros(1, dtype=np.int64)
                with self.copy_stream:
                    self.backend.grouped_sendrecv_device(
                        local,
                        np.asarray([local.size], dtype=np.int64),
                        zero,
                        np.asarray([step.send_to], dtype=np.int32),
                        received,
                        np.asarray([count], dtype=np.int64),
                        zero,
                        np.asarray([step.receive_from], dtype=np.int32),
                        self.comm,
                        stream=self.copy_stream,
                    )
            self.compute_stream.wait_event(self.end(start))
            if after_receive is not None:
                after_receive(step, index, received)
            self.mark_slot(step.slot, launch(step, tile_view(step, received), index))

    def run_window(
        self,
        steps: list,
        slots: list,
        local,
        receive_count: Callable,
        tile_view: Callable,
        launch: Callable,
        send_pinned=None,
        receive_pinned=None,
    ) -> tuple:


        empty64 = np.zeros(0, dtype=np.int64)
        empty32 = np.zeros(0, dtype=np.int32)
        zero = np.zeros(1, dtype=np.int64)
        seconds, received_bytes = 0.0, 0
        for index, step in enumerate(steps):
            slot = slots[step.slot]
            sends = step.send_to >= 0
            receives = step.receive_from >= 0
            count = int(receive_count(step)) if receives else 0
            self.wait_slot(step.slot)
            started = time.perf_counter()
            start = self.begin()
            with self.copy_stream:
                self.backend.grouped_sendrecv_device(
                    local if sends else local[:0],
                    np.asarray([local.size], dtype=np.int64) if sends else empty64,
                    zero if sends else empty64,
                    np.asarray([step.send_to], dtype=np.int32) if sends else empty32,
                    slot if receives else slot[:0],
                    np.asarray([count], dtype=np.int64) if receives else empty64,
                    zero if receives else empty64,
                    np.asarray([step.receive_from], dtype=np.int32)
                    if receives
                    else empty32,
                    self.comm,
                    stream=self.copy_stream,
                    send_pinned=send_pinned if sends else None,
                    recv_pinned=receive_pinned[step.slot]
                    if receives and receive_pinned is not None
                    else None,
                )
            seconds += time.perf_counter() - started
            received_bytes += count * 8
            self.compute_stream.wait_event(self.end(start))
            if receives:
                self.mark_slot(
                    step.slot, launch(step, tile_view(step, slot[:count]), index)
                )
        return seconds, received_bytes

    def run_two_level(
        self,
        local,
        stage,
        gather,
        node_comm,
        domain_rank: int,
        local_sources,
        local_ring_args,
        round_args,
        launch_source: Callable,
        fetch_pairs: list,
        gather_pairs: list,
    ) -> int:


        events = [None, None]

        def wait(slot):

            if events[slot] is not None:
                self.copy_stream.wait_event(events[slot])
                events[slot] = None

        def transfer(send, recv, args, comm):

            with self.copy_stream:
                self.backend.grouped_sendrecv_device(
                    send,
                    args["send_counts"],
                    args["send_displs"],
                    args["dst_ranks"],
                    recv,
                    args["recv_counts"],
                    args["recv_displs"],
                    args["src_ranks"],
                    comm,
                    stream=self.copy_stream,
                )

        def domain_ring(gathered, ring_args, initial, initial_slot, label, ready=None):

            if ready is not None:
                self.compute_stream.wait_event(ready)
            first = int(gathered[domain_rank])
            event = launch_source(first, initial, f"{label}[ring=0,src={first}]")
            if initial_slot is not None:
                events[initial_slot] = event
            send = initial
            for index, args in enumerate(ring_args, start=1):
                slot = 1 if index & 1 else 0
                recv = gather if slot == 1 else stage
                wait(slot)
                start = self.begin()
                if len(args["dst_ranks"]) > 0 or len(args["src_ranks"]) > 0:
                    transfer(
                        send if args["send_source"] >= 0 else send[:0],
                        recv if args["recv_source"] >= 0 else recv[:0],
                        args,
                        node_comm,
                    )
                self.compute_stream.wait_event(self.end(start, gather_pairs))
                source = int(args["recv_source"])
                events[slot] = launch_source(
                    source, recv, f"{label}[ring={index},src={source}]"
                )
                send = recv

        domain_ring(local_sources, local_ring_args, local, None, "hierarchical_local")
        network_bytes = 0
        for round_index, args in enumerate(round_args, start=1):
            wait(0)
            fetched = None
            if args["send_dst"] >= 0 or args["recv_source"] >= 0:
                start = self.begin()
                transfer(
                    local if args["send_dst"] >= 0 else local[:0],
                    stage if args["recv_source"] >= 0 else stage[:0],
                    args,
                    self.comm,
                )
                fetched = self.end(start, fetch_pairs)
                if args["recv_source"] >= 0:
                    network_bytes += int(args["recv_counts"][0]) * 8
            domain_ring(
                args["gathered_sources"],
                args["local_ring_args"],
                stage,
                0,
                f"hierarchical_round[{round_index}]",
                fetched,
            )
        return network_bytes
