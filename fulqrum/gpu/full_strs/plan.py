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

from dataclasses import dataclass, field

from fulqrum.gpu.distributed.exchange_plan import check_same_settings, settings_digest

BLOCK_CUTS = ("row", "membership", "fitted")
SHARE_MODES = ("equal", "balanced", "own")
EXCHANGES = ("stepped", "grouped", "two_level")


@dataclass
class LayoutPlan:


    copies: int = 1
    num_slices: int = 4
    exchange: str = "stepped"
    share_starts: list | None = None
    assignment: list | None = None
    block_cut: str = "row"
    shares: str = "equal"
    coefficients: list | None = None
    entry_cost: float = 0.0
    measured: dict = field(default_factory=dict)

    def check(self):

        if self.copies < 1 or self.num_slices < 1:
            raise ValueError(
                f"copies and num_slices must be positive, got {self.copies} and {self.num_slices}"
            )
        if self.exchange not in EXCHANGES:
            raise ValueError(
                f"exchange must be one of {EXCHANGES}, got {self.exchange!r}"
            )
        if self.block_cut not in BLOCK_CUTS:
            raise ValueError(
                f"block_cut must be one of {BLOCK_CUTS}, got {self.block_cut!r}"
            )
        if self.shares not in SHARE_MODES:
            raise ValueError(
                f"shares must be one of {SHARE_MODES}, got {self.shares!r}"
            )

    @property
    def fixed(self):

        return self.share_starts is not None and (
            self.copies == 1 or self.assignment is not None
        )

    def layout(self):

        return {
            "copies": int(self.copies),
            "num_slices": int(self.num_slices),
            "exchange": self.exchange,
            "share_starts": None
            if self.share_starts is None
            else [int(v) for v in self.share_starts],
            "assignment": None
            if self.assignment is None
            else [[int(v) for v in row] for row in self.assignment],
        }

    @property
    def digest(self):

        return settings_digest(self.layout())

    def to_dict(self):

        out = self.layout()
        out.update(
            {
                "block_cut": self.block_cut,
                "shares": self.shares,
                "coefficients": self.coefficients,
                "entry_cost": self.entry_cost,
                "measured": self.measured,
                "digest": self.digest,
            }
        )
        return out

    @classmethod
    def from_dict(cls, data):

        plan = cls(
            copies=int(data["copies"]),
            num_slices=int(data["num_slices"]),
            exchange=data["exchange"],
            share_starts=data.get("share_starts"),
            assignment=data.get("assignment"),
            block_cut=data.get("block_cut", "row"),
            shares=data.get("shares", "equal"),
            coefficients=data.get("coefficients"),
            entry_cost=float(data.get("entry_cost", 0.0)),
            measured=dict(data.get("measured", {})),
        )
        plan.check()
        if "digest" in data and data["digest"] != plan.digest:
            raise ValueError(
                f"the plan digest {data['digest']} does not match its layout ({plan.digest})"
            )
        return plan


def agree(comm, plan):

    return check_same_settings(comm, plan.layout(), "full-string layout plan")


def volume_objective(metrics):


    return float(metrics["volume_max"])


def choose_copies(candidates, objective=volume_objective, max_imbalance=1.25):


    balanced = [
        t
        for t in sorted(candidates)
        if candidates[t]["work_imbalance"] <= max_imbalance
    ]
    if not balanced:
        return min(sorted(candidates), key=lambda t: candidates[t]["work_imbalance"])
    return min(balanced, key=lambda t: (objective(candidates[t]), t))
