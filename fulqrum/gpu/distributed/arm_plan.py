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

from dataclasses import dataclass

BASELINE = "baseline"
CLIQUE = "clique"
TRANSPOSED_CLIQUE = "transposed_clique"
REMOTE_BETA_CLIQUE = "clique"
REMOTE_NONE = 0
REMOTE_CLIQUE = 2


@dataclass(frozen=True)
class ArmSettings:


    factored: bool
    parent: bool
    remote_beta: str
    remote_alpha: bool


@dataclass(frozen=True)
class SourceFacts:


    is_bra: bool
    same_alpha: bool
    same_beta: bool
    singles_edges: bool
    singles_sell: bool
    doubles_edges: bool
    doubles_sell: bool
    parent_built: bool
    parent_plan: bool
    parent_grouped_edges: bool
    remote_beta_plan: bool
    remote_beta_grouped_edges: bool
    remote_alpha_plan: bool


@dataclass(frozen=True)
class ArmPlan:


    aabb_mode: int
    aabb_path: str
    remote_beta: int
    aaaa_sell: bool


def plan_arms(facts: SourceFacts, settings: ArmSettings) -> ArmPlan:


    if facts.same_beta and facts.doubles_edges and not facts.doubles_sell:
        raise ValueError(
            "a same-beta source tile has alpha double edges but no SELL form of them; "
            "its padded SELL size exceeds int32, so use more ranks"
        )
    mode = aabb_mode(facts, settings)
    remote = _remote_beta(facts, settings, mode)
    return ArmPlan(
        mode, _aabb_path(facts, settings, mode, remote), remote, facts.doubles_sell
    )


def aabb_mode(facts: SourceFacts, settings: ArmSettings) -> int:


    if settings.factored and facts.singles_edges and not facts.singles_sell:
        raise ValueError(
            "a source tile has alpha single edges but no SELL form of them; "
            "its padded SELL size exceeds int32, so use more ranks"
        )
    return 12 if settings.factored and facts.singles_sell else 0


def _remote_beta(facts: SourceFacts, settings: ArmSettings, mode: int) -> int:

    if (
        settings.remote_beta != REMOTE_BETA_CLIQUE
        or facts.is_bra
        or not facts.remote_beta_plan
        or mode != 12
        or not facts.same_alpha
        or facts.same_beta
        or not facts.parent_built
        or not facts.remote_beta_grouped_edges
    ):
        return REMOTE_NONE
    return REMOTE_CLIQUE


def _aabb_path(
    facts: SourceFacts, settings: ArmSettings, mode: int, remote: int
) -> str:

    own_clique = (
        settings.parent
        and facts.parent_plan
        and facts.is_bra
        and mode == 12
        and facts.same_alpha
        and facts.same_beta
    )
    if remote == REMOTE_CLIQUE or (own_clique and facts.parent_grouped_edges):
        return CLIQUE
    if (
        settings.parent
        and settings.remote_alpha
        and facts.remote_alpha_plan
        and not facts.same_alpha
        and facts.same_beta
        and mode == 12
    ):
        return TRANSPOSED_CLIQUE
    return BASELINE
