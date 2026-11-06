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


import dataclasses

import pytest

from fulqrum.gpu.distributed.arm_plan import (
    ArmSettings,
    SourceFacts,
    aabb_mode,
    plan_arms,
)

RECIPE = ArmSettings(
    factored=True, parent=True, remote_beta="clique", remote_alpha=True
)
FULL = SourceFacts(
    is_bra=False,
    same_alpha=False,
    same_beta=False,
    singles_edges=True,
    singles_sell=True,
    doubles_edges=True,
    doubles_sell=True,
    parent_built=True,
    parent_plan=True,
    parent_grouped_edges=True,
    remote_beta_plan=False,
    remote_beta_grouped_edges=False,
    remote_alpha_plan=False,
)


def facts(**changes):
    return dataclasses.replace(FULL, **changes)


def test_a_source_with_packed_alpha_singles_runs_the_aabb_arm():
    assert aabb_mode(facts(), RECIPE) == 12
    assert aabb_mode(facts(singles_edges=False, singles_sell=False), RECIPE) == 0
    assert aabb_mode(facts(), dataclasses.replace(RECIPE, factored=False)) == 0


def test_alpha_singles_without_a_sell_form_raise():
    with pytest.raises(ValueError, match="no SELL form"):
        aabb_mode(facts(singles_sell=False), RECIPE)


@pytest.mark.parametrize(
    "changes,path,remote",
    [
        (dict(is_bra=True, same_alpha=True, same_beta=True), "clique", 0),
        (
            dict(
                is_bra=True, same_alpha=True, same_beta=True, parent_grouped_edges=False
            ),
            "baseline",
            0,
        ),
        (
            dict(
                same_alpha=True, remote_beta_plan=True, remote_beta_grouped_edges=True
            ),
            "clique",
            2,
        ),
        (
            dict(
                same_alpha=True, remote_beta_plan=True, remote_beta_grouped_edges=False
            ),
            "baseline",
            0,
        ),
        (
            dict(
                same_alpha=True,
                remote_beta_plan=True,
                remote_beta_grouped_edges=True,
                parent_built=False,
            ),
            "baseline",
            0,
        ),
        (dict(same_beta=True, remote_alpha_plan=True), "transposed_clique", 0),
        (dict(same_beta=True), "baseline", 0),
        (dict(), "baseline", 0),
    ],
)
def test_each_source_class_gets_its_aabb_path(changes, path, remote):
    plan = plan_arms(facts(**changes), RECIPE)
    assert (plan.aabb_mode, plan.aabb_path, plan.remote_beta) == (12, path, remote)


def test_the_special_aabb_paths_need_aabb_work():
    plan = plan_arms(
        facts(
            is_bra=True,
            same_alpha=True,
            same_beta=True,
            singles_edges=False,
            singles_sell=False,
        ),
        RECIPE,
    )
    assert (plan.aabb_mode, plan.aabb_path) == (0, "baseline")


def test_the_own_tile_clique_needs_beta_edges():
    plan = plan_arms(
        facts(is_bra=True, same_alpha=True, same_beta=True, parent_plan=False), RECIPE
    )
    assert plan.aabb_path == "baseline"


def test_switched_off_arms_give_the_baseline_path():
    off = dataclasses.replace(
        RECIPE, parent=False, remote_beta="baseline", remote_alpha=False
    )
    parent_only_off = dataclasses.replace(RECIPE, parent=False)
    for settings in (off, parent_only_off):
        for changes in (
            dict(is_bra=True, same_alpha=True, same_beta=True),
            dict(same_beta=True, remote_alpha_plan=True),
        ):
            assert plan_arms(facts(**changes), settings).aabb_path == "baseline"
    assert (
        plan_arms(
            facts(
                same_alpha=True, remote_beta_plan=True, remote_beta_grouped_edges=True
            ),
            off,
        ).aabb_path
        == "baseline"
    )


def test_the_aaaa_arm_reads_the_sell_form_when_it_exists():
    assert plan_arms(facts(), RECIPE).aaaa_sell
    assert not plan_arms(facts(doubles_sell=False), RECIPE).aaaa_sell


def test_alpha_doubles_of_a_same_beta_source_without_a_sell_form_raise():
    with pytest.raises(ValueError, match="alpha double edges but no SELL form"):
        plan_arms(facts(same_beta=True, doubles_sell=False), RECIPE)
    assert not plan_arms(
        facts(same_beta=True, doubles_edges=False, doubles_sell=False), RECIPE
    ).aaaa_sell
