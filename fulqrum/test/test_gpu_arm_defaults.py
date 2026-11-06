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

from fulqrum.gpu.distributed.aabb_parents import (
    parent_options,
    remote_alpha_mode,
    remote_beta_options,
)
from fulqrum.gpu.distributed.arm_pair_tables import pair_table_options

VALIDATED = {
    "FQ_AABB_CLIQUE_ENABLE": "1",
    "FQ_AABB_REMOTE_BETA_METHOD": "clique",
    "FQ_AABB_REMOTE_ALPHA_METHOD": "transposed_clique",
    "FQ_AA_PAIR_TABLE": "1",
    "FQ_BB_PAIR_TABLE": "1",
}


def resolve(environ):
    return (
        parent_options(environ),
        remote_beta_options(environ),
        remote_alpha_mode(environ),
        pair_table_options(environ),
    )


def test_empty_environment_selects_validated_configuration():
    assert resolve({}) == resolve(VALIDATED)


def test_parent_arm_off_turns_off_remote_beta_parents():
    assert remote_beta_options({"FQ_AABB_CLIQUE_ENABLE": "0"})[0] == "baseline"


def test_parent_arm_off_turns_off_transposed_remote_alpha():
    assert remote_alpha_mode({"FQ_AABB_CLIQUE_ENABLE": "0"}) == "baseline"
    with pytest.raises(ValueError, match="requires the clique arm"):
        remote_alpha_mode(
            {
                "FQ_AABB_CLIQUE_ENABLE": "0",
                "FQ_AABB_REMOTE_ALPHA_METHOD": "transposed_clique",
            }
        )
    with pytest.raises(ValueError, match="'baseline' or 'transposed_clique'"):
        remote_alpha_mode({"FQ_AABB_REMOTE_ALPHA_METHOD": "direct"})


def test_pair_tables_default_on_and_can_be_switched_off():
    assert pair_table_options({}) == (True, True)
    assert pair_table_options({"FQ_AA_PAIR_TABLE": "0"}) == (False, True)
    assert pair_table_options({"FQ_BB_PAIR_TABLE": "0"}) == (True, False)
    with pytest.raises(ValueError):
        pair_table_options({"FQ_AA_PAIR_TABLE": "2"})


def test_pair_payload_errors_name_each_broken_field():
    from fulqrum.gpu.distributed.arm_pair_tables import (
        ArmPairTablesPayload,
        pair_payload_error,
    )

    good = ArmPairTablesPayload(
        aa_sell_off=1,
        aa_sell_len=1,
        aa_sell_pack=1,
        aa_c0_sell=1,
        aa_t=1,
        bb_word=1,
        bb_t=1,
        aa_enable=1,
        aa_pairs=21,
        aa_t_rows=10,
        bb_enable=1,
        bb_pairs=21,
        bb_t_cols=12,
        aaaa_min_blocks=6,
    )
    assert pair_payload_error(good, 10, 12) is None
    assert "rows" in pair_payload_error(good, 11, 12)
    assert "columns" in pair_payload_error(good, 10, 13)
    assert "device tables" in pair_payload_error(
        dataclasses.replace(good, aa_t=0), 10, 12
    )
    assert "1..65536" in pair_payload_error(
        dataclasses.replace(good, aa_pairs=70000), 10, 12
    )
    assert "minimum blocks" in pair_payload_error(
        dataclasses.replace(good, aaaa_min_blocks=4), 10, 12
    )
