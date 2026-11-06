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


import json

import pytest

from fulqrum.gpu import autotune
from fulqrum.gpu.config import GpuConfig, REGISTRY

GB200 = autotune.DeviceFacts(
    "NVIDIA GB200", (10, 0), 190 << 30, 126 << 20, 228 << 10, 227 << 10, 65536, 2048
)
L40S = autotune.DeviceFacts(
    "NVIDIA L40S", (8, 9), 48 << 30, 96 << 20, 100 << 10, 99 << 10, 65536, 1536
)
SMALL_SHARED = autotune.DeviceFacts(
    "small", (9, 0), 80 << 30, 50 << 20, 100 << 10, 99 << 10, 65536, 2048
)

RECIPE = {
    "mma": True,
    "label_batching": True,
    "source_panel": True,
    "bbbb_class": True,
    "aaaa_class": True,
    "scratch_mib": 512,
    "diag_recompute": True,
}


def _kernel(name, arch, registers=64, stack=0, spill=0, shared=0):
    return {
        "name": name,
        "arch": arch,
        "registers": registers,
        "stack_bytes": stack,
        "static_shared_bytes": shared,
        "local_bytes": 0,
        "spill_store_bytes": spill,
        "spill_load_bytes": spill,
    }


def _facts(arch, spilling=()):
    kernels = []
    for prefetch, alphas in autotune.BB_CANDIDATES:
        pattern = autotune.bb_pattern(prefetch, alphas).replace("\\", "")
        kernels.append(
            _kernel(
                f"_ZN_{pattern}Ev",
                arch,
                spill=96 if (prefetch, alphas) in spilling else 0,
            )
        )
    kernels.append(
        _kernel(
            f"_ZN_{autotune.aa_pattern(True)}Ev",
            arch,
            spill=96 if "aa" in spilling else 0,
        )
    )
    return autotune.KernelFacts((arch,), tuple(kernels), "test")


def _resolve(environ=None, device=GB200, kernels=None, probe=None, **options):
    kernels = _facts(device.arch) if kernels is None else kernels
    return autotune.resolve(
        GpuConfig.from_environ(environ or {}), device, kernels, probe=probe, **options
    )


def test_gb200_at_the_default_tier_is_the_production_recipe():
    result = _resolve()
    assert {key: result.value(key) for key in RECIPE} == RECIPE
    assert result.value("double_buffer") is True
    assert (result.value("bb_prefetch"), result.value("bb_alphas")) == (True, 4)
    assert result.value("clique_rows") == 64 and result.value("aa_prefetch") is True
    assert result.value("hbm_free_fraction") == pytest.approx(1.2)
    assert result.choices["fp64_mma"].source == "table"
    assert not result.errors and not result.warnings


def test_a_slow_probe_turns_the_mma_levers_off_and_a_fast_one_on():
    slow = _resolve(device=L40S, probe=lambda: (2.0, True, "mocked"))
    assert not any(
        slow.value(key)
        for key in ("mma", "label_batching", "source_panel", "bbbb_class", "aaaa_class")
    )
    assert slow.choices["fp64_mma"].source == "probe"
    fast = _resolve(device=L40S, probe=lambda: (5.0, True, "mocked"))
    assert all(
        fast.value(key)
        for key in ("mma", "label_batching", "source_panel", "bbbb_class", "aaaa_class")
    )


def test_an_inexact_or_failing_probe_counts_as_slow():
    assert not _resolve(device=L40S, probe=lambda: (9.0, False, "mocked")).value("mma")

    def broken():
        raise OSError("no library")

    assert not _resolve(device=L40S, probe=broken).value("mma")


@pytest.mark.parametrize(
    "tier,off",
    [
        (0, set()),
        (1, {"stored_diagonal"}),
        (
            3,
            {
                "stored_diagonal",
                "double_buffer",
                "label_batching",
                "source_panel",
                "bbbb_class",
                "aaaa_class",
            },
        ),
    ],
)
def test_each_tier_switches_off_exactly_its_items(tier, off):
    result = _resolve({"FQ_HBM_SAVE_TIER": str(tier)})
    assert result.value("mma") is True
    assert result.value("diag_recompute") is ("stored_diagonal" in off)
    assert result.value("double_buffer") is ("double_buffer" not in off)
    for key in ("label_batching", "source_panel", "bbbb_class", "aaaa_class"):
        assert result.value(key) is (key not in off)
    assert result.value("scratch_mib") == {0: 512, 1: 512, 3: 0}[tier]
    assert result.value("exchange") == ("ring" if tier == 3 else "auto")
    assert result.value("aaaa_slab") == 2048


def test_an_explicit_exchange_beats_tier_3_with_a_warning():
    result = _resolve({"FQ_HBM_SAVE_TIER": "3", "FQ_EXCHANGE": "two_level"})
    assert result.value("exchange") == "two_level" and any(
        "FQ_EXCHANGE" in text for text in result.warnings
    )
    assert (
        _resolve({"FQ_HBM_SAVE_TIER": "3", "FQ_EXCHANGE": "auto"}).value("exchange")
        == "ring"
    )
    assert (
        _resolve({"FQ_TUNE": "off", "FQ_HBM_SAVE_TIER": "3"}).value("exchange")
        == "auto"
    )


def test_tier_2_is_not_accepted():
    with pytest.raises(ValueError, match="FQ_HBM_SAVE_TIER"):
        GpuConfig.from_environ({"FQ_HBM_SAVE_TIER": "2"})


def test_an_explicit_value_beats_the_tier_and_the_platform_rule_with_a_warning():
    result = _resolve(
        {
            "FQ_HBM_SAVE_TIER": "3",
            "FQ_BBBB_CLASS_USE_FP64_GEMM": "1",
            "FQ_EXCHANGE_DOUBLE_BUFFER": "1",
            "FQ_AABB_CLIQUE_SCRATCH_LIMIT_MB": "0",
        }
    )
    assert (
        result.value("bbbb_class") is True
        and result.choices["bbbb_class"].source == "explicit"
    )
    assert result.value("double_buffer") is True
    assert result.value("scratch_mib") == 0
    assert any("FQ_BBBB_CLASS_USE_FP64_GEMM" in text for text in result.warnings)
    forced = _resolve(
        {"FQ_AABB_CLIQUE_USE_FP64_MMA": "1"},
        device=L40S,
        probe=lambda: (1.0, True, "mocked"),
    )
    assert forced.value("mma") is True and forced.warnings and not forced.errors


def test_the_umbrella_sets_the_mma_default_and_a_lever_value_beats_it():
    result = _resolve({"FQ_FP64_MMA": "0", "FQ_BBBB_CLASS_USE_FP64_GEMM": "1"})
    assert result.value("mma") is False and result.value("label_batching") is False
    assert result.value("bbbb_class") is True


def test_an_explicit_lever_without_its_prerequisite_fails():
    result = _resolve(
        {"FQ_AABB_CLIQUE_USE_FP64_MMA": "0", "FQ_AABB_CLIQUE_USE_LABEL_BATCHING": "1"}
    )
    assert any("FQ_AABB_CLIQUE_USE_LABEL_BATCHING" in text for text in result.errors)
    result = _resolve(
        {"FQ_AABB_CLIQUE_USE_FP64_MMA": "0", "FQ_AABB_OWN_CLIQUE_USE_SOURCE_PANEL": "1"}
    )
    assert any("FQ_AABB_OWN_CLIQUE_USE_SOURCE_PANEL" in text for text in result.errors)


def test_a_library_built_for_another_architecture_fails():
    result = _resolve(
        device=L40S, kernels=_facts("sm_100"), probe=lambda: (1.0, True, "mocked")
    )
    assert any("FQ_CUDA_ARCH=sm_89" in text for text in result.errors)


def test_the_bb_screen_takes_the_first_candidate_that_passes():
    result = _resolve(kernels=_facts("sm_100", spilling={(True, 4)}))
    assert (result.value("bb_prefetch"), result.value("bb_alphas")) == (True, 2)
    result = _resolve(
        kernels=_facts("sm_100", spilling={(True, 4), (True, 2), (True, 1)})
    )
    assert (result.value("bb_prefetch"), result.value("bb_alphas")) == (False, 2)
    assert "spills" in result.candidates["bb prefetch=1 alphas=4"]
    result = _resolve(kernels=_facts("sm_100", spilling={"aa"}))
    assert result.value("aa_prefetch") is False


def test_missing_kernel_facts_fail_safe():
    result = _resolve(kernels=autotune.KernelFacts((), (), "none"))
    assert (result.value("bb_prefetch"), result.value("bb_alphas")) == (False, 2)
    assert result.value("aa_prefetch") is False


def test_occupancy_counts_registers_shared_memory_and_threads():
    assert autotune.blocks_per_sm(_kernel("k", "sm_100", registers=64), GB200) == 4
    assert autotune.blocks_per_sm(_kernel("k", "sm_100", registers=72), GB200) == 3
    assert autotune.blocks_per_sm(_kernel("k", "sm_100", registers=32), GB200) == 8
    assert (
        autotune.blocks_per_sm(
            _kernel("k", "sm_100", registers=32, shared=60 << 10), GB200
        )
        == 3
    )
    assert not autotune.screen(_kernel("k", "sm_100", registers=72), GB200)[0]


def test_the_clique_row_tile_follows_the_shared_memory_of_an_sm():
    assert _resolve(device=GB200).value("clique_rows") == 64
    assert (
        _resolve(device=SMALL_SHARED, kernels=_facts("sm_90"), probe=None).value(
            "clique_rows"
        )
        == 32
    )


def test_the_aaaa_block_group_follows_the_l2_size():
    assert autotune.aaaa_block_group(1 << 20, GB200)[0] == 8
    assert autotune.aaaa_block_group(4 << 20, GB200)[0] == 4
    assert autotune.aaaa_block_group(64 << 20, GB200)[0] == 2
    shared = autotune.DeviceFacts(
        *[
            getattr(GB200, name)
            for name in (
                "name",
                "capability",
                "total_bytes",
                "l2_bytes",
                "shared_per_sm",
                "shared_per_block_optin",
                "registers_per_sm",
                "max_threads_per_sm",
            )
        ],
        4,
    )
    assert autotune.aaaa_block_group(4 << 20, shared)[0] == 2


def test_the_scratch_cap_shrinks_on_small_cards():
    small = autotune.DeviceFacts(
        "small", (10, 0), 16 << 30, 50 << 20, 228 << 10, 227 << 10, 65536, 2048
    )
    assert _resolve(device=small).value("scratch_mib") == 163


def test_an_operator_that_cannot_split_its_diagonal_keeps_it_stored():
    assert _resolve(can_split_diagonal=False).value("diag_recompute") is False
    result = _resolve({"FQ_DIAG_RECOMPUTE": "1"}, can_split_diagonal=False)
    assert (
        result.value("diag_recompute") is True
        and result.choices["diag_recompute"].source == "explicit"
    )


def test_tune_off_gives_the_static_values():
    result = _resolve({"FQ_TUNE": "off"})
    for key, name in (
        ("mma", "FQ_AABB_CLIQUE_USE_FP64_MMA"),
        ("bbbb_class", "FQ_BBBB_CLASS_USE_FP64_GEMM"),
        ("diag_recompute", "FQ_DIAG_RECOMPUTE"),
        ("double_buffer", "FQ_EXCHANGE_DOUBLE_BUFFER"),
        ("scratch_mib", "FQ_AABB_CLIQUE_SCRATCH_LIMIT_MB"),
    ):
        assert result.value(key) == REGISTRY[name].default
    assert result.value("hbm_free_fraction") == 1.0
    assert (result.value("bb_prefetch"), result.value("bb_alphas")) == (False, 2)
    result = _resolve({"FQ_TUNE": "off", "FQ_BB_PAIR_PREFETCH_BEFORE_SCREEN": "1"})
    assert result.value("bb_alphas") == 4
    result = _resolve(
        {
            "FQ_TUNE": "off",
            "FQ_AABB_CLIQUE_USE_FP64_MMA": "1",
            "FQ_AABB_CLIQUE_USE_LABEL_BATCHING": "1",
        }
    )
    assert result.value("scratch_mib") == 512
    result = _resolve({"FQ_TUNE": "off", "FQ_AABB_CLIQUE_USE_LABEL_BATCHING": "1"})
    assert result.value("label_batching") is False and result.value("scratch_mib") == 0


def test_auto_is_accepted_and_means_the_rule():
    assert (
        _resolve(
            {"FQ_BBBB_CLASS_USE_FP64_GEMM": "auto", "FQ_DIAG_RECOMPUTE": "AUTO"}
        ).collective()
        == _resolve().collective()
    )


def test_ranks_that_disagree_on_the_mma_rule_turn_it_off_everywhere():
    result = _resolve(agree=lambda value: [value, False])
    assert (
        result.value("mma") is False
        and "rank(s) [1]" in result.choices["fp64_mma"].reason
    )
    result = _resolve(agree=lambda value: [value, value])
    assert result.value("mma") is True


def test_the_record_and_line_name_every_choice():
    result = _resolve()
    json.dumps(result.record())
    assert "mma=1(table)" in result.line() and "scratch_mib=512(tier)" in result.line()
    assert set(result.collective()) <= set(autotune.COLLECTIVE)


def test_the_kernel_record_of_the_build_matches_its_library(tmp_path):
    library = tmp_path / "libdist_matvec_cuda.so"
    library.write_bytes(b"library")
    import hashlib

    record = {
        "sha256": hashlib.sha256(b"library").hexdigest(),
        "archs": ["sm_100"],
        "kernels": [],
    }
    (tmp_path / "kernel_resources.json").write_text(json.dumps(record))
    assert autotune.kernel_facts(library).archs == ("sm_100",)
    library.write_bytes(b"rebuilt")
    assert autotune.kernel_facts(library).archs == ()


def test_the_build_parser_reads_ptxas_and_cuobjdump_output():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "gpu_build.py"
    if not path.is_file():
        pytest.skip("gpu_build.py is not in this tree")
    spec = importlib.util.spec_from_file_location("gpu_build_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ptxas = (
        "ptxas info    : Compiling entry function '_Zk' for 'sm_100'\n"
        "ptxas info    : Function properties for _Zk\n"
        "    256 bytes stack frame, 320 bytes spill stores, 380 bytes spill loads\n"
        "ptxas info    : Used 64 registers, used 0 barriers\n"
    )
    usage = "arch = sm_100\n Function _Zk:\n  REG:64 STACK:256 SHARED:0 LOCAL:0 CONSTANT[0]:1040\n"
    facts = module.parse_ptxas_output(ptxas)
    assert facts[("_Zk", "sm_100")]["spill_store_bytes"] == 320
    assert module.parse_resource_usage(usage)[("_Zk", "sm_100")]["stack_bytes"] == 256
