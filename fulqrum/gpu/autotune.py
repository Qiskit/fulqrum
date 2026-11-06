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

import ctypes
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .config import AUTO, GpuConfig

FAST_MMA_ARCHS = {
    (10, 0): "measured on GB200: DMMA: 39.6 TFLOPS; memory copy: 6.5 TB/s; ratio: 6.1 FLOP/byte."
}
MMA_FLOP_PER_BYTE = 3.0
PROBE_ITERATIONS = 1024
ARM_THREADS = 256
TARGET_BLOCKS_PER_SM = 4
SCRATCH_TIER_MIB = {0: 512, 1: 512}
SCRATCH_HBM_SHARE = 0.01
AAAA_GROUP_L2_SHARE = 0.15
AAAA_GROUP_RANGE = (2, 8)
LBG_FORCED_SCRATCH_MIB = 512
BB_CANDIDATES = ((True, 4), (True, 2), (True, 1), (False, 2))
CLIQUE_ROWS = (64, 32)
CLIQUE_PROBE_BUCKET_MEMBERS = 32
COLLECTIVE = (
    "tier",
    "tune",
    "exchange",
    "fp64_mma",
    "mma",
    "label_batching",
    "source_panel",
    "bbbb_class",
    "aaaa_class",
    "scratch_mib",
    "aaaa_slab",
    "diag_recompute",
    "double_buffer",
    "hbm_free_fraction",
    "clique_rows",
    "bb_prefetch",
    "bb_alphas",
    "aa_prefetch",
    "archs",
)


@dataclass(frozen=True)
class DeviceFacts:


    name: str
    capability: tuple
    total_bytes: int
    l2_bytes: int
    shared_per_sm: int
    shared_per_block_optin: int
    registers_per_sm: int
    max_threads_per_sm: int
    ranks_on_device: int = 1

    @property
    def arch(self) -> str:

        return f"sm_{self.capability[0]}{self.capability[1]}"


@dataclass(frozen=True)
class KernelFacts:


    archs: tuple
    kernels: tuple
    source: str

    def instance(self, pattern: str, arch: str) -> Optional[dict]:

        compiled = re.compile(pattern)
        for entry in self.kernels:
            if entry.get("arch") == arch and compiled.search(entry.get("name", "")):
                return entry
        return None


@dataclass(frozen=True)
class Choice:


    value: Any
    source: str
    reason: str


@dataclass
class Resolution:


    choices: dict = field(default_factory=dict)
    candidates: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def value(self, name: str) -> Any:

        return self.choices[name].value

    def collective(self) -> dict:

        return {
            name: self.choices[name].value
            for name in COLLECTIVE
            if name in self.choices
        }

    def record(self) -> dict:

        return {
            "choices": {
                name: {
                    "value": choice.value,
                    "source": choice.source,
                    "reason": choice.reason,
                }
                for name, choice in sorted(self.choices.items())
            },
            "candidates": self.candidates,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    def line(self) -> str:

        return " ".join(
            f"{name}={_text(choice.value)}({choice.source})"
            for name, choice in sorted(self.choices.items())
        )


def _text(value) -> str:

    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def device_facts(ranks_on_device: int = 1) -> DeviceFacts:


    import cupy  # noqa: PLC0415

    device = cupy.cuda.Device()
    attributes = device.attributes
    properties = cupy.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    name = name.decode() if isinstance(name, bytes) else str(name)
    _, total = cupy.cuda.runtime.memGetInfo()
    capability = str(device.compute_capability)
    return DeviceFacts(
        name=name,
        capability=(int(capability[:-1]), int(capability[-1])),
        total_bytes=int(total),
        l2_bytes=int(attributes["L2CacheSize"]),
        shared_per_sm=int(attributes["MaxSharedMemoryPerMultiprocessor"]),
        shared_per_block_optin=int(attributes["MaxSharedMemoryPerBlockOptin"]),
        registers_per_sm=int(attributes["MaxRegistersPerMultiprocessor"]),
        max_threads_per_sm=int(attributes["MaxThreadsPerMultiProcessor"]),
        ranks_on_device=max(1, int(ranks_on_device)),
    )


def library_path() -> Path:

    return Path(__file__).resolve().parent / "distributed" / "libdist_matvec_cuda.so"


def kernel_facts(library: Optional[Path] = None) -> KernelFacts:


    library = library_path() if library is None else Path(library)
    record_path = library.with_name("kernel_resources.json")
    if not library.is_file() or not record_path.is_file():
        return KernelFacts((), (), f"no kernel_resources.json next to {library.name}")
    record = json.loads(record_path.read_text())
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    if record.get("sha256") != digest:
        return KernelFacts(
            (), (), "kernel_resources.json does not match the library (rebuild)"
        )
    return KernelFacts(
        tuple(record.get("archs", ())),
        tuple(record.get("kernels", ())),
        record_path.name,
    )


def probe_flop_per_byte(
    library: Optional[Path] = None, iterations: int = PROBE_ITERATIONS
) -> tuple:


    lib = ctypes.CDLL(str(library_path() if library is None else library))
    probe = lib.fq_probe_fp64_mma
    probe.restype = ctypes.c_int
    probe.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    mma_ms, scalar_ms, copy_ms = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    flops, copy_bytes, exact = ctypes.c_double(), ctypes.c_double(), ctypes.c_int(0)
    status = probe(
        int(iterations),
        ctypes.byref(mma_ms),
        ctypes.byref(scalar_ms),
        ctypes.byref(copy_ms),
        ctypes.byref(flops),
        ctypes.byref(copy_bytes),
        ctypes.byref(exact),
    )
    if status != 0:
        raise RuntimeError(
            f"FP64 matrix-multiply probe failed with CUDA error {status}"
        )
    rate = flops.value / (mma_ms.value * 1e-3)
    bandwidth = copy_bytes.value / (copy_ms.value * 1e-3)
    ratio = rate / bandwidth
    text = f"probe: DMMA {rate / 1e12:.1f} TFLOPS, copy {bandwidth / 1e12:.2f} TB/s, {ratio:.2f} FLOP per byte"
    return ratio, bool(exact.value), text


def blocks_per_sm(
    entry: dict,
    device: DeviceFacts,
    threads: int = ARM_THREADS,
    dynamic_shared: int = 0,
) -> int:


    warps = (threads + 31) // 32
    registers = int(entry.get("registers") or 0)
    per_warp = ((registers * 32 + 255) // 256) * 256 if registers else 0
    by_registers = device.registers_per_sm // (per_warp * warps) if per_warp else 64
    shared = int(entry.get("static_shared_bytes") or 0) + int(dynamic_shared)
    by_shared = device.shared_per_sm // shared if shared else 64
    by_threads = device.max_threads_per_sm // threads
    return int(min(by_registers, by_shared, by_threads))


def screen(entry: Optional[dict], device: DeviceFacts) -> tuple:


    if entry is None:
        return False, "no such instance in kernel_resources.json"
    stores, loads = entry.get("spill_store_bytes"), entry.get("spill_load_bytes")
    blocks = blocks_per_sm(entry, device)
    if stores is None or loads is None:
        return False, f"spill unknown, {blocks} blocks/SM"
    if stores or loads:
        return False, f"spills {stores}+{loads} B, {blocks} blocks/SM"
    if blocks < TARGET_BLOCKS_PER_SM:
        return False, f"no spill, {blocks} blocks/SM < {TARGET_BLOCKS_PER_SM}"
    return True, f"no spill, {blocks} blocks/SM"


def bb_pattern(prefetch: bool, alphas: int) -> str:

    if prefetch:
        return rf"beta_singles_pair_hoist_kernelILi{alphas}ELi2ELb0E"
    return rf"beta_singles_pair_kernelILi{alphas}ELb0E"


def aa_pattern(prefetch: bool, alphas: int = 4) -> str:

    return rf"alpha_singles_pair_{'hoist_' if prefetch else ''}kernelILi{alphas}E"


def clique_shared_bytes(members: int, edges_per_batch: int, rows: int) -> int:

    from .distributed.aabb_parents import clique_shared_bytes as shared  # noqa: PLC0415

    return int(shared(members, edges_per_batch, rows))


def aaaa_block_group(sell_bytes_per_block: float, device: DeviceFacts) -> tuple:


    l2 = device.l2_bytes / max(1, device.ranks_on_device)
    low, high = AAAA_GROUP_RANGE
    if sell_bytes_per_block <= 0:
        return high, "empty SELL table"
    group = int(AAAA_GROUP_L2_SHARE * l2 // sell_bytes_per_block)
    return (
        max(low, min(high, group)),
        f"0.15 x {l2 / 2**20:.0f} MiB L2 / {sell_bytes_per_block / 2**20:.2f} MiB per block",
    )


class _Settings:


    def __init__(self, config: GpuConfig):

        self.config = config

    def explicit(self, name: str) -> Any:

        if name not in self.config:
            return None
        value = self.config.value(name)
        return None if value == AUTO else value

    def static(self, name: str) -> Any:

        from .config import REGISTRY  # noqa: PLC0415

        return REGISTRY[name].default


def resolve(
    config: GpuConfig,
    device: Optional[DeviceFacts],
    kernels: KernelFacts,
    *,
    probe: Optional[Callable[[], tuple]] = None,
    can_split_diagonal: Optional[bool] = True,
    agree: Optional[Callable[[bool], list]] = None,
) -> Resolution:


    settings = _Settings(config)
    result = Resolution()
    tune = config.value("FQ_TUNE")
    tier = int(config.value("FQ_HBM_SAVE_TIER"))
    result.choices["tune"] = Choice(
        tune, "explicit" if "FQ_TUNE" in config else "static", ""
    )
    result.choices["tier"] = Choice(
        tier, "explicit" if "FQ_HBM_SAVE_TIER" in config else "static", ""
    )
    archs = tuple(kernels.archs)
    result.choices["archs"] = Choice(list(archs), "build", kernels.source)
    if device is not None and archs and device.arch not in archs:
        result.errors.append(
            f"the kernel library was built for {', '.join(archs)}, not for this {device.arch} "
            f"device ({device.name}); rebuild with FQ_CUDA_ARCH={device.arch}"
        )
    static = tune == "off" or device is None
    mma_fast, mma_source, mma_reason = (
        (None, "static", "") if static else _fp64_mma_rule(settings, device, probe)
    )
    if agree is not None:
        everyone = agree(None if static else bool(mma_fast))
        slow = [rank for rank, value in enumerate(everyone) if value is False]
        if mma_fast and slow:
            mma_fast, mma_reason = (
                False,
                f"rank(s) {slow} report slow FP64 matrix multiply",
            )
    if static:
        _resolve_static(
            settings, result, "FQ_TUNE=off" if tune == "off" else "no device"
        )
        return result
    result.choices["fp64_mma"] = Choice(mma_fast, mma_source, mma_reason)
    _resolve_mma_levers(settings, result, tier, mma_fast)
    _resolve_memory(settings, result, tier, device, can_split_diagonal)
    _resolve_kernels(settings, result, device, kernels)
    return result


def _resolve_static(settings: _Settings, result: Resolution, why: str) -> None:


    def pick(key, name):

        value = settings.explicit(name)
        if value is not None:
            return Choice(value, "explicit", "")
        return Choice(settings.static(name), "static", why)

    result.choices["fp64_mma"] = Choice(None, "static", why)
    result.choices["mma"] = pick("mma", "FQ_AABB_CLIQUE_USE_FP64_MMA")
    lbg = pick("label_batching", "FQ_AABB_CLIQUE_USE_LABEL_BATCHING")
    result.choices["label_batching"] = Choice(
        bool(lbg.value) and bool(result.choices["mma"].value), lbg.source, lbg.reason
    )
    result.choices["source_panel"] = pick(
        "source_panel", "FQ_AABB_OWN_CLIQUE_USE_SOURCE_PANEL"
    )
    result.choices["bbbb_class"] = pick("bbbb_class", "FQ_BBBB_CLASS_USE_FP64_GEMM")
    result.choices["aaaa_class"] = pick("aaaa_class", "FQ_AAAA_CLASS_USE_FP64_GEMM")
    scratch = pick("scratch_mib", "FQ_AABB_CLIQUE_SCRATCH_LIMIT_MB")
    if result.choices["label_batching"].value and not scratch.value:
        scratch = Choice(
            LBG_FORCED_SCRATCH_MIB,
            scratch.source,
            "label batching with limit 0 uses 512 MiB",
        )
    result.choices["scratch_mib"] = scratch
    result.choices["aaaa_slab"] = pick("aaaa_slab", "FQ_AAAA_CLASS_BETA_ROWS_PER_BATCH")
    result.choices["diag_recompute"] = pick("diag_recompute", "FQ_DIAG_RECOMPUTE")
    result.choices["double_buffer"] = pick("double_buffer", "FQ_EXCHANGE_DOUBLE_BUFFER")
    result.choices["hbm_free_fraction"] = pick(
        "hbm_free_fraction", "FQ_HBM_FREE_FRACTION"
    )
    exchange = settings.config.value("FQ_EXCHANGE")
    result.choices["exchange"] = Choice(
        exchange, "explicit" if "FQ_EXCHANGE" in settings.config else "static", why
    )
    result.choices["clique_rows"] = pick(
        "clique_rows", "FQ_AABB_CLIQUE_OUTPUT_ROWS_PER_TILE"
    )
    result.choices["aa_prefetch"] = pick(
        "aa_prefetch", "FQ_AA_PAIR_PREFETCH_BEFORE_SCREEN"
    )
    prefetch = pick("bb_prefetch", "FQ_BB_PAIR_PREFETCH_BEFORE_SCREEN")
    result.choices["bb_prefetch"] = prefetch
    alphas = settings.explicit("FQ_BB_PAIR_ALPHAS_PER_THREAD")
    if alphas is not None:
        result.choices["bb_alphas"] = Choice(int(alphas), "explicit", "")
    else:
        alphas = (
            4
            if prefetch.value
            else int(settings.static("FQ_BB_PAIR_ALPHAS_PER_THREAD"))
        )
        result.choices["bb_alphas"] = Choice(alphas, "static", why)
    result.choices["aaaa_group"] = pick("aaaa_group", "FQ_AAAA_ALPHA_BLOCKS_PER_GROUP")


def _fp64_mma_rule(settings: _Settings, device: DeviceFacts, probe) -> tuple:

    umbrella = settings.config.value("FQ_FP64_MMA")
    if umbrella in ("0", "1"):
        return umbrella == "1", "explicit", "FQ_FP64_MMA"
    if device.capability in FAST_MMA_ARCHS:
        return True, "table", f"{device.arch}: {FAST_MMA_ARCHS[device.capability]}"
    if probe is None:
        return (
            False,
            "rule",
            f"{device.arch} is not in the platform table and the probe did not run",
        )
    try:
        ratio, exact, text = probe()
    except (OSError, RuntimeError, AttributeError) as error:
        return (
            False,
            "probe",
            f"the probe failed ({error}); FP64 matrix multiply treated as slow",
        )
    if not exact:
        return False, "probe", f"{text}; the fragment check was not exact"
    return (
        ratio >= MMA_FLOP_PER_BYTE,
        "probe",
        f"{text} (fast from {MMA_FLOP_PER_BYTE:g})",
    )


def _lever(
    settings: _Settings,
    result: Resolution,
    key: str,
    name: str,
    preferred: bool,
    reason: str,
    source: str,
    needs_mma: bool,
) -> None:


    explicit = settings.explicit(name)
    if explicit is None:
        result.choices[key] = Choice(bool(preferred), source, reason)
        return
    if explicit and needs_mma and not result.choices["mma"].value:
        result.errors.append(
            f"{name}=1 needs the clique matrix multiply (FQ_AABB_CLIQUE_USE_FP64_MMA), which is off"
        )
    if bool(explicit) != bool(preferred):
        result.warnings.append(
            f"Using explicit {name}={int(bool(explicit))}; automatic tuning recommends "
            f"{int(bool(preferred))} because {reason}"
        )
    result.choices[key] = Choice(bool(explicit), "explicit", "")


def _resolve_mma_levers(
    settings: _Settings, result: Resolution, tier: int, mma_fast: bool
) -> None:

    source = result.choices["fp64_mma"].source
    why = result.choices["fp64_mma"].reason
    _lever(
        settings,
        result,
        "mma",
        "FQ_AABB_CLIQUE_USE_FP64_MMA",
        mma_fast,
        why,
        source,
        False,
    )
    tier_off = tier >= 3
    heavy = mma_fast and not tier_off
    heavy_source = "tier" if mma_fast and tier_off else source
    heavy_why = "tier 3 turns it off" if mma_fast and tier_off else why
    _lever(
        settings,
        result,
        "label_batching",
        "FQ_AABB_CLIQUE_USE_LABEL_BATCHING",
        heavy and result.choices["mma"].value,
        heavy_why,
        heavy_source,
        True,
    )
    _lever(
        settings,
        result,
        "source_panel",
        "FQ_AABB_OWN_CLIQUE_USE_SOURCE_PANEL",
        result.choices["mma"].value and not tier_off,
        "tier 3 turns it off" if tier_off else "follows the clique matrix multiply",
        "tier" if tier_off else "rule",
        True,
    )
    _lever(
        settings,
        result,
        "bbbb_class",
        "FQ_BBBB_CLASS_USE_FP64_GEMM",
        heavy,
        heavy_why,
        heavy_source,
        False,
    )
    _lever(
        settings,
        result,
        "aaaa_class",
        "FQ_AAAA_CLASS_USE_FP64_GEMM",
        heavy,
        heavy_why,
        heavy_source,
        False,
    )


def _resolve_memory(
    settings: _Settings,
    result: Resolution,
    tier: int,
    device: DeviceFacts,
    can_split_diagonal: Optional[bool],
) -> None:

    explicit = settings.explicit("FQ_AABB_CLIQUE_SCRATCH_LIMIT_MB")
    if explicit is not None:
        result.choices["scratch_mib"] = Choice(
            int(explicit), "explicit", "0 = uncapped" if not explicit else ""
        )
    elif tier >= 3:
        result.choices["scratch_mib"] = Choice(
            0, "tier", "tier 3: no lever uses the scratch"
        )
    else:
        cap = SCRATCH_TIER_MIB[tier]
        small = (
            int(SCRATCH_HBM_SHARE * device.total_bytes / max(1, device.ranks_on_device))
            >> 20
        )
        value = min(cap, max(1, small))
        result.choices["scratch_mib"] = Choice(
            value,
            "tier",
            f"tier {tier}: {cap} MiB"
            + (f", 1% of HBM = {small} MiB" if small < cap else ""),
        )
    explicit = settings.explicit("FQ_AAAA_CLASS_BETA_ROWS_PER_BATCH")
    if explicit is not None:
        result.choices["aaaa_slab"] = Choice(int(explicit), "explicit", "")
    else:
        result.choices["aaaa_slab"] = Choice(
            int(settings.static("FQ_AAAA_CLASS_BETA_ROWS_PER_BATCH")), "static", ""
        )
    explicit = settings.explicit("FQ_DIAG_RECOMPUTE")
    if explicit is not None:
        result.choices["diag_recompute"] = Choice(bool(explicit), "explicit", "")
    elif tier >= 1 and can_split_diagonal is False:
        result.choices["diag_recompute"] = Choice(
            False, "rule", "the operator cannot split its diagonal; stored"
        )
    else:
        result.choices["diag_recompute"] = Choice(tier >= 1, "tier", f"tier {tier}")
    explicit = settings.explicit("FQ_EXCHANGE_DOUBLE_BUFFER")
    if explicit is not None:
        result.choices["double_buffer"] = Choice(bool(explicit), "explicit", "")
    else:
        result.choices["double_buffer"] = Choice(tier < 3, "tier", f"tier {tier}")
    explicit = settings.explicit("FQ_HBM_FREE_FRACTION")
    if explicit is not None:
        result.choices["hbm_free_fraction"] = Choice(float(explicit), "explicit", "")
    else:
        result.choices["hbm_free_fraction"] = Choice(
            1.2, "rule", "the HBM prediction overestimates"
        )
    exchange = settings.config.value("FQ_EXCHANGE")
    if "FQ_EXCHANGE" in settings.config and exchange != "auto":
        if tier >= 3 and exchange != "ring":
            result.warnings.append(
                f"Using explicit FQ_EXCHANGE={exchange}; HBM save tier {tier} recommends ring"
            )
        result.choices["exchange"] = Choice(exchange, "explicit", "")
    elif tier >= 3:
        result.choices["exchange"] = Choice(
            "ring", "tier", "tier 3: the ring exchange with one receive slot"
        )
    else:
        result.choices["exchange"] = Choice(
            "auto", "rule", "two-level when it can run, else ring"
        )


def _resolve_kernels(
    settings: _Settings, result: Resolution, device: DeviceFacts, kernels: KernelFacts
) -> None:

    arch = device.arch
    prefetch = settings.explicit("FQ_BB_PAIR_PREFETCH_BEFORE_SCREEN")
    alphas = settings.explicit("FQ_BB_PAIR_ALPHAS_PER_THREAD")
    screened = {}
    for candidate in BB_CANDIDATES:
        ok, text = screen(kernels.instance(bb_pattern(*candidate), arch), device)
        screened[candidate] = ok
        result.candidates[f"bb prefetch={int(candidate[0])} alphas={candidate[1]}"] = (
            text
        )
    if prefetch is not None or alphas is not None:
        chosen_prefetch = bool(prefetch) if prefetch is not None else None
        chosen_alphas = int(alphas) if alphas is not None and int(alphas) else None
        if chosen_prefetch is None:
            options = [
                c for c in BB_CANDIDATES if c[1] == chosen_alphas and screened[c]
            ]
            chosen_prefetch = options[0][0] if options else False
        if chosen_alphas is None:
            options = [
                c for c in BB_CANDIDATES if c[0] == chosen_prefetch and screened[c]
            ]
            chosen_alphas = options[0][1] if options else (4 if chosen_prefetch else 2)
        result.choices["bb_prefetch"] = Choice(
            chosen_prefetch, "explicit" if prefetch is not None else "rule", ""
        )
        result.choices["bb_alphas"] = Choice(
            chosen_alphas, "explicit" if alphas is not None else "rule", ""
        )
    else:
        passing = [c for c in BB_CANDIDATES if screened[c]]
        chosen = passing[0] if passing else BB_CANDIDATES[-1]
        why = (
            "first candidate that passes the screen"
            if passing
            else "no candidate passes; the screen-first kernel"
        )
        result.choices["bb_prefetch"] = Choice(chosen[0], "rule", why)
        result.choices["bb_alphas"] = Choice(chosen[1], "rule", why)
    explicit = settings.explicit("FQ_AA_PAIR_PREFETCH_BEFORE_SCREEN")
    ok, text = screen(kernels.instance(aa_pattern(True), arch), device)
    result.candidates["aa prefetch=1 alphas=4"] = text
    if explicit is not None:
        result.choices["aa_prefetch"] = Choice(bool(explicit), "explicit", "")
    else:
        result.choices["aa_prefetch"] = Choice(ok, "rule", text)
    explicit = settings.explicit("FQ_AABB_CLIQUE_OUTPUT_ROWS_PER_TILE")
    if explicit is not None:
        result.choices["clique_rows"] = Choice(int(explicit), "explicit", "")
    else:
        edges = int(settings.config.value("FQ_AABB_CLIQUE_EDGES_PER_BATCH"))
        shared = clique_shared_bytes(CLIQUE_PROBE_BUCKET_MEMBERS, edges, CLIQUE_ROWS[0])
        blocks = device.shared_per_sm // shared if shared else 64
        rows = CLIQUE_ROWS[0] if blocks >= 2 else CLIQUE_ROWS[1]
        result.choices["clique_rows"] = Choice(
            rows, "rule", f"{blocks} blocks of the 32-member bucket at 64 rows per SM"
        )
    explicit = settings.explicit("FQ_AAAA_ALPHA_BLOCKS_PER_GROUP")
    if explicit is not None:
        result.choices["aaaa_group"] = Choice(int(explicit), "explicit", "")
    else:
        result.choices["aaaa_group"] = Choice(
            AUTO, "rule", "per source from the L2 size"
        )


def probe_for(device: Optional[DeviceFacts]) -> Optional[Callable[[], tuple]]:

    if device is None or device.capability in FAST_MMA_ARCHS:
        return None
    return probe_flop_per_byte


def describe_table() -> str:

    return "; ".join(
        f"sm_{major}{minor}: fast ({why})"
        for (major, minor), why in sorted(FAST_MMA_ARCHS.items())
    )


def settings_for_record(resolution: Resolution) -> Mapping[str, Any]:

    return {
        name: {
            "value": resolution.choices[name].value,
            "source": resolution.choices[name].source,
        }
        for name in COLLECTIVE
        if name in resolution.choices
    }
