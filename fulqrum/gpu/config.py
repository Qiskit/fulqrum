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

import hashlib
import json
import os
import warnings
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

TUNABLE = "tunable"
DIAGNOSTIC = "diagnostic"
FIXED = "fixed"
TEST = "test"
BUILD = "build"


AUTO = "auto"
HBM_CLASSES = ("none", "tiny", "scratch", "proportional", "vector")

TRUE_TEXTS = ("1", "true", "yes", "on")
FALSE_TEXTS = ("0", "false", "no", "off")

SECTIONS = (
    "Distributed matvec: transport and exchange",
    "Distributed matvec: memory, partitioning, construction",
    "Distributed matvec: same-sector single and double excitations",
    "Distributed matvec: mixed-spin double excitations",
    "Distributed matvec: automatic settings",
    "Distributed PRIMME eigensolver",
    "RDM",
    "Test hooks",
    "Build time",
)


@dataclass(frozen=True)
class Flag:
    name: str
    section: str
    group: str
    kind: str
    default: Any
    description: str
    choices: tuple = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default_text: str = ""
    auto: bool = False
    auto_text: str = ""
    hbm_class: str = "none"
    expert: bool = False


def _flag(name, section, group, kind, default, description, **options):

    return Flag(name, SECTIONS[section], group, kind, default, description, **options)


FLAGS = (
    _flag(
        "FQ_DIST_BACKEND",
        0,
        TUNABLE,
        "choice",
        "auto",
        "Transport for distributed GPU vector exchanges. `host_mpi` stages transfers through host memory by default; `cuda_mpi` requires CUDA-aware MPI; `nccl` uses NCCL. `auto` selects NCCL when all ranks support it and have distinct GPUs, otherwise host MPI. Automatic selection does not choose `cuda_mpi`.",
        choices=("auto", "host_mpi", "cuda_mpi", "nccl"),
    ),
    _flag(
        "FQ_EXCHANGE",
        0,
        TUNABLE,
        "choice",
        "auto",
        "Schedule for exchanging vector tiles. `ring` exchanges tiles in rounds; `two_level` groups communication within and between node domains. `auto` selects two-level only with NCCL, at least two domains and at least one multi-rank domain; otherwise ring. Memory tier 3 selects ring automatically. An explicitly requested unsupported schedule raises.",
        choices=("auto", "ring", "two_level"),
    ),
    _flag(
        "FQ_EXCHANGE_DOUBLE_BUFFER",
        0,
        TUNABLE,
        "bool",
        True,
        "Overlap ring transfers and computation using two receive buffers. `0` uses one buffer and reduces GPU memory at the cost of overlap. Applies only to ring exchange; two-level exchange retains its own two-buffer schedule.",
        auto=True,
        hbm_class="vector",
        auto_text="`1` at tiers 0 and 1; `0` at tier 3. Only ring schedules use this setting",
    ),
    _flag(
        "FQ_EXCHANGE_SYNC",
        0,
        DIAGNOSTIC,
        "bool",
        False,
        "Synchronize the CUDA device around NCCL exchanges for diagnostics. Read when the NCCL backend module is imported.",
    ),
    _flag(
        "FQ_HOST_MPI_CHUNK_MiB",
        0,
        TUNABLE,
        "float",
        256.0,
        "Maximum host-staged MPI transfer chunk size, in MiB. All ranks must use the same value.",
        minimum=0.001,
    ),
    _flag(
        "FQ_HOST_MPI_TIMEOUT",
        0,
        TUNABLE,
        "float",
        900.0,
        "Maximum seconds without progress before a host-MPI transfer raises an error. `0` waits indefinitely. Errors identify the rank, peer and transfer state.",
        minimum=0.0,
    ),
    _flag(
        "FQ_HOST_MPI_USE_CUDA_IPC",
        0,
        TUNABLE,
        "bool",
        False,
        "Use CUDA inter-process communication for direct GPU-to-GPU copies within a node on the host-MPI backend. Requires all participating GPUs to be visible to each rank. A startup capability check disables this optimization collectively when unavailable and records the reason.",
    ),
    _flag(
        "FQ_COMM_TIMING",
        0,
        DIAGNOSTIC,
        "bool",
        False,
        "Measure GPU communication with CUDA events and include the measurements in matvec timing records.",
    ),
    _flag(
        "FQ_HBM_FREE_FRACTION",
        1,
        TUNABLE,
        "float",
        1.0,
        "Maximum ratio of predicted allocation to currently free GPU memory. `1` requires the prediction to fit; values below `1` reserve headroom and values above `1` relax the prediction check.",
        minimum=0.0,
        auto=True,
        auto_text="`1.20`, relaxing the check for conservative allocation predictions",
    ),
    _flag(
        "FQ_HBM_LOG",
        1,
        DIAGNOSTIC,
        "bool",
        False,
        "Log rank-0 GPU and host-memory reports at DEBUG level. Use Detailed or Trace logging to display them.",
    ),
    _flag(
        "FQ_DIAG_RECOMPUTE",
        1,
        TUNABLE,
        "bool",
        False,
        "Recompute diagonal contributions during matvec instead of storing a full diagonal vector on the GPU. Explicit `1` raises if the diagonal cannot be represented in the required factored form.",
        auto=True,
        hbm_class="vector",
        auto_text="`0` at tier 0; `1` at tiers 1 and 3 when the diagonal can be factored, otherwise stored",
    ),
    _flag(
        "FQ_ARM_TABLE_SOURCE",
        1,
        TUNABLE,
        "choice",
        "auto",
        "Where same-sector connection values are built. `host` computes them on the CPU during construction; `kernel` computes them on the GPU during the first matvec. `auto` uses host construction when supported, otherwise kernel construction. An unsupported explicit `host` request raises.",
        choices=("auto", "host", "kernel"),
    ),
    _flag(
        "FQ_BUILD_THREADS",
        1,
        TUNABLE,
        "int",
        0,
        "CPU threads used to construct GPU tile data. `0` uses the constructor's `build_threads` argument, then `OMP_NUM_THREADS`.",
        minimum=0,
    ),
    _flag(
        "FQ_TILE_RESIDENT_ARM_TIMING",
        1,
        DIAGNOSTIC,
        "bool",
        False,
        "Collect GPU device-clock timings for individual matvec contributions inside the kernels.",
    ),
    _flag(
        "FQ_AA_PAIR_TABLE",
        2,
        FIXED,
        "bool",
        None,
        "Use the alpha single-excitation pair table when the required coefficient caches are available. Each source is checked for exact agreement with the standard representation; failed checks use the fallback kernel.",
        default_text="enabled when the required caches are available",
    ),
    _flag(
        "FQ_BB_PAIR_TABLE",
        2,
        FIXED,
        "bool",
        None,
        "Use the corresponding pair-table representation for beta single excitations when its required caches are available.",
        default_text="enabled when the required caches are available",
    ),
    _flag(
        "FQ_AAAA_ALPHA_BLOCKS_PER_GROUP",
        2,
        TUNABLE,
        "int",
        8,
        "Number of alpha blocks processed together by the alpha double-excitation kernel to reuse sparse tables in the GPU L2 cache. `0` uses the standard two-dimensional block order.",
        minimum=0,
        auto=True,
        auto_text="15% of GPU L2 capacity divided by sparse-table bytes per alpha block, rounded down and limited to 2–8 blocks",
    ),
    _flag(
        "FQ_BBBB_ALPHA_BLOCKS_PER_GROUP",
        2,
        TUNABLE,
        "int",
        1,
        "Number of alpha blocks processed together by the beta double-excitation kernel to reuse source data in the GPU L2 cache. `0` uses the standard two-dimensional block order.",
        minimum=0,
        expert=True,
    ),
    _flag(
        "FQ_BBBB_CLASS_USE_FP64_GEMM",
        2,
        TUNABLE,
        "bool",
        False,
        "Apply grouped beta double-excitation connections as dense FP64 matrix multiplications. Smaller groups retain the sparse kernel. Requires additional GPU storage for class edges and can change floating-point accumulation order. Non-finite inputs use the scalar fallback.",
        auto=True,
        hbm_class="proportional",
        auto_text="enabled by the FP64 matrix-multiply rule, disabled automatically at tier 3",
    ),
    _flag(
        "FQ_BBBB_CLASS_MIN_SIZE",
        2,
        TUNABLE,
        "int",
        24,
        "Minimum class size, in rows and columns, for the beta double-excitation matrix-multiply kernel. Smaller classes use the sparse kernel.",
        minimum=1,
        expert=True,
    ),
    _flag(
        "FQ_BBBB_CLASS_ALPHA_CHUNKS_PER_GROUP",
        2,
        TUNABLE,
        "int",
        4,
        "Number of 64-alpha-string chunks processed together by the beta class kernel to reuse source and output data in the GPU L2 cache.",
        minimum=1,
        expert=True,
    ),
    _flag(
        "FQ_AAAA_CLASS_USE_FP64_GEMM",
        2,
        TUNABLE,
        "bool",
        False,
        "Apply grouped alpha double-excitation connections as dense FP64 matrix multiplications over batches of beta rows. Smaller classes retain the sparse kernel. Uses additional class metadata and batch scratch memory, and can change floating-point accumulation order.",
        auto=True,
        hbm_class="proportional",
        auto_text="enabled by the FP64 matrix-multiply rule, disabled automatically at tier 3",
    ),
    _flag(
        "FQ_AAAA_CLASS_MIN_SIZE",
        2,
        TUNABLE,
        "int",
        24,
        "Minimum class size, in rows and columns, for the alpha double-excitation matrix-multiply kernel.",
        minimum=1,
        expert=True,
    ),
    _flag(
        "FQ_AAAA_CLASS_TRANSPOSE_RESIDUAL",
        2,
        TUNABLE,
        "bool",
        True,
        "With alpha class matrix multiplication enabled, process connections outside the classes in the transposed beta-row batches. `0` processes them with the standard alpha sparse kernel afterwards. May change floating-point accumulation order.",
        expert=True,
    ),
    _flag(
        "FQ_AAAA_CLASS_BETA_ROWS_PER_BATCH",
        2,
        TUNABLE,
        "int",
        2048,
        "Beta rows per alpha class-multiply batch. Scratch storage contains this many rows times the total source and output alpha-string counts, in float64 elements.",
        minimum=64,
        hbm_class="scratch",
        expert=True,
    ),
    _flag(
        "FQ_BB_PAIR_ALPHA_BLOCKS_PER_GROUP",
        2,
        TUNABLE,
        "int",
        1,
        "Number of alpha blocks processed together by the beta pair-table kernel for cache reuse. `0` keeps the standard two-dimensional block order.",
        minimum=0,
        expert=True,
    ),
    _flag(
        "FQ_BB_PAIR_ALPHAS_PER_THREAD",
        2,
        TUNABLE,
        "int",
        2,
        "Alpha strings processed by each beta pair-table thread. `0` selects four; the delta representation also uses four. Automatic selection considers this value together with beta prefetching.",
        minimum=0,
        maximum=8,
        auto=True,
        auto_text="selected together with beta prefetching from compiled-kernel resource information",
    ),
    _flag(
        "FQ_AA_PAIR_PREFETCH_BEFORE_SCREEN",
        2,
        TUNABLE,
        "bool",
        True,
        "Load alpha pair-table coefficients and source amplitudes before testing whether a connection contributes. `0` tests the connection first. Automatic selection uses kernel resource information.",
        auto=True,
        auto_text="enabled when the kernel has no register spills and permits at least four blocks per GPU multiprocessor",
    ),
    _flag(
        "FQ_BB_PAIR_PREFETCH_BEFORE_SCREEN",
        2,
        TUNABLE,
        "bool",
        False,
        "Load beta pair-table connection data and source amplitudes before testing whether they contribute. `0` tests first. Automatic selection also chooses the alpha strings processed per thread.",
        auto=True,
        auto_text="first viable (prefetch, alphas/thread) candidate: (1,4), (1,2), (1,1), (0,2); requires no register spills and four blocks per GPU multiprocessor",
    ),
    _flag(
        "FQ_AABB_CLIQUE_ENABLE",
        3,
        FIXED,
        "bool",
        None,
        "Use grouped mixed-spin connections that share source loads, with a sparse residual for remaining connections. Enabled when the supported packed sparse-table configuration is available.",
        default_text="enabled for the supported default packed sparse-table configuration",
    ),
    _flag(
        "FQ_AABB_CLIQUE_MIN_MEMBERS",
        3,
        TUNABLE,
        "int",
        6,
        "Minimum number of half strings in a clique for the grouped kernel. Smaller groups remain on the residual path.",
        minimum=2,
        maximum=32,
    ),
    _flag(
        "FQ_AABB_TIME_CLIQUE_KERNELS",
        3,
        DIAGNOSTIC,
        "bool",
        False,
        "Measure mixed-spin clique kernel durations with CUDA events.",
    ),
    _flag(
        "FQ_AABB_CLIQUE_GROUPING_KIND",
        3,
        TUNABLE,
        "choice",
        "auto",
        "Build cliques from shared occupied-orbital sets: `union` uses an N+1-electron parent, `intersection` an N-1-electron parent. N is the sector electron count. `auto` chooses the grouping with greater connection coverage.",
        choices=("auto", "union", "intersection"),
        expert=True,
    ),
    _flag(
        "FQ_AABB_CLIQUE_OUTPUT_ROWS_PER_TILE",
        3,
        TUNABLE,
        "int",
        64,
        "Output rows processed by each clique tile: beta rows for local and remote-beta sources, alpha rows for transposed remote-alpha sources.",
        minimum=1,
        maximum=1024,
        auto=True,
        auto_text="64 rows, reduced to 32 when shared-memory capacity cannot support two blocks for the 32-member group at 64 rows",
    ),
    _flag(
        "FQ_AABB_CLIQUE_EDGES_PER_BATCH",
        3,
        TUNABLE,
        "choice",
        "4",
        "Connections processed together in each clique-kernel batch; choose four or eight.",
        choices=("4", "8"),
        expert=True,
    ),
    _flag(
        "FQ_AABB_OWN_CLIQUES_PER_GROUP",
        3,
        TUNABLE,
        "int",
        0,
        "Number of parent cliques whose output-row tiles are processed together for local-source work. `0` uses the standard parent-first order. Changes scheduling and atomic accumulation order.",
        minimum=0,
        expert=True,
    ),
    _flag(
        "FQ_AABB_CLIQUE_USE_FP64_MMA",
        3,
        TUNABLE,
        "bool",
        False,
        "Use FP64 tensor-core matrix multiplication for mixed-spin clique contributions. May change accumulation order. Non-finite input chunks use the scalar fallback.",
        auto=True,
        auto_text="selected by the FP64 matrix-multiply rule at every memory tier",
    ),
    _flag(
        "FQ_AABB_OWN_CLIQUE_USE_SOURCE_PANEL",
        3,
        TUNABLE,
        "bool",
        False,
        "With clique matrix multiplication enabled, pack local source amplitudes into a reusable GPU panel. Uses temporary device memory controlled by the clique scratch limit. Non-finite panels use the standard matrix-multiply path; remote sources retain that path.",
        auto=True,
        hbm_class="scratch",
        auto_text="enabled with clique matrix multiplication, disabled automatically at tier 3",
    ),
    _flag(
        "FQ_AABB_CLIQUE_USE_LABEL_BATCHING",
        3,
        TUNABLE,
        "bool",
        False,
        "With clique matrix multiplication enabled, batch connections sharing a beta pair across output rows and combine their results through temporary panels. Applies to local and remote-beta sources with at most 65,536 output rows. Larger ranges, remote-alpha sources and non-finite inputs use the standard kernels. Shares the clique scratch budget.",
        auto=True,
        hbm_class="scratch",
        auto_text="enabled by the FP64 matrix-multiply rule, disabled automatically at tier 3",
    ),
    _flag(
        "FQ_AABB_CLIQUE_SCRATCH_LIMIT_MB",
        3,
        TUNABLE,
        "int",
        0,
        "Maximum mixed-spin panel scratch size per rank, in MiB. Batches are sized to fit this buffer. Explicit `0` allows the largest full panel; with tuning off, label batching interprets `0` as 512 MiB. Smaller limits reduce memory use but can require more launches.",
        minimum=0,
        auto=True,
        hbm_class="scratch",
        auto_text="512 MiB at tiers 0 and 1, limited to 1% of total GPU memory; zero scratch at tier 3",
    ),
    _flag(
        "FQ_AABB_CLIQUE_RESIDUAL_DEGREE_SORT",
        3,
        TUNABLE,
        "bool",
        True,
        "Sort residual sparse rows by connection count and omit empty rows so nearby GPU threads process similar lengths. Uses row-index metadata and, for remote-alpha sources, a sorted sparse-table copy.",
        expert=True,
    ),
    _flag(
        "FQ_AABB_REMOTE_BETA_METHOD",
        3,
        FIXED,
        "choice",
        None,
        "Kernel family for mixed-spin connections from remote-beta sources: grouped cliques when enabled, otherwise the standard kernel.",
        choices=("baseline", "clique"),
        default_text="`clique` when clique processing is enabled, otherwise `baseline`",
        expert=True,
    ),
    _flag(
        "FQ_AABB_REMOTE_ALPHA_METHOD",
        3,
        FIXED,
        "choice",
        None,
        "Kernel family for mixed-spin connections from remote-alpha sources: transposed beta cliques when enabled, otherwise the standard sparse kernel.",
        choices=("baseline", "transposed_clique"),
        default_text="`transposed_clique` when clique processing is enabled, otherwise `baseline`",
        expert=True,
    ),
    _flag(
        "FQ_AABB_TIME_EACH_SOURCE",
        3,
        DIAGNOSTIC,
        "bool",
        False,
        "Collect separate mixed-spin timing measurements for each source tile.",
    ),
    _flag(
        "FQ_TUNE",
        4,
        TUNABLE,
        "choice",
        "profile",
        "Resolve automatic settings during Hamiltonian construction. `profile` uses GPU architecture rules, startup probes and the memory-saving tier; `off` uses static defaults. This setting does not enable profiling output.",
        choices=("profile", "off"),
    ),
    _flag(
        "FQ_HBM_SAVE_TIER",
        4,
        TUNABLE,
        "choice",
        "1",
        "Memory-saving level for tensor-product GPU matvecs. `0` permits all automatic optimizations and stores the diagonal. `1` recomputes eligible diagonals. `3` also selects a single-buffer ring and disables automatic label batching, source panels, class matrix multiplies and cross-term scratch. Explicit feature settings take precedence. Does not apply to FullStrsHamiltonian.",
        choices=("0", "1", "3"),
    ),
    _flag(
        "FQ_FP64_MMA",
        4,
        TUNABLE,
        "choice",
        "auto",
        "Default preference for FP64 matrix-multiply optimizations. `auto` uses a known-architecture preset or a startup probe with a minimum ratio of 3 FLOP per byte. `0` and `1` set the default preference explicitly. Memory-saving tiers and explicit per-feature settings still apply.",
        choices=("auto", "0", "1"),
    ),
    _flag(
        "FQ_PRIMME_MPI_SO",
        5,
        TEST,
        "text",
        "",
        "Test-only path to an alternative compiled PRIMME MPI extension.",
    ),
    _flag(
        "FQ_PRIMME_GPU_PRECOND",
        5,
        TUNABLE,
        "bool",
        False,
        "Apply the shifted Jacobi preconditioner on the GPU while using the host PRIMME backend. Requires a supplied local diagonal.",
    ),
    _flag(
        "FQ_PRIMME_GPU_PRECOND_TILE",
        5,
        TUNABLE,
        "int",
        1 << 24,
        "Local vector elements processed per GPU Jacobi-preconditioner chunk on the host PRIMME backend.",
        minimum=1,
    ),
    _flag(
        "FQ_PRIMME_MANAGED_BASIS",
        5,
        TUNABLE,
        "bool",
        False,
        "Use CUDA unified memory for GPU PRIMME basis vectors and their Hamiltonian products, allowing storage in both GPU and host memory. Skips the usual device-only basis memory precheck.",
    ),
    _flag(
        "FQ_PRIMME_MANAGED_BASIS_LOG",
        5,
        DIAGNOSTIC,
        "bool",
        False,
        "Write native managed-basis budget and allocation reports to stderr on each rank. Independent of Python logging; CUDA failure warnings remain enabled without this setting.",
    ),
    _flag(
        "FQ_PRIMME_HBM_BUDGET_GB",
        5,
        TUNABLE,
        "float",
        None,
        "GPU-preferred storage budget for managed PRIMME basis arrays, in GiB. Remaining managed storage is assigned a host-memory preference.",
        minimum=0.0,
        default_text="free GPU memory at first allocation minus the reserve",
    ),
    _flag(
        "FQ_PRIMME_HBM_RESERVE_GB",
        5,
        TUNABLE,
        "float",
        None,
        "GPU memory excluded from the automatically computed managed-basis budget, in GiB.",
        minimum=0.0,
        default_text="4 GiB plus three local float64 vectors",
    ),
    _flag(
        "FQ_PRIMME_GPU_DENSE",
        5,
        TUNABLE,
        "bool",
        False,
        "Offload large dense basis updates from the host PRIMME backend to the GPU when local vector length is at least 1,048,576. Eligible operations that decline offload use the host update; GPU execution failures raise.",
    ),
    _flag(
        "FQ_PRIMME_GPU_DENSE_TILE",
        5,
        TUNABLE,
        "int",
        0,
        "Rows per GPU dense-update chunk on the host PRIMME backend. `0` derives the chunk size from the workspace budget.",
        minimum=0,
    ),
    _flag(
        "FQ_PRIMME_GPU_DENSE_WS_MB",
        5,
        TUNABLE,
        "int",
        1024,
        "GPU workspace budget for dense updates from the host PRIMME backend, in MiB.",
        minimum=1,
    ),
    _flag(
        "FQ_PRIMME_GPU_ORTHO",
        5,
        TUNABLE,
        "bool",
        False,
        "Offload block orthogonalization from the host PRIMME backend to the GPU for blocks containing more than one vector.",
    ),
    _flag(
        "FQ_PRIMME_PINNED_BASIS",
        5,
        TUNABLE,
        "bool",
        True,
        "Store the host PRIMME basis and its Hamiltonian products in page-locked host memory for direct GPU transfers.",
    ),
    _flag(
        "FQ_PRIMME_OVERLAP_MATVEC",
        5,
        TUNABLE,
        "bool",
        True,
        "Overlap a GPU matvec for a new basis vector with CPU orthogonalization on the host PRIMME backend. Requires block size one. With multiple ranks, requires NCCL or MPI_THREAD_MULTIPLE support.",
    ),
    _flag(
        "FQ_PRIMME_SPEC_MATVEC",
        5,
        TUNABLE,
        "bool",
        True,
        "With host-backend matvec overlap enabled, start the next correction matvec early while the CPU performs subsequent solver work. Applies to supported GD+k and GD+Olsen configurations with one eigenpair and either clamped Jacobi or no preconditioner.",
    ),
    _flag(
        "FQ_RDM_VERBOSE",
        6,
        DIAGNOSTIC,
        "bool",
        False,
        "Log rank-0 RDM stage timings at DEBUG level. Use Detailed or Trace logging to display them. Stage measurements synchronize the GPU.",
    ),
    _flag(
        "_FULQRUM_DEBUG_MATVEC",
        7,
        TEST,
        "text",
        "",
        "Test-only per-rank vector fingerprints and debug output. Accepts a true value, `tail` or `per_source`.",
    ),
    _flag(
        "_FULQRUM_NO_OVERLAP",
        7,
        TEST,
        "bool",
        False,
        "Test-only synchronous host-MPI transfers without CUDA stream overlap.",
    ),
    _flag(
        "_FULQRUM_TWO_LEVEL_ANY_BACKEND",
        7,
        TEST,
        "bool",
        False,
        "Test-only permission to run two-level exchange with other backends, including ranks sharing one GPU.",
    ),
    _flag(
        "FQ_BUILD_GPU",
        8,
        BUILD,
        "bool",
        False,
        "Build the complete CUDA execution stack and distributed PRIMME extension. `0` keeps the CPU build configuration.",
    ),
    _flag(
        "FQ_CUDA_ARCH",
        8,
        BUILD,
        "text",
        "",
        "Explicit CUDA architecture target for nvcc, such as `sm_90`. Unset selects visible GPU architectures or toolkit-supported fallback targets.",
        default_text="visible GPU architectures, otherwise supported fallback targets",
    ),
    _flag(
        "FQ_ENABLE_PRIMME_MPI",
        8,
        BUILD,
        "bool",
        False,
        "Build distributed host PRIMME support without enabling CUDA kernels. GPU builds include this extension automatically.",
    ),
    _flag(
        "FQ_BUILD_PARALLEL",
        8,
        BUILD,
        "int",
        1,
        "Positive number of parallel build jobs; defaults to one.",
        minimum=1,
    ),
    _flag(
        "FQ_CUDA_HOST_COMPILER",
        8,
        BUILD,
        "text",
        "",
        "Host C++ compiler executable used by nvcc. Defaults to CXX when set, otherwise nvcc selects its compiler.",
    ),
    _flag(
        "FQ_DISABLE_DIST_BUILD",
        8,
        BUILD,
        "bool",
        False,
        "GPU builds always include the distributed CUDA library. Explicit `1` is rejected when FQ_BUILD_GPU is enabled; use `0` or leave unset.",
    ),
    _flag(
        "FQ_CUDA_MATH_DIR",
        8,
        BUILD,
        "text",
        "",
        "CUDA math-library root containing cuBLAS/cuSPARSE headers and libraries, for split toolkit installations such as the NVIDIA HPC SDK.",
    ),
    _flag(
        "FQ_CUDA_LIBDIR",
        8,
        BUILD,
        "text",
        "",
        "Override CUDA shared-library search directories. Separate multiple paths with colons on Linux; the list must provide cudart, cuBLAS and cuSPARSE.",
    ),
    _flag(
        "FQ_PRIMME_BLASINT64",
        8,
        BUILD,
        "bool",
        True,
        "Distributed PRIMME always uses 64-bit BLAS integers (ILP64). Omit this setting or use `1`; explicit `0` is rejected when PRIMME support is enabled.",
    ),
    _flag(
        "FQ_BLAS_LIB",
        8,
        BUILD,
        "text",
        "",
        "Path to the compatible OpenBLAS ILP64 shared library used by PRIMME. Requires 64-bit integers and the `_64_` BLAS/LAPACK symbol suffix.",
    ),
)

REGISTRY = {flag.name: flag for flag in FLAGS}

NOT_VARIABLES = (
    "These `FQ_*` names in `fulqrum/gpu/` are compile-time macros or constants, so the environment does not "
    "change them: `FQ_GPU_DENSE_ABI`, and `FQ_PRIMME_GPU`, a C define "
    "that the build sets with `FQ_BUILD_GPU=1`."
)


def parse_bool(name: str, text: str) -> bool:

    value = text.strip().lower()
    if value in TRUE_TEXTS:
        return True
    if value in FALSE_TEXTS:
        return False
    raise ValueError(f"{name}={text!r}: use one of {TRUE_TEXTS + FALSE_TEXTS}")


def parse(flag: Flag, text: str) -> Any:

    if flag.auto and text.strip().lower() == AUTO:
        return AUTO
    if flag.kind == "bool":
        return parse_bool(flag.name, text)
    if flag.kind == "text":
        return text
    if flag.kind == "choice":
        value = text.strip().lower()
        if value not in flag.choices:
            raise ValueError(f"{flag.name}={text!r}: use one of {flag.choices}")
        return value
    try:
        value = int(text) if flag.kind == "int" else float(text)
    except ValueError as error:
        raise ValueError(
            f"{flag.name}={text!r}: not a{'n int' if flag.kind == 'int' else ' number'}"
        ) from error
    if flag.minimum is not None and value < flag.minimum:
        raise ValueError(f"{flag.name}={text!r}: smaller than {flag.minimum}")
    if flag.maximum is not None and value > flag.maximum:
        raise ValueError(f"{flag.name}={text!r}: larger than {flag.maximum}")
    return value


def read(
    name: str, environ: Optional[Mapping[str, str]] = None, value: Any = None
) -> Any:

    flag = REGISTRY[name]
    if value is not None:
        return value
    environ = os.environ if environ is None else environ
    text = environ.get(name)
    if text is None or text.strip() == "":
        return flag.default
    return parse(flag, text)


def normalize(flag: Flag, value: Any) -> str:

    if isinstance(value, bool):
        value = "1" if value else "0"
    text = str(value)
    parsed = parse(flag, text)
    if parsed == AUTO and flag.auto:
        return AUTO
    if flag.kind == "bool":
        return "1" if parsed else "0"
    if flag.kind == "choice":
        return parsed
    return text.strip() if flag.kind != "text" else text


class GpuConfig(MappingABC):
    """Stores user-supplied GPU settings.

    Create with ``from_environ()`` to validate values. ``cfg[name]`` returns
    the supplied value as text; ``cfg.value(name)`` returns its parsed value
    or the default if omitted.

    Automatic GPU tuning happens when the Hamiltonian is constructed.
    Its final settings are available in ``run_record``.
    """

    def __init__(self, settings: Mapping[str, str]):

        self._settings = dict(settings)

    @classmethod
    def from_environ(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "GpuConfig":
        """Build a validated snapshot from runtime environment settings and overrides.

        ``environ`` defaults to ``os.environ``. Only registered runtime settings
        are captured; build-time environment settings are handled by the build
        system. Explicit overrides take precedence, and None overrides are ignored.
        Unknown override names raise KeyError; invalid values raise ValueError.
        """
        environ = os.environ if environ is None else environ
        settings = {}
        for flag in FLAGS:
            if flag.group == BUILD:
                continue
            text = environ.get(flag.name)
            if text is not None and text.strip() != "":
                settings[flag.name] = normalize(flag, text)
        for name, value in (overrides or {}).items():
            if value is not None:
                settings[name] = normalize(REGISTRY[name], value)
        return cls(settings)

    def __getitem__(self, name: str) -> str:

        return self._settings[name]

    def __iter__(self) -> Iterator[str]:

        return iter(self._settings)

    def __len__(self) -> int:

        return len(self._settings)

    def value(self, name: str) -> Any:
        """Return a typed configured value, or its registry default when unspecified.

        Does not apply hardware-dependent automatic tuning; resolved execution
        choices are available in the Hamiltonian's ``run_record``.
        """
        return read(name, self._settings)

    def digest(self) -> str:
        """Return a short hash of the settings explicitly present in this snapshot."""
        text = json.dumps(sorted(self._settings.items()))
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def record(self) -> dict:
        """Return the settings explicitly present in this snapshot as a plain dict."""
        return dict(sorted(self._settings.items()))


def unknown_names(environ: Optional[Mapping[str, str]] = None) -> list:

    environ = os.environ if environ is None else environ
    return sorted(
        name
        for name in environ
        if (name.startswith("FQ_") or name.startswith("FULQRUM_"))
        and name not in REGISTRY
    )


def warn_unknown_names(environ: Optional[Mapping[str, str]] = None) -> list:

    names = unknown_names(environ)
    if names:
        warnings.warn(
            f"Unrecognized GPU environment variables will be ignored: {', '.join(names)}. "
            "Check registered names in fulqrum/gpu/config.py.",
            RuntimeWarning,
            stacklevel=2,
        )
    return names
