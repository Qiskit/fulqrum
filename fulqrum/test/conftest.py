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

"""Pytest helpers for GPU and MPI-GPU tests.

`gpu_available()` and `mpi_gpu_available()`: so the GPU test
files can decorate themselves with `@pytest.mark.skipif(not gpu_available(),
...)` without raising at collection time when the GPU build is absent.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Return True when the GPU build is present and a CUDA device is visible."""
    try:
        importlib.import_module("fulqrum.gpu.distributed.kernel_launch")
        import cupy  # noqa: PLC0415

        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001
        return False


@functools.lru_cache(maxsize=1)
def mpi_gpu_available() -> bool:
    """True iff GPU is available AND mpi4py + _primme_mpi import cleanly."""
    if not gpu_available():
        return False
    if importlib.util.find_spec("mpi4py") is None:
        return False
    try:
        importlib.import_module("fulqrum.gpu.eigensolver._primme_mpi")
    except Exception:
        return False
    return True


FE4S4_SAMPLES = ("head", "uniform", "random")
# Two alpha 8-bit prefixes that are common in the bundled Fe4S4 samples,
# for the tests that pick two clusters of strings.
FE4S4_CLUSTER_PREFIXES = ("00001100", "01000001")


def fe4s4_data_dir():
    """Return the directory of the bundled compressed Fe4S4 test data."""
    from pathlib import Path  # noqa: PLC0415

    return Path(__file__).parent / "data" / "fe4s4"


def fe4s4_fcidump(directory):
    """Decompress the bundled Fe4S4 FCIDUMP into directory and return its path."""
    import lzma  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    path = Path(directory) / "fcidump_Fe4S4_MO.txt"
    if not path.exists():
        path.write_bytes(
            lzma.decompress((fe4s4_data_dir() / "fcidump_Fe4S4_MO.txt.xz").read_bytes())
        )
    return path


def fe4s4_determinants():
    """Return the 72-bit (beta then alpha) determinants of the bundled Fe4S4 samples."""
    import lzma  # noqa: PLC0415

    out = []
    for sample in FE4S4_SAMPLES:
        with lzma.open(fe4s4_data_dir() / f"fe4s4_1000_{sample}.txt.xz", "rt") as handle:
            out.extend(handle.read().split())
    return out


def fe4s4_alpha_strings():
    """Return the distinct 36-bit alpha strings of the bundled Fe4S4 samples, in first-seen order."""
    return list(dict.fromkeys(det[36:] for det in fe4s4_determinants()))


def make_lih_hsub():
    """Return the LiH subspace Hamiltonian at two electrons per spin and its alpha and beta strings."""
    from itertools import combinations  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import fulqrum as fq  # noqa: PLC0415

    op = fq.FermionicOperator.from_json(
        Path(__file__).parent / "data/lih.json"
    ).extended_jw_transformation()
    op.set_type(2)
    half = op.width // 2
    strings = []
    for ones in combinations(range(half), 2):
        bits = ["0"] * half
        for i in ones:
            bits[half - 1 - i] = "1"
        strings.append("".join(bits))
    strings.sort()
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([strings, list(strings)]))
    return hsub, strings, list(strings)
