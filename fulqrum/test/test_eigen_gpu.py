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

"""GPU eigenvalue tests on chemistry-realistic subspaces.

Covers the non-MPI ``GPUHamiltonian`` and the MPI-distributed
``TileGPUResidentHamiltonian`` paths. Mirrors the structure of test_eigen_LiH.py for
the CPU path. We use a particle-conserving subspace (each alpha/beta
half-string with Hamming weight = number of electrons per spin).

Skipped cleanly when the GPU build (FQ_BUILD_GPU=1) or a CUDA device is
unavailable. The MPI test additionally requires mpi4py + the
``_primme_mpi`` extension built with ``FQ_ENABLE_PRIMME_MPI=1``.
"""

import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg as spla

import fulqrum as fq

from .conftest import gpu_available, mpi_gpu_available


# Skip the whole module at collection time when GPU isn't usable.
pytestmark = [
    pytest.mark.skipif(
        not gpu_available(), reason="GPU build / CUDA device unavailable"
    ),
    pytest.mark.gpu,
]

_DATA = Path(__file__).parent / "data" / "lih.json"
LIH_GROUND_ENERGY = -7.875652564927877
N_ELECTRONS_PER_SPIN = 2  # LiH: 4 electrons total, 2 alpha + 2 beta


def _hwn_strings(width: int, weight: int) -> list[str]:
    """Hamming-weight `weight` strings of length `width`, LSB-on-right convention."""
    out = []
    for ones in combinations(range(width), weight):
        bits = ["0"] * width
        for i in ones:
            bits[width - 1 - i] = "1"
        out.append("".join(bits))
    return sorted(out)


def _lih_setup():
    OP = fq.FermionicOperator.from_json(_DATA).extended_jw_transformation()
    OP.set_type(2)
    half = OP.width // 2
    alpha = _hwn_strings(half, N_ELECTRONS_PER_SPIN)
    beta = _hwn_strings(half, N_ELECTRONS_PER_SPIN)
    return OP, alpha, beta


def test_lih_gpu_eigenvalue():
    """GPUHamiltonian lowest eigenvalue on 4e LiH."""
    import cupy

    from fulqrum.gpu import GPUHamiltonian

    gpu_ids = tuple(range(int(cupy.cuda.runtime.getDeviceCount())))

    OP, alpha, beta = _lih_setup()
    S = fq.Subspace([list(alpha), list(beta)])
    hsub = fq.SubspaceHamiltonian(OP, S)
    H_gpu = GPUHamiltonian(hsub, gpu_ids=gpu_ids)
    n = H_gpu.shape[0]
    evals, _ = spla.eigsh(H_gpu, k=1, which="SA", v0=np.ones(n))
    assert np.isclose(evals[0], LIH_GROUND_ENERGY, atol=1e-8), (
        f"GPU eig {evals[0]} differs from {LIH_GROUND_ENERGY}"
    )


def test_lih_gpu_matches_cpu_matvec():
    """GPU matvec matches CPU matvec on a fixed random vector."""
    from fulqrum.gpu import GPUHamiltonian

    OP, alpha, beta = _lih_setup()
    # CPU reference: full-string subspace in the same beta-outer ordering
    full_bs = [b + a for b in beta for a in alpha]
    S_cpu = fq.Subspace([list(full_bs)])
    H_cpu = fq.SubspaceHamiltonian(OP, S_cpu)

    S_gpu = fq.Subspace([list(alpha), list(beta)])
    hsub = fq.SubspaceHamiltonian(OP, S_gpu)
    H_gpu = GPUHamiltonian(hsub, gpu_ids=(0,))

    rng = np.random.default_rng(7)
    x = rng.standard_normal(len(S_cpu))
    y_cpu = H_cpu.matvec(x)
    y_gpu = H_gpu.matvec(x)
    assert np.allclose(y_cpu, y_gpu, atol=1e-8), (
        f"CPU/GPU matvec max diff {np.max(np.abs(y_cpu - y_gpu)):.3e}"
    )


def test_gpu_rejects_non_type_2_operator():
    """The GPU pre-requisite guard must reject operators that aren't type=2."""
    from fulqrum.exceptions import FulqrumError
    from fulqrum.gpu import GPUHamiltonian

    OP = fq.FermionicOperator.from_json(_DATA).extended_jw_transformation()
    # Force type != 2.
    OP.set_type(1)
    half = OP.width // 2
    alpha = _hwn_strings(half, N_ELECTRONS_PER_SPIN)
    beta = _hwn_strings(half, N_ELECTRONS_PER_SPIN)
    S = fq.Subspace([list(alpha), list(beta)])
    hsub = fq.SubspaceHamiltonian(OP, S)
    with pytest.raises(FulqrumError, match="type=2"):
        GPUHamiltonian(hsub, gpu_ids=(0,))


def test_gpu_rejects_non_lazy_subspace():
    """The GPU pre-requisite guard must reject subspaces without lazy-build."""
    from fulqrum.exceptions import FulqrumError
    from fulqrum.gpu import GPUHamiltonian

    OP, alpha, beta = _lih_setup()
    # Eager full-string subspace (half_strs_mode==0) -- must be rejected.
    full_bs = [b + a for b in beta for a in alpha]
    S = fq.Subspace([list(full_bs)])
    hsub = fq.SubspaceHamiltonian(OP, S)
    with pytest.raises(FulqrumError, match="half-string Subspace"):
        GPUHamiltonian(hsub, gpu_ids=(0,))


_MPI_DRIVER = """
from itertools import combinations
from pathlib import Path
import numpy as np
from mpi4py import MPI
import fulqrum as fq
from fulqrum.gpu import distributed_primme_eigsh
from fulqrum.gpu.distributed.dist_matvec_tile_resident import TileGPUResidentHamiltonian

def hwn(width, weight):
    out = []
    for ones in combinations(range(width), weight):
        bits = ['0']*width
        for i in ones:
            bits[width-1-i] = '1'
        out.append(''.join(bits))
    return sorted(out)

OP = fq.FermionicOperator.from_json(Path('__DATA_PATH__')).extended_jw_transformation()
OP.set_type(2)
half = OP.width // 2
alpha = hwn(half, 2); beta = hwn(half, 2)
S = fq.Subspace([list(alpha), list(beta)])
hsub = fq.SubspaceHamiltonian(OP, S)
H = TileGPUResidentHamiltonian(
    hsub, alpha, beta, comm=MPI.COMM_WORLD, process_grid=(MPI.COMM_WORLD.Get_size(), 1), comm_backend='host_mpi'
)
diag = H.diag_local
lowest = MPI.COMM_WORLD.allreduce((float(diag.min()), MPI.COMM_WORLD.Get_rank()), op=MPI.MINLOC)[1]
v0 = np.zeros((diag.size, 1))
if MPI.COMM_WORLD.Get_rank() == lowest:
    v0[int(np.argmin(diag)), 0] = 1.0
evals, _ = distributed_primme_eigsh(H, k=1, which='SA', precond_diag_local=diag, v0_local=v0)
if MPI.COMM_WORLD.Get_rank() == 0:
    print(f'__RESULT__ {float(evals[0]):.15e}')
"""


@pytest.mark.mpi
@pytest.mark.skipif(
    not mpi_gpu_available(), reason="mpi4py + _primme_mpi + GPU required"
)
def test_lih_mpi_gpu_eigenvalue(tmp_path):
    """mpirun -n 2 with TileGPUResidentHamiltonian + distributed_primme_eigsh.

    Both ranks share the local GPU. The driver
    runs in a subprocess so the parent pytest process stays single-rank.
    """
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        # Fall back to the same bin/ as the current python.
        candidate = Path(sys.executable).parent / "mpirun"
        if candidate.exists():
            mpirun = str(candidate)
    if mpirun is None:
        pytest.skip("mpirun not on PATH and not next to sys.executable")
    driver = tmp_path / "mpi_driver.py"
    driver.write_text(_MPI_DRIVER.replace("__DATA_PATH__", str(_DATA)))
    result = subprocess.run(
        [mpirun, "-n", "2", "--oversubscribe", sys.executable, str(driver)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"mpirun -n 2 failed (rc={result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    marker = "__RESULT__"
    eig = None
    for line in result.stdout.splitlines():
        if marker in line:
            eig = float(line.split(marker, 1)[1].strip())
            break
    assert eig is not None, f"driver did not emit {marker!r}:\n{result.stdout}"
    assert np.isclose(eig, LIH_GROUND_ENERGY, atol=1e-8), (
        f"MPI GPU eig {eig} differs from {LIH_GROUND_ENERGY}"
    )


def test_gpu_warns_on_low_particle_subspace():
    """The bug guard must warn when a half-string has Hamming weight < 2."""
    import warnings

    from fulqrum.gpu import GPUHamiltonian

    OP = fq.FermionicOperator.from_json(_DATA).extended_jw_transformation()
    OP.set_type(2)
    half = OP.width // 2
    # Mix in a Hamming-weight-1 string to trigger the warning.
    alpha = sorted(_hwn_strings(half, 2) + ["000001"])
    beta = _hwn_strings(half, 2)
    S = fq.Subspace([list(alpha), list(beta)])
    hsub = fq.SubspaceHamiltonian(OP, S)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            GPUHamiltonian(hsub, gpu_ids=(0,))
        except Exception:
            pass  # we only care that the warning fired
        assert any("Hamming weight" in str(w.message) for w in wlist), (
            "Expected low-particle RuntimeWarning mentioning Hamming weight"
        )
