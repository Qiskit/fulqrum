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


from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

try:
    import cupy
except ImportError:
    cupy = None

from fulqrum.gpu.distributed import _collective as col


def _has_cupy() -> bool:
    return cupy is not None


def _gpu_backend_available() -> bool:
    try:
        from fulqrum.gpu.eigensolver.primme_distributed import _load_backend

        return bool(_load_backend().gpu_backend_available())
    except ImportError:
        return False


def _hwn_strings(width: int, weight: int) -> list[str]:
    out = []
    for ones in combinations(range(width), weight):
        bits = ["0"] * width
        for i in ones:
            bits[width - 1 - i] = "1"
        out.append("".join(bits))
    return sorted(out)


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
@pytest.mark.skipif(
    not _gpu_backend_available(),
    reason="_primme_mpi built without the cuBLAS backend",
)
@pytest.mark.parametrize("use_precond", [False, True], ids=["noprecond", "jacobi"])
def test_gpu_eigsh_matches_host_lih(use_precond):
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    import fulqrum as fq
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )
    from fulqrum.gpu.eigensolver.primme_distributed import (
        distributed_primme_eigsh,
    )

    lih_path = Path(__file__).parent / "data/lih.json"
    OP = fq.FermionicOperator.from_json(lih_path).extended_jw_transformation()
    OP.set_type(2)
    half = OP.width // 2
    alpha_strs = _hwn_strings(half, 2)
    beta_strs = list(alpha_strs)
    sub = fq.Subspace([alpha_strs, beta_strs])
    hsub = fq.SubspaceHamiltonian(OP, sub)

    col.reset_for_testing()
    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )
    assert H.supports_device_matvec is True

    diag = np.asarray(H.diag_local, dtype=np.float64)
    v0 = np.zeros(H.n_local)
    v0[int(diag.argmin())] = 1.0

    precond = diag if use_precond else None
    results = {}
    for gpu in (False, True):
        evals, evecs = distributed_primme_eigsh(
            H,
            k=1,
            tol=1e-9,
            which="SA",
            conv="abs",
            precond_diag_local=precond,
            v0_local=v0,
            return_eigenvectors=True,
            gpu=gpu,
        )
        assert isinstance(evecs, np.ndarray)  # GPU path converts back to host
        assert evecs.shape == (H.n_local, 1)
        results[gpu] = float(evals[0])

    assert results[True] == pytest.approx(results[False], abs=1e-10)


def _pageable_memory_access() -> bool:
    if cupy is None:
        return False
    try:
        runtime = cupy.cuda.runtime
        return bool(
            runtime.deviceGetAttribute(runtime.cudaDevAttrPageableMemoryAccess, 0)
        )
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


@pytest.mark.skipif(
    not _pageable_memory_access(), reason="GPU without pageable host memory access"
)
def test_host_gpu_preconditioner_keeps_its_copy_of_a_strided_diagonal():
    import gc

    from fulqrum.gpu.eigensolver.primme_distributed import _make_host_gpu_precond

    n = 50_000
    rng = np.random.default_rng(5)
    storage = np.zeros(2 * n)
    storage[::2] = rng.uniform(-3.0, 3.0, n)
    diag = storage[::2].copy()
    precond = _make_host_gpu_precond(storage[::2], n, 7001)
    held = [
        cell.cell_contents
        for cell in precond.__closure__
        if isinstance(cell.cell_contents, np.ndarray)
    ]
    assert any(
        array.flags.c_contiguous and np.array_equal(array, diag) for array in held
    )
    gc.collect()
    filler = [np.full(n, 1e300) for _ in range(8)]
    x = np.asfortranarray(rng.uniform(-1.0, 1.0, (n, 2)))
    y = np.zeros((n, 2), order="F")
    shifts = [0.25, -1.5]
    precond(x.ctypes.data, n, y.ctypes.data, n, 2, shifts)
    for j, shift in enumerate(shifts):
        t = diag - shift
        np.testing.assert_array_equal(
            y[:, j], x[:, j] / np.copysign(np.maximum(np.abs(t), 1e-10), t)
        )
    assert len(filler) == 8


@pytest.mark.skipif(
    not _pageable_memory_access(), reason="GPU without pageable host memory access"
)
def test_host_leg_gpu_preconditioner_on_a_strided_diagonal_matches_the_host_preconditioner(
    monkeypatch,
):
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )
    from fulqrum.gpu.eigensolver.primme_distributed import distributed_primme_eigsh

    from .conftest import make_lih_hsub

    hsub, alpha_strs, beta_strs = make_lih_hsub()
    col.reset_for_testing()
    H = TileGPUResidentHamiltonian(
        hsub, alpha_strs, beta_strs, comm=MPI.COMM_SELF, comm_backend="host_mpi"
    )
    diag = np.asarray(H.diag_local, dtype=np.float64)
    v0 = np.zeros(H.n_local)
    v0[int(diag.argmin())] = 1.0
    results = {}
    for setting in ("0", "1"):
        monkeypatch.setenv("FQ_PRIMME_GPU_PRECOND", setting)
        storage = np.zeros(2 * H.n_local, dtype=np.float32)
        storage[::2] = diag
        evals, stats = distributed_primme_eigsh(
            H,
            k=1,
            tol=1e-9,
            precond_diag_local=storage[::2],
            v0_local=v0,
            return_eigenvectors=False,
            return_stats=True,
            gpu=False,
        )
        results[setting] = (float(evals[0]), stats["numMatvecs"], stats["numPreconds"])
    assert results["1"][1:] == results["0"][1:]
    assert results["1"][0] == pytest.approx(results["0"][0], abs=1e-12)
