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

import os
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .. import config

logger = logging.getLogger(__name__)
__all__ = ["distributed_primme_eigsh", "is_primme_mpi_available"]


def _env_flag(name: str) -> bool:

    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _load_backend():


    alt = os.environ.get("FQ_PRIMME_MPI_SO", "").strip()
    if alt:
        import importlib.util  # noqa: PLC0415
        import sys  # noqa: PLC0415

        name = f"{__package__}._primme_mpi"
        mod = sys.modules.get(name)
        if mod is None or getattr(mod, "__file__", None) != alt:
            spec = importlib.util.spec_from_file_location(name, alt)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules[name] = mod
        return mod
    try:
        from . import _primme_mpi
    except ImportError as exc:
        raise ImportError(
            "distributed_primme_eigsh requires the `_primme_mpi` Cython "
            "extension. GPU builds include it automatically, from the patched PRIMME "
            "submodule at `third-party/primme/` (pinned to v3.2.3).\n"
            "To build it:\n"
            "  1. Initialize the submodule: `git submodule update --init --recursive`\n"
            '  2. Rebuild fulqrum: `FQ_BUILD_GPU=1 python -m pip install -e ".[gpu]"`\n'
            f"Original import error: {exc}"
        ) from exc
    return _primme_mpi


def is_primme_mpi_available() -> bool:
    """Return whether the optional native PRIMME MPI extension imports. This checks import
    availability only; it does not verify CUDA hardware or execution."""

    try:
        _load_backend()
    except ImportError:
        return False
    return True


def distributed_primme_eigsh(
    H,
    k: int,
    tol: float = 1e-3,
    maxiter: int | None = None,
    ncv: int | None = None,
    which: str = "SA",
    method: str = "PRIMME_GD_Olsen_plusK",
    conv: str = "abs",
    precond_diag_local: np.ndarray | None = None,
    v0_local: np.ndarray | None = None,
    return_eigenvectors: bool = True,
    return_stats: bool = False,
    return_unconverged: bool = False,
    maxBlockSize: int | None = None,
    minRestartSize: int | None = None,
    maxPrevRetain: int | None = None,
    printLevel: int = 0,
    monitor=None,
    gpu: bool | str = "auto",
    initBasisMode: str | None = None,
    energy_tol: float = 1e-6,
    energy_window: int = 3,
    resnorm_cap: float = 1e-2,
):
    """Compute eigenpairs of a distributed real Hermitian operator with PRIMME.

    Uses Fulqrum's native PRIMME MPI extension. Every solver rank in ``H._comm``
    must participate with the same solver options; starting vectors and
    preconditioner data are local to each rank. ``TileGPUResidentHamiltonian``
    is one supported operator. Operators with separate worker ranks restrict
    solver calls to their primary ranks; workers use the operator's worker API.

    ``H`` must expose global square ``shape``, local length ``n_local``, an MPI
    communicator ``_comm`` and ``_matvec(v_local)`` returning the local product
    in the same row order as its input. The distributed solver calls this local
    interface directly. Inherited SciPy ``LinearOperator.matvec`` checks the
    global vector length and is not the multi-rank local-vector interface.
    A device-capable operator additionally accepts CuPy vectors and keyword
    ``out=buffer``, and advertises ``supports_device_matvec``.

    For a GPU-backed ``H``, eigensolving combines GPU matrix-vector products
    with PRIMME linear algebra steps, such as basis orthogonalization and dense
    operations. ``gpu=False`` runs the linear-algebra steps on the host;
    ``gpu=True`` runs the steps on GPUs. Matrix-vector products stay on the GPUs
    in both modes. Returned eigenvalues and eigenvectors are host NumPy arrays.

    Convergence uses the Ritz value ``theta`` and residual norm ``r``:
        - ``"abs"``: require ``r < tol``.
        - ``"rel"``: require ``r <= tol * max(abs(theta), 4e-11)``.
        - ``"norm"``: use PRIMME's operator-norm-scaled convergence test.
        - ``"energy"``: accept the relative residual test or stability of the
          tracked Ritz value within ``energy_tol``, subject to ``resnorm_cap``.
          Energy-based stopping can accept a residual larger than ``tol``.

    Args:
        H: Distributed real Hermitian operator.
        k: Positive number of eigenpairs to request.
        tol: Convergence tolerance interpreted by ``conv``; defaults to 1e-3.
        maxiter: Maximum matrix-vector products, passed to PRIMME's
            ``maxMatvecs``. ``None`` keeps PRIMME's default limit.
        ncv: Maximum number of basis vectors. ``None`` lets PRIMME choose.
        which: ``"SA"`` for smallest algebraic values (default), ``"LA"`` for
            largest algebraic, ``"SM"`` for smallest magnitude or ``"LM"`` for
            largest magnitude.
        method: PRIMME preset name; defaults to ``"PRIMME_GD_Olsen_plusK"``.
        conv: ``"abs"`` (default), ``"rel"``, ``"norm"`` or ``"energy"``.
        precond_diag_local: Optional real host diagonal of shape ``(H.n_local,)``
            in the operator's local row order, for a shifted Jacobi preconditioner.
            ``None`` disables this preconditioner; ``H.diag_local`` can supply it.
        v0_local: Optional real host starting vector, shape ``(H.n_local,)``, or
            block, shape ``(H.n_local, m)``. Converted to Fortran-order float64;
            at most ``k`` columns are used. ``None`` supplies no starting vectors.
        return_eigenvectors: Include local eigenvectors; defaults to True.
        return_stats: Include a statistics dictionary; defaults to False.
        return_unconverged: Allow PRIMME status -3, including non-convergence,
            and expose its current result buffers; defaults to False. Buffers retain
            ``k`` entries and are not filtered to converged eigenpairs. Other PRIMME
            failures still raise.
        maxBlockSize: Optional maximum PRIMME block size.
        minRestartSize: Optional number of basis vectors retained at restart.
        maxPrevRetain: Optional maximum number of previous search directions
            retained. These three options preserve preset defaults when None.
        printLevel: PRIMME verbosity level; defaults to 0.
        monitor: Optional ``monitor(ritz_value, residual_norm)`` callback during
            convergence checks for ``abs``, ``rel`` and ``energy``. It is unused
            for ``norm``; defaults to None.
        gpu: Placement of PRIMME's linear-algebra steps: True for GPU, False
            for host. With a GPU-backed ``H``, matvecs run on GPUs in both modes.
            ``"auto"`` (default) chooses GPU when the native extension and
            operator advertise device support. GPU mode requires CuPy.
        initBasisMode: Optional PRIMME initialization constant name. ``None``
            keeps the preset's initialization mode.
        energy_tol: Ritz-value stability tolerance for ``energy`` stopping;
            defaults to 1e-6.
        energy_window: Number of previous distinct matrix-vector-count samples
            compared with the current Ritz value; defaults to 3.
        resnorm_cap: Residual upper bound for the energy-stability criterion;
            defaults to 1e-2. These energy options must be positive, with
            ``energy_window >= 1``, when ``conv="energy"``.

    Returns:
        By default, ``(evals, evecs_local)``. ``evals`` is a float64 array of
        shape ``(k,)``, identical on all solver ranks. ``evecs_local`` is a
        host float64 array of shape ``(H.n_local, k)`` containing this
        rank's rows; column j belongs to eigenvalue j. Eigenvectors remain
        distributed rather than being gathered into a global array. Global norms
        and residuals require contributions from all ranks.

        With eigenvectors disabled, returns ``evals``. With statistics enabled,
        returns ``(evals, evecs_local, stats)`` or ``(evals, stats)``. ``stats``
        includes PRIMME counts/timings, residual norms in ``rnorms``, convergence
        details and ``gpu_eigsh``. Rank-local timings or memory fields may differ.
        When unconverged returns are allowed, these shapes remain unchanged;
        the contents are not guaranteed to be converged eigenpairs.

    Raises:
        TypeError: The operator lacks the required local-vector interface.
        ValueError: Dimensions, local input shapes, target, convergence mode,
            GPU selection or a PRIMME option are invalid.
        ImportError: The native PRIMME extension or a required Python dependency
            is unavailable.
        RuntimeError: PRIMME fails, cannot use the requested backend, or does
            not converge and unconverged returns are disabled.

    Multi-rank GPU matvec/preconditioner callback failures abort the MPI job to
    prevent peers from waiting indefinitely in collective operations.
    """

    _check_operator(H)
    options = EigshOptions(
        k=k,
        tol=tol,
        maxiter=maxiter,
        ncv=ncv,
        which=which,
        method=method,
        conv=conv,
        return_eigenvectors=return_eigenvectors,
        return_stats=return_stats,
        return_unconverged=return_unconverged,
        maxBlockSize=maxBlockSize,
        minRestartSize=minRestartSize,
        maxPrevRetain=maxPrevRetain,
        printLevel=printLevel,
        monitor=monitor,
        initBasisMode=initBasisMode,
        energy_tol=energy_tol,
        energy_window=energy_window,
        resnorm_cap=resnorm_cap,
    )
    precond_diag_local, v0_local = _local_inputs(H, precond_diag_local, v0_local)
    backend = _load_backend()
    use_gpu = choose_leg(H, gpu, backend)
    callbacks = make_callbacks(use_gpu, H, options, precond_diag_local, v0_local)
    started = time.perf_counter()
    result = run_primme(backend, H, options, callbacks)
    if H._comm.Get_rank() == 0:
        logger.info(
            "PRIMME solve returned: requested_eigenpairs=%d, leg=%s, elapsed=%.2f s",
            k,
            "GPU" if use_gpu else "host",
            time.perf_counter() - started,
            extra={"gpu_rank": 0},
        )
    return _host_result(result, use_gpu, options)


@dataclass(frozen=True)
class EigshOptions:


    k: int
    tol: float
    maxiter: int | None
    ncv: int | None
    which: str
    method: str
    conv: str
    return_eigenvectors: bool
    return_stats: bool
    return_unconverged: bool
    maxBlockSize: int | None
    minRestartSize: int | None
    maxPrevRetain: int | None
    printLevel: int
    monitor: object
    initBasisMode: str | None = None
    energy_tol: float = 1e-6
    energy_window: int = 3
    resnorm_cap: float = 1e-2

    def __post_init__(self):

        if self.which not in {"SA", "LA", "SM", "LM"}:
            raise ValueError("which must be one of 'SA', 'LA', 'SM', or 'LM'")
        if self.conv not in {"abs", "rel", "energy", "norm"}:
            raise ValueError("conv must be 'abs', 'rel', 'energy' or 'norm'")
        if self.conv == "energy" and not (
            self.energy_tol > 0 and self.energy_window >= 1 and self.resnorm_cap > 0
        ):
            raise ValueError(
                "conv='energy' needs energy_tol > 0, energy_window >= 1 and resnorm_cap > 0"
            )


@dataclass
class SolverCallbacks:


    precond_diag_local: np.ndarray | None
    v0_local: np.ndarray | None
    gpu_evecs: object = None
    gpu_matvec: object = None
    gpu_precond: object = None
    gpu_init_size: int = 0
    overlap_matvec: object = None
    pinned_basis: bool = False
    primme_comm: object = None
    spec_matvec: bool = False


def _check_operator(H):


    if hasattr(H, "_is_primary") and not H._is_primary:
        raise RuntimeError(
            "distributed_primme_eigsh must only be called on primary ranks "
            "(task_id == 0).  Worker ranks must call H.run_workers() instead."
        )
    if not hasattr(H, "_matvec") or not hasattr(H, "n_local"):
        raise TypeError("H must expose H._matvec(v_local) and H.n_local")
    if not hasattr(H, "_comm"):
        raise TypeError("H must expose H._comm, usually mpi4py.MPI.COMM_WORLD")
    if len(H.shape) != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got shape={H.shape!r}")


def _local_inputs(H, precond_diag_local, v0_local):


    n_local = int(H.n_local)
    if precond_diag_local is not None:
        precond_diag_local = np.asarray(precond_diag_local, dtype=np.float64)
        if precond_diag_local.shape != (n_local,):
            raise ValueError(
                "precond_diag_local must have shape "
                f"({n_local},), got {precond_diag_local.shape}"
            )
    if v0_local is not None:
        v0_local = np.asarray(v0_local, dtype=np.float64, order="F")
        if v0_local.ndim == 1:
            v0_local = v0_local.reshape((n_local, 1), order="F")
        if v0_local.shape[0] != n_local:
            raise ValueError(
                f"v0_local must have {n_local} local rows, got {v0_local.shape}"
            )
    return precond_diag_local, v0_local


def choose_leg(H, gpu, backend) -> bool:


    if isinstance(gpu, str):
        if gpu != "auto":
            raise ValueError(f"gpu must be True, False or 'auto', got {gpu!r}")
        return (
            bool(getattr(H, "supports_device_matvec", False))
            and backend.gpu_backend_available()
        )
    use_gpu = bool(gpu)
    if use_gpu and not getattr(H, "supports_device_matvec", False):
        raise ValueError(
            "gpu=True but H does not support device matvecs "
            "(H.supports_device_matvec is falsy)"
        )
    return use_gpu


def make_callbacks(
    use_gpu, H, options, precond_diag_local, v0_local
) -> SolverCallbacks:


    n_local = int(H.n_local)
    if use_gpu:
        callbacks = _gpu_leg_callbacks(
            H, n_local, options, precond_diag_local, v0_local
        )
    else:
        callbacks = _host_leg_callbacks(H, n_local, precond_diag_local, v0_local)
    callbacks.gpu_matvec = _abort_on_rank_error(callbacks.gpu_matvec, H._comm, "matvec")
    callbacks.gpu_precond = _abort_on_rank_error(
        callbacks.gpu_precond, H._comm, "preconditioner"
    )
    if callbacks.overlap_matvec is not None:
        callbacks.overlap_matvec = _GuardedOverlap(callbacks.overlap_matvec, H._comm)
    return callbacks


def _gpu_leg_callbacks(
    H, n_local, options, precond_diag_local, v0_local
) -> SolverCallbacks:


    import cupy  # noqa: PLC0415

    cupy.get_default_memory_pool().free_all_blocks()
    managed_basis = _env_flag("FQ_PRIMME_MANAGED_BASIS")
    if options.ncv:
        _check_gpu_basis_fits(n_local, int(options.ncv), managed_basis)
    callbacks = SolverCallbacks(None, None)
    callbacks.gpu_evecs = cupy.zeros(
        (n_local, int(options.k)), dtype=cupy.float64, order="F"
    )
    if v0_local is not None:
        callbacks.gpu_init_size = min(int(options.k), int(v0_local.shape[1]))
        callbacks.gpu_evecs[:, : callbacks.gpu_init_size] = cupy.asarray(
            v0_local[:, : callbacks.gpu_init_size]
        )
    if precond_diag_local is not None:
        callbacks.gpu_precond = _make_gpu_precond(precond_diag_local, n_local)
    callbacks.gpu_matvec = _make_gpu_block_matvec(H, n_local, stage=managed_basis)
    return callbacks


def _check_gpu_basis_fits(n_local: int, ncv: int, managed_basis: bool):


    import cupy  # noqa: PLC0415

    _int32_max = 2**31 - 1
    if n_local > _int32_max:
        raise ValueError(
            f"GPU PRIMME path: n_local = {n_local:,} exceeds INT32_MAX "
            f"({_int32_max:,}). Use more ranks (lower per-rank n_local) or "
            "pass gpu=False."
        )
    _free, _total = cupy.cuda.runtime.memGetInfo()
    _need = 2 * n_local * ncv * 8
    if _need > 0.95 * _free and not managed_basis:
        raise RuntimeError(
            f"GPU PRIMME path: device basis V+W needs ~{_need / 1e9:.1f} "
            f"GB but only {_free / 1e9:.1f} GB HBM is free on this "
            f"rank. Use ncv <= {int(0.95 * _free / (16 * n_local))}, "
            "use more ranks, or pass gpu=False."
        )


def _host_leg_callbacks(H, n_local, precond_diag_local, v0_local) -> SolverCallbacks:


    callbacks = SolverCallbacks(precond_diag_local, v0_local)
    if precond_diag_local is not None and os.environ.get(
        "FQ_PRIMME_GPU_PRECOND", ""
    ).lower() in ("1", "true", "yes", "on"):
        import cupy  # noqa: PLC0415

        cupy.get_default_memory_pool().free_all_blocks()
        _pc_tile = int(os.environ.get("FQ_PRIMME_GPU_PRECOND_TILE", 1 << 24))
        callbacks.gpu_precond = _make_host_gpu_precond(
            precond_diag_local, n_local, _pc_tile
        )
        callbacks.precond_diag_local = None
    if getattr(H, "supports_device_matvec", False):
        direct = _make_host_direct_matvec(H, n_local)
        callbacks.gpu_matvec = direct
        callbacks.pinned_basis = bool(config.read("FQ_PRIMME_PINNED_BASIS"))
        overlap, reason = _host_overlap_mode(H)
        if overlap:
            callbacks.overlap_matvec = direct
            callbacks.spec_matvec = bool(config.read("FQ_PRIMME_SPEC_MATVEC"))
            if H._comm.Get_size() > 1:
                callbacks.primme_comm = H._comm.Dup()
        if H._comm.Get_rank() == 0:
            logger.info(
                "PRIMME host leg: page_locked_basis=%s, matvec_overlap=%s (%s), spec_matvec=%s",
                "on" if callbacks.pinned_basis else "off",
                "on" if overlap else "off",
                reason,
                "on" if callbacks.spec_matvec else "off",
                extra={"gpu_rank": 0},
            )
    return callbacks


def _host_overlap_mode(H):


    if not config.read("FQ_PRIMME_OVERLAP_MATVEC"):
        return False, "FQ_PRIMME_OVERLAP_MATVEC is off"
    comm = H._comm
    if comm.Get_size() == 1:
        return True, "one rank"
    from ..distributed.diagnostics import _debug_matvec_enabled  # noqa: PLC0415
    from ..distributed.local_comm import LocalComm  # noqa: PLC0415

    if isinstance(comm, LocalComm):
        return False, "in-process ranks"
    from mpi4py import MPI  # noqa: PLC0415

    backend = getattr(H, "_backend_name", "")
    settings = getattr(H, "_recorder_settings", None)
    diagnostics = _debug_matvec_enabled() or (
        settings is not None and any(vars(settings).values())
    )
    multiple = MPI.Query_thread() == MPI.THREAD_MULTIPLE
    if diagnostics:
        local, reason = False, "matvec diagnostics run MPI collectives"
    elif backend == "nccl":
        local, reason = True, "nccl backend"
    elif multiple:
        local, reason = True, f"{backend} backend with MPI_THREAD_MULTIPLE"
    else:
        local, reason = False, f"{backend} backend without MPI_THREAD_MULTIPLE"
    if not comm.allreduce(int(local), op=MPI.MIN):
        return False, reason if not local else "off on another rank"
    return True, reason


def run_primme(backend, H, options, callbacks):


    return backend.eigsh(
        H,
        int(options.k),
        float(options.tol),
        -1 if options.maxiter is None else int(options.maxiter),
        0 if options.ncv is None else int(options.ncv),
        options.which,
        options.method,
        options.conv,
        callbacks.precond_diag_local,
        callbacks.v0_local,
        bool(options.return_eigenvectors),
        bool(options.return_stats),
        bool(options.return_unconverged),
        0 if options.maxBlockSize is None else int(options.maxBlockSize),
        0 if options.minRestartSize is None else int(options.minRestartSize),
        0 if options.maxPrevRetain is None else int(options.maxPrevRetain),
        int(options.printLevel),
        options.monitor,
        callbacks.gpu_evecs,
        callbacks.gpu_matvec,
        callbacks.gpu_init_size,
        callbacks.gpu_precond,
        "" if options.initBasisMode is None else str(options.initBasisMode),
        callbacks.overlap_matvec,
        bool(callbacks.pinned_basis),
        callbacks.primme_comm,
        bool(callbacks.spec_matvec),
        float(options.energy_tol),
        int(options.energy_window),
        float(options.resnorm_cap),
    )


def _host_result(result, use_gpu, options):


    if use_gpu and options.return_eigenvectors:
        import cupy  # noqa: PLC0415

        if isinstance(result, tuple):
            result = tuple(
                cupy.asnumpy(part) if isinstance(part, cupy.ndarray) else part
                for part in result
            )
        elif isinstance(result, cupy.ndarray):
            result = cupy.asnumpy(result)
    if (
        options.return_stats
        and isinstance(result, tuple)
        and isinstance(result[-1], dict)
    ):
        result[-1]["gpu_eigsh"] = bool(use_gpu)
    return result


def _abort_on_rank_error(fn, comm, what: str):


    if fn is None or comm.Get_size() == 1:
        return fn

    def _guarded(*args):

        try:
            return fn(*args)
        except BaseException:  # noqa: BLE001
            import sys  # noqa: PLC0415
            import traceback  # noqa: PLC0415

            sys.stderr.write(
                f"PRIMME {what} callback failed on rank {comm.Get_rank()}. "
                "Aborting the MPI job to prevent other ranks from waiting indefinitely in collective operations.\n"
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            comm.Abort(1)
            raise

    return _guarded


def _device_vector_wrapper(n_local: int):

    import cupy  # noqa: PLC0415

    n = int(n_local)

    def _wrap(ptr):

        mem = cupy.cuda.UnownedMemory(int(ptr), n * 8, owner=None)
        return cupy.ndarray(
            (n,), dtype=cupy.float64, memptr=cupy.cuda.MemoryPointer(mem, 0)
        )

    return _wrap


def _make_gpu_block_matvec(H, n_local: int, stage: bool = False):


    import cupy  # noqa: PLC0415

    _wrap = _device_vector_wrapper(n_local)
    d_x = cupy.empty(int(n_local), dtype=cupy.float64) if stage else None
    d_y = cupy.empty(int(n_local), dtype=cupy.float64) if stage else None

    def _block_matvec(x_ptr, ldx, y_ptr, ldy, block_size):


        cupy.cuda.runtime.deviceSynchronize()
        for j in range(block_size):
            x_j = _wrap(int(x_ptr) + 8 * int(ldx) * j)
            y_j = _wrap(int(y_ptr) + 8 * int(ldy) * j)
            if stage:
                d_x[...] = x_j
                H._matvec(d_x, out=d_y)
                y_j[...] = d_y
            else:
                H._matvec(x_j, out=y_j)
        cupy.cuda.runtime.deviceSynchronize()

    return _block_matvec


class _HostDirectMatvec:


    def __init__(self, H, n_local: int):

        import ctypes  # noqa: PLC0415

        import cupy  # noqa: PLC0415

        self._H = H
        self._cupy = cupy
        n = int(n_local)
        self._d_x = cupy.empty(n, dtype=cupy.float64)
        self._d_y = cupy.empty(n, dtype=cupy.float64)
        self._col_t = ctypes.c_double * n
        self._device = cupy.cuda.Device().id
        self._thread = None
        self._input_ready = threading.Event()
        self._error = None
        self._accepted = 0
        self._discarded = 0
        self._wait_seconds = 0.0
        self._thread_seconds = 0.0
        self._spec = False
        self._backend = _load_backend()
        self._dot = 0.0
        self._d_dot = cupy.zeros(1, dtype=cupy.float64)

    def _host(self, ptr):

        return np.frombuffer(self._col_t.from_address(int(ptr)), dtype=np.float64)

    def __call__(self, x_ptr, ldx, y_ptr, ldy, block_size):


        for j in range(block_size):
            self._d_x.set(self._host(int(x_ptr) + 8 * int(ldx) * j))
            self._H._matvec(self._d_x, out=self._d_y)
            self._cupy.cuda.runtime.deviceSynchronize()
            self._d_y.get(out=self._host(int(y_ptr) + 8 * int(ldy) * j))

    def start(self, x_ptr, y_ptr, spec=False):


        if self._thread is not None:
            raise RuntimeError("an overlapped matvec is still running")
        self._error = None
        self._spec = bool(spec)
        self._input_ready.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(int(x_ptr), int(y_ptr), self._spec),
            name="fq-primme-matvec",
            daemon=True,
        )
        self._thread.start()

    def _run(self, x_ptr, y_ptr, spec):


        started = time.perf_counter()
        try:
            with self._cupy.cuda.Device(self._device):
                try:
                    if spec:
                        self._copy_in_chunks(x_ptr)
                    else:
                        self._d_x.set(self._host(x_ptr))
                finally:
                    self._input_ready.set()
                self._H._matvec(self._d_x, out=self._d_y)
                if spec:

                    from cupy import cublas  # noqa: PLC0415

                    cublas.dot(self._d_x, self._d_y, out=self._d_dot)
                self._cupy.cuda.runtime.deviceSynchronize()
                if spec:


                    self._backend.spec_wait_output_free()
                self._d_y.get(out=self._host(y_ptr))
                if spec:
                    self._dot = float(self._d_dot.get()[0])
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
        finally:
            self._input_ready.set()
            self._thread_seconds += time.perf_counter() - started

    def _copy_in_chunks(self, x_ptr):


        n = self._d_x.size
        chunk = max(1, int(self._backend.spec_chunk_rows()))
        host = self._host(x_ptr)
        for r0 in range(0, n, chunk):
            r1 = min(n, r0 + chunk)
            if not self._backend.spec_wait_rows(r1):
                raise RuntimeError("the speculative pass was aborted")
            self._d_x[r0:r1].set(host[r0:r1])
        if n == 0 and not self._backend.spec_wait_rows(0):
            raise RuntimeError("the speculative pass was aborted")

    def dot(self):

        return self._dot

    def input_done(self):

        self._input_ready.wait()
        if self._error is not None:
            raise self._error

    def wait(self, accept):


        started = time.perf_counter()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._wait_seconds += time.perf_counter() - started
        if accept:
            self._accepted += 1
        else:
            self._discarded += 1
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    def close(self):

        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def stats(self):

        return {
            "overlapMatvecs": self._accepted,
            "overlapDiscarded": self._discarded,
            "overlapWaitSeconds": self._wait_seconds,
            "overlapThreadSeconds": self._thread_seconds,
        }


class _GuardedOverlap:


    def __init__(self, inner, comm):

        self.start = _abort_on_rank_error(inner.start, comm, "overlapped matvec")
        self.input_done = _abort_on_rank_error(
            inner.input_done, comm, "overlapped matvec"
        )
        self.wait = _abort_on_rank_error(inner.wait, comm, "overlapped matvec")
        self.close = inner.close
        self.stats = inner.stats
        self.dot = inner.dot


def _make_host_direct_matvec(H, n_local: int):


    return _HostDirectMatvec(H, n_local)


def _clamped_jacobi_kernel():


    import cupy  # noqa: PLC0415

    return cupy.ElementwiseKernel(
        "float64 x, float64 d, float64 s",
        "float64 y",
        "double t = d - s; double a = fabs(t);"
        " a = (a != a || a > 1e-10) ? a : 1e-10; y = x / copysign(a, t);",
        "fq_clamped_jacobi",
    )


def _make_host_gpu_precond(precond_diag_local, n_local: int, tile_rows: int):


    import cupy

    n = int(n_local)
    T = max(1, min(int(tile_rows), n if n > 0 else 1))
    diag_host = np.ascontiguousarray(np.asarray(precond_diag_local, dtype=np.float64))
    fused = _clamped_jacobi_kernel()

    def _coherent(ptr, count):

        mem = cupy.cuda.UnownedMemory(int(ptr), int(count) * 8, owner=None)
        return cupy.ndarray(
            (int(count),),
            dtype=cupy.float64,
            memptr=cupy.cuda.MemoryPointer(mem, 0),
        )

    def _block_precond(x_ptr, ldx, y_ptr, ldy, block_size, shifts):


        diag_ptr = int(diag_host.ctypes.data)
        for j in range(block_size):
            xj = int(x_ptr) + 8 * int(ldx) * j
            yj = int(y_ptr) + 8 * int(ldy) * j
            s = float(shifts[j])
            for off in range(0, n, T):
                m = min(T, n - off)
                d_x = _coherent(xj + 8 * off, m)
                d_y = _coherent(yj + 8 * off, m)
                d_d = _coherent(diag_ptr + 8 * off, m)
                fused(d_x, d_d, s, d_y)
        cupy.cuda.runtime.deviceSynchronize()

    return _block_precond


def _make_gpu_precond(precond_diag_local, n_local: int):


    import cupy  # noqa: PLC0415

    _wrap = _device_vector_wrapper(n_local)
    d_diag = cupy.asarray(np.asarray(precond_diag_local, dtype=np.float64))
    fused = _clamped_jacobi_kernel()

    def _block_precond(x_ptr, ldx, y_ptr, ldy, block_size, shifts):


        for j in range(block_size):
            d_x = _wrap(int(x_ptr) + 8 * int(ldx) * j)
            d_y = _wrap(int(y_ptr) + 8 * int(ldy) * j)
            fused(d_x, d_diag, float(shifts[j]), d_y)
        cupy.cuda.runtime.deviceSynchronize()

    return _block_precond
