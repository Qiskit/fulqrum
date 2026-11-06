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

# cython: language_level=3, boundscheck=False, wraparound=False


import warnings

import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmax
from libc.stdint cimport int64_t


cdef extern from "../../core/src/checked_size.hpp":
    size_t checked_multiply(size_t, size_t) except +

ctypedef int64_t PRIMME_INT


cdef extern from "primme.h":
    ctypedef struct primme_params:
        void *matrix

    ctypedef int primme_preset_method
    ctypedef int primme_params_label

    int dprimme(double *evals, double *evecs, double *resNorms, primme_params *primme) nogil
    primme_params* primme_params_create()
    int primme_params_destroy(primme_params *primme)
    void primme_free(primme_params *primme)
    int primme_set_method(primme_preset_method method, primme_params *params)
    int primme_get_member(primme_params *primme, primme_params_label label, void *value)
    int primme_set_member(primme_params *primme, primme_params_label label, void *value)
    int primme_constant_info(const char* label_name, int *value)


cdef extern from *:
    """
    #ifdef FQ_PRIMME_GPU
    static int fq_cublas_dprimme_(double *evals, double *evecs,
                                  double *resNorms, primme_params *primme)
    { return cublas_dprimme(evals, evecs, resNorms, primme); }
    static int fq_primme_gpu_available_(void) { return 1; }
    /* Must be the SAME libcublas instance the extension
     * (and the static libprimme.a inside it) calls into. */
    extern "C" int cublasCreate_v2(void **handle);
    extern "C" int cublasDestroy_v2(void *handle);
    static void *fq_primme_cublas_create_(void) {
        void *h = 0;
        return cublasCreate_v2(&h) == 0 ? h : (void *)0;
    }
    static int fq_primme_cublas_destroy_(void *h) {
        return h ? cublasDestroy_v2(h) : 0;
    }
    extern "C" const char *fq_primme_gpu_last_error(void);
    extern "C" void fq_primme_gpu_last_error_clear(void);
    static const char *fq_primme_gpu_last_error_(void) { return fq_primme_gpu_last_error(); }
    static void fq_primme_gpu_last_error_clear_(void) { fq_primme_gpu_last_error_clear(); }
    #else
    static int fq_cublas_dprimme_(double *evals, double *evecs,
                                  double *resNorms, primme_params *primme)
    { (void)evals; (void)evecs; (void)resNorms; (void)primme; return -2000; }
    static int fq_primme_gpu_available_(void) { return 0; }
    static void *fq_primme_cublas_create_(void) { return (void *)0; }
    static int fq_primme_cublas_destroy_(void *h) { (void)h; return 0; }
    static const char *fq_primme_gpu_last_error_(void) { return ""; }
    static void fq_primme_gpu_last_error_clear_(void) { }
    #endif
    """
    int fq_cublas_dprimme_(double *evals, double *evecs, double *resNorms, primme_params *primme)
    int fq_primme_gpu_available_()
    void *fq_primme_cublas_create_()
    int fq_primme_cublas_destroy_(void *h)
    const char *fq_primme_gpu_last_error_()
    void fq_primme_gpu_last_error_clear_()


cdef extern from *:
    """
    extern "C" void fq_primme_host_clamped_jacobi(const double *d, double shift,
                                                  const double *x, double *y, long long n);
    """
    void fq_primme_host_clamped_jacobi(const double *d, double shift, const double *x,
                                       double *y, long long n)


cdef extern from *:
    """
    #ifdef FQ_PRIMME_GPU
    extern "C" void fq_primme_host_pinned_set(int enabled);
    extern "C" void fq_primme_host_pinned_stats(double *seconds, double *gib, int *registered,
                                                int *failed);
    static void fq_primme_host_pinned_set_(int enabled) { fq_primme_host_pinned_set(enabled); }
    static void fq_primme_host_pinned_stats_(double *seconds, double *gib, int *registered,
                                             int *failed)
    { fq_primme_host_pinned_stats(seconds, gib, registered, failed); }
    #else
    static void fq_primme_host_pinned_set_(int enabled) { (void)enabled; }
    static void fq_primme_host_pinned_stats_(double *seconds, double *gib, int *registered,
                                             int *failed)
    { *seconds = 0.0; *gib = 0.0; *registered = 0; *failed = 0; }
    #endif
    typedef int (*fq_matvec_start_fn)(double *, double *, primme_params *);
    typedef int (*fq_matvec_input_done_fn)(primme_params *);
    typedef int (*fq_matvec_wait_fn)(primme_params *, int);
    static fq_matvec_start_fn fq_matvec_start_cb = 0;
    static fq_matvec_input_done_fn fq_matvec_input_done_cb = 0;
    static fq_matvec_wait_fn fq_matvec_wait_cb = 0;
    static void fq_matvec_overlap_set_(fq_matvec_start_fn start, fq_matvec_input_done_fn input_done,
                                       fq_matvec_wait_fn wait)
    {
        fq_matvec_start_cb = start;
        fq_matvec_input_done_cb = input_done;
        fq_matvec_wait_cb = wait;
    }
    extern "C" int fq_primme_matvec_start(double *x, double *y, primme_params *primme)
    { return fq_matvec_start_cb ? fq_matvec_start_cb(x, y, primme) : 1; }
    extern "C" int fq_primme_matvec_input_done(primme_params *primme)
    { return fq_matvec_input_done_cb ? fq_matvec_input_done_cb(primme) : 0; }
    extern "C" int fq_primme_matvec_wait(primme_params *primme, int accept)
    { return fq_matvec_wait_cb ? fq_matvec_wait_cb(primme, accept) : 0; }
    typedef int (*fq_spec_start_fn)(double *, double *, primme_params *);
    typedef double (*fq_matvec_dot_fn)(primme_params *);
    static fq_spec_start_fn fq_spec_start_cb = 0;
    static fq_matvec_dot_fn fq_matvec_dot_cb = 0;
    static void fq_spec_overlap_set_(fq_spec_start_fn start, fq_matvec_dot_fn dot)
    {
        fq_spec_start_cb = start;
        fq_matvec_dot_cb = dot;
    }
    extern "C" int fq_primme_spec_matvec_start(double *x, double *y, primme_params *primme)
    { return fq_spec_start_cb ? fq_spec_start_cb(x, y, primme) : 1; }
    extern "C" double fq_primme_matvec_dot(primme_params *primme)
    { return fq_matvec_dot_cb ? fq_matvec_dot_cb(primme) : 0.0; }
    #ifdef FQ_PRIMME_GPU
    extern "C" long long fq_primme_spec_chunk_rows(void);
    extern "C" int fq_primme_spec_wait_rows(long long rows);
    extern "C" void fq_primme_spec_wait_output_free(void);
    static long long fq_primme_spec_chunk_rows_(void) { return fq_primme_spec_chunk_rows(); }
    static int fq_primme_spec_wait_rows_(long long rows) { return fq_primme_spec_wait_rows(rows); }
    static void fq_primme_spec_wait_output_free_(void) { fq_primme_spec_wait_output_free(); }
    extern "C" void fq_primme_spec_set_enabled(int enabled, const double *diag);
    extern "C" void fq_primme_spec_release(void);
    extern "C" void fq_primme_spec_stats(long long *counts, double *seconds);
    static void fq_primme_spec_set_enabled_(int enabled, const double *diag)
    { fq_primme_spec_set_enabled(enabled, diag); }
    static void fq_primme_spec_release_(void) { fq_primme_spec_release(); }
    static void fq_primme_spec_stats_(long long *counts, double *seconds)
    { fq_primme_spec_stats(counts, seconds); }
    #else
    static long long fq_primme_spec_chunk_rows_(void) { return 0; }
    static int fq_primme_spec_wait_rows_(long long rows) { (void)rows; return 0; }
    static void fq_primme_spec_wait_output_free_(void) { }
    static void fq_primme_spec_set_enabled_(int enabled, const double *diag)
    { (void)enabled; (void)diag; }
    static void fq_primme_spec_release_(void) { }
    static void fq_primme_spec_stats_(long long *counts, double *seconds)
    { int i; for (i = 0; i < 5; i++) counts[i] = 0; for (i = 0; i < 3; i++) seconds[i] = 0.0; }
    #endif
    """
    ctypedef int (*fq_matvec_start_fn)(double *, double *, primme_params *) noexcept
    ctypedef int (*fq_matvec_input_done_fn)(primme_params *) noexcept
    ctypedef int (*fq_matvec_wait_fn)(primme_params *, int) noexcept
    void fq_primme_host_pinned_set_(int enabled)
    void fq_primme_host_pinned_stats_(double *seconds, double *gib, int *registered, int *failed)
    void fq_matvec_overlap_set_(fq_matvec_start_fn start, fq_matvec_input_done_fn input_done,
                                fq_matvec_wait_fn wait)
    ctypedef int (*fq_spec_start_fn)(double *, double *, primme_params *) noexcept
    ctypedef double (*fq_matvec_dot_fn)(primme_params *) noexcept
    void fq_spec_overlap_set_(fq_spec_start_fn start, fq_matvec_dot_fn dot)
    void fq_primme_spec_set_enabled_(int enabled, const double *diag)
    void fq_primme_spec_release_()
    void fq_primme_spec_stats_(long long *counts, double *seconds)
    long long fq_primme_spec_chunk_rows_() nogil
    int fq_primme_spec_wait_rows_(long long rows) nogil
    void fq_primme_spec_wait_output_free_() nogil


cdef enum:
    PRIMME_n = 1
    PRIMME_matrixMatvec = 2
    PRIMME_applyPreconditioner = 4
    PRIMME_numProcs = 8
    PRIMME_procID = 9
    PRIMME_nLocal = 11
    PRIMME_globalSumReal = 12
    PRIMME_broadcastReal = 14
    PRIMME_numEvals = 16
    PRIMME_target = 17
    PRIMME_locking = 20
    PRIMME_initSize = 21
    PRIMME_maxBasisSize = 23
    PRIMME_minRestartSize = 24
    PRIMME_maxBlockSize = 25
    PRIMME_maxMatvecs = 26
    PRIMME_eps = 32
    PRIMME_printLevel = 35
    PRIMME_matrix = 37
    PRIMME_ShiftsForPreconditioner = 40
    PRIMME_initBasisMode = 41
    PRIMME_restartingParams_maxPrevRetain = 43
    PRIMME_correctionParams_precondition = 44
    PRIMME_stats_numOuterIterations = 55
    PRIMME_stats_numRestarts = 56
    PRIMME_stats_numMatvecs = 57
    PRIMME_stats_numPreconds = 58
    PRIMME_stats_numGlobalSum = 59
    PRIMME_stats_volumeGlobalSum = 60
    PRIMME_stats_numBroadcast = 61
    PRIMME_stats_volumeBroadcast = 62
    PRIMME_stats_elapsedTime = 65
    PRIMME_stats_timeMatvec = 66
    PRIMME_stats_timePrecond = 67
    PRIMME_stats_timeOrtho = 68
    PRIMME_stats_timeGlobalSum = 69
    PRIMME_stats_timeBroadcast = 70
    PRIMME_stats_timeDense = 71
    PRIMME_stats_estimateMinEVal = 72
    PRIMME_stats_estimateMaxEVal = 73
    PRIMME_stats_estimateLargestSVal = 74
    PRIMME_convTestFun = 80
    PRIMME_queue = 88


_TARGETS = {
    "SA": "primme_smallest",
    "LA": "primme_largest",
    "SM": "primme_closest_abs",
    "LM": "primme_largest_abs",
}


def gpu_backend_available():

    return bool(fq_primme_gpu_available_())


cdef class _Context:


    cdef public object H
    cdef public object comm
    cdef public object MPI
    cdef public object precond_diag
    cdef public object exc
    cdef public object conv
    cdef public double tol
    cdef public Py_ssize_t n_local
    cdef public object monitor
    cdef public object overlap
    cdef public double energy_tol
    cdef public int energy_window
    cdef public double resnorm_cap
    cdef public double sign
    cdef public list history
    cdef public object accepted
    cdef public object reason
    cdef public int checks

    cdef public object gpu_matvec

    cdef public object gpu_precond

    cdef public bint host_precond
    cdef public size_t precond_diag_ptr

    def __init__(self, H, precond_diag, conv, double tol):


        self.H = H
        self.comm = H._comm
        from mpi4py import MPI
        self.MPI = MPI
        self.precond_diag = precond_diag
        self.overlap = None
        self.exc = None
        self.conv = conv
        self.tol = tol
        self.history = []
        self.accepted = None
        self.reason = None
        self.checks = 0
        self.n_local = int(H.n_local)
        self.host_precond = precond_diag is not None
        self.precond_diag_ptr = 0
        if self.host_precond:
            self.precond_diag = np.ascontiguousarray(precond_diag, dtype=np.float64)
            self.precond_diag_ptr = <size_t>self.precond_diag.ctypes.data


cdef void _set_int(primme_params *pp, int label, int value) except *:

    cdef PRIMME_INT v = <PRIMME_INT>value
    if primme_set_member(pp, <primme_params_label>label, &v) != 0:
        raise RuntimeError(f"PRIMME rejected integer parameter label {label}")


cdef void _set_int64(primme_params *pp, int label, PRIMME_INT value) except *:

    cdef PRIMME_INT v = value
    if primme_set_member(pp, <primme_params_label>label, &v) != 0:
        raise RuntimeError(f"PRIMME rejected integer parameter label {label}")


cdef void _set_double(primme_params *pp, int label, double value) except *:

    cdef double v = value
    if primme_set_member(pp, <primme_params_label>label, &v) != 0:
        raise RuntimeError(f"PRIMME rejected double parameter label {label}")


cdef void _set_pointer(primme_params *pp, int label, void *value) except *:

    if primme_set_member(pp, <primme_params_label>label, value) != 0:
        raise RuntimeError(f"PRIMME rejected pointer parameter label {label}")


cdef int _constant(str name) except -2147483648:

    cdef bytes b = name.encode("ascii")
    cdef int value = -1
    if primme_constant_info(<const char *>b, &value) != 0 or value < 0:
        raise ValueError(f"Unknown PRIMME constant {name!r}")
    return value


cdef inline _Context _context(primme_params *primme):

    return <_Context>primme.matrix


cdef object _as_local_block(double *ptr, PRIMME_INT ld, int block_size, Py_ssize_t n_local):


    return np.asarray(<double[:ld:1, :block_size]>ptr)[:n_local, :]


cdef void c_matvec(
    void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
    primme_params *primme, int *ierr
) noexcept with gil:


    cdef _Context ctx = _context(primme)
    cdef int j
    cdef object xin
    cdef object yview
    cdef object yout
    try:
        xin = _as_local_block(<double *>x, ldx[0], blockSize[0], ctx.n_local)
        yview = _as_local_block(<double *>y, ldy[0], blockSize[0], ctx.n_local)
        for j in range(blockSize[0]):
            yout = ctx.H._matvec(np.asarray(xin[:, j], dtype=np.float64))
            yview[:, j] = np.asarray(yout, dtype=np.float64)
        ierr[0] = 0
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        ierr[0] = 1


cdef void c_matvec_gpu(
    void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
    primme_params *primme, int *ierr
) noexcept with gil:


    cdef _Context ctx = _context(primme)
    try:
        ctx.gpu_matvec(
            <size_t>x, <long long>ldx[0], <size_t>y, <long long>ldy[0],
            blockSize[0],
        )
        ierr[0] = 0
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        ierr[0] = 1


cdef int c_overlap_start(double *x, double *y, primme_params *primme) noexcept with gil:


    cdef _Context ctx = _context(primme)
    if ctx.overlap is None:
        return 1
    try:
        ctx.overlap.start(<size_t>x, <size_t>y)
        return 0
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        return -1


cdef int c_overlap_input_done(primme_params *primme) noexcept with gil:


    cdef _Context ctx = _context(primme)
    try:
        ctx.overlap.input_done()
        return 0
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        return 1


cdef int c_overlap_wait(primme_params *primme, int accept) noexcept with gil:


    cdef _Context ctx = _context(primme)
    try:
        ctx.overlap.wait(bool(accept))
        return 0
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        return 1


cdef int c_spec_start(double *x, double *y, primme_params *primme) noexcept with gil:


    cdef _Context ctx = _context(primme)
    if ctx.overlap is None:
        return 1
    try:
        ctx.overlap.start(<size_t>x, <size_t>y, True)
        return 0
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        return -1


cdef double c_matvec_dot(primme_params *primme) noexcept with gil:

    cdef _Context ctx = _context(primme)
    if ctx.overlap is None:
        return 0.0
    try:
        return float(ctx.overlap.dot())
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        return float("nan")


fq_matvec_overlap_set_(c_overlap_start, c_overlap_input_done, c_overlap_wait)
fq_spec_overlap_set_(c_spec_start, c_matvec_dot)


def spec_chunk_rows():

    return int(fq_primme_spec_chunk_rows_())


def spec_wait_rows(long long rows):

    cdef int err
    with nogil:
        err = fq_primme_spec_wait_rows_(rows)
    return err == 0


def spec_wait_output_free():

    with nogil:
        fq_primme_spec_wait_output_free_()


cdef void c_precond_gpu(
    void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
    primme_params *primme, int *ierr
) noexcept with gil:


    cdef _Context ctx = _context(primme)
    cdef double *shifts = NULL
    cdef int j
    try:
        if primme_get_member(
            primme, <primme_params_label>PRIMME_ShiftsForPreconditioner, &shifts
        ) != 0 or shifts == NULL:
            raise RuntimeError("PRIMME did not provide ShiftsForPreconditioner")
        shift_list = [shifts[j] for j in range(blockSize[0])]
        ctx.gpu_precond(
            <size_t>x, <long long>ldx[0], <size_t>y, <long long>ldy[0],
            blockSize[0], shift_list,
        )
        ierr[0] = 0
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        ierr[0] = 1


cdef void c_precond(
    void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
    primme_params *primme, int *ierr
) noexcept with gil:


    cdef _Context ctx = _context(primme)
    cdef double *shifts = NULL
    cdef int j
    try:
        if primme_get_member(
            primme, <primme_params_label>PRIMME_ShiftsForPreconditioner, &shifts
        ) != 0 or shifts == NULL:
            raise RuntimeError("PRIMME did not provide ShiftsForPreconditioner")
        if not ctx.host_precond:
            raise RuntimeError("the host preconditioner has no diagonal")
        for j in range(blockSize[0]):
            fq_primme_host_clamped_jacobi(
                <const double *>ctx.precond_diag_ptr, shifts[j],
                (<const double *>x) + ldx[0] * j, (<double *>y) + ldy[0] * j,
                <long long>ctx.n_local,
            )
        ierr[0] = 0
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        ierr[0] = 1


cdef void c_global_sum_real(
    void *sendBuf, void *recvBuf, int *count, primme_params *primme, int *ierr
) noexcept with gil:


    cdef _Context ctx = _context(primme)
    cdef object send_arr
    cdef object recv_arr
    try:
        if count[0] <= 0:
            ierr[0] = 0
            return
        if recvBuf == NULL:
            raise RuntimeError("PRIMME called globalSumReal with a null recvBuf")
        recv_arr = np.asarray(<double[:count[0]]>recvBuf)
        if sendBuf == recvBuf:
            ctx.comm.Allreduce(ctx.MPI.IN_PLACE, [recv_arr, ctx.MPI.DOUBLE], op=ctx.MPI.SUM)
        else:
            if sendBuf == NULL:
                raise RuntimeError("PRIMME called globalSumReal with a null sendBuf")
            send_arr = np.asarray(<double[:count[0]]>sendBuf)
            ctx.comm.Allreduce([send_arr, ctx.MPI.DOUBLE], [recv_arr, ctx.MPI.DOUBLE], op=ctx.MPI.SUM)
        ierr[0] = 0
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        ierr[0] = 1


cdef void c_broadcast_real(
    void *buffer, int *count, primme_params *primme, int *ierr
) noexcept with gil:


    cdef _Context ctx = _context(primme)
    cdef object buf
    try:
        if count[0] <= 0:
            ierr[0] = 0
            return
        if buffer == NULL:
            raise RuntimeError("PRIMME called broadcastReal with a null buffer")
        buf = np.asarray(<double[:count[0]]>buffer)
        ctx.comm.Bcast([buf, ctx.MPI.DOUBLE], root=0)
        ierr[0] = 0
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        ierr[0] = 1


cdef double _EPS23 = 4.0e-11


cdef bint _energy_stable(_Context ctx):

    cdef Py_ssize_t n = len(ctx.history)
    cdef Py_ssize_t j
    cdef double latest
    if n <= ctx.energy_window:
        return False
    latest = ctx.history[n - 1][1]
    for j in range(n - 1 - ctx.energy_window, n - 1):
        if fabs(ctx.history[j][1] - latest) >= ctx.energy_tol:
            return False
    return True


cdef bint _conv_decision(_Context ctx, double value, double resnorm, long long matvecs) except? -1:


    cdef bint target = True
    cdef bint isconv
    if ctx.conv == "abs":
        isconv = resnorm < ctx.tol
        if isconv:
            ctx.reason = "residual"
        return isconv
    isconv = resnorm <= ctx.tol * fmax(_EPS23, fabs(value))
    if ctx.conv == "energy":
        last = len(ctx.history) - 1
        if last < 0 or ctx.history[last][0] != matvecs:
            ctx.history.append([int(matvecs), value])
        elif ctx.sign * value < ctx.sign * ctx.history[last][1]:
            ctx.history[last][1] = value
        else:
            target = False
        if target and not isconv and resnorm < ctx.resnorm_cap and _energy_stable(ctx):
            isconv = True
            ctx.reason = "energy"
            ctx.accepted = (int(matvecs), value)
        elif (not isconv and ctx.accepted is not None and ctx.accepted[0] == matvecs
              and resnorm < ctx.resnorm_cap and fabs(value - ctx.accepted[1]) <= ctx.energy_tol):


            isconv = True
    if isconv and ctx.reason is None:
        ctx.reason = "residual"
    return isconv


def _test_conv_sequence(conv, double tol, double energy_tol, int energy_window, double resnorm_cap,
                        steps, double sign=1.0):


    cdef _Context ctx = _Context.__new__(_Context)
    ctx.conv = conv
    ctx.tol = tol
    ctx.energy_tol = energy_tol
    ctx.energy_window = energy_window
    ctx.resnorm_cap = resnorm_cap
    ctx.sign = sign
    ctx.history = []
    ctx.reason = None
    ctx.accepted = None
    decisions = [bool(_conv_decision(ctx, v, r, m)) for v, r, m in steps]
    return decisions, ctx.reason


cdef void c_conv(
    double *eval, void *evec, double *resNorm, int *isconv,
    primme_params *primme, int *ierr
) noexcept with gil:


    cdef _Context ctx = _context(primme)
    cdef PRIMME_INT matvecs = 0
    cdef double value = eval[0]
    try:
        ctx.checks += 1
        if ctx.conv == "energy":
            primme_get_member(primme, <primme_params_label>PRIMME_stats_numMatvecs, &matvecs)
        isconv[0] = 1 if _conv_decision(ctx, value, resNorm[0], int(matvecs)) else 0
        if ctx.monitor is not None:
            ctx.monitor(value, resNorm[0])
        ierr[0] = 0
    except BaseException as exc:  # noqa: BLE001
        ctx.exc = exc
        ierr[0] = 1


cdef dict _stats(primme_params *pp, np.ndarray rnorms, int nconv):


    cdef PRIMME_INT i64
    cdef double d
    out = {}
    for key, label in [
        ("numOuterIterations", PRIMME_stats_numOuterIterations),
        ("numRestarts", PRIMME_stats_numRestarts),
        ("numMatvecs", PRIMME_stats_numMatvecs),
        ("numPreconds", PRIMME_stats_numPreconds),
        ("numGlobalSum", PRIMME_stats_numGlobalSum),
        ("volumeGlobalSum", PRIMME_stats_volumeGlobalSum),
        ("numBroadcast", PRIMME_stats_numBroadcast),
        ("volumeBroadcast", PRIMME_stats_volumeBroadcast),
    ]:
        i64 = 0
        primme_get_member(pp, <primme_params_label>label, &i64)
        out[key] = int(i64)
    for key, label in [
        ("elapsedTime", PRIMME_stats_elapsedTime),
        ("timeMatvec", PRIMME_stats_timeMatvec),
        ("timePrecond", PRIMME_stats_timePrecond),
        ("timeOrtho", PRIMME_stats_timeOrtho),
        ("timeGlobalSum", PRIMME_stats_timeGlobalSum),
        ("timeBroadcast", PRIMME_stats_timeBroadcast),
        ("timeDense", PRIMME_stats_timeDense),
        ("estimateMinEVal", PRIMME_stats_estimateMinEVal),
        ("estimateMaxEVal", PRIMME_stats_estimateMaxEVal),
        ("estimateLargestSVal", PRIMME_stats_estimateLargestSVal),
    ]:
        d = 0.0
        primme_get_member(pp, <primme_params_label>label, &d)
        out[key] = float(d)
    out["rnorms"] = np.array(rnorms[:nconv], copy=True)
    return out


def eigsh(
    H,
    int k,
    double tol,
    int maxiter,
    int ncv,
    str which,
    str method,
    str conv,
    precond_diag_local,
    v0_local,
    bint return_eigenvectors,
    bint return_stats,
    bint return_unconverged,
    int maxBlockSize,
    int minRestartSize,
    int maxPrevRetain,
    int printLevel,
    monitor=None,
    gpu_evecs=None,
    gpu_matvec=None,
    int gpu_init_size=0,
    gpu_precond=None,
    str initBasisMode="",
    overlap_matvec=None,
    bint pinned_basis=False,
    primme_comm=None,
    bint spec_matvec=False,
    double energy_tol=0.0,
    int energy_window=0,
    double resnorm_cap=0.0,
):


    cdef primme_params *pp = NULL
    cdef np.ndarray[np.double_t, ndim=1] evals
    cdef np.ndarray[np.double_t, ndim=1] rnorms
    cdef np.ndarray[np.double_t, ndim=2, mode="fortran"] evecs
    cdef int err
    cdef int init_size
    cdef int basis_size = 0
    cdef int target
    cdef PRIMME_INT init_size_out
    cdef double *evals_ptr
    cdef double *evecs_ptr
    cdef double *rnorms_ptr
    cdef double pinned_seconds = 0.0
    cdef double pinned_gib = 0.0
    cdef int pinned_registered = 0
    cdef int pinned_failed = 0
    cdef long long spec_counts[5]
    cdef double spec_seconds[3]
    cdef int spec_i
    cdef PRIMME_INT n_global = <PRIMME_INT>H.shape[0]
    cdef PRIMME_INT n_local = <PRIMME_INT>H.n_local
    cdef _Context ctx

    cdef bint use_gpu = gpu_evecs is not None
    cdef size_t cublas_handle_slot = 0
    cdef size_t gpu_evecs_ptr = 0
    cdef void *own_cublas_handle = NULL

    for spec_i in range(5):
        spec_counts[spec_i] = 0
    for spec_i in range(3):
        spec_seconds[spec_i] = 0.0
    if k <= 0:
        raise ValueError("k must be positive")
    if n_local < 0 or n_global < 0 or n_local > n_global:
        raise ValueError("invalid distributed dimensions")
    checked_multiply(checked_multiply(n_local, k), sizeof(double))
    if ncv > 0:
        checked_multiply(checked_multiply(n_local, ncv), sizeof(double))
        checked_multiply(checked_multiply(ncv, ncv), sizeof(double))
    if use_gpu and not fq_primme_gpu_available_():
        raise RuntimeError(
            "GPU PRIMME path requested but this _primme_mpi build lacks "
            "the cuBLAS backend; rebuild with FQ_BUILD_GPU=1 and a CUDA "
            "toolkit on PATH."
        )

    pp = primme_params_create()
    if pp == NULL:
        raise MemoryError("could not allocate primme_params")

    try:
        ctx = _Context(H, precond_diag_local, conv, tol)
        ctx.monitor = monitor
        ctx.energy_tol = energy_tol
        ctx.energy_window = energy_window
        ctx.resnorm_cap = resnorm_cap
        ctx.sign = -1.0 if which == "LA" else 1.0
        if primme_comm is not None:
            ctx.comm = primme_comm
        if not use_gpu:
            ctx.overlap = overlap_matvec
        ctx.gpu_matvec = gpu_matvec
        ctx.gpu_precond = gpu_precond
        _set_pointer(pp, PRIMME_matrix, <void *>ctx)

        _set_int64(pp, PRIMME_n, n_global)
        _set_int64(pp, PRIMME_nLocal, n_local)
        _set_int(pp, PRIMME_numProcs, int(H._comm.Get_size()))
        _set_int(pp, PRIMME_procID, int(H._comm.Get_rank()))
        _set_int(pp, PRIMME_numEvals, k)
        _set_double(pp, PRIMME_eps, tol)
        _set_int(pp, PRIMME_printLevel, printLevel)
        if use_gpu:
            _set_pointer(pp, PRIMME_matrixMatvec, <void *>c_matvec_gpu)
            own_cublas_handle = fq_primme_cublas_create_()
            if own_cublas_handle == NULL:
                raise RuntimeError(
                    "cublasCreate failed while setting up the GPU "
                    "PRIMME path (no CUDA device visible to this rank?)"
                )
            cublas_handle_slot = <size_t>own_cublas_handle
            _set_pointer(pp, PRIMME_queue, <void *>&cublas_handle_slot)
        elif gpu_matvec is not None:
            _set_pointer(pp, PRIMME_matrixMatvec, <void *>c_matvec_gpu)
        else:
            _set_pointer(pp, PRIMME_matrixMatvec, <void *>c_matvec)
        _set_pointer(pp, PRIMME_globalSumReal, <void *>c_global_sum_real)
        _set_pointer(pp, PRIMME_broadcastReal, <void *>c_broadcast_real)

        if gpu_precond is not None:
            _set_int(pp, PRIMME_correctionParams_precondition, 1)
            _set_pointer(pp, PRIMME_applyPreconditioner, <void *>c_precond_gpu)
        elif precond_diag_local is not None:
            _set_int(pp, PRIMME_correctionParams_precondition, 1)
            _set_pointer(pp, PRIMME_applyPreconditioner, <void *>c_precond)
        else:
            _set_int(pp, PRIMME_correctionParams_precondition, 0)

        if conv in ("abs", "rel", "energy"):
            _set_pointer(pp, PRIMME_convTestFun, <void *>c_conv)

        target = _constant(_TARGETS[which])
        _set_int(pp, PRIMME_target, target)

        if method:
            if primme_set_method(<primme_preset_method>_constant(method), pp) != 0:
                raise ValueError(f"PRIMME rejected method={method!r}")

        if maxiter >= 0:
            _set_int64(pp, PRIMME_maxMatvecs, <PRIMME_INT>maxiter)
        if ncv > 0:
            _set_int(pp, PRIMME_maxBasisSize, ncv)
        if maxBlockSize > 0:
            _set_int(pp, PRIMME_maxBlockSize, maxBlockSize)
        if minRestartSize > 0:
            _set_int(pp, PRIMME_minRestartSize, minRestartSize)
        if maxPrevRetain > 0:
            _set_int(pp, PRIMME_restartingParams_maxPrevRetain, maxPrevRetain)
        if initBasisMode:
            _set_int(pp, PRIMME_initBasisMode, _constant(initBasisMode))

        if primme_get_member(pp, <primme_params_label>PRIMME_maxBasisSize, &basis_size) == 0 and basis_size > 0:
            checked_multiply(checked_multiply(n_local, basis_size), sizeof(double))
            checked_multiply(checked_multiply(basis_size, basis_size), sizeof(double))

        evals = np.zeros(k, dtype=np.float64)
        rnorms = np.zeros(k, dtype=np.float64)
        fq_primme_gpu_last_error_clear_()

        if use_gpu:
            if gpu_init_size > 0:
                _set_int(pp, PRIMME_initSize, gpu_init_size)
            gpu_evecs_ptr = <size_t>int(gpu_evecs.data.ptr)
            err = fq_cublas_dprimme_(
                &evals[0], <double *>gpu_evecs_ptr, &rnorms[0], pp
            )
        else:
            evecs = np.zeros((int(n_local), k), dtype=np.float64, order="F")

            if v0_local is not None:
                init_size = min(k, v0_local.shape[1])
                _set_int(pp, PRIMME_initSize, init_size)
                evecs[:, :init_size] = np.asarray(v0_local[:, :init_size], dtype=np.float64, order="F")

            evals_ptr = &evals[0]
            evecs_ptr = &evecs[0, 0]
            rnorms_ptr = &rnorms[0]
            fq_primme_host_pinned_set_(1 if pinned_basis else 0)
            fq_primme_spec_set_enabled_(
                1 if (spec_matvec and ctx.overlap is not None) else 0,
                <const double *>ctx.precond_diag_ptr if (ctx.host_precond and gpu_precond is None) else NULL,
            )
            with nogil:
                err = dprimme(evals_ptr, evecs_ptr, rnorms_ptr, pp)
            fq_primme_spec_stats_(spec_counts, spec_seconds)
            fq_primme_spec_set_enabled_(0, NULL)
            fq_primme_host_pinned_stats_(&pinned_seconds, &pinned_gib, &pinned_registered, &pinned_failed)
            fq_primme_host_pinned_set_(0)
        if err != 0 and not (err == -3 and return_unconverged):
            gpu_error = fq_primme_gpu_last_error_().decode()
            message = f"PRIMME failed with error code {err}"
            if gpu_error:
                message += f"; GPU offload: {gpu_error}"
            if ctx.exc is not None:
                raise RuntimeError(message) from ctx.exc
            raise RuntimeError(message)

        result = [np.array(evals[:k], copy=True)]
        if return_eigenvectors:
            if use_gpu:
                result.append(gpu_evecs[:, :k].copy(order="F"))
            else:
                result.append(np.array(evecs[:, :k], copy=True, order="F"))
        if return_stats:
            stats = _stats(pp, rnorms, k)
            if not use_gpu:
                stats["pinnedBasisGiB"] = float(pinned_gib)
                stats["pinnedBasisSeconds"] = float(pinned_seconds)
                stats["pinnedBasisArrays"] = int(pinned_registered)
                stats["pinnedBasisFailed"] = int(pinned_failed)
                if spec_matvec:
                    stats["specMatvecs"] = {
                        "started": int(spec_counts[0]), "taken": int(spec_counts[1]),
                        "kept": int(spec_counts[2]), "discarded": int(spec_counts[3]),
                        "onePassOrtho": int(spec_counts[4]), "passSeconds": float(spec_seconds[0]),
                        "windowSeconds": float(spec_seconds[1]), "allocSeconds": float(spec_seconds[2]),
                    }
                if ctx.overlap is not None:
                    stats.update(ctx.overlap.stats())
            stats["convergence"] = {"conv": conv, "reason": ctx.reason, "checks": ctx.checks,
                                    "history": [tuple(entry) for entry in ctx.history]}
            result.append(stats)

        if len(result) == 1:
            return result[0]
        return tuple(result)
    finally:
        if not use_gpu:
            fq_primme_host_pinned_set_(0)
            if overlap_matvec is not None:
                overlap_matvec.close()
            fq_primme_spec_set_enabled_(0, NULL)
            fq_primme_spec_release_()
        if own_cublas_handle != NULL and fq_primme_cublas_destroy_(own_cublas_handle) != 0:
            warnings.warn("Failed to destroy the PRIMME GPU cuBLAS handle during cleanup", RuntimeWarning)
        if pp != NULL:
            primme_free(pp)
            primme_params_destroy(pp)
