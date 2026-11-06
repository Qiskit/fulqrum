# Installing Fulqrum with distributed GPU support

GPU support includes all CPU modules, the CUDA matvec kernels, Fulqrum's
distributed PRIMME interface, and the pinned PRIMME source with Fulqrum's patches
and cuBLAS backend. One build switch, `FQ_BUILD_GPU=1`, enables the complete stack.
PRIMME's BLAS integer width and symbol convention are fixed by the build:
64-bit integers with `_64_` BLAS/LAPACK symbols.

## Requirements

- Linux on a 64-bit system, Python 3.10 or newer, and NVIDIA GPUs.
- An NVIDIA driver compatible with the selected CUDA toolkit.
- The CUDA toolkit, including `nvcc`, headers, cuBLAS, cuSPARSE, and cudart.
  A driver or CuPy wheel alone does not provide the complete build toolkit.
- A CUDA-compatible C++17 compiler with OpenMP support, a C compiler, GNU Make,
  Git (for applying PRIMME patches).
- Open MPI and a matching mpi4py installation and `mpirun` launcher.
- OpenBLAS ILP64 with LAPACK and the `_64_` symbol convention. Ordinary LP64 BLAS
  and private NumPy/SciPy wheel libraries are not compatible replacements.

The `gpu` extra installs `cupy-cuda13x>=14,<15`. Install `mpi4py>=4.1,<5`
separately against your Open MPI installation, as shown below.
The local reference stack is CUDA 13.3, CuPy 14.1, mpi4py 4.1, OpenBLAS ILP64
0.3.34, and H100 (`sm_90`). This reference is not a guarantee for every platform
or driver combination. For CUDA 12, install a compatible `cupy-cuda12x` explicitly
and install Fulqrum without the `gpu` extra; never install both CuPy variants.
The build filters its fallback targets against the selected toolkit's supported
architectures. NVIDIA HPC SDK split toolkit/math-library layouts are detected.

## Recommended environment with conda or mamba

Install the CUDA 13 toolkit through your system administrator or NVIDIA's installer.
Ensure `nvcc` is on `PATH`, or set `CUDA_HOME` to its toolkit root.
Create the Python, MPI, and BLAS environment using:

```bash
conda create -n fulqrum-gpu --override-channels -c conda-forge \
  python=3.12 pip openmpi "mpi=*=openmpi" "mpi4py>=4.1,<5" \
  openblas-ilp64=0.3.34 "numpy>=2.0" "scipy>=1.14.1"
conda activate fulqrum-gpu
```

The command selects Open MPI and matching mpi4py from conda-forge and includes
OpenBLAS ILP64. It does not install the driver or CUDA toolkit. Replace `conda`
with `mamba` to use mamba. On clusters with an existing MPI stack, use the
existing-Open-MPI instructions below instead of installing a second MPI.

From the Fulqrum checkout:

```bash
git submodule update --init --recursive qiskit-addon-sqd-hpc third-party/primme
FQ_BUILD_GPU=1 python -m pip install ".[gpu]"
```

Pip's standard isolated build is supported. It installs its build dependencies
separately; MPI and CuPy are runtime dependencies. Fulqrum discovers OpenBLAS in
the active conda environment, the interpreter prefix, the default local OpenBLAS
prefix below, standard system directories, or `LD_LIBRARY_PATH`.

## Without conda

Install the CUDA toolkit and Open MPI through your operating system or cluster.
For example, Ubuntu/Debian users can install build tools and Open MPI:

```bash
sudo apt-get install build-essential python3-dev gfortran git libopenmpi-dev openmpi-bin
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip

# Build mpi4py explicitly against available Open MPI.
MPICC=/usr/bin/mpicc.openmpi python -m pip install \
  --no-binary=mpi4py --no-cache-dir "mpi4py>=4.1,<5"
```

Install OpenBLAS ILP64 as described below, then install the Python dependencies
and Fulqrum using the same `FQ_BUILD_GPU=1 python -m pip install ".[gpu]"` command.
Initialize both required submodules first. The `gpu` extra does not install mpi4py;
the explicit step above supplies it.

### Build OpenBLAS ILP64 from source

Use a fresh source tree from a pinned [OpenBLAS release](https://github.com/OpenMathLib/OpenBLAS/releases).
GCC, gfortran, and GNU Make are required for this recipe. LAPACK must remain enabled.
For example, use OpenBLAS 0.3.34 to match the reference BLAS version.

From the extracted OpenBLAS directory:

```bash
make -j4 CC=gcc FC=gfortran \
  INTERFACE64=1 SYMBOLSUFFIX=64_ LIBNAMESUFFIX=64_ \
  DYNAMIC_ARCH=1 USE_THREAD=1 USE_OPENMP=0

make install PREFIX="$HOME/.local/opt/openblas-ilp64" \
  INTERFACE64=1 SYMBOLSUFFIX=64_ LIBNAMESUFFIX=64_ \
  DYNAMIC_ARCH=1 USE_THREAD=1 USE_OPENMP=0
```

`INTERFACE64=1` selects 64-bit integers; `SYMBOLSUFFIX=64_` produces symbols such
as `dgemm_64_`; `LIBNAMESUFFIX=64_` gives the shared library a distinct filename.
The installer automatically searches this default prefix. Root access is not
required to install there.

Fulqrum checks OpenBLAS's 64-bit configuration and representative BLAS/LAPACK
symbols using Python, then compiles and links a small host/GPU solver probe
against its built PRIMME archive. The probe is not executed and needs no GPU.
No `nm` tool is required. For a different installation prefix,
set `FQ_BLAS_LIB=/absolute/path/to/libopenblas64_.so`. The library and its compiler
runtime dependencies, including libgfortran, must be available on every compute
node. Fulqrum records the discovered library directory in the extension's runtime
search path; if you move the environment or library, rebuild Fulqrum.

### Use an existing Open MPI installation

Use paths to your installed compiler, CUDA toolkit, and Open MPI. Replace the
example paths below with yours. If the tools are already on `PATH` and their
libraries are discoverable, skip the path exports. Use `lib64` instead of `lib`
for Open MPI if that matches its installation.

```bash
export PATH=/opt/gcc/bin:/opt/cuda/bin:/opt/openmpi/bin:$PATH
export CUDA_HOME=/opt/cuda
export CC=/opt/gcc/bin/gcc
export CXX=/opt/gcc/bin/g++
export LD_LIBRARY_PATH=/opt/openmpi/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}

# Check that the tools come from the intended installations.
"$CC" --version
nvcc --version
mpicc --showme
mpirun --version

# Build mpi4py against this Open MPI installation.
MPICC=/opt/openmpi/bin/mpicc python -m pip install \
  --force-reinstall --no-binary=mpi4py --no-cache-dir "mpi4py>=4.1,<5"

# From the Fulqrum checkout, with OpenBLAS ILP64 installed:
git submodule update --init --recursive qiskit-addon-sqd-hpc third-party/primme
FQ_BUILD_GPU=1 python -m pip install ".[gpu]"

# Verify MPI.
mpirun -n 2 python -c \
  'from mpi4py import MPI; print(MPI.COMM_WORLD.rank, MPI.Get_library_version())'
```

Use this Open MPI installation's `mpirun` or the scheduler's prescribed launcher.
Rebuilding mpi4py as shown replaces an installation built against another MPI.

MPI is required for coordination. CUDA-aware MPI is optional: automatic backend
selection uses NCCL when available and otherwise uses the supported MPI fallback.
For multi-node NCCL, install the site's NCCL/network support and validate it on
compute nodes. A basic host MPI installation does not guarantee optimal GPU
communication performance.

## Verification

Run these checks on a GPU compute node after installation:

```bash
python - <<'PYCODE'
import cupy
import fulqrum
from fulqrum.gpu.eigensolver import _primme_mpi
assert _primme_mpi.gpu_backend_available()
assert cupy.cuda.runtime.getDeviceCount() > 0
print("Fulqrum", fulqrum.__version__)
print("CuPy", cupy.__version__)
print("PRIMME cuBLAS backend available")
PYCODE

mpirun -n 2 python -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size, MPI.Get_library_version())'
```

These are installation checks. Before a production multi-GPU run, validate a
small distributed eigenproblem using the same GPUs, launcher, and communication
backend as the intended job. Follow your scheduler's GPU-per-rank assignment.

The first run compiles CuPy kernels once and caches them in
`~/.cupy/kernel_cache` (default); later runs reuse them. If your home directory is not persistent or not shared across nodes, set `CUPY_CACHE_DIR` to a directory that is.

## Optional build controls

| Variable | Default | Purpose |
| --- | --- | --- |
| `FQ_BUILD_GPU` | `0` | `=1` enables the complete GPU stack; CPU modules are always built. |
| `FQ_BUILD_PARALLEL` | `1` | `>1` parallelizes the build process (faster). |
| `FQ_CUDA_ARCH` | Detected, otherwise supported fallback targets | Target compute nodes, for example `sm_90`, when building without a GPU. |

Advanced overrides are available for unusual installations: `FQ_BLAS_LIB`,
`CUDA_HOME` (also legacy `CUDADIR`/`CUDA_PATH`), standard `CC`/`CXX`,
`FQ_CUDA_HOST_COMPILER`, `FQ_CUDA_MATH_DIR`, and `FQ_CUDA_LIBDIR`.
Fulqrum builds the pinned PRIMME sources with its patches and mandatory ILP64
settings in an isolated build directory; prebuilt PRIMME overrides are not supported.
The normal installation does not require these overrides.

`FQ_ENABLE_PRIMME_MPI=1` enables a CPU distributed-PRIMME build.
GPU builds enable this feature and ILP64 automatically. Explicitly disabling
`FQ_ENABLE_PRIMME_MPI` or `FQ_PRIMME_BLASINT64` while requesting GPU support is
an error. Partial distributed-kernel builds are no longer supported.
Developers can enable the GPU kernel test library with `FQ_BUILD_GPU_TESTS=1`.

## Which PRIMME do I need?

Fulqrum can use two separate PRIMME builds. They do not conflict, so you can
install both in the same environment.

| | Fulqrum's PRIMME | PyPI's `primme` |
| --- | --- | --- |
| Used by | `distributed_primme_eigsh(...)`: distributed GPU solves on one or more nodes | `primme.eigsh(...)`: CPU solves, single-node GPU solves with `GPUHamiltonian`, and tests |
| Installed by | Fulqrum's GPU build (`FQ_BUILD_GPU=1`) | Its own pip command, or Fulqrum's `primme` / `test` extras |
| Needs | OpenBLAS **ILP64** (64-bit integers, `dgemm_64_`) | Ordinary **LP64** BLAS/LAPACK (32-bit integers, `dgemm_`) |

Each build needs its own BLAS type. The ILP64 library cannot replace the LP64
one, and the LP64 library cannot replace the ILP64 one.

`GPUHamiltonian` does not need PyPI's `primme`. You can also pass it to
`scipy.sparse.linalg.eigsh`.

### Fulqrum's PRIMME

The GPU install above builds it. You do not need `pip install primme` for
distributed solves.

### PyPI's `primme`

PyPI provides PRIMME only as source code, so pip compiles it. This needs a C
compiler and LP64 BLAS/LAPACK development libraries.

Tests that import `primme` are not collected until it is installed.

## Troubleshooting

- **OpenBLAS ILP64 not found or rejected:** install `openblas-ilp64`, follow the
  source-build recipe above, or set `FQ_BLAS_LIB` to the full path of its shared
  library, such as `/path/to/libopenblas64_.so`. The library must use 64-bit
  integers, provide the required BLAS and LAPACK functions with `_64_` names,
  and be loadable on the machine running the build. Ordinary LP64 OpenBLAS is
  incompatible.
- **CUDA compiler or libraries missing:** install the complete CUDA toolkit,
  including its compiler and development libraries. Check that `nvcc --version`
  works. If the toolkit is installed outside the usual locations, set
  `CUDA_HOME` to its root directory, such as `/opt/cuda`.
- **C++ compiler unsupported by CUDA:** use a C++ compiler version supported by
  your CUDA toolkit. If several compilers are installed, set
  `FQ_CUDA_HOST_COMPILER` to the full path of the compatible compiler. Fulqrum
  checks whether nvcc can use it before compiling the GPU kernels.
- **Importing mpi4py or launching an MPI job fails:** use mpi4py built against
  the same Open MPI installation as your `mpirun` launcher. Follow the
  existing-Open-MPI instructions above to rebuild mpi4py if needed, then run
  the MPI verification command.
- **Importing Fulqrum reports a missing shared library:** Fulqrum's compiled
  modules load MPI, CUDA, and OpenBLAS libraries from your system or environment.
  Python can raise `ImportError` with a message such as
  `libopenblas64_.so: cannot open shared object file` if one of these libraries
  is no longer available at a location it searches. This can happen after
  renaming or copying a conda environment, moving a library installation to
  another directory, changing loaded cluster modules, or running on a compute
  node where the libraries are absent. Activate the intended environment or
  load the required cluster modules, and check that the libraries are available
  on every compute node. If their installation paths have changed permanently,
  rebuild and reinstall Fulqrum using the new paths.

During installation, Fulqrum copies the PRIMME source into a separate build
directory, applies its patches there, and compiles it. The original files in
`third-party/primme` remain unchanged. The build can reuse previously compiled
files, but it checks the build settings, source code, and patches first. If any
of these change, it recompiles the affected files instead of reusing the old
ones.
