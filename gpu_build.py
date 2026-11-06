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

"""Native build support; importing this module never compiles dependencies."""

import ctypes
import glob
import hashlib
import json
import io
import importlib.machinery
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile

import setuptools

ROOT = Path(__file__).resolve().parent


def build_parallelism():
    """Bound default build concurrency; accept a positive explicit override."""
    value = os.environ.get("FQ_BUILD_PARALLEL", "1")
    try:
        workers = int(value)
    except ValueError as exc:
        raise ValueError("FQ_BUILD_PARALLEL must be a positive integer") from exc
    if workers < 1:
        raise ValueError("FQ_BUILD_PARALLEL must be a positive integer")
    return workers


def env_flag(name):
    value = os.environ.get(name, "0").strip().lower()
    if value not in ("0", "false", "no", "off", "", "1", "true", "yes", "on"):
        raise ValueError(f"{name} must be 0 or 1 (or false/true)")
    return value in ("1", "true", "yes", "on")


def gpu_enabled():
    return env_flag("FQ_BUILD_GPU")


def primme_mpi_enabled():
    """GPU includes distributed PRIMME; retain the legacy CPU-PRIMME opt-in."""
    enabled = gpu_enabled()
    if (
        enabled
        and "FQ_ENABLE_PRIMME_MPI" in os.environ
        and not env_flag("FQ_ENABLE_PRIMME_MPI")
    ):
        raise ValueError(
            "FQ_BUILD_GPU=1 includes MPI PRIMME; remove FQ_ENABLE_PRIMME_MPI=0"
        )
    distributed = enabled or env_flag("FQ_ENABLE_PRIMME_MPI")
    if (
        distributed
        and "FQ_PRIMME_BLASINT64" in os.environ
        and not env_flag("FQ_PRIMME_BLASINT64")
    ):
        raise ValueError(
            "Distributed PRIMME requires ILP64; remove FQ_PRIMME_BLASINT64=0"
        )
    if enabled and env_flag("FQ_DISABLE_DIST_BUILD"):
        raise ValueError(
            "FQ_BUILD_GPU=1 includes distributed kernels; remove FQ_DISABLE_DIST_BUILD=1"
        )
    return distributed


def _detect_cuda_archs():
    """Return nvcc -gencode flags for every GPU found via nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    caps = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            caps.add(line)

    if not caps:
        return None

    flags = []
    for cap in sorted(caps):
        sm = "sm_" + cap.replace(".", "")
        compute = "compute_" + cap.replace(".", "")
        flags += ["-gencode", f"arch={compute},code={sm}"]

    highest_compute = "compute_" + sorted(
        caps, key=lambda cap: tuple(map(int, cap.split(".")))
    )[-1].replace(".", "")
    flags += ["-gencode", f"arch={highest_compute},code={highest_compute}"]

    return flags


FALLBACK_ARCHS = ("sm_80", "sm_86", "sm_89", "sm_90", "sm_100")


def _supported_cuda_archs():
    result = subprocess.run(
        [_nvcc(), "--list-gpu-code"], capture_output=True, text=True, check=True
    )
    return set(re.findall(r"sm_[0-9]+", result.stdout))


def _fallback_arch_flags():
    """Only request architectures supported by the selected toolkit."""
    supported = _supported_cuda_archs()
    archs = [arch for arch in FALLBACK_ARCHS if arch in supported]
    if not archs:
        raise RuntimeError(
            "No supported fallback GPU target; set FQ_CUDA_ARCH for the compute nodes"
        )
    flags = []
    for arch in archs:
        flags += ["-gencode", f"arch=compute_{arch[3:]},code={arch}"]
    flags += ["-gencode", f"arch=compute_{archs[-1][3:]},code=compute_{archs[-1][3:]}"]
    return flags


def _nvcc():
    root = _detect_cuda_root()
    candidate = Path(root) / "bin/nvcc" if root else None
    if candidate and candidate.is_file():
        return str(candidate)
    raise RuntimeError(
        "CUDA compiler nvcc not found; install the CUDA toolkit or set CUDA_HOME. See GPU_INSTALL.md"
    )


def _build_cuda_library(output_dir, library_name, sources, label, *, verbose=False):
    out = str(Path(output_dir) / library_name)
    cmd = [
        _nvcc(),
        "-shared",
        "-O3",
        f"--threads={build_parallelism()}",
        "-Xcompiler",
        "-fPIC",
        "-lineinfo",
        "-DNDEBUG",
        *_nvcc_host_compiler_flags(),
        "-std=c++17",
        "--expt-relaxed-constexpr",
        *(
            ["-Xptxas", "-v", "-Xptxas", "-warn-spills"]
            if verbose
            else ["-Xptxas=-warn-spills"]
        ),
        *_nvcc_arch_flags(),
        "-o",
        out,
        *[str(ROOT / source) for source in sources],
    ]
    print(f"Building {label}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label} build failed:\n{result.stderr}")
    if not verbose and result.stderr.strip():
        print(result.stderr)
    print(f"Built {out}")
    return out, result.stdout + result.stderr


def _build_dist_cuda_lib(output_dir):
    out, compiler_output = _build_cuda_library(
        output_dir,
        "libdist_matvec_cuda.so",
        KERNEL_SOURCES,
        "distributed CUDA library",
        verbose=True,
    )
    _write_kernel_resources(out, compiler_output)


HOT_KERNEL_NAMES = (
    "cross_clique",
    "cross_lbg",
    "cross_rsl",
    "beta_doubles_class",
    "alpha_doubles_class",
)

_PTXAS_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '(sm_[0-9]+[a-z]?)'")
_PTXAS_FRAME = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
)
_PTXAS_REGS = re.compile(r"Used (\d+) registers")
_RESOURCE_FUNCTION = re.compile(r"^\s*Function ([^:\s]+):\s*$")
_RESOURCE_USAGE = re.compile(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")
_RESOURCE_ARCH = re.compile(r"arch = (sm_[0-9]+[a-z]?)")


def parse_ptxas_output(text):
    """Return the ptxas facts of each kernel instance from the verbose compiler output."""
    facts = {}
    key = None
    for line in text.splitlines():
        entry = _PTXAS_ENTRY.search(line)
        if entry:
            key = (entry.group(1), entry.group(2))
            facts.setdefault(key, {})
            continue
        if key is None:
            continue
        frame = _PTXAS_FRAME.search(line)
        if frame:
            stack, stores, loads = (int(value) for value in frame.groups())
            facts[key].update(
                stack_bytes=stack, spill_store_bytes=stores, spill_load_bytes=loads
            )
            continue
        regs = _PTXAS_REGS.search(line)
        if regs:
            facts[key]["registers"] = int(regs.group(1))
            key = None
    return facts


def parse_resource_usage(text):
    """Return the resource usage of each kernel instance from ``cuobjdump --dump-resource-usage``."""
    usage = {}
    arch = None
    name = None
    for line in text.splitlines():
        found_arch = _RESOURCE_ARCH.search(line)
        if found_arch:
            arch = found_arch.group(1)
        function = _RESOURCE_FUNCTION.match(line)
        if function:
            name = function.group(1)
            continue
        found = _RESOURCE_USAGE.search(line)
        if found and name is not None and arch is not None:
            regs, stack, shared, local = (int(value) for value in found.groups())
            usage[(name, arch)] = {
                "registers": regs,
                "stack_bytes": stack,
                "static_shared_bytes": shared,
                "local_bytes": local,
            }
            name = None
    return usage


def kernel_resources(library, compiler_output, resource_output):
    """Return the kernel resource record of a built library."""
    ptxas = parse_ptxas_output(compiler_output)
    usage = parse_resource_usage(resource_output)
    kernels = []
    for key in sorted(set(usage) | set(ptxas)):
        name, arch = key
        entry = {"name": name, "arch": arch}
        entry.update(usage.get(key, {}))
        compiled = ptxas.get(key, {})
        entry.setdefault("registers", compiled.get("registers"))
        entry.setdefault("stack_bytes", compiled.get("stack_bytes"))
        entry["spill_store_bytes"] = compiled.get("spill_store_bytes")
        entry["spill_load_bytes"] = compiled.get("spill_load_bytes")
        kernels.append(entry)
    digest = hashlib.sha256(Path(library).read_bytes()).hexdigest()
    archs = sorted({entry["arch"] for entry in kernels})
    return {
        "library": Path(library).name,
        "sha256": digest,
        "archs": archs,
        "kernels": kernels,
    }


def _write_kernel_resources(library, compiler_output):
    """Write kernel_resources.json next to a built library and print the spills of hot kernels."""
    cuobjdump = Path(_nvcc()).with_name("cuobjdump")
    resource_output = ""
    if cuobjdump.is_file():
        found = subprocess.run(
            [str(cuobjdump), "--dump-resource-usage", str(library)],
            capture_output=True,
            text=True,
        )
        if found.returncode == 0:
            resource_output = found.stdout
    record = kernel_resources(library, compiler_output, resource_output)
    target = Path(library).with_name("kernel_resources.json")
    target.write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"Wrote {target}: {len(record['kernels'])} kernel instances, archs {record['archs']}"
    )
    for entry in record["kernels"]:
        spilled = (entry["spill_store_bytes"] or 0) + (entry["spill_load_bytes"] or 0)
        if spilled and any(hot in entry["name"] for hot in HOT_KERNEL_NAMES):
            print(
                f"WARNING: hot kernel spills {spilled} bytes on {entry['arch']}: {entry['name']}"
            )


KERNEL_SOURCES = [
    os.path.join("fulqrum", "gpu", "kernels", name)
    for name in (
        "diagonal.cu",
        "alpha_singles.cu",
        "beta_singles.cu",
        "alpha_doubles.cu",
        "beta_doubles.cu",
        "beta_doubles_class.cu",
        "cross_sell.cu",
        "cross_direct.cu",
        "cross_clique.cu",
        "cross_lbg.cu",
        "cache_build.cu",
        "param_sizes.cu",
    )
]

FULL_STRS_KERNEL_SOURCES = [
    os.path.join("fulqrum", "gpu", "kernels", "full_strs", name)
    for name in (
        "diagonal.cu",
        "groups.cu",
        "reduce.cu",
        "param_sizes.cu",
        "rdm_groups.cu",
    )
]

RDM_KERNEL_SOURCES = [
    os.path.join("fulqrum", "gpu", "kernels", "rdm", name)
    for name in ("same_spin.cu", "one_body.cu", "alpha_beta.cu", "param_sizes.cu")
]


def _nvcc_arch_flags():
    explicit = os.environ.get("FQ_CUDA_ARCH")
    if explicit:
        if not re.fullmatch(r"sm_[0-9]+[a-z]?", explicit):
            raise ValueError("FQ_CUDA_ARCH must be an architecture such as sm_90")
        result = subprocess.run(
            [_nvcc(), "--list-gpu-code"], capture_output=True, text=True, check=True
        )
        if explicit not in result.stdout.split():
            raise RuntimeError(f"The CUDA toolkit does not support {explicit}")
        return [f"-arch={explicit}"]
    detected = _detect_cuda_archs()
    if detected:
        supported = _supported_cuda_archs()
        requested = set(re.findall(r"code=(sm_[0-9]+)", " ".join(detected)))
        if requested <= supported:
            return detected
    return _fallback_arch_flags()


def _nvcc_host_compiler_flags():
    # Respect explicit choices; nvcc selects its default compiler otherwise.
    compiler = os.environ.get("FQ_CUDA_HOST_COMPILER") or os.environ.get("CXX")
    return ["-ccbin", compiler] if compiler else []


def _build_kernel_test_lib(output_dir):
    """Compile the GPU kernel-library test kernels into libfq_kernel_tests.so."""
    root = str(ROOT / "fulqrum/gpu/kernels/tests")
    out = str(Path(output_dir) / "libfq_kernel_tests.so")
    cmd = [
        _nvcc(),
        "-shared",
        "-O3",
        "-Xcompiler",
        "-fPIC",
        "-lineinfo",
        "-DNDEBUG",
        *_nvcc_host_compiler_flags(),
        "-std=c++17",
        "--expt-relaxed-constexpr",
        *_nvcc_arch_flags(),
        "-o",
        out,
        os.path.join(root, "primitives_test.cu"),
    ]
    print(f"Building GPU kernel test library: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GPU kernel test library build failed:\n{result.stderr}")
    print(f"Built {out}")


def _build_full_strs_cuda_lib(output_dir):
    _build_cuda_library(
        output_dir,
        "libfull_strs_cuda.so",
        FULL_STRS_KERNEL_SOURCES,
        "full-string CUDA library",
    )


def _build_rdm_cuda_lib(output_dir):
    _build_cuda_library(
        output_dir, "librdm_cuda.so", RDM_KERNEL_SOURCES, "RDM CUDA library"
    )


def _detect_cuda_root():
    """Return the CUDA toolkit root for PRIMME's GPU (cuBLAS) backend, or None."""
    for var in ("CUDADIR", "CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root and os.path.isdir(os.path.join(root, "include")):
            return root
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    return None


def _detect_cuda_math_root(cuda_root):
    """Return the NVIDIA HPC SDK ``math_libs`` root matching ``cuda_root``, or None."""
    explicit = os.environ.get("FQ_CUDA_MATH_DIR")
    if explicit:
        return explicit
    if not cuda_root:
        return None
    # cuda_root like .../26.3/cuda/12.9  -> .../26.3/math_libs/12.9
    parent = os.path.dirname(cuda_root)  # .../cuda
    ver = os.path.basename(cuda_root)  # 12.9
    if os.path.basename(parent) == "cuda":
        cand = os.path.join(os.path.dirname(parent), "math_libs", ver)
        if os.path.isdir(cand):
            return cand
    # cuda_root is the unversioned symlink dir .../26.3/cuda -> sibling math_libs
    if os.path.basename(cuda_root) == "cuda":
        cand = os.path.join(parent, "math_libs")
        if os.path.isdir(cand):
            return cand
    return None


def _detect_cuda_link(cuda_root):
    """Locate cuBLAS/cuSPARSE/cudart for linking _primme_mpi."""
    import glob  # noqa: PLC0415

    names = ("cusparse", "cublas", "cudart")

    explicit = os.environ.get("FQ_CUDA_LIBDIR")
    if explicit:
        candidates = [d for d in explicit.split(os.pathsep) if d]
    else:
        candidates = []
        if cuda_root:
            candidates += [
                os.path.join(cuda_root, "lib64"),
                os.path.join(cuda_root, "lib"),
                os.path.join(cuda_root, "targets", "x86_64-linux", "lib"),
            ]
            # HPC SDK: cuBLAS/cuSPARSE live in the sibling math_libs tree.
            math_root = _detect_cuda_math_root(cuda_root)
            if math_root:
                candidates += [
                    os.path.join(math_root, "lib64"),
                    os.path.join(math_root, "lib"),
                ]
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidates.append(os.path.join(conda_prefix, "lib"))
        candidates.append("/usr/lib/x86_64-linux-gnu")
        # pip nvidia-*-cu12 wheels next to this interpreter
        for mod in names:
            candidates += glob.glob(
                os.path.join(
                    os.path.dirname(os.__file__), "site-packages", "nvidia", mod, "lib"
                )
            )
        # Last resort: use ldconfig to find where libcublas lives.
        try:
            out = subprocess.run(
                ["ldconfig", "-p"], capture_output=True, text=True
            ).stdout
            for line in out.splitlines():
                if "libcublas.so" in line and "=>" in line:
                    candidates.append(os.path.dirname(line.split("=>")[1].strip()))
        except OSError:
            pass

    # Resolve each library independently; a given lib may live in a different
    # candidate dir than the others (HPC SDK splits cudart from cuBLAS/cuSPARSE).
    library_dirs, libraries, extra = [], [], []
    for name in names:
        for libdir in candidates:
            unversioned = os.path.join(libdir, f"lib{name}.so")
            if os.path.exists(unversioned):
                libraries.append(name)
                if libdir not in library_dirs:
                    library_dirs.append(libdir)
                break
            versioned = sorted(glob.glob(unversioned + ".*"))
            if versioned:
                extra.append(versioned[-1])  # link the versioned .so directly
                break
        else:
            raise RuntimeError(
                f"FQ_BUILD_GPU=1 with the PRIMME GPU backend, but lib{name} was "
                "not found. Locate cuBLAS/cuSPARSE/cudart "
                "(`ldconfig -p | grep cublas`) and set FQ_CUDA_LIBDIR=<dir> "
                "(os.pathsep-separated for multiple dirs), or set CUDADIR to the "
                "CUDA toolkit root. For the NVIDIA HPC SDK, point CUDADIR at "
                ".../<ver>/cuda/<cudaver> and FQ_CUDA_MATH_DIR at the sibling "
                ".../<ver>/math_libs/<cudaver>."
            )
    return library_dirs, libraries, extra


BLAS_SYMBOLS = (
    "dgemm_64_",
    "dscal_64_",
    "dgemv_64_",
    "dsyevx_64_",
    "dgesvd_64_",
    "zgemm_64_",
    "zheevx_64_",
)


def _validate_blas(path):
    """Require the supported OpenBLAS integer ABI and BLAS/LAPACK exports."""
    try:
        library = ctypes.CDLL(str(path))
        missing = [name for name in BLAS_SYMBOLS if not hasattr(library, name)]
        if missing:
            raise RuntimeError(
                "missing ILP64 BLAS/LAPACK symbols: " + ", ".join(missing)
            )
        config = library.openblas_get_config64_
        config.restype = ctypes.c_char_p
        description = config().decode()
        if "USE64BITINT" not in description:
            raise RuntimeError("OpenBLAS does not declare USE64BITINT")
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f"Cannot load compatible OpenBLAS ILP64: {exc}") from exc
    return library


def _detect_blas_link():
    """Resolve a validated ILP64 provider without an LP64 fallback."""
    explicit = os.environ.get("FQ_BLAS_LIB")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"FQ_BLAS_LIB={explicit} does not exist")
        _validate_blas(path)
        return [], [], [str(path)]
    prefixes = [
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
        str(Path.home() / ".local/opt/openblas-ilp64"),
    ]
    dirs = [
        str(Path(prefix) / libdir)
        for prefix in prefixes
        if prefix
        for libdir in ("lib", "lib64")
    ]
    dirs += [
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
    ]
    dirs += ["/opt/homebrew/opt/openblas-ilp64/lib", "/opt/homebrew/lib"]
    dirs += [
        directory
        for directory in (
            os.environ.get("LD_LIBRARY_PATH", "")
            + os.pathsep
            + os.environ.get("DYLD_LIBRARY_PATH", "")
        ).split(os.pathsep)
        if directory
    ]
    seen, rejected = set(), []
    for directory in dict.fromkeys(dirs):
        for candidate in sorted(
            glob.glob(str(Path(directory) / "libopenblas*.so*"))
            + glob.glob(str(Path(directory) / "libopenblas*.dylib")),
            key=lambda name: ("64" not in Path(name).name, len(name)),
        ):
            path = Path(candidate).resolve()
            if path in seen:
                continue
            seen.add(path)
            try:
                _validate_blas(path)
            except RuntimeError as exc:
                rejected.append(f"{path}: {exc}")
                continue
            return [], [], [str(path)]
    detail = "\n" + "\n".join(rejected) if rejected else ""
    raise RuntimeError(
        "Compatible OpenBLAS ILP64 not found. Install conda-forge openblas-ilp64, "
        "or follow GPU_INSTALL.md to build OpenBLAS with INTERFACE64=1 and SYMBOLSUFFIX=64_. "
        "For a custom prefix, set FQ_BLAS_LIB to libopenblas64_.so." + detail
    )


PRIMME_REVISION = "aa98cd27c13b2dbd41daa25d574c8837b0779a11"


def copy_primme_sources(destination):
    """Export the pinned upstream source, preserving local submodule edits."""
    source = ROOT / "third-party/primme"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (source / ".git").exists():
        data = subprocess.check_output(
            ["git", "-C", str(source), "archive", PRIMME_REVISION]
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive.getmembers():
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or member.issym()
                    or member.islnk()
                ):
                    raise RuntimeError("Unsafe path in PRIMME source archive")
                target = destination / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
                    target.chmod(member.mode)
    else:
        # A release sdist already contains the pinned upstream export.
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", "*.o", "*.a", "*.so*", "__pycache__"),
        )


def _primme_build_key(source, cuda_root, blas_path):
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*")):
        if (
            path.is_file()
            and ".git" not in path.parts
            and path.suffix not in (".o", ".a", ".so")
            and "lib" not in path.relative_to(source).parts
        ):
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    for path in sorted((ROOT / "patches/primme").glob("*.patch")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    settings = {
        name: os.environ.get(name, "")
        for name in ("CC", "CFLAGS", "CPPFLAGS", "LDFLAGS", "CPATH", "FQ_CUDA_MATH_DIR")
    }
    settings.update(
        cuda_root=cuda_root, blas=blas_path, ilp64=True, revision=PRIMME_REVISION
    )
    if sys.platform == "darwin":
        settings["MACOSX_DEPLOYMENT_TARGET"] = os.environ.get(
            "MACOSX_DEPLOYMENT_TARGET"
        ) or sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")
    blas_stat = Path(blas_path).stat()
    settings["blas_identity"] = [blas_stat.st_size, blas_stat.st_mtime_ns]
    compiler = shlex.split(os.environ.get("CC", "cc"))
    settings["compiler_version"] = subprocess.check_output(
        compiler + ["--version"], text=True
    )
    if cuda_root:
        settings["cuda_version"] = subprocess.check_output(
            [_nvcc(), "--version"], text=True
        )
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def _apply_primme_patches(destination):
    for patch in sorted((ROOT / "patches/primme").glob("*.patch")):
        # git is a GPU build prerequisite.
        common = ["git", "apply", "--unsafe-paths", "--directory=" + str(destination)]
        check = subprocess.run(
            common + ["--reverse", "--check", str(patch)],
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            continue
        result = subprocess.run(
            common + ["--check", str(patch)], capture_output=True, text=True
        )
        if result.returncode:
            raise RuntimeError(
                f"PRIMME patch {patch.name} does not apply; use the pinned submodule sources.\n{result.stderr}"
            )
        subprocess.check_call(common + [str(patch)])


def _resolve_primme_dir(build_temp, cuda_root, blas_path):
    source = ROOT / "third-party/primme"
    if not (source / "include/primme.h").is_file():
        raise RuntimeError(
            "PRIMME sources missing. Run git submodule update --init --recursive third-party/primme; see GPU_INSTALL.md"
        )
    key = _primme_build_key(source, cuda_root, blas_path)
    destination = Path(build_temp).resolve() / "primme" / key
    archive = destination / "lib/libprimme.a"
    marker = destination / ".fulqrum-complete"
    if marker.is_file() and archive.is_file():
        return destination
    copy_primme_sources(destination)
    _apply_primme_patches(destination)
    env = os.environ.copy()
    if sys.platform == "darwin" and not env.get("MACOSX_DEPLOYMENT_TARGET"):
        target = sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")
        if target:
            env["MACOSX_DEPLOYMENT_TARGET"] = str(target)
    # Correctness settings are mandatory, including when callers provide CFLAGS.
    flags = env.get("CFLAGS", "-O3")
    if re.search(r"-DPRIMME_BLAS(INT_SIZE|_SUFFIX)", flags):
        raise ValueError(
            "PRIMME's ILP64 ABI is configured by Fulqrum; remove PRIMME BLAS definitions from CFLAGS"
        )
    env["CFLAGS"] = (
        flags + " -fPIC -DNDEBUG -DPRIMME_BLASINT_SIZE=64 -DPRIMME_BLAS_SUFFIX=_64"
    )
    math_root = _detect_cuda_math_root(cuda_root)
    if math_root:
        env["CPATH"] = (
            str(Path(math_root) / "include") + os.pathsep + env.get("CPATH", "")
        )
    command = ["make", "lib", f"-j{build_parallelism()}", f"PYTHON={sys.executable}"]
    if cuda_root:
        command += [
            f"CUDADIR={cuda_root}",
            "PRIMME_WITH_CUDA=yes",
            "PRIMME_WITH_MAGMA=no",
            "PRIMME_WITH_HIPBLAS=no",
        ]
    else:
        command += [
            "PRIMME_WITH_CUDA=no",
            "PRIMME_WITH_MAGMA=no",
            "PRIMME_WITH_HIPBLAS=no",
        ]
    print(
        f"[fulqrum] Building PRIMME {PRIMME_REVISION[:12]} + Fulqrum patches (ILP64) in {destination}",
        flush=True,
    )
    subprocess.check_call(command, cwd=destination, env=env)
    marker.write_text(key + "\n")
    return destination


def _validate_primme_archive(archive, blas_path, gpu):
    """Link the Fulqrum-built host/GPU solvers against validated ILP64 OpenBLAS."""
    _validate_blas(blas_path)
    archive = Path(archive).resolve()
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = shlex.split(os.environ.get("CFLAGS", ""))
    link_flags = shlex.split(os.environ.get("LDFLAGS", ""))
    source = """#include <primme.h>
_Static_assert(sizeof(PRIMME_INT) == 8, "PRIMME requires 64-bit integers");
int main(void) {
    int status = dprimme(0, 0, 0, 0);
    status += zprimme(0, 0, 0, 0);
"""
    if gpu:
        source += "    status += cublas_dprimme(0, 0, 0, 0);\n"
        source += "    status += cublas_zprimme(0, 0, 0, 0);\n"
    source += "    return status;\n}\n"
    if sys.platform == "darwin":
        for patch in sorted((ROOT / "patches/primme").glob("*.patch")):
            added = "\n".join(
                line[1:]
                for line in patch.read_text().splitlines()
                if line.startswith("+")
            )
            for declaration in re.findall(r"extern\s+[^;]+?fq_\w+\s*\([^;]+;", added):
                if "__attribute__((weak))" in declaration:
                    name = re.search(r"fq_\w+", declaration).group()
                    link_flags.append(f"-Wl,-U,_{name}")
    cuda_flags = []
    if gpu:
        dirs, libraries, extra = _detect_cuda_link(_detect_cuda_root())
        cuda_flags = [f"-L{directory}" for directory in dirs]
        cuda_flags += [f"-l{library}" for library in libraries] + extra
    with tempfile.TemporaryDirectory(
        prefix="link-check-", dir=archive.parent
    ) as temporary:
        path = Path(temporary)
        probe = path / "primme_link.c"
        probe.write_text(source)
        result = subprocess.run(
            compiler
            + flags
            + [
                "-std=c11",
                "-DPRIMME_INT_SIZE=64",
                "-I" + str(archive.parent.parent / "include"),
                str(probe),
                str(archive),
                str(blas_path),
            ]
            + cuda_flags
            + link_flags
            + ["-lm", "-o", str(path / "primme_link")],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError("PRIMME solver link check failed:\n" + result.stderr)


def prepare_native_build(command):
    """Prepare dependencies, then compile into the setuptools build tree."""
    gpu = gpu_enabled()
    primme_mpi_enabled()  # Validate legacy flags before any writes.
    if gpu and sys.platform != "linux":
        raise RuntimeError(
            "Fulqrum's GPU build currently supports Linux; see GPU_INSTALL.md"
        )
    blas_path = _detect_blas_link()[2][0]
    cuda_root = _detect_cuda_root() if gpu else None
    if gpu:
        _nvcc()
        cuda_dirs, cuda_libs, cuda_extra = _detect_cuda_link(cuda_root)
        host_flags = _nvcc_host_compiler_flags()
        # nvcc validates compiler compatibility without compiling the kernels yet.
        probe = subprocess.run(
            [_nvcc(), *host_flags, "-x", "cu", "-E", "-"],
            input="",
            text=True,
            capture_output=True,
        )
        if probe.returncode:
            raise RuntimeError(
                "CUDA host compiler preflight failed. Select a CUDA-compatible compiler with FQ_CUDA_HOST_COMPILER.\n"
                + probe.stderr
            )
        targets = _nvcc_arch_flags()
        print(f"[fulqrum] CUDA: {cuda_root}; targets: {' '.join(targets)}", flush=True)
    for tool in ("make", "git"):
        if not shutil.which(tool):
            raise RuntimeError(
                f"{tool} is required to build patched PRIMME; see GPU_INSTALL.md"
            )
    print(
        f"[fulqrum] OpenBLAS ILP64: {blas_path}; build jobs: {build_parallelism()}",
        flush=True,
    )
    directory = _resolve_primme_dir(command.build_temp, cuda_root, blas_path)
    archive = directory / "lib/libprimme.a"
    _validate_primme_archive(archive, blas_path, gpu)
    for extension in command.extensions:
        if extension.name.endswith("._primme_mpi"):
            extension.include_dirs += [str(directory / "include")]
            extension.extra_objects = [str(archive)]
            extension.extra_link_args += [blas_path]
            extension.runtime_library_dirs = [str(Path(blas_path).parent)]
            if gpu:
                extension.define_macros.append(("FQ_PRIMME_GPU", "1"))
                extension.include_dirs.append(str(Path(cuda_root) / "include"))
                math_root = _detect_cuda_math_root(cuda_root)
                if math_root:
                    extension.include_dirs.append(str(Path(math_root) / "include"))
                extension.library_dirs += cuda_dirs
                extension.libraries += cuda_libs
                extension.extra_link_args += cuda_extra
                extension.runtime_library_dirs += cuda_dirs + [
                    str(Path(path).parent) for path in cuda_extra
                ]
    if gpu:
        output = Path(command.build_temp).resolve() / "cuda"
        output.mkdir(parents=True, exist_ok=True)
        _build_dist_cuda_lib(output)
        _build_full_strs_cuda_lib(output)
        _build_rdm_cuda_lib(output)
        copies = [
            ("distributed", "libdist_matvec_cuda.so"),
            ("full_strs", "libfull_strs_cuda.so"),
            ("rdm", "librdm_cuda.so"),
        ]
        if env_flag("FQ_BUILD_GPU_TESTS"):
            _build_kernel_test_lib(output)
            copies.append(("kernels/tests", "libfq_kernel_tests.so"))
        copies.append(("distributed", "kernel_resources.json"))
        for subdir, name in copies:
            targets = [Path(command.build_lib) / "fulqrum/gpu" / subdir]
            if command.inplace:
                targets.append(ROOT / "fulqrum/gpu" / subdir)
            for target in targets:
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output / name, target / name)
        for extension in command.extensions:
            if extension.name.endswith(".kernel_launch"):
                extension.library_dirs = [str(output)]


def prune_inplace_native(extensions):
    enabled = {extension.name for extension in extensions}
    for source in (ROOT / "fulqrum/gpu").rglob("*.pyx"):
        module = source.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")
        if module not in enabled:
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                source.with_name(source.stem + suffix).unlink(missing_ok=True)
    if not gpu_enabled():
        for relative in (
            "distributed/libdist_matvec_cuda.so",
            "distributed/kernel_resources.json",
            "full_strs/libfull_strs_cuda.so",
            "rdm/librdm_cuda.so",
            "kernels/tests/libfq_kernel_tests.so",
        ):
            (ROOT / "fulqrum/gpu" / relative).unlink(missing_ok=True)


def gpu_extensions(root, include_dirs, compile_args, link_args, define_macros):
    """Return the Cython extensions of the GPU module."""
    exts = []

    def _ext(module, source, **kw):
        kw.setdefault(
            "include_dirs",
            list(include_dirs)
            + [
                str(ROOT / Path(source).parent),
                str(ROOT / Path(source).parent / "src"),
                str(ROOT / "fulqrum/core/src"),
            ],
        )
        kw.setdefault("define_macros", list(define_macros))
        extra_link = kw.pop("extra_link_args_ext", [])
        return setuptools.Extension(
            module,
            sources=[source] + kw.pop("extra_sources", []),
            extra_compile_args=list(compile_args),
            extra_link_args=list(link_args) + extra_link,
            language="c++",
            **kw,
        )

    exts.append(
        _ext(
            f"{root}.gpu.distributed.operator_check",
            f"{root}/gpu/distributed/operator_check.pyx",
        )
    )
    exts.append(
        _ext(
            f"{root}.gpu.distributed.tile_tables",
            f"{root}/gpu/distributed/tile_tables.pyx",
        )
    )
    exts.append(
        _ext(
            f"{root}.gpu.full_strs.host",
            f"{root}/gpu/full_strs/host.pyx",
        )
    )
    exts.append(
        _ext(
            f"{root}.gpu.common.operator_arrays",
            f"{root}/gpu/common/operator_arrays.pyx",
        )
    )
    if gpu_enabled():
        exts.append(
            _ext(
                f"{root}.gpu.distributed.tile_build",
                f"{root}/gpu/distributed/tile_build.pyx",
            )
        )
        exts.append(
            _ext(
                f"{root}.gpu.distributed.kernel_launch",
                f"{root}/gpu/distributed/kernel_launch.pyx",
                library_dirs=[f"{root}/gpu/distributed"],
                libraries=[":libdist_matvec_cuda.so"],
                extra_link_args_ext=["-Wl,-rpath,$ORIGIN"],
            )
        )
        exts.append(
            _ext(
                f"{root}.gpu.full_strs.kernel_launch",
                f"{root}/gpu/full_strs/kernel_launch.pyx",
                library_dirs=[f"{root}/gpu/full_strs"],
                libraries=[":libfull_strs_cuda.so"],
                extra_link_args_ext=["-Wl,-rpath,$ORIGIN"],
            )
        )
        exts.append(
            _ext(
                f"{root}.gpu.rdm.kernel_launch",
                f"{root}/gpu/rdm/kernel_launch.pyx",
                library_dirs=[f"{root}/gpu/rdm"],
                libraries=[":librdm_cuda.so"],
                extra_link_args_ext=["-Wl,-rpath,$ORIGIN"],
            )
        )

    if primme_mpi_enabled():
        exts.append(_primme_mpi_extension(root, _ext))
    return exts


def _primme_mpi_extension(root, make_ext):
    # Declare sources now; dependency discovery and linkage are deferred to build_ext.
    sources = [
        f"{root}/gpu/eigensolver/host_gemv.cpp",
        f"{root}/gpu/eigensolver/host_dense.cpp",
    ]
    if gpu_enabled():
        sources += [
            f"{root}/gpu/eigensolver/dense_gpu_offload.cpp",
            f"{root}/gpu/eigensolver/managed_basis.cpp",
            f"{root}/gpu/eigensolver/host_pinned_basis.cpp",
            f"{root}/gpu/eigensolver/host_spec.cpp",
        ]
    return make_ext(
        f"{root}.gpu.eigensolver._primme_mpi",
        f"{root}/gpu/eigensolver/_primme_mpi.pyx",
        extra_sources=sources,
        libraries=[],
        library_dirs=[],
        extra_objects=[],
    )
