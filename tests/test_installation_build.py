"""Packaging checks without importing Fulqrum or requiring GPU hardware."""

import ctypes
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fulqrum_gpu_build", ROOT / "gpu_build.py"
)
gpu_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_build)


@pytest.fixture(autouse=True)
def clear_build_flags(monkeypatch):
    for name in (
        "FQ_BUILD_GPU",
        "FQ_ENABLE_PRIMME_MPI",
        "FQ_PRIMME_BLASINT64",
        "FQ_DISABLE_DIST_BUILD",
        "FQ_BUILD_PARALLEL",
        "FQ_BLAS_LIB",
    ):
        monkeypatch.delenv(name, raising=False)


def test_cpu_extensions_need_no_native_toolchain(monkeypatch):
    monkeypatch.setattr(
        gpu_build.subprocess,
        "run",
        Mock(side_effect=AssertionError("unexpected subprocess")),
    )
    assert not gpu_build.primme_mpi_enabled()
    extensions = gpu_build.gpu_extensions("fulqrum", [], [], [], [])
    assert all(not ext.name.endswith("_primme_mpi") for ext in extensions)


def test_one_gpu_switch_declares_distributed_solver_without_building(monkeypatch):
    monkeypatch.setenv("FQ_BUILD_GPU", "1")
    monkeypatch.setattr(
        gpu_build.subprocess,
        "run",
        Mock(side_effect=AssertionError("metadata compiled dependencies")),
    )
    assert gpu_build.primme_mpi_enabled()
    extensions = gpu_build.gpu_extensions("fulqrum", [], [], [], [])
    assert any(ext.name.endswith("_primme_mpi") for ext in extensions)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("FQ_ENABLE_PRIMME_MPI", "0"),
        ("FQ_PRIMME_BLASINT64", "0"),
        ("FQ_DISABLE_DIST_BUILD", "1"),
    ],
)
def test_incomplete_gpu_configuration_is_rejected(monkeypatch, flag, value):
    monkeypatch.setenv("FQ_BUILD_GPU", "1")
    monkeypatch.setenv(flag, value)
    with pytest.raises(ValueError, match=flag):
        gpu_build.primme_mpi_enabled()


def test_legacy_cpu_distributed_solver_still_enabled(monkeypatch):
    monkeypatch.setenv("FQ_ENABLE_PRIMME_MPI", "1")
    assert gpu_build.primme_mpi_enabled()
    assert not gpu_build.gpu_enabled()


def test_missing_ilp64_fails_before_compilation(monkeypatch):
    monkeypatch.setattr(gpu_build.sys, "platform", "linux")
    monkeypatch.setenv("FQ_BUILD_GPU", "1")
    monkeypatch.setattr(
        gpu_build, "_detect_blas_link", Mock(side_effect=RuntimeError("ILP64 missing"))
    )
    compile_mock = Mock()
    monkeypatch.setattr(gpu_build, "_resolve_primme_dir", compile_mock)
    with pytest.raises(RuntimeError, match="ILP64 missing"):
        gpu_build.prepare_native_build(SimpleNamespace())
    compile_mock.assert_not_called()


def test_openblas_integer_width_is_checked(monkeypatch):
    library = SimpleNamespace(**{name: Mock() for name in gpu_build.BLAS_SYMBOLS})
    library.openblas_get_config64_ = Mock(return_value=b"OpenBLAS USE64BITINT")
    monkeypatch.setattr(ctypes, "CDLL", Mock(return_value=library))
    gpu_build._validate_blas("example.so")
    library.openblas_get_config64_.return_value = b"OpenBLAS"
    with pytest.raises(RuntimeError, match="USE64BITINT"):
        gpu_build._validate_blas("example.so")


def test_lp64_and_blas_only_libraries_are_rejected(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", Mock(return_value=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="missing ILP64 BLAS/LAPACK"):
        gpu_build._validate_blas("example.so")


def test_conda_ilp64_is_selected_before_system_lp64(monkeypatch, tmp_path):
    libdir = tmp_path / "lib"
    libdir.mkdir()
    candidate = libdir / "libopenblas64_.so"
    candidate.touch()
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    validate = Mock()
    monkeypatch.setattr(gpu_build, "_validate_blas", validate)
    assert gpu_build._detect_blas_link()[2] == [str(candidate)]
    validate.assert_called_once_with(candidate)


def test_toolkit_fallback_filters_unsupported_architectures(monkeypatch):
    monkeypatch.setattr(gpu_build, "_supported_cuda_archs", lambda: {"sm_80", "sm_90"})
    flags = gpu_build._fallback_arch_flags()
    assert "arch=compute_90,code=sm_90" in flags
    assert all("100" not in flag for flag in flags)


def test_build_jobs_validate_resource_control(monkeypatch):
    assert gpu_build.build_parallelism() == 1
    monkeypatch.setenv("FQ_BUILD_PARALLEL", "0")
    with pytest.raises(ValueError, match="positive"):
        gpu_build.build_parallelism()


def test_primme_copy_keeps_source_and_uses_configured_jobs(monkeypatch, tmp_path):
    source = tmp_path / "third-party/primme"
    (source / "include").mkdir(parents=True)
    (source / "include/primme.h").write_text("original")
    monkeypatch.setattr(gpu_build, "ROOT", tmp_path)
    monkeypatch.setattr(gpu_build, "_primme_build_key", lambda *args: "example")
    monkeypatch.setattr(
        gpu_build,
        "_apply_primme_patches",
        lambda path: (path / "include/primme.h").write_text("patched"),
    )
    monkeypatch.setenv("FQ_BUILD_PARALLEL", "2")

    def make(command, cwd, env):
        assert "-j2" in command
        assert "-DPRIMME_BLASINT_SIZE=64" in env["CFLAGS"]
        assert "-DPRIMME_BLAS_SUFFIX=_64" in env["CFLAGS"]
        (cwd / "lib").mkdir()
        (cwd / "lib/libprimme.a").touch()

    monkeypatch.setattr(gpu_build.subprocess, "check_call", make)
    destination = gpu_build._resolve_primme_dir(tmp_path / "build", None, "blas.so")
    assert (source / "include/primme.h").read_text() == "original"
    assert (destination / "include/primme.h").read_text() == "patched"


def test_setup_metadata_has_no_native_build_side_effects(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "setup.py"), "--name"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "fulqrum"
    assert "Building" not in result.stdout


def test_macos_ilp64_library_discovery(monkeypatch, tmp_path):
    libdir = tmp_path / "lib"
    libdir.mkdir()
    candidate = libdir / "libopenblas64_.dylib"
    candidate.touch()
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    validate = Mock()
    monkeypatch.setattr(gpu_build, "_validate_blas", validate)
    assert gpu_build._detect_blas_link()[2] == [str(candidate)]
    validate.assert_called_once_with(candidate)


@pytest.mark.parametrize("gpu", [False, True])
def test_native_solver_link_probe(monkeypatch, tmp_path, gpu):
    monkeypatch.setattr(gpu_build, "_validate_blas", Mock())
    monkeypatch.setattr(gpu_build, "_detect_cuda_root", lambda: "/cuda")
    monkeypatch.setattr(
        gpu_build, "_detect_cuda_link", lambda _: (["/cuda/lib"], ["cublas"], [])
    )
    archive = tmp_path / "lib" / "libprimme.a"
    archive.parent.mkdir()
    sources = []

    def compile_probe(command, **kwargs):
        source = next(Path(arg) for arg in command if arg.endswith("primme_link.c"))
        sources.append(source.read_text())
        assert str(archive) in command
        assert "blas.so" in command
        assert ("-lcublas" in command) == gpu
        return SimpleNamespace(returncode=0, stderr="")

    run = Mock(side_effect=compile_probe)
    monkeypatch.setattr(gpu_build.subprocess, "run", run)
    gpu_build._validate_primme_archive(archive, "blas.so", gpu)
    run.assert_called_once()
    assert "dprimme(0" in sources[0] and "zprimme(0" in sources[0]
    assert ("cublas_dprimme(0" in sources[0]) == gpu
    assert not list(archive.parent.glob("link-check-*"))


def test_cpu_inplace_rebuild_removes_stale_native_modules(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_build, "ROOT", tmp_path)
    directory = tmp_path / "fulqrum/gpu/eigensolver"
    directory.mkdir(parents=True)
    (directory / "_primme_mpi.pyx").touch()
    binary = directory / (
        "_primme_mpi" + gpu_build.importlib.machinery.EXTENSION_SUFFIXES[0]
    )
    binary.write_bytes(b"old native build")
    library = tmp_path / "fulqrum/gpu/distributed/libdist_matvec_cuda.so"
    library.parent.mkdir()
    library.write_bytes(b"old CUDA build")
    gpu_build.prune_inplace_native([])
    assert not binary.exists()
    assert not library.exists()
    assert (directory / "_primme_mpi.pyx").is_file()


def test_gpu_registration_preserves_every_native_source(monkeypatch):
    monkeypatch.setenv("FQ_BUILD_GPU", "1")
    extensions = gpu_build.gpu_extensions("fulqrum", [], [], [], [])
    names = {extension.name for extension in extensions}
    expected = {
        "fulqrum.gpu.common.operator_arrays",
        "fulqrum.gpu.distributed.operator_check",
        "fulqrum.gpu.distributed.tile_tables",
        "fulqrum.gpu.distributed.tile_build",
        "fulqrum.gpu.distributed.kernel_launch",
        "fulqrum.gpu.full_strs.host",
        "fulqrum.gpu.full_strs.kernel_launch",
        "fulqrum.gpu.rdm.kernel_launch",
        "fulqrum.gpu.eigensolver._primme_mpi",
    }
    assert names == expected
    native = next(
        extension for extension in extensions if extension.name.endswith("._primme_mpi")
    )
    assert {
        Path(source).name for source in native.sources if source.endswith(".cpp")
    } == {
        "host_gemv.cpp",
        "host_dense.cpp",
        "dense_gpu_offload.cpp",
        "managed_basis.cpp",
        "host_pinned_basis.cpp",
        "host_spec.cpp",
    }


def test_native_solver_link_failure_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_build, "_validate_blas", Mock())
    monkeypatch.setattr(
        gpu_build.subprocess,
        "run",
        Mock(
            return_value=SimpleNamespace(returncode=1, stderr="undefined solver symbol")
        ),
    )
    with pytest.raises(RuntimeError, match="undefined solver symbol"):
        gpu_build._validate_primme_archive(tmp_path / "libprimme.a", "blas.so", False)
    assert not list(tmp_path.glob("link-check-*"))


def test_native_cache_tracks_the_macos_deployment_target(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "Make_flags").write_text("pristine")
    blas = tmp_path / "blas.dylib"
    blas.touch()
    monkeypatch.setattr(gpu_build.sys, "platform", "darwin")
    monkeypatch.setattr(gpu_build.sysconfig, "get_config_var", lambda _: "11.0")
    monkeypatch.setattr(
        gpu_build.subprocess, "check_output", lambda *args, **kwargs: "clang version"
    )
    monkeypatch.delenv("MACOSX_DEPLOYMENT_TARGET", raising=False)
    first = gpu_build._primme_build_key(source, None, str(blas))
    monkeypatch.setenv("MACOSX_DEPLOYMENT_TARGET", "12.0")
    assert gpu_build._primme_build_key(source, None, str(blas)) != first
