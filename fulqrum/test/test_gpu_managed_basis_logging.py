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


import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native_probe(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("A C++ compiler is required for native reporting checks")
    directory = tmp_path_factory.mktemp("managed_logging")
    (directory / "cuda_runtime.h").write_text(r"""
#pragma once
#include <cstddef>
#include <cstdlib>
#define CUDART_VERSION 13000
using cudaError_t = int;
using cudaMemoryAdvise = int;
constexpr int cudaSuccess = 0;
constexpr int cudaMemAdviseSetAccessedBy = 1;
constexpr int cudaMemAdviseSetPreferredLocation = 2;
constexpr int cudaMemLocationTypeDevice = 1;
constexpr int cudaMemLocationTypeHostNuma = 2;
constexpr int cudaMemLocationTypeHost = 3;
constexpr int cudaMemAttachGlobal = 1;
constexpr int cudaDevAttrHostNumaId = 1;
struct cudaMemLocation { int type; int id; };
inline int cudaGetDevice(int* device) {
    *device = 0; return std::getenv("MOCK_CUDA_FAILURE") ? 1 : 0;
}
inline int cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    *free_bytes = *total_bytes = 16ULL << 30; return 0;
}
inline int cudaMallocManaged(void** pointer, size_t bytes, int) {
    *pointer = std::malloc(bytes); return *pointer ? 0 : 1;
}
inline int cudaDeviceGetAttribute(int* value, int, int) { *value = -1; return 0; }
inline int cudaMemAdvise(const void*, size_t, int, cudaMemLocation) { return 0; }
inline int cudaMemPrefetchAsync(const void*, size_t, cudaMemLocation, int, int) { return 0; }
inline int cudaDeviceSynchronize() { return 0; }
inline int cudaFree(void* pointer) { std::free(pointer); return 0; }
inline int cudaGetLastError() { return 0; }
inline const char* cudaGetErrorName(int) { return "mock_error"; }
inline const char* cudaGetErrorString(int) { return "mock CUDA failure"; }
""")
    (directory / "main.cpp").write_text(r"""
#include <cstdlib>
extern "C" int fq_primme_gpu_basis_malloc(void**, size_t, long long, int);
extern "C" void fq_primme_gpu_basis_free(void*);
int main() {
    void* pointer = nullptr;
    int status = fq_primme_gpu_basis_malloc(&pointer, 96, 3, 4);
    if(pointer) { fq_primme_gpu_basis_free(pointer); std::free(pointer); }
    return status;
}
""")
    source = Path(__file__).resolve().parents[1] / "gpu/eigensolver/managed_basis.cpp"
    binary = directory / "probe"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(directory),
            str(source),
            str(directory / "main.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    return binary


def _run(binary, value=None, failure=False):
    env = dict(os.environ, FQ_PRIMME_MANAGED_BASIS="1", OMPI_COMM_WORLD_RANK="3")
    env.pop("FQ_PRIMME_MANAGED_BASIS_LOG", None)
    env.pop("MOCK_CUDA_FAILURE", None)
    if value is not None:
        env["FQ_PRIMME_MANAGED_BASIS_LOG"] = value
    if failure:
        env["MOCK_CUDA_FAILURE"] = "1"
    return subprocess.run(
        [str(binary)], capture_output=True, text=True, env=env, check=False
    )


@pytest.mark.parametrize("value", [None, "0", "false", " off "])
def test_managed_allocations_are_quiet_by_default(native_probe, value):
    result = _run(native_probe, value)
    assert result.returncode == 0
    assert result.stdout == "" and result.stderr == ""


@pytest.mark.parametrize("value", ["1", "true", " YES ", "On"])
def test_native_reporting_is_explicit_and_identifies_rank(native_probe, value):
    result = _run(native_probe, value)
    assert result.returncode == 0
    assert "PRIMME managed basis budget: rank=3" in result.stderr
    assert "PRIMME managed basis allocated: rank=3" in result.stderr
    assert "host_numa_node=unavailable" in result.stderr
    assert "[fq_" not in result.stderr


def test_cuda_failure_warning_is_visible_with_reporting_off(native_probe):
    result = _run(native_probe, failure=True)
    assert result.returncode == 1
    assert "CUDA operation cudaGetDevice failed" in result.stderr
    assert "mock CUDA failure" in result.stderr
