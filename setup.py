# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fulqrum : A generalized framework for quantum subspace diagonalization"""

import os
from pathlib import Path
import sys

import numpy as np
import setuptools
from Cython.Build import cythonize
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_build  # noqa: E402


class FulqrumBuild(build):
    """Keep CPU and GPU artifacts separate when rebuilding the same checkout."""

    def finalize_options(self):
        super().finalize_options()
        variant = (
            "gpu"
            if gpu_build.gpu_enabled()
            else ("primme" if gpu_build.primme_mpi_enabled() else "cpu")
        )
        self.build_purelib += f"-{variant}"
        self.build_platlib += f"-{variant}"
        self.build_lib += f"-{variant}"
        self.build_temp += f"-{variant}"


class FulqrumBuildExt(build_ext):
    """Defer PRIMME and CUDA builds until build_ext runs."""

    def run(self):
        self.parallel = gpu_build.build_parallelism()
        if gpu_build.primme_mpi_enabled():
            gpu_build.prepare_native_build(self)
        if self.inplace:
            gpu_build.prune_inplace_native(self.extensions)
        generated = cythonize(
            self.extensions,
            build_dir=str(Path(self.build_temp) / "cython"),
            nthreads=0 if self.inplace else self.parallel,
            language_level=3,
            compiler_directives={"embedsignature": True},
        )
        generated_root = Path(self.build_temp) / "cython"
        for header in generated_root.rglob("*.hpp"):
            original = gpu_build.ROOT / header.relative_to(generated_root)
            if original.is_file():
                forwarding = '#include "' + original.as_posix() + '"\n'
                if header.read_text() != forwarding:
                    header.write_text(forwarding)

        originals = {extension.name: extension for extension in self.extensions}
        for extension in generated:
            originals[extension.name].__dict__.update(extension.__dict__)
        super().run()


class FulqrumSdist(sdist):
    """Ship original pinned PRIMME sources even if the submodule is dirty."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        source = Path(base_dir) / "third-party/primme"
        if not (gpu_build.ROOT / "third-party/primme/include/primme.h").is_file():
            raise RuntimeError(
                "A source distribution requires the pinned PRIMME sources; initialize third-party/primme first"
            )
        if (gpu_build.ROOT / "third-party/primme/include/primme.h").is_file():
            for path in source.rglob("*"):
                if path.is_file():
                    path.unlink()
            gpu_build.copy_primme_sources(source)


CPU_MODULES = [
    "core.subspace",
    "core.bitset",
    "core.bitset_view",
    "core.qubit_operator",
    "core.fermi_operator",
    "core.spmv",
    "core.csr",
    "core.csrlike",
    "core.constants",
    "convert.qiskit",
    "convert.integrals",
    "convert.openfermion",
    "utils.matrix",
    "core.sqd",
    "ramps.simple",
    "ramps.open",
]
INCLUDE_DIRS = [np.get_include(), "fulqrum/include", "qiskit-addon-sqd-hpc/include"]
COMPILE_ARGS = ["-O3", "-std=c++17", "-fopenmp"]
LINK_ARGS = ["-fopenmp"]
DEFINE_MACROS = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    ("BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS", None),
]

architecture = os.environ.get("FQ_ARCH")
if architecture and sys.platform != "win32":
    COMPILE_ARGS += (
        [f"-mcpu={architecture}"]
        if sys.platform == "darwin"
        else [f"-march={architecture}", f"-mtune={architecture}"]
    )
if gpu_build.env_flag("FQ_TBB"):
    DEFINE_MACROS.append(("FQ_TBB", "1"))
    LINK_ARGS.append("-ltbb")
if gpu_build.env_flag("FQ_DEBUG"):
    DEFINE_MACROS.append(("_GLIBCXX_ASSERTIONS", None))

EXTENSIONS = [
    setuptools.Extension(
        f"fulqrum.{module}",
        sources=[f"fulqrum/{module.replace('.', '/')}.pyx"],
        include_dirs=list(INCLUDE_DIRS)
        + [
            str(gpu_build.ROOT / "fulqrum/core/src"),
            str(gpu_build.ROOT / f"fulqrum/{module.split('.')[0]}/src"),
        ],
        extra_compile_args=list(COMPILE_ARGS),
        extra_link_args=list(LINK_ARGS),
        define_macros=list(DEFINE_MACROS),
        language="c++",
    )
    for module in CPU_MODULES
]
EXTENSIONS += gpu_build.gpu_extensions(
    "fulqrum",
    INCLUDE_DIRS,
    COMPILE_ARGS,
    LINK_ARGS,
    DEFINE_MACROS,
)

if __name__ == "__main__":
    setuptools.setup(
        ext_modules=EXTENSIONS,
        cmdclass={
            "build": FulqrumBuild,
            "build_ext": FulqrumBuildExt,
            "sdist": FulqrumSdist,
        },
    )
