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

"""GPU-accelerated Fulqrum operators; CUDA components load on first access."""

import logging as _logging

from .logging import configure_gpu_logging, gpu_profile_record, write_gpu_profile


_logging.getLogger(__name__).addHandler(_logging.NullHandler())


def __getattr__(name):


    if name == "GPUHamiltonian":
        from .gpu_hamiltonian import GPUHamiltonian  # noqa: PLC0415

        return GPUHamiltonian
    if name == "distributed_primme_eigsh":
        from .eigensolver.primme_distributed import (  # noqa: PLC0415
            distributed_primme_eigsh,
        )

        return distributed_primme_eigsh
    if name == "is_primme_mpi_available":
        from .eigensolver.primme_distributed import (  # noqa: PLC0415
            is_primme_mpi_available,
        )

        return is_primme_mpi_available
    if name == "TileGPUResidentHamiltonian":
        from .distributed.dist_matvec_tile_resident import (  # noqa: PLC0415
            TileGPUResidentHamiltonian,
        )

        return TileGPUResidentHamiltonian
    if name == "DistributedMPIGPUHamiltonian":
        from .distributed.dist_matvec import (  # noqa: PLC0415
            DistributedMPIGPUHamiltonian,
        )

        return DistributedMPIGPUHamiltonian
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GPUHamiltonian",
    "distributed_primme_eigsh",
    "is_primme_mpi_available",
    "TileGPUResidentHamiltonian",
    "DistributedMPIGPUHamiltonian",
    "configure_gpu_logging",
    "gpu_profile_record",
    "write_gpu_profile",
]
