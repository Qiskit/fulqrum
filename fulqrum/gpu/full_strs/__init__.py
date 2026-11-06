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

"""GPU operators for subspaces of full determinant strings."""


def __getattr__(name):


    if name == "FullStrsHamiltonian":
        from .hamiltonian import FullStrsHamiltonian  # noqa: PLC0415

        return FullStrsHamiltonian
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
