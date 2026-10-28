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

"""Fulqrum"""

__version__ = "0.5.0"


from .core import (
    QubitOperator,
    Subspace,
    SubspaceHamiltonian,
    FermionicOperator,
    Bitset,
)

from fulqrum.core import constants

__all__ = [
    "Bitset",
    "FermionicOperator",
    "QubitOperator",
    "Subspace",
    "SubspaceHamiltonian",
    "constants",
]


def about():
    """About Fulqrum"""
    print()
    print("███████ ██    ██ ██       ██████  ██████  ██    ██   ████    ████")
    print("██      ██    ██ ██      ██    ██ ██   ██ ██    ██  ██  ██  ██  ██")
    print("█████   ██    ██ ██      ██    ██ ██████  ██    ██ ██    ████    ██")
    print("██      ██    ██ ██      ██ ▄▄ ██ ██   ██ ██    ██ ██    ████    ██")
    print("██       ██████  ███████  ██████  ██   ██  ██████  ██    ████    ██")
    print()
    print("(C) Copyright IBM 2024")
    print("Paul D. Nation, Abdullah Saki, and Hwajung Kang")
    print(f"Fulqrum version: {__version__}")


def get_include():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path + os.sep + "include"
