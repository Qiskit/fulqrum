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

"""Conversion utilities"""

from .openfermion import (
    openfermion_fermi_op_to_fulqrum,
    openfermion_qubit_op_to_fulqrum,
)
from .qiskit import (
    sparsepauli_to_fulqrum,
    qiskit_nature_fermi_op_to_fulqrum,
)
from .integrals import integrals_to_fq_fermionic_op, fcidump_to_fq_fermionic_op

from .fcidump import read_fcidump

__all__ = [
    "openfermion_fermi_op_to_fulqrum",
    "openfermion_qubit_op_to_fulqrum",
    "sparsepauli_to_fulqrum",
    "qiskit_nature_fermi_op_to_fulqrum",
    "integrals_to_fq_fermionic_op",
    "fcidump_to_fq_fermionic_op",
    "read_fcidump",
]
