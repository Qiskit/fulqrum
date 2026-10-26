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
# cython: c_string_type=unicode, c_string_encoding=UTF-8

"""PySCF conversion utilities"""

from pathlib import Path
import time
import numpy as np
from ..core.fermi_operator cimport FermionicOperator
from ..convert.fcidump cimport FCIDumpData, parse_fcidump

import logging
logger = logging.getLogger(__name__)

include "../core/includes/types.pxi"


def integrals_to_fq_fermionic_op(double_or_complex[:,::1] one_body_integrals, double_or_complex[:,:,:,::1] two_body_integrals, 
                                 complex constant=0, double EQ_TOLERANCE=1e-12) -> FermionicOperator:
    """Convert one- and two-body integrals as numpy arrays into Fulqrum
        fermionic operator.

    Parameters:
        one_body_integrals (np.ndarray): One body integrals.
        two_body_integrals (np.ndarray): Two body integrals. The integrals must
            be in chemist's notation (a_p^ a_q a_r^ a_s) as used by PySCF.
        constant (float or complex): A constant term such as nuclear repulsion energy.
        EQ_TOLERANCE (float): Equality tolerance.

    Returns:
        FermionicOperator: Converted operator.
    """
    two_body_integrals = np.ascontiguousarray(np.asarray(two_body_integrals).transpose(0, 2, 3, 1))
    # Go to flat arrays in prep for doing calculation in C++
    cdef double_or_complex[::1] flat_one_body_integrals = np.asarray(one_body_integrals).ravel()
    cdef double_or_complex[::1] flat_two_body_integrals = np.asarray(two_body_integrals).ravel()

    num_qubits = int(2 * np.sqrt(flat_one_body_integrals.shape[0]))
    cdef FermionicOperator fop = FermionicOperator(num_qubits)
    
    if double_or_complex is double:
        fop.oper = pyscf_integrals_to_fermionic[double](&flat_one_body_integrals[0], &flat_two_body_integrals[0],
                                            flat_one_body_integrals.shape[0], flat_two_body_integrals.shape[0], 
                                            constant, EQ_TOLERANCE)
    else:
        fop.oper = pyscf_integrals_to_fermionic[complex](&flat_one_body_integrals[0], &flat_two_body_integrals[0],
                                            flat_one_body_integrals.shape[0], flat_two_body_integrals.shape[0], 
                                            constant, EQ_TOLERANCE)

    return fop


def fcidump_to_fq_fermionic_op(fcidump_path: str | Path, double EQ_TOLERANCE=1e-12) -> FermionicOperator:
    """Load one- and two-body integrals as numpy arrays into Fulqrum
        fermionic operator from FCIDUMP file.

    Parameters:
        fcidump_path (str | Path): The FCIDUMP file.
        EQ_TOLERANCE (float): Equality tolerance.

    Returns:
        FermionicOperator: Converted operator.
    """
    logger.info("Starting import of FCIDump file")
    cdef double st = time.perf_counter()
    cdef FCIDumpData data = FCIDumpData(str(fcidump_path))
    cdef int norb = data.NORB
    cdef int norb2 = norb * norb
    cdef FermionicOperator fop = FermionicOperator(2*norb)
    cdef vector[double] two_body_ints = data.two_body_integrals(1)
    fop.oper = pyscf_integrals_to_fermionic[double](&data.data.H1[0], &two_body_ints[0], norb2, norb2 * norb2,
                                                                      data.ECORE, EQ_TOLERANCE)
    cdef double ft = time.perf_counter()
    logger.info("FCIDump to FermionicOperator time: %s ms", round((ft - st) * 1000, 3))
    return fop
