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

# distutils: language = c++
# cython: language_level=3


from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.vector cimport vector

from ...core.constants cimport width_t
from ...core.qubit_operator cimport QubitOperator_t
from ...core.spmv cimport FulqrumSpMV


cdef extern from "../../core/src/sector_layout.hpp":
    cdef cppclass CheckLayout "SectorLayout":
        bool valid

    CheckLayout check_make_layout "make_sector_layout"(vector[uint8_t] sector_of) except +


cdef extern from "../../core/src/operator_tables.hpp":
    cdef enum class CheckTier "GroupTier"(uint8_t):
        generic
        cross_sector_double

    cdef enum class CheckReason "GenericReason"(uint8_t):
        forced
        no_sector_layout
        crosses_sectors
        mixed_own_factors
        double_with_other_factor
        cross_with_projector
        cross_sign_not_separable

    cdef cppclass CheckGroupData "GroupData":
        CheckTier tier
        CheckReason reason
        double complex cross_values[4]

    cdef cppclass CheckTables "OperatorTables":
        CheckTables()
        size_t num_groups()
        const CheckGroupData& group(size_t group)

    CheckTables check_build_tables "OperatorTables::build"(const QubitOperator_t& oper,
                                                           const size_t* group_ptrs,
                                                           const vector[vector[width_t]]& group_flips,
                                                           const CheckLayout& layout,
                                                           bool force_generic) except +


GENERIC_REASONS = {
    <int>CheckReason.forced: "forced",
    <int>CheckReason.no_sector_layout: "no_sector_layout",
    <int>CheckReason.crosses_sectors: "crosses_sectors",
    <int>CheckReason.mixed_own_factors: "mixed_own_factors",
    <int>CheckReason.double_with_other_factor: "double_with_other_factor",
    <int>CheckReason.cross_with_projector: "cross_with_projector",
    <int>CheckReason.cross_sign_not_separable: "cross_sign_not_separable",
}
TIER_GENERIC = <int>CheckTier.generic
TIER_CROSS = <int>CheckTier.cross_sector_double
DIRECTION_TOLERANCE = 1e-12


cdef CheckTables block_tables(FulqrumSpMV spmv, int num_alpha_qubits) except *:


    cdef int width = spmv.oper.width
    cdef int qubit
    cdef vector[uint8_t] sector_of
    if not 0 < num_alpha_qubits < width:
        raise ValueError(f"num_alpha_qubits must be in (0, {width}); got {num_alpha_qubits}")
    sector_of.resize(width, 0)
    for qubit in range(num_alpha_qubits, width):
        sector_of[qubit] = 1
    return check_build_tables(
        spmv.oper, &spmv.group_ptrs[0], spmv.group_offdiag_inds, check_make_layout(sector_of), False
    )


def block_tier_counts(FulqrumSpMV spmv, int num_alpha_qubits):


    cdef CheckTables tables
    cdef size_t group
    if spmv.oper.type != 2 or spmv.num_groups == 0:
        return {}
    tables = block_tables(spmv, num_alpha_qubits)
    counts = {}
    for group in range(tables.num_groups()):
        tier = <int>tables.group(group).tier
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def cross_group_values(FulqrumSpMV spmv, int num_alpha_qubits):


    cdef CheckTables tables
    cdef CheckGroupData plan
    cdef size_t group
    cdef int pattern
    if spmv.oper.type != 2 or spmv.num_groups == 0:
        return {}
    tables = block_tables(spmv, num_alpha_qubits)
    values = {}
    for group in range(tables.num_groups()):
        plan = tables.group(group)
        if <int>plan.tier == TIER_CROSS:
            values[group] = [complex(plan.cross_values[pattern]) for pattern in range(4)]
    return values


def unsupported_groups(FulqrumSpMV spmv, int num_alpha_qubits, gpu_coefficients=None):


    cdef size_t group
    cdef int pattern
    cdef CheckTables tables
    cdef CheckGroupData plan
    cdef int tier
    if spmv.oper.type != 2 or spmv.num_groups == 0:
        return []
    tables = block_tables(spmv, num_alpha_qubits)
    found = []
    for group in range(tables.num_groups()):
        plan = tables.group(group)
        tier = <int>plan.tier
        flips = list(spmv.group_offdiag_inds[group])
        if tier == TIER_GENERIC:
            reason = <int>plan.reason
            found.append((group, flips, GENERIC_REASONS.get(reason, f"reason {reason}")))
            continue
        if tier != TIER_CROSS:
            continue
        values = [complex(plan.cross_values[pattern]) for pattern in range(4)]
        if any(value.imag != 0.0 for value in values):
            found.append((group, flips, "complex_cross_value"))
            continue
        scale = max(abs(value) for value in values)
        if any(abs(value - values[0]) > DIRECTION_TOLERANCE * scale for value in values):
            found.append((group, flips, "direction_dependent_cross_value"))
            continue
        if gpu_coefficients is None:
            continue
        expected = abs(float(gpu_coefficients[group]))
        if abs(expected - scale) > DIRECTION_TOLERANCE * max(expected, scale):
            found.append((group, flips, "gpu_coefficient_mismatch"))
    return found
