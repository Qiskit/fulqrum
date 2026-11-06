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
from libcpp cimport bool
from libc.stdint cimport uint8_t
from libcpp.vector cimport vector
from .constants cimport width_t
from .subspace cimport Subspace_t
from .qubit_operator cimport QubitOperator_t
from .element_evaluator cimport TermMasks

cdef extern from "src/operator_tables.hpp":
    cdef cppclass OperatorTables:
        OperatorTables()
        size_t num_groups()
        size_t count(GroupTier)
        const SectorLayout& layout()

cdef extern from "src/tensor_product/engine.hpp":
    cdef enum class CrossMode(int):
        automatic
        stored
        formed

cdef extern from "src/tensor_product/cpu.hpp":
    cdef cppclass TensorProductCPU[T]:
        size_t tp_groups
        size_t bytes

    OperatorTables tensor_product_cpu_operator_tables(const QubitOperator_t&, const QubitOperator_t&,
        const size_t*, const vector[vector[width_t]]&, const Subspace_t&, bool, bool) except +
    TensorProductCPU[T]* build_tensor_product_cpu[T](const OperatorTables&, const size_t*,
        const vector[vector[width_t]]&, const TermMasks&, const Subspace_t&, CrossMode, bool, bool) except +
    void tensor_product_cpu_apply[T](const TensorProductCPU[T]&, const Subspace_t&, const T*, int, const T*, T*) except + nogil
    void tensor_product_cpu_csr[I,T](const TensorProductCPU[T]&, const Subspace_t&, const T*, int, I*, I*, T*, int) except + nogil
    void tensor_product_cpu_csrlike[T,I](const TensorProductCPU[T]&, const Subspace_t&, const T*, int, vector[vector[I]]&, vector[vector[T]]&) except + nogil
    void tensor_product_cpu_diagonal[T](const Subspace_t&, const TermMasks&, bool, T*) except + nogil
    size_t tensor_product_cpu_residual_groups[T](const TensorProductCPU[T]&)
    bool tensor_product_cpu_tp_ready[T](const TensorProductCPU[T]&)
    bool tensor_product_cpu_cross_formed[T](const TensorProductCPU[T]&)


cdef extern from "<array>":
    cdef cppclass OrbitalCounts "std::array<width_t, 2>":
        width_t& operator[](size_t)

cdef extern from "src/sector_layout.hpp":
    cdef cppclass SectorLayout:
        bool valid
        vector[uint8_t] sector_of
        OrbitalCounts num_orbitals

cdef extern from "src/operator_tables.hpp":
    cdef enum class GroupTier(uint8_t):
        generic


cdef extern from "src/checked_size.hpp":
    size_t checked_multiply(size_t, size_t) except +
    size_t checked_add(size_t, size_t) except +
    I checked_index[I](size_t) except +
