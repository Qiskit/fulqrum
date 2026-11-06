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


cimport openmp
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from cython.operator cimport dereference as deref

import numpy as np

from ...core.constants cimport width_t
from ...core.spmv cimport FulqrumSpMV
from ...core.half_key_view cimport HalfKeyView
from ...core.tensor_product_cpu cimport OperatorTables, SectorLayout, GroupTier


cdef extern from "<array>":
    cdef cppclass PairCounts "std::array<std::size_t, 2>":
        PairCounts()
        size_t& operator[](size_t)

cdef extern from "../../core/src/sector_layout.hpp":
    vector[uint8_t] tile_block_sectors "block_sectors"(width_t width, width_t split)


cdef extern from "../../core/src/tile_connections.hpp":
    cdef cppclass TileSectorGroupIndex "SectorGroupIndex":
        pass

    cdef cppclass TileConnections[T]:
        vector[uint64_t] single_offsets
        vector[uint64_t] single_packed_entries
        vector[T] single_values
        vector[uint64_t] double_offsets
        vector[uint32_t] double_columns
        vector[T] double_values

    TileSectorGroupIndex tile_index_groups "index_sector_groups"(const OperatorTables& tables,
                                                                 const PairCounts& num_pairs) except +
    vector[TileConnections[T]] build_tile_connections[T](const OperatorTables& tables,
                                                         const HalfKeyView& store,
                                                         const TileSectorGroupIndex& groups,
                                                         int sector,
                                                         const PairCounts& num_pairs,
                                                         size_t output_begin,
                                                         size_t output_end,
                                                         const vector[size_t]& block_bounds) except +
    vector[T] build_tile_cross_table[T](const OperatorTables& tables,
                                        const HalfKeyView& store,
                                        const TileSectorGroupIndex& groups,
                                        int sector,
                                        const PairCounts& num_pairs,
                                        size_t other_begin,
                                        size_t other_end) except +
    vector[T] tile_cross_sector_table "build_cross_sector_table"[T](const OperatorTables& tables,
                                                                    const PairCounts& num_pairs) except +


cdef extern from "../kernels/formats/sell.hpp" namespace "fq::sell":
    const int kSliceRows
    const long long kMaxSlots


cdef extern from "../kernels/formats/edge_word.hpp" namespace "fq::edge_word":
    const int kCrossPairShift
    const int kCrossSignShift
    const uint32_t kCrossMaxPair
    const uint32_t kSignedPairNegative
    uint64_t pack_cross(uint32_t column, uint32_t pair, int sign)


SELL_SLICE_ROWS = kSliceRows
SELL_MAX_SLOTS = kMaxSlots
CROSS_PAIR_SHIFT = kCrossPairShift
CROSS_SIGN_SHIFT = kCrossSignShift
CROSS_MAX_PAIR = kCrossMaxPair
SIGNED_PAIR_NEGATIVE = kSignedPairNegative
MAX_DIRECTED_PAIRS = 0xFFFF


def cross_words(columns, pairs, signs):

    columns = np.asarray(columns, dtype=np.uint32)
    pairs = np.asarray(pairs, dtype=np.uint32)
    signs = np.asarray(signs, dtype=np.int32)
    out = np.empty(columns.size, dtype=np.uint64)
    cdef uint64_t[::1] view = out
    cdef size_t index
    for index in range(columns.size):
        view[index] = pack_cross(columns[index], pairs[index], signs[index])
    return out


cdef const OperatorTables* tables_of(FulqrumSpMV spmv) except NULL:

    spmv._prepare_gpu_tables()
    return &spmv.tables


cdef PairCounts pairs_of(FulqrumSpMV spmv):

    cdef const SectorLayout* layout = &tables_of(spmv).layout()
    cdef PairCounts pairs
    cdef size_t sector
    cdef size_t orbitals
    for sector in range(2):
        orbitals = layout.num_orbitals[sector]
        pairs[sector] = orbitals * (orbitals - 1) // 2 if orbitals > 1 else 0
    return pairs


def tile_input_error(FulqrumSpMV spmv, int num_alpha_qubits, alpha_strings, beta_strings):


    cdef const OperatorTables* tables = tables_of(spmv)
    cdef unique_ptr[HalfKeyView] owned_store = unique_ptr[HalfKeyView](new HalfKeyView(spmv.subspace.subspace))
    cdef HalfKeyView* store = owned_store.get()
    cdef const SectorLayout* layout
    cdef PairCounts pairs
    cdef vector[uint8_t] sector_of
    cdef int sector
    cdef size_t index
    if spmv.is_real != 1:
        return "the operator has complex values"
    if tables.num_groups() == 0:
        return "the operator has no CPU operator tables"
    if not store.is_tensor_product():
        return "the subspace is not a half-string (tensor-product) subspace"
    if store.split_bit() != num_alpha_qubits:
        return f"the subspace splits at qubit {store.split_bit()}, not at {num_alpha_qubits}"
    layout = &tables.layout()
    sector_of = layout.sector_of
    if not layout.valid or sector_of != tile_block_sectors(store.width(), store.split_bit()):
        return "the operator tables do not use the block layout of the subspace"
    if tables.count(GroupTier.generic) != 0:
        return f"{tables.count(GroupTier.generic)} groups are in the generic tier"
    pairs = pairs_of(spmv)
    for sector in range(2):
        if 2 * pairs[sector] > MAX_DIRECTED_PAIRS:
            return f"sector {sector} has more than {MAX_DIRECTED_PAIRS // 2} orbital pairs"
        if store.half_words(sector) != 1:
            return f"sector {sector} is wider than 64 qubits"
    for sector, strings in enumerate((alpha_strings, beta_strings)):
        strings = np.asarray(strings, dtype=np.uint64)
        if store.num_half_keys(sector) != strings.size:
            return f"sector {sector} has {store.num_half_keys(sector)} strings in the subspace, not {strings.size}"
        keys = np.empty(strings.size, dtype=np.uint64)
        for index in range(strings.size):
            keys[index] = store.half_key(sector, index)[0]
        if not np.array_equal(keys, strings):
            return f"the sector {sector} strings of the subspace differ from the GPU strings"
    return None


cdef void validate_tile(FulqrumSpMV spmv, HalfKeyView* store, int sector) except *:
    cdef PairCounts pairs
    if sector not in (0, 1):
        raise ValueError("sector must be 0 or 1")
    if not store.is_tensor_product() or spmv.is_real != 1:
        raise ValueError("GPU tile tables require a real half-string operator")
    if tables_of(spmv).count(GroupTier.generic) != 0 or not tables_of(spmv).layout().valid:
        raise ValueError("GPU tile tables require supported groups and a valid block layout")
    pairs = pairs_of(spmv)
    for half in range(2):
        if store.half_words(half) != 1 or 2 * pairs[half] > MAX_DIRECTED_PAIRS:
            raise ValueError("GPU tile sector exceeds the packed format limits")
        if store.num_half_keys(half) > 0x100000000:
            raise ValueError("GPU tile columns must fit in uint32")

def sector_tiles(FulqrumSpMV spmv, int sector, size_t bra_begin, size_t bra_end, block_bounds,
                 int threads=0):


    cdef unique_ptr[HalfKeyView] owned_store = unique_ptr[HalfKeyView](new HalfKeyView(spmv.subspace.subspace))
    cdef HalfKeyView* store = owned_store.get()
    validate_tile(spmv, store, sector)
    cdef PairCounts pairs = pairs_of(spmv)
    cdef TileSectorGroupIndex groups = tile_index_groups(tables_of(spmv)[0], pairs)
    if bra_begin > bra_end or bra_end > store.num_half_keys(sector):
        raise ValueError("Invalid bra string range")
    if len(block_bounds) < 2 or any(int(bound) != bound or bound < 0 for bound in block_bounds):
        raise ValueError("Block boundaries must be nonnegative integer indices")
    if any(left > right for left, right in zip(block_bounds[:-1], block_bounds[1:])) or block_bounds[-1] > store.num_half_keys(sector):
        raise ValueError("Block boundaries must be sorted and inside the sector")
    cdef vector[size_t] bounds = [int(bound) for bound in block_bounds]
    cdef vector[TileConnections[double]] tiles
    cdef int previous = openmp.omp_get_max_threads()
    if threads > 0:
        openmp.omp_set_num_threads(threads)
    try:
        tiles = build_tile_connections[double](tables_of(spmv)[0], deref(store), groups, sector,
                                               pairs, bra_begin, bra_end, bounds)
    finally:
        openmp.omp_set_num_threads(previous)
    return [_tile_arrays(tiles[block]) for block in range(tiles.size())]


def cross_table(FulqrumSpMV spmv, int sector, size_t other_begin, size_t other_end, int threads=0):


    cdef unique_ptr[HalfKeyView] owned_store = unique_ptr[HalfKeyView](new HalfKeyView(spmv.subspace.subspace))
    cdef HalfKeyView* store = owned_store.get()
    validate_tile(spmv, store, sector)
    cdef PairCounts pairs = pairs_of(spmv)
    cdef TileSectorGroupIndex groups = tile_index_groups(tables_of(spmv)[0], pairs)
    if other_begin > other_end or other_end > store.num_half_keys(1 - sector):
        raise ValueError("Invalid other-sector string range")
    cdef vector[double] table
    cdef int previous = openmp.omp_get_max_threads()
    if threads > 0:
        openmp.omp_set_num_threads(threads)
    try:
        table = build_tile_cross_table[double](tables_of(spmv)[0], deref(store), groups, sector,
                                               pairs, other_begin, other_end)
    finally:
        openmp.omp_set_num_threads(previous)
    return np.asarray(_doubles(table)).reshape(other_end - other_begin, 2 * pairs[sector])


def cross_sector_table(FulqrumSpMV spmv):


    cdef PairCounts pairs = pairs_of(spmv)
    cdef vector[double] table = tile_cross_sector_table[double](tables_of(spmv)[0], pairs)
    return np.asarray(_doubles(table)).reshape(2 * pairs[1], 4 * pairs[0])


def sector_pairs(FulqrumSpMV spmv):

    cdef PairCounts pairs = pairs_of(spmv)
    return int(pairs[0]), int(pairs[1])


cdef dict _tile_arrays(TileConnections[double]& tile):

    packed_entries = np.asarray(_u64(tile.single_packed_entries))
    return dict(
        single_offsets=np.asarray(_u64(tile.single_offsets)),
        columns=(packed_entries & np.uint64(0xFFFFFFFF)).astype(np.uint32),
        directed_pairs=((packed_entries >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.uint16),
        own_signs=((packed_entries >> np.uint64(48)) & np.uint64(0xFF)).astype(np.uint8).view(np.int8),
        jw_signs=((packed_entries >> np.uint64(56)) & np.uint64(0xFF)).astype(np.uint8).view(np.int8),
        single_values=np.asarray(_doubles(tile.single_values)),
        double_offsets=np.asarray(_u64(tile.double_offsets)),
        double_columns=np.asarray(_u32(tile.double_columns)),
        double_values=np.asarray(_doubles(tile.double_values)),
    )


cdef object _u64(vector[uint64_t]& values):

    out = np.empty(values.size(), dtype=np.uint64)
    cdef uint64_t[::1] view = out
    cdef size_t index
    for index in range(values.size()):
        view[index] = values[index]
    return out


cdef object _u32(vector[uint32_t]& values):

    out = np.empty(values.size(), dtype=np.uint32)
    cdef uint32_t[::1] view = out
    cdef size_t index
    for index in range(values.size()):
        view[index] = values[index]
    return out


cdef object _doubles(vector[double]& values):

    out = np.empty(values.size(), dtype=np.float64)
    cdef double[::1] view = out
    cdef size_t index
    for index in range(values.size()):
        view[index] = values[index]
    return out
