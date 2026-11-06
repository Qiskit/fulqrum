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


from cpython.ref cimport Py_INCREF
from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint64_t
from libc.string cimport memcpy
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

import numpy as np
cimport numpy as cnp

from ...core.qubit_operator cimport QubitOperator, QubitOperator_t

cnp.import_array()


cdef extern from "src/host.hpp":
    cdef cppclass CppKeyStore "fq::full_strs::KeyStore<1>":
        int num_orbitals
        int num_alpha
        int num_beta
        vector[uint64_t] keys
        vector[int64_t] line_start
        vector[int64_t] column_start
        vector[int64_t] column_rows
        int64_t num_rows()
        int64_t num_lines()
        int64_t num_columns()

    cdef cppclass CppKinds "fq::full_strs::GroupKinds":
        int alpha
        int beta

    cdef cppclass CppGroups "fq::full_strs::GroupStore":
        CppKinds kinds
        vector[int64_t] row_slot_start
        vector[int64_t] row_slots

    cdef cppclass CppClassStats "fq::full_strs::ClassStats":
        int64_t groups
        int64_t outputs
        int64_t members
        int64_t pairs
        int64_t largest
        int64_t size_groups[7]
        int64_t size_members[7]

    cdef cppclass CppChunkClass "fq::full_strs::ChunkClass":
        vector[int32_t] members
        vector[uint16_t] sizes
        int64_t bucket_start[8]

    cdef cppclass CppLayout "fq::full_strs::HostLayout<1>":
        CppLayout(const uint64_t* keys, int64_t num_rows, int num_orbitals, int alpha_kind, int beta_kind) except + nogil
        CppKeyStore keys
        CppGroups groups
        CppClassStats stats(int cls) nogil
        vector[int64_t] stored_pairs(int cls) nogil
        vector[int64_t] scanned_pairs(int cls) nogil
        CppChunkClass chunk_class(int num_slices, int slice, int cls) except + nogil
        pair[int64_t, uint64_t] pair_digest_sum(int cls) nogil

    cdef cppclass CppCounts "fq::full_strs::HostCounts":
        int64_t num_rows
        CppKinds kinds
        CppClassStats classes[5]
        double key_seconds
        double group_seconds

    CppCounts cpp_count_groups "fq::full_strs::count_groups<1>"(const uint64_t* keys, int64_t num_rows, int num_orbitals,
                                                              int alpha_kind, int beta_kind, int num_slices) except + nogil

    cdef cppclass CppChunk "fq::full_strs::GroupChunk":
        CppChunkClass classes[5]
        int64_t num_members()
        int64_t num_groups()

    cdef cppclass CppSlice "fq::full_strs::GroupSlice":
        int count
        int index

    cdef cppclass CppRecordBatch "fq::full_strs::RecordBatch":
        vector[uint64_t] words
        vector[int64_t] counts

    cdef cppclass CppReturnBatch "fq::full_strs::ReturnBatch":
        vector[uint64_t] headers
        vector[int64_t] header_counts
        vector[uint64_t] members
        vector[int64_t] member_counts

    CppRecordBatch cpp_emit_records "fq::full_strs::emit_records<1>"(
        const uint64_t* keys, int64_t num_rows, int64_t row_start, int num_orbitals, const CppKinds& kinds,
        const CppSlice& slice, int num_ranks) except + nogil
    CppReturnBatch cpp_join_runs "fq::full_strs::join_runs<1>"(
        const uint64_t* records, int64_t num_records, const int64_t* block_starts, int num_ranks) except + nogil

    cdef cppclass CppPieces "fq::full_strs::GroupPieces<1>":
        CppPieces() nogil
        CppPieces(const CppPieces&) nogil
        int64_t share_begin
        int64_t share_end
        vector[int64_t] halo_rows
        vector[uint64_t] halo_keys
        vector[CppChunk] chunks[2]
        int64_t num_own()
        vector[int64_t] pair_list(int kind, int cls) nogil
        pair[int64_t, uint64_t] pair_digest_sum(int kind, int cls) nogil
        void release_chunk(int kind, int chunk) except + nogil
        void release_halo() nogil

    cdef cppclass CppWorkProfile "fq::full_strs::WorkProfile":
        vector[double] row_total
        vector[double] row_own
        vector[double] block_work
        vector[int64_t] row_memberships
        vector[int64_t] block_entries

    cdef cppclass CppReceiver "fq::full_strs::GroupReceiver<1>":
        CppReceiver(const uint64_t* keys, int64_t num_rows, int64_t first_row, int num_slices,
                    int64_t total_rows) except +
        vector[int64_t] halo_rows
        vector[uint64_t] halo_keys
        vector[CppChunk] chunks
        int64_t num_own()
        void add_slice(int slice, const uint64_t* headers, int64_t num_headers, const uint64_t* members,
                       int64_t num_members) except + nogil
        void finish() except + nogil
        void finish_global(const int64_t* block_starts, int num_blocks, int64_t stride) except + nogil
        pair[int64_t, uint64_t] pair_digest_sum(int cls) nogil
        vector[int32_t] halo_chunk_counts() nogil
        vector[int64_t] output_memberships() nogil
        void release_chunk(int chunk) except + nogil
        void release_halo() nogil
        vector[uint64_t] halo_bitmap(int64_t total_rows) except + nogil
        uint64_t* halo_key_buffer() except +
        void verify_halo_keys() except + nogil
        void extend_halo(const uint64_t* bits, size_t num_words) except + nogil
        vector[int64_t] pair_list(int cls) nogil
        CppPieces pieces(int64_t share_begin, int64_t share_end, const uint8_t* assigned) except + nogil
        CppWorkProfile work_profile(const double* coefficients, const int32_t* halo_block, int num_blocks,
                                    int own_block) nogil

    cdef cppclass CppTables "fq::full_strs::CoefficientTables<1, double>":
        int num_orbitals
        int64_t num_pairs
        int64_t num_quads
        uint64_t reference[2]
        vector[double] single_value
        vector[double] single_delta
        vector[double] single_reference
        vector[double] double_value
        vector[double] mixed_value

    cdef cppclass CppTerms "fq::full_strs::OperatorTerms<1>":
        int64_t num_groups()

    cdef cppclass CppHost "fq::full_strs::HostOperator<1, double>":
        CppHost(const QubitOperator_t& op, const uint64_t* keys, int64_t num_rows, int alpha_kind, int beta_kind,
                int num_slices) except + nogil
        CppKeyStore keys
        CppTerms terms
        CppTables tables
        vector[double] diagonal
        CppKinds kinds
        CppGroups groups
        vector[CppChunk] chunks
        void matvec(const double* x, double* y) except + nogil
        void chunk_matvec(const double* x, double* y) except + nogil
        void chunk_partials(const double* x, int chunk, int cls, double* partials) except + nogil
        double table_element(const uint64_t* out, const uint64_t* source) nogil
        double term_element(const uint64_t* out, const uint64_t* source) nogil

    cdef cppclass CppRank "fq::full_strs::RankOperator<1, double>":
        CppRank(const QubitOperator_t& op, const uint64_t* own_keys, int64_t num_own,
                const uint64_t* reference) except + nogil
        CppKeyStore own
        CppTables tables
        vector[double] diagonal

    cdef cppclass CppReference "std::array<uint64_t, 2>":
        uint64_t& operator[](size_t)

    CppReference cpp_block_reference "fq::full_strs::block_reference<1>"(const uint64_t* keys, int64_t num_rows,
                                                                         int num_orbitals) except + nogil


CLASSES = ("alpha_single", "alpha_double", "beta_single", "beta_double", "mixed")
KINDS = ("intersection", "union")


cdef cnp.ndarray owned_view(object owner, void* data, cnp.npy_intp count, int typenum):


    cdef cnp.npy_intp dims[1]
    dims[0] = count
    cdef cnp.ndarray out = cnp.PyArray_SimpleNewFromData(1, dims, typenum, data)
    Py_INCREF(owner)
    cnp.PyArray_SetBaseObject(out, owner)
    cnp.PyArray_CLEARFLAGS(out, cnp.NPY_ARRAY_WRITEABLE)
    return out


cdef cnp.ndarray key_array(object keys):


    cdef cnp.ndarray out = np.ascontiguousarray(keys, dtype=np.uint64)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(f"keys must have shape (n, 2) (alpha word, beta word); got {np.shape(out)}")
    return out


cdef class HostOperator:


    cdef unique_ptr[CppHost] host

    def __cinit__(self, QubitOperator operator, keys, alpha_kind=None, beta_kind=None, int num_slices=0):
        cdef cnp.ndarray rows = key_array(keys)
        cdef const uint64_t* data = <const uint64_t*>cnp.PyArray_DATA(rows)
        cdef int64_t num_rows = rows.shape[0]
        cdef int alpha = -1 if alpha_kind is None else KINDS.index(alpha_kind)
        cdef int beta = -1 if beta_kind is None else KINDS.index(beta_kind)
        with nogil:
            self.host.reset(new CppHost(operator.oper, data, num_rows, alpha, beta, num_slices))

    @property
    def kinds(self):

        return (KINDS[self.host.get().kinds.alpha], KINDS[self.host.get().kinds.beta])

    @property
    def num_chunks(self):

        return self.host.get().chunks.size()

    @property
    def num_orbitals(self):

        return self.host.get().keys.num_orbitals

    def matvec(self, x):


        return self._apply(x, False)

    def chunk_matvec(self, x):


        return self._apply(x, True)

    cdef _apply(self, x, bint from_chunks):


        cdef cnp.ndarray xs = np.ascontiguousarray(x, dtype=np.float64)
        if xs.ndim != 1 or xs.shape[0] != self.num_rows:
            raise ValueError(f"x must have shape ({self.num_rows},); got {np.shape(xs)}")
        y = np.empty(self.num_rows, dtype=np.float64)
        cdef double[::1] ys = y
        cdef const double* xp = <const double*>cnp.PyArray_DATA(xs)
        with nogil:
            if from_chunks:
                self.host.get().chunk_matvec(xp, &ys[0])
            else:
                self.host.get().matvec(xp, &ys[0])
        return y

    def chunk_partials(self, x, int chunk, int cls):


        cdef cnp.ndarray xs = np.ascontiguousarray(x, dtype=np.float64)
        if xs.ndim != 1 or xs.shape[0] != self.num_rows:
            raise ValueError(f"x must have shape ({self.num_rows},); got {np.shape(xs)}")
        members, _, _ = self.chunk_views(chunk, cls)
        out = np.empty(members.size, dtype=np.float64)
        cdef double[::1] view = out
        cdef double* outp = &view[0] if members.size else NULL
        cdef const double* xp = <const double*>cnp.PyArray_DATA(xs)
        with nogil:
            self.host.get().chunk_partials(xp, chunk, cls, outp)
        return out

    def key_words(self):

        cdef vector[uint64_t]* words = &self.host.get().keys.keys
        return owned_view(self, words.data(), words.size(), cnp.NPY_UINT64)

    def diagonal_view(self):

        cdef vector[double]* values = &self.host.get().diagonal
        return owned_view(self, values.data(), values.size(), cnp.NPY_FLOAT64)

    def table_views(self):


        cdef CppTables* t = &self.host.get().tables
        return {
            "single_value": owned_view(self, t.single_value.data(), t.single_value.size(), cnp.NPY_FLOAT64),
            "single_delta": owned_view(self, t.single_delta.data(), t.single_delta.size(), cnp.NPY_FLOAT64),
            "single_reference": owned_view(self, t.single_reference.data(), t.single_reference.size(), cnp.NPY_FLOAT64),
            "double_value": owned_view(self, t.double_value.data(), t.double_value.size(), cnp.NPY_FLOAT64),
            "mixed_value": owned_view(self, t.mixed_value.data(), t.mixed_value.size(), cnp.NPY_FLOAT64),
            "reference": (int(t.reference[0]), int(t.reference[1])),
            "num_pairs": t.num_pairs,
            "num_quads": t.num_quads,
            "num_orbitals": t.num_orbitals,
        }

    def chunk_views(self, int chunk, int cls):


        if not 0 <= chunk < <int>self.host.get().chunks.size():
            raise IndexError(f"chunk {chunk} out of range")
        if not 0 <= cls < len(CLASSES):
            raise IndexError(f"class {cls} out of range")
        cdef CppChunkClass* c = &self.host.get().chunks[chunk].classes[cls]
        members = owned_view(self, c.members.data(), c.members.size(), cnp.NPY_INT32)
        sizes = owned_view(self, c.sizes.data(), c.sizes.size(), cnp.NPY_UINT16)
        return members, sizes, [c.bucket_start[k] for k in range(8)]

    @property
    def num_rows(self):

        return self.host.get().keys.num_rows()

    @property
    def num_lines(self):

        return self.host.get().keys.num_lines()

    @property
    def num_columns(self):

        return self.host.get().keys.num_columns()

    @property
    def num_groups(self):

        return self.host.get().terms.num_groups()

    @property
    def electrons(self):

        return (self.host.get().keys.num_alpha, self.host.get().keys.num_beta)

    def sorted_keys(self):

        cdef vector[uint64_t]* words = &self.host.get().keys.keys
        out = np.empty((words.size() // 2, 2), dtype=np.uint64)
        cdef uint64_t[:, ::1] view = out
        cdef size_t k
        for k in range(words.size()):
            view[k // 2, k % 2] = words[0][k]
        return out

    def diagonal(self):

        cdef vector[double]* values = &self.host.get().diagonal
        out = np.empty(values.size(), dtype=np.float64)
        cdef double[::1] view = out
        cdef size_t k
        for k in range(values.size()):
            view[k] = values[0][k]
        return out

    def table_elements(self, out_keys, source_keys):


        return self._elements(out_keys, source_keys, True)

    def term_elements(self, out_keys, source_keys):


        return self._elements(out_keys, source_keys, False)

    cdef _elements(self, out_keys, source_keys, bint from_tables):


        cdef cnp.ndarray outs = key_array(out_keys)
        cdef cnp.ndarray sources = key_array(source_keys)
        if outs.shape[0] != sources.shape[0]:
            raise ValueError("out_keys and source_keys must have the same number of rows")
        cdef int64_t count = outs.shape[0]
        values = np.empty(count, dtype=np.float64)
        cdef double[::1] view = values
        cdef const uint64_t* out_data = <const uint64_t*>cnp.PyArray_DATA(outs)
        cdef const uint64_t* source_data = <const uint64_t*>cnp.PyArray_DATA(sources)
        cdef int64_t k
        cdef CppHost* host = self.host.get()
        with nogil:
            for k in range(count):
                if from_tables:
                    view[k] = host.table_element(out_data + 2 * k, source_data + 2 * k)
                else:
                    view[k] = host.term_element(out_data + 2 * k, source_data + 2 * k)
        return values


cdef dict class_dict(CppClassStats& c):


    return {"groups": c.groups, "outputs": c.outputs, "members": c.members, "pairs": c.pairs,
            "largest": c.largest, "size_groups": [c.size_groups[k] for k in range(7)],
            "size_members": [c.size_members[k] for k in range(7)]}


cdef object pair_array(vector[int64_t]& flat):


    out = np.empty((flat.size() // 2, 2), dtype=np.int64)
    cdef int64_t[:, ::1] view = out
    cdef size_t k
    for k in range(flat.size()):
        view[k // 2, k % 2] = flat[k]
    return out


cdef class HostLayout:


    cdef unique_ptr[CppLayout] layout

    def __cinit__(self, keys, int num_orbitals, alpha_kind=None, beta_kind=None):
        cdef cnp.ndarray rows = key_array(keys)
        cdef const uint64_t* data = <const uint64_t*>cnp.PyArray_DATA(rows)
        cdef int64_t num_rows = rows.shape[0]
        cdef int alpha = -1 if alpha_kind is None else KINDS.index(alpha_kind)
        cdef int beta = -1 if beta_kind is None else KINDS.index(beta_kind)
        with nogil:
            self.layout.reset(new CppLayout(data, num_rows, num_orbitals, alpha, beta))

    @property
    def num_rows(self):

        return self.layout.get().keys.num_rows()

    @property
    def kinds(self):

        return (KINDS[self.layout.get().groups.kinds.alpha], KINDS[self.layout.get().groups.kinds.beta])

    def stats(self):


        cdef CppClassStats c
        cdef int cls
        out = {}
        for cls in range(len(CLASSES)):
            with nogil:
                c = self.layout.get().stats(cls)
            out[CLASSES[cls]] = class_dict(c)
        return out

    def stored_pairs(self, int cls):

        cdef vector[int64_t] flat
        with nogil:
            flat = self.layout.get().stored_pairs(cls)
        return pair_array(flat)

    def scanned_pairs(self, int cls):

        cdef vector[int64_t] flat
        with nogil:
            flat = self.layout.get().scanned_pairs(cls)
        return pair_array(flat)

    def chunk_class(self, int num_slices, int index, int cls):


        cdef CppChunkClass c
        with nogil:
            c = self.layout.get().chunk_class(num_slices, index, cls)
        return chunk_class_arrays(c)

    def pair_digest(self, int cls):

        cdef pair[int64_t, uint64_t] result
        with nogil:
            result = self.layout.get().pair_digest_sum(cls)
        return int(result.first), int(result.second)


def count_groups(keys, int num_orbitals, int num_slices=1, alpha_kind=None, beta_kind=None):


    cdef cnp.ndarray rows = key_array(keys)
    cdef const uint64_t* data = <const uint64_t*>cnp.PyArray_DATA(rows)
    cdef int64_t num_rows = rows.shape[0]
    cdef int alpha = -1 if alpha_kind is None else KINDS.index(alpha_kind)
    cdef int beta = -1 if beta_kind is None else KINDS.index(beta_kind)
    cdef CppCounts counts
    with nogil:
        counts = cpp_count_groups(data, num_rows, num_orbitals, alpha, beta, num_slices)
    classes = {}
    cdef int cls
    for cls in range(len(CLASSES)):
        classes[CLASSES[cls]] = class_dict(counts.classes[cls])
    return {"rows": counts.num_rows, "kinds": (KINDS[counts.kinds.alpha], KINDS[counts.kinds.beta]),
            "key_seconds": counts.key_seconds, "group_seconds": counts.group_seconds, "classes": classes}


cdef object chunk_class_arrays(CppChunkClass& c):

    members = np.empty(c.members.size(), dtype=np.int32)
    sizes = np.empty(c.sizes.size(), dtype=np.uint16)
    if c.members.size():
        memcpy(cnp.PyArray_DATA(members), c.members.data(), c.members.size() * sizeof(int32_t))
    if c.sizes.size():
        memcpy(cnp.PyArray_DATA(sizes), c.sizes.data(), c.sizes.size() * sizeof(uint16_t))
    return members, sizes, [c.bucket_start[k] for k in range(8)]


cdef class WordBuffer:


    cdef vector[uint64_t] words


cdef object take_words(vector[uint64_t]& words, int width):


    if words.size() == 0:
        return np.zeros((0, width), dtype=np.uint64)
    cdef WordBuffer buffer = WordBuffer()
    buffer.words.swap(words)
    return owned_view(buffer, buffer.words.data(), buffer.words.size(), cnp.NPY_UINT64).reshape(-1, width)


cdef object word_array(vector[uint64_t]& words, int width):

    out = np.empty((words.size() // width, width), dtype=np.uint64)
    if words.size():
        memcpy(cnp.PyArray_DATA(out), words.data(), words.size() * sizeof(uint64_t))
    return out


cdef object count_array(vector[int64_t]& counts):

    return np.array([counts[k] for k in range(counts.size())], dtype=np.int64)


cdef cnp.ndarray word_rows(object words, int width, str name):

    cdef cnp.ndarray out = np.ascontiguousarray(words, dtype=np.uint64)
    if out.ndim != 2 or out.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width}); got {np.shape(out)}")
    return out


def emit_records(keys, int64_t row_start, int num_orbitals, int alpha_kind, int beta_kind, int num_slices,
                 int index, int num_ranks):


    cdef cnp.ndarray rows = key_array(keys)
    cdef const uint64_t* data = <const uint64_t*>cnp.PyArray_DATA(rows)
    cdef int64_t num_rows = rows.shape[0]
    cdef CppKinds kinds
    kinds.alpha = alpha_kind
    kinds.beta = beta_kind
    cdef CppSlice slice_
    slice_.count = num_slices
    slice_.index = index
    cdef CppRecordBatch batch
    with nogil:
        batch = cpp_emit_records(data, num_rows, row_start, num_orbitals, kinds, slice_, num_ranks)
    return take_words(batch.words, 3), count_array(batch.counts)


def join_runs(records, block_starts):


    cdef cnp.ndarray rows = word_rows(records, 3, "records")
    cdef cnp.ndarray starts = np.ascontiguousarray(block_starts, dtype=np.int64)
    cdef const uint64_t* data = <const uint64_t*>cnp.PyArray_DATA(rows)
    cdef const int64_t* start_data = <const int64_t*>cnp.PyArray_DATA(starts)
    cdef int64_t num_records = rows.shape[0]
    cdef int num_ranks = starts.shape[0] - 1
    cdef CppReturnBatch batch
    with nogil:
        batch = cpp_join_runs(data, num_records, start_data, num_ranks)
    return (take_words(batch.headers, 3), count_array(batch.header_counts), take_words(batch.members, 1),
            count_array(batch.member_counts))


cdef class GroupPieces:


    cdef unique_ptr[CppPieces] pieces

    @property
    def num_own(self):

        return self.pieces.get().num_own()

    def num_chunks(self):

        return self.pieces.get().chunks[0].size()

    def ranges(self, int kind):

        cdef CppPieces* p = self.pieces.get()
        own = p.num_own()
        if kind == 0:
            return (p.share_begin, p.share_end, 0, own)
        return (0, own, own, own + <int64_t>p.halo_rows.size())

    def halo_rows(self):

        cdef CppPieces* p = self.pieces.get()
        out = np.empty(p.halo_rows.size(), dtype=np.int64)
        if p.halo_rows.size():
            memcpy(cnp.PyArray_DATA(out), p.halo_rows.data(), p.halo_rows.size() * sizeof(int64_t))
        return out

    def halo_key_words(self):

        cdef vector[uint64_t]* words = &self.pieces.get().halo_keys
        return owned_view(self, words.data(), words.size(), cnp.NPY_UINT64)

    def chunk_views(self, int chunk, int cls, int kind):

        if not 0 <= kind < 2:
            raise IndexError(f"kind {kind} out of range")
        if not 0 <= chunk < <int>self.pieces.get().chunks[kind].size():
            raise IndexError(f"chunk {chunk} out of range")
        if not 0 <= cls < len(CLASSES):
            raise IndexError(f"class {cls} out of range")
        cdef CppChunkClass* c = &self.pieces.get().chunks[kind][chunk].classes[cls]
        members = owned_view(self, c.members.data(), c.members.size(), cnp.NPY_INT32)
        sizes = owned_view(self, c.sizes.data(), c.sizes.size(), cnp.NPY_UINT16)
        return members, sizes, [c.bucket_start[k] for k in range(8)]

    def release_chunk(self, int chunk, int kind):


        self.pieces.get().release_chunk(kind, chunk)

    def release_halo(self):

        self.pieces.get().release_halo()

    def pairs(self, int kind, int cls):

        cdef vector[int64_t] flat
        with nogil:
            flat = self.pieces.get().pair_list(kind, cls)
        return pair_array(flat)

    def pair_digest(self, int kind, int cls):

        cdef pair[int64_t, uint64_t] result
        with nogil:
            result = self.pieces.get().pair_digest_sum(kind, cls)
        return int(result.first), int(result.second)


cdef class GroupReceiver:


    cdef unique_ptr[CppReceiver] receiver

    def __cinit__(self, keys, int64_t row_start, int num_slices, int64_t total_rows=0):
        cdef cnp.ndarray rows = key_array(keys)
        self.receiver.reset(new CppReceiver(<const uint64_t*>cnp.PyArray_DATA(rows), rows.shape[0], row_start,
                                            num_slices, total_rows))

    def add_slice(self, int index, headers, members):


        cdef cnp.ndarray h = word_rows(headers, 3, "headers")
        cdef cnp.ndarray m = word_rows(members, 1, "members")
        cdef const uint64_t* hd = <const uint64_t*>cnp.PyArray_DATA(h)
        cdef const uint64_t* md = <const uint64_t*>cnp.PyArray_DATA(m)
        cdef int64_t nh = h.shape[0]
        cdef int64_t nm = m.shape[0]
        with nogil:
            self.receiver.get().add_slice(index, hd, nh, md, nm)

    def finish(self):


        with nogil:
            self.receiver.get().finish()

    def finish_global(self, block_starts, int64_t stride):


        cdef cnp.ndarray starts = np.ascontiguousarray(block_starts, dtype=np.int64)
        cdef const int64_t* data = <const int64_t*>cnp.PyArray_DATA(starts)
        cdef int count = starts.shape[0] - 1
        with nogil:
            self.receiver.get().finish_global(data, count, stride)

    @property
    def num_own(self):

        return self.receiver.get().num_own()

    def halo_rows(self):

        cdef CppReceiver* r = self.receiver.get()
        out = np.empty(r.halo_rows.size(), dtype=np.int64)
        if r.halo_rows.size():
            memcpy(cnp.PyArray_DATA(out), r.halo_rows.data(), r.halo_rows.size() * sizeof(int64_t))
        return out

    def halo_row_view(self):

        cdef vector[int64_t]* rows = &self.receiver.get().halo_rows
        if rows.size() == 0:
            return np.zeros(0, dtype=np.int64)
        return owned_view(self, rows.data(), rows.size(), cnp.NPY_INT64)

    def halo_keys(self):

        return word_array(self.receiver.get().halo_keys, 2)

    def halo_key_words(self):

        cdef vector[uint64_t]* words = &self.receiver.get().halo_keys
        return owned_view(self, words.data(), words.size(), cnp.NPY_UINT64)

    def num_chunks(self):

        return self.receiver.get().chunks.size()

    def chunk_class(self, int chunk, int cls):

        if not 0 <= chunk < <int>self.receiver.get().chunks.size():
            raise IndexError(f"chunk {chunk} out of range")
        if not 0 <= cls < len(CLASSES):
            raise IndexError(f"class {cls} out of range")
        return chunk_class_arrays(self.receiver.get().chunks[chunk].classes[cls])

    def chunk_views(self, int chunk, int cls):

        if not 0 <= chunk < <int>self.receiver.get().chunks.size():
            raise IndexError(f"chunk {chunk} out of range")
        if not 0 <= cls < len(CLASSES):
            raise IndexError(f"class {cls} out of range")
        cdef CppChunkClass* c = &self.receiver.get().chunks[chunk].classes[cls]
        members = owned_view(self, c.members.data(), c.members.size(), cnp.NPY_INT32)
        sizes = owned_view(self, c.sizes.data(), c.sizes.size(), cnp.NPY_UINT16)
        return members, sizes, [c.bucket_start[k] for k in range(8)]

    def pair_digest(self, int cls):

        cdef pair[int64_t, uint64_t] result
        with nogil:
            result = self.receiver.get().pair_digest_sum(cls)
        return int(result.first), int(result.second)

    def pairs(self, int cls):

        cdef vector[int64_t] flat
        with nogil:
            flat = self.receiver.get().pair_list(cls)
        return pair_array(flat)

    def pieces(self, int64_t share_begin, int64_t share_end, assigned):


        cdef cnp.ndarray flags = np.ascontiguousarray(assigned, dtype=np.uint8)
        if flags.shape[0] != <Py_ssize_t>self.receiver.get().halo_rows.size():
            raise ValueError(f"assigned needs one flag per halo row ({self.receiver.get().halo_rows.size()}), "
                             f"got {flags.shape[0]}")
        cdef const uint8_t* data = <const uint8_t*>cnp.PyArray_DATA(flags) if flags.shape[0] else NULL
        cdef uint8_t empty = 0
        if data == NULL:
            data = &empty
        cdef GroupPieces out = GroupPieces()
        with nogil:
            out.pieces.reset(new CppPieces(self.receiver.get().pieces(share_begin, share_end, data)))
        return out

    def work_profile(self, coefficients, halo_block, int num_blocks, int own_block):


        cdef cnp.ndarray coef = np.ascontiguousarray(coefficients, dtype=np.float64).reshape(-1)
        cdef cnp.ndarray blocks = np.ascontiguousarray(halo_block, dtype=np.int32).reshape(-1)
        if coef.shape[0] != 35:
            raise ValueError(f"coefficients need 5 x 7 values, got {coef.shape[0]}")
        if blocks.shape[0] != <Py_ssize_t>self.receiver.get().halo_rows.size():
            raise ValueError("halo_block needs one block per halo row")
        if blocks.shape[0] and (blocks.min() < 0 or blocks.max() >= num_blocks):
            raise ValueError("a halo block is out of range")
        if not 0 <= own_block < num_blocks:
            raise ValueError(f"own_block {own_block} out of range")
        cdef const double* coef_data = <const double*>cnp.PyArray_DATA(coef)
        cdef int32_t dummy = 0
        cdef const int32_t* block_data = <const int32_t*>cnp.PyArray_DATA(blocks) if blocks.shape[0] else &dummy
        cdef CppWorkProfile profile
        with nogil:
            profile = self.receiver.get().work_profile(coef_data, block_data, num_blocks, own_block)
        return (np.array(profile.row_total, dtype=np.float64), np.array(profile.row_own, dtype=np.float64),
                np.array(profile.row_memberships, dtype=np.int64), np.array(profile.block_work, dtype=np.float64),
                np.array(profile.block_entries, dtype=np.int64))

    def output_memberships(self):

        cdef vector[int64_t] counts
        with nogil:
            counts = self.receiver.get().output_memberships()
        out = np.empty(counts.size(), dtype=np.int64)
        if counts.size():
            memcpy(cnp.PyArray_DATA(out), counts.data(), counts.size() * sizeof(int64_t))
        return out

    def release_chunk(self, int chunk):


        self.receiver.get().release_chunk(chunk)

    def release_halo(self):

        self.receiver.get().release_halo()

    def halo_key_buffer(self):


        cdef uint64_t* data = self.receiver.get().halo_key_buffer()
        cdef cnp.npy_intp dims[1]
        dims[0] = 2 * <cnp.npy_intp>self.receiver.get().halo_rows.size()
        if dims[0] == 0:
            return np.zeros((0, 2), dtype=np.uint64)
        cdef cnp.ndarray out = cnp.PyArray_SimpleNewFromData(1, dims, cnp.NPY_UINT64, data)
        Py_INCREF(self)
        cnp.PyArray_SetBaseObject(out, self)
        return out.reshape(-1, 2)

    def verify_halo_keys(self):


        with nogil:
            self.receiver.get().verify_halo_keys()

    def halo_bitmap(self, int64_t total_rows):


        cdef vector[uint64_t] bits
        with nogil:
            bits = self.receiver.get().halo_bitmap(total_rows)
        return word_array(bits, 1).reshape(-1)

    def extend_halo(self, bits):


        cdef cnp.ndarray words = np.ascontiguousarray(bits, dtype=np.uint64).reshape(-1)
        cdef const uint64_t* data = <const uint64_t*>cnp.PyArray_DATA(words)
        cdef size_t count = words.shape[0]
        with nogil:
            self.receiver.get().extend_halo(data, count)

    def halo_chunk_counts(self):

        cdef vector[int32_t] counts
        with nogil:
            counts = self.receiver.get().halo_chunk_counts()
        return np.array([counts[k] for k in range(counts.size())], dtype=np.int32)


def block_reference(keys, int num_orbitals):


    cdef cnp.ndarray rows = key_array(keys)
    cdef const uint64_t* data = <const uint64_t*>cnp.PyArray_DATA(rows)
    cdef int64_t num_rows = rows.shape[0]
    cdef CppReference reference
    with nogil:
        reference = cpp_block_reference(data, num_rows, num_orbitals)
    return np.array([reference[0], reference[1]], dtype=np.uint64)


cdef class RankOperator:


    cdef unique_ptr[CppRank] rank

    def __cinit__(self, QubitOperator operator, own_keys, reference):
        cdef cnp.ndarray own = key_array(own_keys)
        cdef cnp.ndarray ref = np.ascontiguousarray(reference, dtype=np.uint64).reshape(2)
        cdef const uint64_t* own_data = <const uint64_t*>cnp.PyArray_DATA(own)
        cdef const uint64_t* ref_data = <const uint64_t*>cnp.PyArray_DATA(ref)
        cdef int64_t num_own = own.shape[0]
        with nogil:
            self.rank.reset(new CppRank(operator.oper, own_data, num_own, ref_data))

    @property
    def num_own(self):

        return self.rank.get().own.num_rows()

    def key_words(self):

        cdef vector[uint64_t]* words = &self.rank.get().own.keys
        return owned_view(self, words.data(), words.size(), cnp.NPY_UINT64)

    def diagonal_view(self):

        cdef vector[double]* values = &self.rank.get().diagonal
        return owned_view(self, values.data(), values.size(), cnp.NPY_FLOAT64)

    def table_views(self):

        cdef CppTables* t = &self.rank.get().tables
        return {
            "single_value": owned_view(self, t.single_value.data(), t.single_value.size(), cnp.NPY_FLOAT64),
            "single_delta": owned_view(self, t.single_delta.data(), t.single_delta.size(), cnp.NPY_FLOAT64),
            "single_reference": owned_view(self, t.single_reference.data(), t.single_reference.size(), cnp.NPY_FLOAT64),
            "double_value": owned_view(self, t.double_value.data(), t.double_value.size(), cnp.NPY_FLOAT64),
            "mixed_value": owned_view(self, t.mixed_value.data(), t.mixed_value.size(), cnp.NPY_FLOAT64),
            "reference": (int(t.reference[0]), int(t.reference[1])),
            "num_pairs": t.num_pairs,
            "num_quads": t.num_quads,
            "num_orbitals": t.num_orbitals,
        }
