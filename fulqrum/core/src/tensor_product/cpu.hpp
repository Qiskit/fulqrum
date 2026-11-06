/**
 * This code is part of Fulqrum.
 *
 * (C) Copyright IBM 2026.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../diag.hpp"
#include "../element_evaluator.hpp"
#include "../checked_size.hpp"
#include "../operator_tables.hpp"
#include "../row_outputs.hpp"
#include "../subspace.hpp"
#include "engine.hpp"

struct TensorProductResidualGroup
{
    std::size_t term_begin;
    std::size_t term_end;
    std::size_t word_offset;
};

inline OperatorTables tensor_product_cpu_operator_tables(const QubitOperator_t& oper,
                                              const QubitOperator_t& diagonal,
                                              const std::size_t* group_ptrs,
                                              const std::vector<std::vector<width_t>>& flips,
                                              const Subspace_t& space,
                                              bool residual_only,
                                              bool force_tp)
{
    const std::size_t bytes =
        checked_add(checked_multiply(flips.size(), sizeof(GroupData)),
                       checked_multiply(oper.terms.size(),
                                        checked_add(checked_multiply(checked_multiply(3, words_for_width(oper.width)), sizeof(KeyWord)),
                                            sizeof(std::complex<double>))));
    (void)bytes;
    const HalfKeyView view(space);
    const SectorLayout layout =
        choose_sector_layout(SpinLayout::automatic, view, flips, oper.width);
    const bool table_packing = checked_add(checked_multiply(oper.terms.size(), 6),
                                              checked_multiply(flips.size(), 2)) <= UINT32_MAX;
    bool hermitian =
        flips.empty() || first_non_hermitian_group(oper, group_ptrs, flips.size()) == flips.size();
    if(!diagonal.terms.empty())
    {
        const std::size_t ptrs[2] = {0, diagonal.terms.size()};
        hermitian = hermitian && first_non_hermitian_group(diagonal, ptrs, 1) == 1;
    }
    if(force_tp && (!layout.valid || !hermitian || !table_packing))
        throw std::invalid_argument("Forced tensor-product execution requires a valid block layout "
                                    "and Hermitian operator.");
    return OperatorTables::build(
        oper, group_ptrs, flips, layout, residual_only || !hermitian || !table_packing);
}

template <typename T>
struct TensorProductCPU
{
    TensorProductEngine<T> tier;
    std::vector<TensorProductResidualGroup> residual;
    std::vector<KeyWord> group_words;
    const TermMasks* masks = nullptr;
    std::size_t tp_groups = 0;
    std::size_t bytes = 0;
};

template <typename T>
TensorProductCPU<T>* build_tensor_product_cpu(const OperatorTables& tables,
                                 const std::size_t* group_ptrs,
                                 const std::vector<std::vector<width_t>>& flips,
                                 const TermMasks& masks,
                                 const Subspace_t& space,
                                 CrossMode cross_mode,
                                 bool residual_only,
                                 bool force_tp)
{
    auto state = std::make_unique<TensorProductCPU<T>>();
    state->masks = &masks;
    const HalfKeyView view(space);
    if(!residual_only)
        state->tier = build_tensor_product_engine<T>(tables, view, cross_mode);
    if(force_tp && !state->tier.ready)
        throw std::invalid_argument("Tensor-product packing or layout is ineligible.");
    const std::size_t words = masks.num_words;
    for(std::size_t group = 0; group < flips.size(); group++)
    {
        if(state->tier.ready && !state->tier.generic_groups[group])
        {
            state->tp_groups++;
            continue;
        }
        if(group_ptrs[group] >= group_ptrs[group + 1])
            continue;
        const std::size_t offset = state->group_words.size();
        state->residual.push_back({group_ptrs[group], group_ptrs[group + 1], offset});
        state->group_words.resize(checked_add(offset, checked_multiply(3, words)), 0);
        for(const width_t bit : flips[group])
            state->group_words[offset + bit / KEY_WORD_BITS] |= KeyWord(1) << (bit % KEY_WORD_BITS);
        KeyWord* bits = state->group_words.data() + offset;
        const std::size_t stride = 3 * words;
        const KeyWord* first = masks.words.data() + group_ptrs[group] * stride;
        for(std::size_t word = 0; word < words; word++)
        {
            KeyWord tested = bits[word], differs = 0;
            for(std::size_t term = group_ptrs[group]; term < group_ptrs[group + 1]; term++)
            {
                const KeyWord* entry = masks.words.data() + term * stride;
                tested &= entry[word];
                differs |= entry[words + word] ^ first[words + word];
            }
            bits[words + word] = tested & ~differs;
            bits[2 * words + word] = first[words + word] & bits[words + word];
        }
    }
    state->bytes = checked_add(
        tensor_product_engine_bytes(state->tier),
        checked_add(checked_multiply(state->group_words.size(), sizeof(KeyWord)),
                       checked_multiply(state->residual.size(), sizeof(TensorProductResidualGroup))));
    return state.release();
}

template <typename T>
struct TensorProductCPURowWalk
{
    const TensorProductCPU<T>& state;
    HalfKeyView view;
    KeyBitset row;
    std::vector<KeyWord> candidate, alpha, beta;

    TensorProductCPURowWalk(const TensorProductCPU<T>& cache, const Subspace_t& space)
        : state(cache)
        , view(space)
        , row(view.width())
        , candidate(view.num_words())
        , alpha(view.half_words(0))
        , beta(view.half_words(1))
    {}

    template <typename Emit>
    void residual_elements(std::size_t index, Emit&& emit)
    {
        if(state.residual.empty())
            return;
        view.load_key(index, row);
        const std::size_t words = view.num_words();
        for(const TensorProductResidualGroup& group : state.residual)
        {
            const KeyWord* bits = state.group_words.data() + group.word_offset;
            KeyWord mismatch = 0;
            for(std::size_t word = 0; word < words; word++)
            {
                mismatch |= (row.m_bits[word] & bits[words + word]) ^ bits[2 * words + word];
                candidate[word] = row.m_bits[word] ^ bits[word];
            }
            if(mismatch)
                continue;
            copy_bit_range(candidate.data(), 0, view.split_bit(), alpha.data(), alpha.size());
            copy_bit_range(candidate.data(),
                           view.split_bit(),
                           view.width() - view.split_bit(),
                           beta.data(),
                           beta.size());
            const std::size_t ai = view.find_half_key(0, alpha.data());
            if(ai == KEY_NOT_FOUND)
                continue;
            const std::size_t bi = view.find_half_key(1, beta.data());
            if(bi == KEY_NOT_FOUND)
                continue;
            const T value =
                term_sum<T>(*state.masks, group.term_begin, group.term_end, row.m_bits.data());
            if(std::abs(value) > ATOL)
                emit(bi * view.num_half_keys(0) + ai, value);
        }
    }

    template <typename Emit>
    void operator()(std::size_t index, Emit&& emit)
    {
        if(state.tier.ready)
            tensor_product_row_elements(state.tier, static_cast<const T*>(nullptr), index, emit);
        residual_elements(index, emit);
    }
};

template <typename T>
void tensor_product_cpu_apply(const TensorProductCPU<T>& state,
                   const Subspace_t& space,
                   const T* diag,
                   int has_diag,
                   const T* x,
                   T* y)
{
    if(state.tier.ready)
        tensor_product_apply(state.tier, has_diag ? diag : nullptr, x, y);
    else
    {
#pragma omp parallel for schedule(static)
        for(std::size_t row = 0; row < space.size; row++)
            y[row] = has_diag ? diag[row] * x[row] : T(0);
    }
    if(state.residual.empty())
        return;
#pragma omp parallel
    {
        TensorProductCPURowWalk<T> walk(state, space);
#pragma omp for schedule(dynamic, 256)
        for(std::size_t row = 0; row < space.size; row++)
        {
            T sum = y[row];
            walk.residual_elements(row,
                                   [&](std::size_t column, T value) { sum += value * x[column]; });
            y[row] = sum;
        }
    }
}

template <typename I, typename T>
void tensor_product_cpu_csr(const TensorProductCPU<T>& state,
                 const Subspace_t& space,
                 const T* diag,
                 int has_diag,
                 I* indptr,
                 I* indices,
                 T* data,
                 int compute_values)
{
    auto make_walk = [&]() { return TensorProductCPURowWalk<T>(state, space); };
    csr_from_rows(
        space.size, has_diag ? diag : nullptr, make_walk, indptr, indices, data, compute_values);
}

template <typename T, typename I>
void tensor_product_cpu_csrlike(const TensorProductCPU<T>& state,
                     const Subspace_t& space,
                     const T* diag,
                     int has_diag,
                     std::vector<std::vector<I>>& cols,
                     std::vector<std::vector<T>>& data)
{
    auto make_walk = [&]() { return TensorProductCPURowWalk<T>(state, space); };
    csrlike_from_rows(space.size, has_diag ? diag : nullptr, make_walk, cols, data);
}

template <typename T>
void tensor_product_cpu_diagonal(const Subspace_t& space, const TermMasks& masks, bool fast, T* out)
{
    std::vector<std::size_t> ptrs(1, 0);
    if(fast)
        for(std::size_t i = 0; i < space.num_qubits; i++)
            ptrs.push_back(ptrs.back() + space.num_qubits - i);
#pragma omp parallel
    {
        KeyBitset key;
#pragma omp for schedule(static)
        for(std::size_t row = 0; row < space.size; row++)
        {
            space.load_key(row, key);
            T value;
            if(fast)
                single_bitstring_diagonal_fast(key, masks, ptrs, value);
            else
                single_bitstring_diagonal(key, masks, value);
            out[row] += value;
        }
    }
}

template <typename T>
std::size_t tensor_product_cpu_residual_groups(const TensorProductCPU<T>& state)
{
    return state.residual.size();
}
template <typename T>
bool tensor_product_cpu_tp_ready(const TensorProductCPU<T>& state)
{
    return state.tier.ready;
}
template <typename T>
bool tensor_product_cpu_cross_formed(const TensorProductCPU<T>& state)
{
    return state.tier.cross_formed;
}
