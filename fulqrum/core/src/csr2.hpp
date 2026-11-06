/**
 * This code is part of Fulqrum.
 *
 * (C) Copyright IBM 2024.
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
#include <array>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "checked_size.hpp"
#include "external/hash_set8.hpp"

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "element_evaluator.hpp"
#include "elements.hpp"
#include "offdiag_grouping.hpp"
#include "type2_group_table.hpp"
#include <boost/dynamic_bitset.hpp>

// Builds the CSR structure for a type-2 subspace Hamiltonian.
//
// The group handling mirrors csrlike_builder2. See type2_group_table.hpp for the
// buckets and for the reason each one is correct. This file holds the row loops
// and the store action only.
//
// Called twice: compute_values == 0 counts entries per row and fills indptr;
// compute_values == 1 fills indices/data using the indptr from the first call.
// The two calls run the same code in the same order, thus the counts and the
// writes agree.
//
// T is the index type, U is the value type, e.g. (int, complex).
template <typename T, typename U>
void csr_matrix_builder2(const std::vector<OperatorTerm_t>& terms,
                         const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                         const U* __restrict diag_vec,
                         const width_t width,
                         const std::size_t subspace_dim,
                         const int has_nonzero_diag,
                         const std::size_t* __restrict group_ptrs,
                         const std::size_t* __restrict group_ladder_ptrs,
                         const width_t* __restrict group_rowint_length,
                         const std::vector<std::vector<width_t>>& group_offdiag_inds,
                         const std::size_t num_groups,
                         const unsigned int ladder_offset,
                         T* __restrict indptr,
                         T* __restrict indices,
                         U* __restrict data,
                         const int compute_values,
                         const TermMasks* prepared_masks = nullptr)
{
    if(subspace_dim) checked_index<T>(subspace_dim - 1);
    checked_add(subspace_dim, 1);
    const TermMasks owned_masks =
        prepared_masks ? TermMasks{} : build_term_masks(terms, words_for_width(width));
    const TermMasks& masks = prepared_masks ? *prepared_masks : owned_masks;

    std::size_t kk;
    std::size_t temp, _sum;

    const auto* bitsets = subspace.get_bitsets();

    // See csrlike_builder2.hpp.
    const Type2GroupTable table =
        build_type2_group_table(terms, group_ptrs, group_offdiag_inds, num_groups, width);

    // See csrlike_builder2.hpp.
    const std::size_t _ladder_len = static_cast<std::size_t>(num_groups) * ladder_offset + 1;
    const bool _ladder_fits = terms.size() <= UINT32_MAX;
    std::vector<std::uint32_t> _ladder32;
    if(_ladder_fits)
    {
        _ladder32.resize(_ladder_len);
        for(std::size_t i = 0; i < _ladder_len; ++i)
        {
            _ladder32[i] = static_cast<std::uint32_t>(group_ladder_ptrs[i]);
        }
    }

    // See csrlike_builder2.hpp.
    std::size_t BLK = 128;
    if(const char* _blk_env = std::getenv("FQ_BLK"))
    {
        long _v = std::atol(_blk_env);
        if(_v > 0)
        {
            BLK = static_cast<std::size_t>(_v);
        }
    }
    const std::size_t rsb_w = width; // one uint8 per qubit
    const std::size_t num_blocks = (subspace_dim + BLK - 1) / BLK;

    // The shared coefficient of each direct group, in the kernel value type.
    std::vector<U> direct_value(num_groups, U(0));
    for(const auto& group : table.direct_unpaired_groups)
    {
        direct_value[group] = direct_coeff_as<U>(table.direct_coeff[group]);
    }
    for(const auto& entries : table.paired_by_low_pair)
    {
        for(const auto& entry : entries)
        {
            direct_value[entry.group] = direct_coeff_as<U>(table.direct_coeff[entry.group]);
        }
    }

    const Type2RowInputs<U> inputs{terms,
                                   masks,
                                   subspace,
                                   table,
                                   bitsets,
                                   direct_value.data(),
                                   group_rowint_length,
                                   _ladder32.data(),
                                   group_ladder_ptrs,
                                   ladder_offset,
                                   _ladder_fits};

    // Prefilter for the direct paired groups. See type2_prefilter_pairs.
    emhash8::HashSet<std::uint64_t> low_half_set;
    emhash8::HashSet<std::uint64_t> high_half_set;
    if(table.has_paired_groups())
    {
        for(std::size_t s = 0; s < subspace_dim; ++s)
        {
            const auto& bs = bitsets[s].first;
            low_half_set.insert(table.half_key(bs, 0, table.split_point));
            high_half_set.insert(table.half_key(bs, table.split_point, width));
        }
    }

    // process diagonal
    std::vector<std::size_t> row_nnz_s(subspace_dim, 0);
    if(has_nonzero_diag)
    {
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
        for(kk = 0; kk < subspace_dim; kk++)
        { // begin loop over all rows
            std::size_t& row_nnz = row_nnz_s[kk]; // reference T& is critical
            T& elem_start = indptr[kk];
            // do diagonal first, if any
            if(diag_vec[kk] != 0.0)
            {
                if(compute_values)
                {
                    indices[elem_start + row_nnz] = kk;
                    data[elem_start + row_nnz] = diag_vec[kk];
                }
                row_nnz += 1;
            }
        }
    }

    // See csrlike_builder2.hpp for details.
#pragma omp parallel if(subspace_dim > 4096)
    {
        // Per-thread scratch, reused across blocks (no per-block realloc).
        std::vector<uint8_t> rsb_buf;
        std::vector<char> low_pair_ok(table.low_pairs.size());
        std::vector<char> high_pair_ok(table.high_pairs.size());
        boost::dynamic_bitset<std::size_t> col_vec;
        std::size_t col_idx;
        U val;

#pragma omp for schedule(dynamic)
        for(std::size_t blk = 0; blk < num_blocks; ++blk)
        {
            const std::size_t r0 = blk * BLK;
            const std::size_t r1 = std::min(r0 + BLK, subspace_dim);
            const std::size_t bn = r1 - r0;

            fill_block_row_bits(bitsets, r0, bn, rsb_w, rsb_buf);

            // Stores a single (row r0+row_in_block, col_idx, v) entry.
            // row_nnz_s[r0+row_in_block] and indptr[r0+row_in_block] give the CSR write position.
            auto store_element = [&](std::size_t row_in_block, std::size_t cidx, const U& v) {
                std::size_t& row_nnz = row_nnz_s[r0 + row_in_block];
                if(compute_values)
                {
                    const T elem_start = indptr[r0 + row_in_block];
                    indices[elem_start + row_nnz] = cidx;
                    data[elem_start + row_nnz] = v;
                }
                row_nnz += 1;
            };

            // Standard groups (group-outer, row-inner).
            for(const auto& g : table.standard_groups)
            {
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                {
                    const uint8_t* row_set_bits = rsb_buf.data() + row_in_block * rsb_w;
                    if(type2_standard_element(
                           inputs, r0 + row_in_block, row_set_bits, g, col_vec, col_idx, val))
                    {
                        store_element(row_in_block, col_idx, val);
                    }
                }
            }

            // Direct paired groups, ROW-major within the block to preserve the
            // per-row prefilter skip.
            if(table.has_paired_groups())
            {
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                {
                    const boost::dynamic_bitset<std::size_t>& row =
                        bitsets[r0 + row_in_block].first;
                    const uint8_t* row_set_bits = rsb_buf.data() + row_in_block * rsb_w;
                    type2_prefilter_pairs(table,
                                          row,
                                          row_set_bits,
                                          low_half_set,
                                          high_half_set,
                                          width,
                                          low_pair_ok.data(),
                                          high_pair_ok.data());

                    for(std::size_t i = 0; i < table.low_pairs.size(); ++i)
                    {
                        if(!low_pair_ok[i])
                        {
                            continue;
                        }
                        for(const auto& entry : table.paired_by_low_pair[i])
                        {
                            if(!high_pair_ok[entry.high_pair_id])
                            {
                                continue;
                            }
                            if(type2_direct_element(inputs,
                                                    r0 + row_in_block,
                                                    row_set_bits,
                                                    entry.group,
                                                    col_vec,
                                                    col_idx,
                                                    val))
                            {
                                store_element(row_in_block, col_idx, val);
                            }
                        }
                    }
                }
            }

            // Direct groups that do not split two and two (group-outer, row-inner).
            for(const auto& g : table.direct_unpaired_groups)
            {
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                {
                    const uint8_t* row_set_bits = rsb_buf.data() + row_in_block * rsb_w;
                    if(type2_direct_element(
                           inputs, r0 + row_in_block, row_set_bits, g, col_vec, col_idx, val))
                    {
                        store_element(row_in_block, col_idx, val);
                    }
                }
            }
        } // end loop over blocks
    } // end parallel region

    if(!compute_values) // Done with all rows so accumulate for correct indptr structure
    {
        _sum = 0;
        for(kk = 0; kk < (subspace_dim); kk++)
        {
            temp = checked_add(_sum, static_cast<std::size_t>(row_nnz_s[kk]));
            indptr[kk] = checked_index<T>(_sum);
            _sum = temp;
        }
        indptr[subspace_dim] = checked_index<T>(_sum);
    }
}
