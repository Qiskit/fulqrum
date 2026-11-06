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

#include "external/hash_set8.hpp"

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "csr_utils.hpp"
#include "csrlike.hpp"
#include "element_evaluator.hpp"
#include "elements.hpp"
#include "offdiag_grouping.hpp"
#include "type2_group_table.hpp"
#include <boost/dynamic_bitset.hpp>

// Builds the CSR-like structure for a type-2 subspace Hamiltonian.
//
// ``build_type2_group_table`` sorts the off-diagonal groups into buckets and
// builds the look-up tables. ``type2_standard_element`` and ``type2_direct_element``
//  evaluate one group for one row. See type2_group_table.hpp for the buckets.
//
//  (1) standard groups: ``accum_element`` evaluates the element term by term.
//  (2) direct groups: The element is ``coeff * sign``, where the sign comes from
//      the parity between two consecutive ladder operators. A direct group that
//      has two flips in each half of the bitset also uses the half-key prefilter
//      for early reject.
//
// T is the data type, U is in the index type, e.g (complex, int)
template <typename T, typename U>
void csrlike_builder2(const std::vector<OperatorTerm_t>& terms,
                      const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                      const T* __restrict diag_vec,
                      const width_t width,
                      const std::size_t subspace_dim,
                      const int has_nonzero_diag,
                      const std::size_t* __restrict group_ptrs,
                      const std::size_t* __restrict group_ladder_ptrs,
                      const width_t* __restrict group_rowint_length,
                      const std::vector<std::vector<width_t>>& group_offdiag_inds,
                      const std::size_t num_groups,
                      const unsigned int ladder_offset,
                      std::vector<std::vector<U>>& cols,
                      std::vector<std::vector<T>>& data,
                      const TermMasks* prepared_masks = nullptr)
{
    const TermMasks owned_masks =
        prepared_masks ? TermMasks{} : build_term_masks(terms, words_for_width(width));
    const TermMasks& masks = prepared_masks ? *prepared_masks : owned_masks;

    std::size_t kk;
    const auto* bitsets = subspace.get_bitsets();

    // Sort the groups into buckets and build the look-up tables. The table also
    // flattens group_offdiag_inds into one contiguous array.
    const Type2GroupTable table =
        build_type2_group_table(terms, group_ptrs, group_offdiag_inds, num_groups, width);

    // Make ladder table leaner: size_t -> uint32 once.
    // A lean table is more cache-friendly.
    // Any reasonable operator will have <= UINT32_MAX terms.
    // For the (very unlikely) > 2^32-term operator we skip the lean table
    // and read the original size_t table, so we never silently truncate an offset.
    const std::size_t _ladder_len = num_groups * ladder_offset + 1;
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

    // BLK defines number of rows that we process per group in one iter.
    // Group middle x block of rows inner iteration order cut down cache misses
    // significantly. BLK=128 is a good middle-ground.
    // A too large of BLK may overflow cache.
    // User can override it through env var FQ_BLK.
    std::size_t BLK = 128;
    if(const char* _blk_env = std::getenv("FQ_BLK"))
    {
        long _blk = std::atol(_blk_env);
        if(_blk > 0)
        {
            BLK = static_cast<std::size_t>(_blk);
        }
    }

    cols.resize(subspace_dim);
    data.resize(subspace_dim);

    // The shared coefficient of each direct group.
    std::vector<T> direct_value(num_groups, T(0));
    for(const auto& group : table.direct_unpaired_groups)
    {
        direct_value[group] = direct_coeff_as<T>(table.direct_coeff[group]);
    }
    for(const auto& entries : table.paired_by_low_pair)
    {
        for(const auto& entry : entries)
        {
            direct_value[entry.group] = direct_coeff_as<T>(table.direct_coeff[entry.group]);
        }
    }

    const Type2RowInputs<T> inputs{terms,
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

    // Populate diagonal elements first separately
    if(has_nonzero_diag)
    {
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
        for(kk = 0; kk < subspace_dim; kk++)
        { // begin loop over all rows

            if(diag_vec[kk] != 0.0)
            {
                cols[kk].push_back(kk);
                data[kk].push_back(diag_vec[kk]);
            }
        }
    }

    const std::size_t rsb_w = width; // one uint8 per qubit
    const std::size_t num_blocks = (subspace_dim + BLK - 1) / BLK;

    // New loop order: block-of-rows outer, group middle, row inner
    // for block_of_rows in rows:
    //   for g in groups:
    //     for row in block_of_rows:
    //       ...
    // In group_ladder_ptrs[g * ladder_offset + row_int] lookup, row_int varies only over [0, ladder_offset - 1]
    // (= [0,3] if ladder_width=2, [0,15] if ladder_width=4). For a fixed g, the first
    // row in the block of rows fetches a cache-line of group_ladder_ptrs[] that can be resued by remaining rows
    // while being cache resident.
#pragma omp parallel if(subspace_dim > 4096)
    {
        // Per-thread, reused across blocks.
        std::vector<uint8_t> rsb_buf;
        std::vector<char> low_pair_ok(table.low_pairs.size());
        std::vector<char> high_pair_ok(table.high_pairs.size());
        boost::dynamic_bitset<std::size_t> col_vec;
        std::size_t col_idx;
        T val;

#pragma omp for schedule(dynamic)
        for(std::size_t blk = 0; blk < num_blocks; ++blk)
        {
            const std::size_t r0 = blk * BLK;
            const std::size_t r1 = std::min(r0 + BLK, subspace_dim);
            const std::size_t bn = r1 - r0;

            fill_block_row_bits(bitsets, r0, bn, rsb_w, rsb_buf);

            // Standard groups.
            for(const auto& g : table.standard_groups)
            {
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                {
                    const uint8_t* row_set_bits = rsb_buf.data() + row_in_block * rsb_w;
                    if(type2_standard_element(
                           inputs, r0 + row_in_block, row_set_bits, g, col_vec, col_idx, val))
                    {
                        cols[r0 + row_in_block].push_back(col_idx);
                        data[r0 + row_in_block].push_back(val);
                    }
                }
            }

            // Direct paired groups.
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
                                cols[r0 + row_in_block].push_back(col_idx);
                                data[r0 + row_in_block].push_back(val);
                            }
                        }
                    }
                }
            }

            // Direct groups that do not split two and two.
            for(const auto& g : table.direct_unpaired_groups)
            {
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                {
                    const uint8_t* row_set_bits = rsb_buf.data() + row_in_block * rsb_w;
                    if(type2_direct_element(
                           inputs, r0 + row_in_block, row_set_bits, g, col_vec, col_idx, val))
                    {
                        cols[r0 + row_in_block].push_back(col_idx);
                        data[r0 + row_in_block].push_back(val);
                    }
                }
            }
        }
    }

    sort_paired(cols, data);
} // end function
