/**
 * This code is a part of Fulqrum.
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
#include <boost/dynamic_bitset.hpp>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../core/src/base.hpp"
#include "../../core/src/bitset_hashmap.hpp"
#include "../../core/src/bitset_utils.hpp"
#include "../../core/src/constants.hpp"
#include "../../core/src/diag.hpp"
#include "../../core/src/elements.hpp"

double simple_restricted(const QubitOperator& oper,
                         const bitset_map_namespace::BitsetHashMapWrapper& restricted_subspace,
                         bitset_map_namespace::BitsetHashMapWrapper& out_subspace,
                         QubitOperator& diag_oper,
                         const width_t width,
                         const std::size_t subspace_dim,
                         const int has_nonzero_diag,
                         const std::size_t* __restrict group_ptrs,
                         const std::size_t* __restrict group_ladder_ptrs,
                         const width_t* __restrict group_rowint_length,
                         const std::vector<std::vector<width_t>>& group_offdiag_inds,
                         const std::size_t num_groups,
                         const unsigned int ladder_offset,
                         const double target_energy,
                         const unsigned int max_recursion,
                         const double tol,
                         const TermMasks* prepared_masks = nullptr,
                         const TermMasks* prepared_diag_masks = nullptr)
{
    const TermMasks owned_masks =
        prepared_masks ? TermMasks{} : build_term_masks(oper.terms, words_for_width(width));
    const TermMasks& masks = prepared_masks ? *prepared_masks : owned_masks;
    const TermMasks owned_diag_masks =
        prepared_diag_masks ? TermMasks{}
                            : build_term_masks(diag_oper.terms, words_for_width(width));
    const TermMasks& diag_masks = prepared_diag_masks ? *prepared_diag_masks : owned_diag_masks;

    std::size_t recur, kk, current;
    // do stuff if the diagonal can be evaluated quickly
    bool do_fast_diag = fast_diag_compatible(diag_oper);
    // alpha (spin-up) sector is the lower half of the bitset, beta the upper half
    const width_t half_width = width / 2;
    std::vector<std::size_t> row_ptrs;
    if(do_fast_diag)
    {
        row_ptrs.reserve(diag_oper.width + 1);
        row_ptrs.push_back(0);
        current = 0;
        for(kk = 0; kk < diag_oper.width; kk++)
        {
            current += (diag_oper.width - kk);
            row_ptrs.push_back(current);
        }
    }
    const auto* input_bitsets = restricted_subspace.get_bitsets();
    auto* output_bitsets = out_subspace.get_bitsets();

    std::vector<std::size_t> current_rows;
    std::vector<std::size_t> next_rows = {*restricted_subspace.get_ptr(output_bitsets[0].first)};

    std::vector<double> current_prefactors;
    std::vector<double> next_prefactors = {1.0 / target_energy};

    double est_energy = target_energy;
    double col_energy = 0;
    double energy_amp = 0;

    boost::dynamic_bitset<std::size_t> col_vec;
    const std::vector<width_t>* group_inds;
    std::size_t* col_ptr;
    std::size_t* out_col_ptr;
    std::size_t group;
    std::size_t group_int_start, group_int_stop;
    std::size_t num_inserted_bitsets = 1;
    std::complex<double> val = 0;
    unsigned int row_int;
    int do_col_search;

    struct Candidate
    {
        boost::dynamic_bitset<std::size_t> col_vec;
        std::size_t restricted_row;
        double est_delta;
        double next_prefactor;
    };

    for(recur = 0; recur < max_recursion; recur++)
    {
        // Set current terms from previous recursion results
        current_rows = next_rows;
        current_prefactors = next_prefactors;
        next_rows.clear();
        next_prefactors.clear();
        std::vector<std::vector<Candidate>> pending_candidates(current_rows.size());
// Loop over all rows in the current set
#pragma omp parallel for private(col_vec,                                                          \
                                     group_inds,                                                   \
                                     col_ptr,                                                      \
                                     out_col_ptr,                                                  \
                                     group,                                                        \
                                     group_int_start,                                              \
                                     group_int_stop,                                               \
                                     val,                                                          \
                                     row_int,                                                      \
                                     do_col_search,                                                \
                                     col_energy,                                                   \
                                     energy_amp) schedule(guided)
        for(kk = 0; kk < current_rows.size(); kk++)
        {
            const boost::dynamic_bitset<size_t>& row = input_bitsets[current_rows[kk]].first;
            std::vector<uint8_t> row_set_bits(row.size(), 0);
            std::vector<Candidate> row_candidates;
            bitset_to_bitvec(row, row_set_bits);

            // Row diagonal energy + occupied-orbital list, computed once per row.
            // Each in-subspace connected col's diagonal energy is then obtained
            // incrementally across the 2-/4-bit flip in O(nflip * nelec) instead of
            // the full O(nelec^2) single_bitstring_diagonal_fast sweep.
            double e_row = 0;
            std::vector<width_t> row_occ;
            if(do_fast_diag)
            {
                single_bitstring_diagonal_fast(row, diag_masks, row_ptrs, e_row);
                row_occ = set_bit_indices(row);
            }

            for(group = 0; group < num_groups; group++)
            {
                group_inds = &group_offdiag_inds[group];

                // Cheap reject (ported from csrlike_builder2.hpp): Single excitation
                // (2 inds): the two positions must differ; double excitation (4 inds):
                // exactly two of four must be occupied.
                // Rejects before the ladder lookup, col flip and probe.
                {
                    const width_t _p = (*group_inds)[0];
                    const width_t _q = (*group_inds)[1];
                    if(group_inds->size() == 2)
                    {
                        if(row_set_bits[_p] == row_set_bits[_q])
                        {
                            continue;
                        }
                    }
                    else if(group_inds->size() == 4)
                    {
                        const width_t _r = (*group_inds)[2];
                        const width_t _s = (*group_inds)[3];
                        if(_q < half_width && _r >= half_width)
                        {
                            // aabb: exactly one occupied in
                            // the alpha pair and exactly one in the beta pair.
                            if(row_set_bits[_p] + row_set_bits[_q] != 1 ||
                               row_set_bits[_r] + row_set_bits[_s] != 1)
                            {
                                continue;
                            }
                        }
                        // aaaa / bbbb: exactly two of four occupied
                        else if(row_set_bits[_p] + row_set_bits[_q] + row_set_bits[_r] +
                                    row_set_bits[_s] !=
                                2)
                        {
                            continue;
                        }
                    }
                }

                row_int = bitset_ladder_int(
                    row_set_bits.data(), group_inds->data(), group_rowint_length[group]);
                group_int_start = group_ladder_ptrs[group * ladder_offset + row_int];
                group_int_stop = group_ladder_ptrs[group * ladder_offset + row_int + 1];

                if(group_int_start >= group_int_stop)
                {
                    continue;
                }

                col_vec = row;
                flip_bits(col_vec, group_inds->data(), group_inds->size());
                col_ptr = restricted_subspace.get_ptr(col_vec);
                do_col_search = col_ptr == nullptr;
                val = do_col_search
                          ? std::complex<double>(0)
                          : term_sum<std::complex<double>>(
                                masks, group_int_start, group_int_stop, row.m_bits.data());

                if(!do_col_search)
                {
                    // This column is in the subspace: compute its diagonal energy.
                    // In delta-fast mode do it incrementally from the row's diagonal
                    // energy across the group's bit-flip; otherwise evaluate fully.
                    if(do_fast_diag)
                    {
                        col_energy = single_bitstring_diagonal_delta_fast(row,
                                                                          col_vec,
                                                                          row_occ,
                                                                          diag_masks,
                                                                          row_ptrs,
                                                                          group_inds->data(),
                                                                          group_inds->size(),
                                                                          e_row);
                    }
                    else
                    {
                        single_bitstring_diagonal(
                            input_bitsets[*col_ptr].first, diag_masks, col_energy);
                    }
                    energy_amp = current_prefactors[kk] * std::pow(std::abs(val), 2) /
                                 (target_energy - col_energy + 1e-15);
                    // If the amplitude is larger than tol
                    if(std::abs(energy_amp) > tol)
                    {
                        // if col not in out_subspace then add the column to the output subspace
                        // and add the col_ptr to the next rows array and add a new prefactor
                        // to the next_prefactors
                        out_col_ptr = out_subspace.get_ptr2(col_vec);
                        if(out_col_ptr == nullptr)
                        {
                            row_candidates.push_back(
                                {col_vec,
                                 *col_ptr,
                                 target_energy * energy_amp,
                                 energy_amp / (target_energy - col_energy + 1e-15)});
                        }
                    }
                }
            }
            pending_candidates[kk] = std::move(row_candidates);
        }

        for(kk = 0; kk < pending_candidates.size(); kk++)
        {
            for(const Candidate& candidate : pending_candidates[kk])
            {
                out_col_ptr = out_subspace.get_ptr2(candidate.col_vec);
                if(out_col_ptr == nullptr)
                {
                    est_energy += candidate.est_delta;
                    next_rows.push_back(candidate.restricted_row);
                    next_prefactors.push_back(candidate.next_prefactor);
                    out_subspace.emplace(candidate.col_vec, num_inserted_bitsets);
                    num_inserted_bitsets += 1;
                }
            }
        }
    }
    out_subspace.set_bucket_occupancy();
    return est_energy;
}
