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
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "base.hpp"
#include "checked_size.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "element_evaluator.hpp"
#include "elements.hpp"
#include <boost/dynamic_bitset.hpp>

/**
 * Populate the diagonal vector for a given diagonal operator
 *
 *
 * @param data The subspace data
 * @param diag_vec The diagonal vector to store information to
 * @param val Variable storing the element value
 * @param diag_oper The diagonal operator
 * @param width The width of the operator
 * @param subspace_dim The dimension of the subspace
 */
template <typename T>
inline void compute_diag_vector(const bitset_map_namespace::BitsetHashMapWrapper& data,
                                T* __restrict diag_vec,
                                const QubitOperator& diag_oper,
                                const std::size_t subspace_dim,
                                const TermMasks* prepared_masks = nullptr)
{
    std::size_t kk;
    const TermMasks owned_masks =
        prepared_masks ? TermMasks{}
                       : build_term_masks(diag_oper.terms, words_for_width(diag_oper.width));
    const TermMasks& masks = prepared_masks ? *prepared_masks : owned_masks;
    const auto* bitsets = data.get_bitsets();

#pragma omp parallel for if(subspace_dim > 4096)
    for(kk = 0; kk < subspace_dim; kk++)
    {
        T val = 0;
        const boost::dynamic_bitset<size_t>& row = bitsets[kk].first;
        val = term_sum<T>(masks, 0, masks.real_values.size(), row.m_bits.data());
        diag_vec[kk] += val; // += here since const_energy (if any) is already included in diag
    }
}

/**
 * Compute the diagonal matrix-element for a single bit-string
 *
 *
 * @param row The row bit-string
 * @param diag_terms The diagonal operator
 * @param val Variable storing the element value
 */
template <typename T>
inline void single_bitstring_diagonal(const boost::dynamic_bitset<size_t>& row,
                                      const std::vector<OperatorTerm>& diag_terms,
                                      T& val)
{
    const TermMasks masks = build_term_masks(diag_terms, row.num_blocks());
    val = term_sum<T>(masks, 0, diag_terms.size(), row.m_bits.data());
}

/**
* Is diagonal fast_proj compatible
* Here we assume any constant offset terms are removed before this
* function is called
*
*/
inline bool fast_diag_compatible(const QubitOperator& oper)
{
    bool out = true;
    if(oper.type != 2)
    {
        out = false;
    }
    else if(!oper.is_diagonal())
    {
        out = false;
    }
    // Number of non-constant terms in diag must be W * (W + 1) / 2
    else if(oper.size() != static_cast<std::size_t>(oper.width) * (static_cast<std::size_t>(oper.width) + 1) / 2)
    {
        out = false;
    }
    else
    {
        std::size_t kk;
        for(auto term : oper.terms)
        {
            // if condition is already false, or current term has no projectors, e.g. constant term,
            // 'Z' ops only
            if(!out)
            {
                break;
            }
            else if(term.proj_indices.size() == 0)
            {
                out = false;
                break;
            }
            // All projectors must be '1' for this to work so break if '0' is found
            for(kk = 0; kk < term.proj_indices.size(); kk++)
            {
                if(term.proj_bits[kk] == 0)
                {
                    out = false;
                    break;
                }
            }
        }
    }
    return out;
}

/**
 * Compare terms based on the index of their 1st projector index, if any
 * 
 * This is used for sorting terms in a diagonal Hamiltonian so that computing
 * the energy along the diagonal is more efficient for type=2 Hamiltonians
 * 
 * If there are no projectors then the index value = -1, i.e. no projector terms
 * come first
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int proj_index_term_comp(OperatorTerm& term1, OperatorTerm& term2)
{
    int term1_index = -1;
    int term2_index = -1;
    if(term1.proj_indices.size())
    {
        term1_index = term1.proj_indices[0];
    }
    if(term2.proj_indices.size())
    {
        term2_index = term2.proj_indices[0];
    }
    return term1_index < term2_index;
}

inline int proj_second_index_term_comp(OperatorTerm& term1, OperatorTerm& term2)
{
    int term1_index = -1;
    int term2_index = -1;
    if(term1.proj_indices.size() > 1)
    {
        term1_index = term1.proj_indices[1];
    }
    if(term2.proj_indices.size() > 1)
    {
        term2_index = term2.proj_indices[1];
    }
    return term1_index < term2_index;
}

/**
* In-place sorting of terms by projector index
*/
inline QubitOperator& diag_proj_index_sort(QubitOperator& oper)
{
    if(oper.type != 2)
    {
        throw std::runtime_error("Operator must be type=2 for this to make sense");
    }
    std::sort(oper.terms.begin(), oper.terms.end(), proj_index_term_comp);
    oper.off_weight_sorted = 0;
    oper.weight_sorted = 0;
    oper.sorted = 0;
    return oper;
}

inline void fast_diag_term_sort(QubitOperator& oper)
{
    if(!oper.terms.size())
    {
        return; // return if there are no terms
    }
    if(!oper.is_diagonal())
    {
        throw std::runtime_error("Operator must be diagonal");
    }
    width_t width = oper.width;
    diag_proj_index_sort(oper);
    std::vector<std::size_t> ptrs = {0};
    std::size_t current = 0;
    for(width_t kk = 0; kk < width; kk++)
    {
        current += (width - kk);
        ptrs.push_back(current);
    }
#pragma omp parallel for if(width > 31) schedule(dynamic)
    for(std::size_t ll = 0; ll < (ptrs.size() - 1); ll++)
    {
        std::size_t start = ptrs[ll];
        std::size_t stop = ptrs[ll + 1];
        // sort by group index within the start and stop indices
        std::sort(
            oper.terms.begin() + start, oper.terms.begin() + stop, proj_second_index_term_comp);
    }
}

/**
 * Compute the diagonal matrix-element for a single bit-string
 *
 *
 * @param row The row bit-string
 * @param diag_terms The diagonal operator
 * @param val Variable storing the element value
 */
template <typename T>
inline void single_bitstring_diagonal_fast(const boost::dynamic_bitset<size_t>& row,
                                           const TermMasks& masks,
                                           const std::vector<std::size_t>& row_ptrs,
                                           T& val)
{
    val = 0;
    //const std::size_t num_terms = diag_terms.size();
    std::size_t start, offset;

    std::vector<width_t> set_bits = set_bit_indices(row);

    for(std::size_t ll = 0; ll < set_bits.size(); ll++)
    {
        width_t current_bit = set_bits[ll];
        start = row_ptrs[current_bit];
        for(std::size_t mm = ll; mm < set_bits.size(); mm++)
        {
            offset = set_bits[mm] - current_bit;
            val += term_sum<T>(masks, start + offset, start + offset + 1, row.m_bits.data());
        }
    }
}

/**
 * Incremental fast-diagonal energy col bitset. E(row) must be known.
 *
 * row and col bitsets only differs on offdiag_grp_inds positions (flip_inds here).
 * When we know the diagonal energy for a row bitset, E(row), we can incrementally
 * correct the E(row) to compute the E(col).
 * 
 * single_bitstring_diagonal_fast uses two nested loops over set bits indices (= num
 * of electrons) of a bitset, which takes nelecs * (nelecs + 1) / 2 iterations. 
 * Consider a row bitset with set bits at [0, 2, 3, 5] (nelecs = 4). E(row) iterates
 * over or has contribution from following 10 (set_bits[ll], set_bits[mm]) pairs:
 * (0, 0), (0, 2), (0, 3), (0, 5), (2, 2), (2, 3), (2, 5), (3, 3), (3, 5), and (5, 5).
 * 
 * Now, consider a group with flip_inds [2, 4]. The corresponding col for this group
 * and row will have set bits at [0, 3, 4, 5] ("1" bit at row pos 2 flips to "0", and
 * "0" bit at row pos 4 flips to "1"). E(col) needs contribution from following 10
 * pairs:
 * (0, 0), (0, 3), (0, 4), (0, 5), (3, 3), (3, 4), (3, 5), (4, 4), (4, 5), and (5, 5).
 * 
 * If we inspect pairs from E(row) and E(col), we can find three buckets of pairs:
 * A. Common in both row and col: (0, 0), (0, 3), (0, 5), (3, 3), (3, 5), and (5, 5).
 * B. In col, but not in row: (0, 4), (3, 4), (4, 4), (4, 5)
 * C. Not in col, but in row: (0, 2), (2, 2), (2, 3), (2, 5)
 * 
 * To compute E(col) by editing E(row), we need to subtract contributions from pairs
 * that are not in col but in row (Bucket # C), and add from pairs that are in col but not in row (Bucket # B).
 * 
 * E(col) = E(row) + delta
 *        = E(row) + Bucket B - Bucket C
 * 
 * When we already know E(row), we just need to compute delta to find E(col).
 * Incremental delta needs 4 pairs in Bucket B + 4 pairs in Bucket C = 8 pairs
 * instead of full 10 pairs from scratch to get E(col).
 * 
 * In general, incremental delta needs nflips * nelecs steps instead of
 * (nelecs * (nelecs + 1)) / 2 steps, which can pay more dividends when
 * nelecs is large.
 * For example, Fe4S4 has 54 electrons requiring (54 * 55) / 2 = 1485 steps per det.
 * With incremental logic that drops to 2 * 54 = 108 for single excitation groups
 * and 4 * 54 + 2 = 218 for double excitation groups.
 *
 * `row_occ` must be set_bit_indices(row) (passed in so it can be reused across all
 * connected cols of a row).`flip_inds` need not be
 * sorted; nflip must be <=4 (2 or 4).
 */
template <typename T>
inline T single_bitstring_diagonal_delta_fast(const boost::dynamic_bitset<size_t>& row,
                                              const boost::dynamic_bitset<size_t>& col,
                                              const std::vector<width_t>& row_occ,
                                              const TermMasks& masks,
                                              const std::vector<std::size_t>& row_ptrs,
                                              const width_t* flip_inds,
                                              const unsigned int nflip,
                                              const T e_row)
{
    auto g = [&](width_t a, width_t b, const boost::dynamic_bitset<size_t>& det) -> T {
        const width_t lo = a < b ? a : b;
        const width_t hi = a < b ? b : a;
        const std::size_t index = row_ptrs[lo] + (hi - lo);
        return term_sum<T>(masks, index, index + 1, det.m_bits.data());
    };

    // Split flips into positions added (0->1) and removed (1->0) going row -> col.
    // Type-2 off-diagonal groups are single (nflip==2) or double (nflip==4)
    // excitations, so na + nr == nflip <= 4 and each buffer holds at most 4.
    assert(nflip <= 4);
    width_t added[4];
    width_t removed[4];
    unsigned int na = 0, nr = 0;
    for(unsigned int f = 0; f < nflip; f++)
    {
        const width_t p = flip_inds[f];
        if(row[p])
        {
            removed[nr++] = p;
        }
        else
        {
            added[na++] = p;
        }
    }

    auto is_removed = [&](width_t y) {
        for(unsigned int i = 0; i < nr; i++)
        {
            if(removed[i] == y)
            {
                return true;
            }
        }
        return false;
    };

    T delta = 0;
    for(unsigned int i = 0; i < na; i++)
    {
        const width_t a = added[i];
        for(std::size_t t = 0; t < row_occ.size(); t++)
        {
            const width_t y = row_occ[t];
            if(!is_removed(y))
            {
                delta += g(a, y, col);
            }
        }
        for(unsigned int j = 0; j < na; j++) // added-added (incl self g(a,a))
        {
            delta += g(a, added[j], col);
        }
    }
    for(unsigned int i = 0; i < na; i++) // undo double-counted added-added pairs
    {
        for(unsigned int j = i + 1; j < na; j++)
        {
            delta -= g(added[i], added[j], col);
        }
    }

    for(unsigned int i = 0; i < nr; i++)
    {
        const width_t r = removed[i];
        for(std::size_t t = 0; t < row_occ.size(); t++)
        {
            delta -= g(r, row_occ[t], row);
        }
    }
    for(unsigned int i = 0; i < nr; i++) // undo double-counted removed-removed pairs
    {
        for(unsigned int j = i + 1; j < nr; j++)
        {
            delta += g(removed[i], removed[j], row);
        }
    }

    return e_row + delta;
}

/**
 * Compute the diagonal matrix-element for a single bit-string using fast projector mode
 *
 */
template <typename T>
inline void compute_diag_vector_fast(const bitset_map_namespace::BitsetHashMapWrapper& data,
                                     T* __restrict diag_vec,
                                     const QubitOperator& diag_oper,
                                     const std::size_t subspace_dim,
                                     const TermMasks* prepared_masks = nullptr)
{
    std::size_t kk;
    width_t width = diag_oper.width;
    const TermMasks owned_masks =
        prepared_masks ? TermMasks{}
                       : build_term_masks(diag_oper.terms, words_for_width(diag_oper.width));
    const TermMasks& masks = prepared_masks ? *prepared_masks : owned_masks;
    const auto* bitsets = data.get_bitsets();

    // set row_pointers
    std::vector<std::size_t> row_ptrs;
    row_ptrs.reserve(width + 1);
    row_ptrs.push_back(0);
    std::size_t current = 0;
    for(width_t kk = 0; kk < width; kk++)
    {
        current += (width - kk);
        row_ptrs.push_back(current);
    }

#pragma omp parallel for if(subspace_dim > 4096)
    for(kk = 0; kk < subspace_dim; kk++)
    {
        T val = 0;
        const boost::dynamic_bitset<size_t>& row = bitsets[kk].first;
        single_bitstring_diagonal_fast(row, masks, row_ptrs, val);
        diag_vec[kk] += val; // += here since const_energy (if any) is already included in diag
    }
}

template <typename T>
inline void
single_bitstring_diagonal(const boost::dynamic_bitset<size_t>& row, const TermMasks& masks, T& val)
{
    val = term_sum<T>(masks, 0, masks.real_values.size(), row.m_bits.data());
}

template <typename T>
inline void single_bitstring_diagonal_fast(const boost::dynamic_bitset<size_t>& row,
                                           const std::vector<OperatorTerm>& diag_terms,
                                           const std::vector<std::size_t>& row_ptrs,
                                           T& val)
{
    const TermMasks masks = build_term_masks(diag_terms, row.num_blocks());
    single_bitstring_diagonal_fast(row, masks, row_ptrs, val);
}

template <typename T>
inline T single_bitstring_diagonal_delta_fast(const boost::dynamic_bitset<size_t>& row,
                                              const boost::dynamic_bitset<size_t>& col,
                                              const std::vector<width_t>& row_occ,
                                              const std::vector<OperatorTerm>& diag_terms,
                                              const std::vector<std::size_t>& row_ptrs,
                                              const width_t* flip_inds,
                                              const unsigned int nflip,
                                              const T e_row)
{
    const TermMasks masks = build_term_masks(diag_terms, row.num_blocks());
    return single_bitstring_diagonal_delta_fast(
        row, col, row_occ, masks, row_ptrs, flip_inds, nflip, e_row);
}

/**
 * Count "cross" terms (those touching both alpha and beta halves) in a diagonal
 * operator. Used by the BLAS path to size the factor buffers before calling
 * compute_diag_factors().
 */
inline std::size_t count_diag_cross_terms(const QubitOperator& diag_oper,
                                          const unsigned int num_alpha)
{
    std::size_t n_cross = 0;
    for(const auto& term : diag_oper.terms)
    {
        bool has_alpha = false, has_beta = false;
        for(auto idx : term.indices)
            (idx < num_alpha) ? (has_alpha = true) : (has_beta = true);
        for(auto idx : term.proj_indices)
            (idx < num_alpha) ? (has_alpha = true) : (has_beta = true);
        if(has_alpha && has_beta)
            ++n_cross;
    }
    return n_cross;
}

/**
 * Compute the four factor arrays of the product-space diagonal decomposition.
 */
template <typename T>
inline void compute_diag_factors(const unsigned long long* __restrict alpha_uint,
                                 const unsigned long long* __restrict beta_uint,
                                 const QubitOperator& diag_oper,
                                 const unsigned int num_alpha,
                                 const std::size_t a0,
                                 const std::size_t a1,
                                 const std::size_t n_b,
                                 T* __restrict d_alpha_out,
                                 T* __restrict d_beta_out,
                                 T* __restrict Fc_out,
                                 T* __restrict G_out)
{
    if(a0 > a1) throw std::invalid_argument("Invalid diagonal alpha range.");
    const std::size_t na_local = a1 - a0;
    const std::size_t n_terms = diag_oper.terms.size();
    const std::size_t cross_count = count_diag_cross_terms(diag_oper, num_alpha);
    if(Fc_out != nullptr) checked_multiply(checked_multiply(na_local, cross_count), sizeof(T));
    if(G_out != nullptr) checked_multiply(checked_multiply(n_b, cross_count), sizeof(T));

    struct HalfInfo
    {
        std::vector<unsigned int> inds;
        std::vector<unsigned char> vals;
        std::vector<unsigned int> proj_inds;
        std::vector<unsigned int> proj_bits;
    };
    struct CrossTermInfo
    {
        double weighted_coeff;
        HalfInfo alpha_part;
        HalfInfo beta_part;
    };

    std::vector<std::size_t> alpha_term_idx, beta_term_idx;
    std::vector<CrossTermInfo> cross_terms;
    cross_terms.reserve(n_terms);

    for(std::size_t t = 0; t < n_terms; t++)
    {
        const auto& term = diag_oper.terms[t];
        bool has_alpha = false, has_beta = false;
        for(auto idx : term.indices)
            (idx < num_alpha) ? (has_alpha = true) : (has_beta = true);
        for(auto idx : term.proj_indices)
            (idx < num_alpha) ? (has_alpha = true) : (has_beta = true);

        if(!has_beta)
        {
            alpha_term_idx.push_back(t);
        }
        else if(!has_alpha)
        {
            beta_term_idx.push_back(t);
        }
        else
        {
            CrossTermInfo ct;
            ct.weighted_coeff = term.coeff.real() * static_cast<double>(term.real_phase);
            for(std::size_t k = 0; k < term.indices.size(); k++)
            {
                unsigned int idx = term.indices[k];
                unsigned char v = term.values[k];
                if(idx < num_alpha)
                {
                    ct.alpha_part.inds.push_back(idx);
                    ct.alpha_part.vals.push_back(v);
                }
                else
                {
                    ct.beta_part.inds.push_back(idx - num_alpha);
                    ct.beta_part.vals.push_back(v);
                }
            }
            for(std::size_t k = 0; k < term.proj_indices.size(); k++)
            {
                unsigned int idx = term.proj_indices[k];
                unsigned int bit = term.proj_bits[k];
                if(idx < num_alpha)
                {
                    ct.alpha_part.proj_inds.push_back(idx);
                    ct.alpha_part.proj_bits.push_back(bit);
                }
                else
                {
                    ct.beta_part.proj_inds.push_back(idx - num_alpha);
                    ct.beta_part.proj_bits.push_back(bit);
                }
            }
            cross_terms.push_back(std::move(ct));
        }
    }

    const std::size_t n_cross = cross_terms.size();

    auto eval_half = [](unsigned long long bits, const HalfInfo& hi) -> double {
        for(std::size_t k = 0; k < hi.proj_inds.size(); k++)
        {
            if(((bits >> hi.proj_inds[k]) & 1u) != hi.proj_bits[k])
                return 0.0;
        }
        double prod = 1.0;
        for(std::size_t k = 0; k < hi.inds.size(); k++)
        {
            unsigned int bit = (bits >> hi.inds[k]) & 1u;
            prod *= REAL_OPER_ELEMENTS[4u * hi.vals[k] + 3u * bit];
        }
        return prod;
    };

#pragma omp parallel for schedule(static)
    for(std::size_t ia = 0; ia < na_local; ia++)
    {
        const unsigned long long a_bits = alpha_uint[a0 + ia];
        double da = 0.0;
        for(std::size_t ti : alpha_term_idx)
        {
            const auto& term = diag_oper.terms[ti];
            bool pass = true;
            for(std::size_t k = 0; k < term.proj_indices.size(); k++)
            {
                if(((a_bits >> term.proj_indices[k]) & 1u) != (unsigned int)term.proj_bits[k])
                {
                    pass = false;
                    break;
                }
            }
            if(!pass)
                continue;
            double prod = 1.0;
            for(std::size_t k = 0; k < term.indices.size(); k++)
            {
                unsigned int bit = (a_bits >> term.indices[k]) & 1u;
                prod *= REAL_OPER_ELEMENTS[4u * term.values[k] + 3u * bit];
            }
            da += static_cast<double>(term.real_phase) * term.coeff.real() * prod;
        }
        d_alpha_out[ia] = static_cast<T>(da);
        if(n_cross > 0 && Fc_out != nullptr)
        {
            T* fc_row = Fc_out + ia * n_cross;
            for(std::size_t ct = 0; ct < n_cross; ct++)
                fc_row[ct] = static_cast<T>(cross_terms[ct].weighted_coeff *
                                            eval_half(a_bits, cross_terms[ct].alpha_part));
        }
    }

#pragma omp parallel for schedule(static)
    for(std::size_t ib = 0; ib < n_b; ib++)
    {
        const unsigned long long b_bits = beta_uint[ib];
        double db = 0.0;
        for(std::size_t ti : beta_term_idx)
        {
            const auto& term = diag_oper.terms[ti];
            bool pass = true;
            for(std::size_t k = 0; k < term.proj_indices.size(); k++)
            {
                unsigned int shifted = term.proj_indices[k] - num_alpha;
                if(((b_bits >> shifted) & 1u) != (unsigned int)term.proj_bits[k])
                {
                    pass = false;
                    break;
                }
            }
            if(!pass)
                continue;
            double prod = 1.0;
            for(std::size_t k = 0; k < term.indices.size(); k++)
            {
                unsigned int bit = (b_bits >> (term.indices[k] - num_alpha)) & 1u;
                prod *= REAL_OPER_ELEMENTS[4u * term.values[k] + 3u * bit];
            }
            db += static_cast<double>(term.real_phase) * term.coeff.real() * prod;
        }
        d_beta_out[ib] = static_cast<T>(db);
        if(n_cross > 0 && G_out != nullptr)
        {
            T* g_row = G_out + ib * n_cross;
            for(std::size_t ct = 0; ct < n_cross; ct++)
                g_row[ct] = static_cast<T>(eval_half(b_bits, cross_terms[ct].beta_part));
        }
    }
}


template <typename T>
inline bool compute_diag_cross_factors(const QubitOperator& diag_oper,
                                       const unsigned int num_alpha,
                                       const unsigned int num_beta,
                                       T* __restrict V_out,
                                       T* __restrict da_lin_out,
                                       T* __restrict db_lin_out,
                                       T* __restrict const_out)
{
    checked_multiply(checked_multiply(num_alpha, num_beta), sizeof(T));
    checked_multiply(checked_add(num_alpha, num_beta), sizeof(T));

    double c_accum = 0.0;
    for(const auto& term : diag_oper.terms)
    {
        int a_orb = -1, b_orb = -1;
        bool a_multi = false, b_multi = false;
        for(auto idx : term.indices)
            if(idx < num_alpha) { if(a_orb < 0) a_orb = (int)idx; else if(a_orb != (int)idx) a_multi = true; }
            else { int q = (int)(idx - num_alpha); if(b_orb < 0) b_orb = q; else if(b_orb != q) b_multi = true; }
        for(auto idx : term.proj_indices)
            if(idx < num_alpha) { if(a_orb < 0) a_orb = (int)idx; else if(a_orb != (int)idx) a_multi = true; }
            else { int q = (int)(idx - num_alpha); if(b_orb < 0) b_orb = q; else if(b_orb != q) b_multi = true; }

        if(a_orb < 0 || b_orb < 0)
            continue;
        if(a_multi || b_multi)
            return false;

        double a0 = 1.0, a1 = 1.0, b0 = 1.0, b1 = 1.0;
        for(std::size_t k = 0; k < term.proj_indices.size(); k++)
        {
            const unsigned int bit = term.proj_bits[k];
            if(term.proj_indices[k] < num_alpha) { if(bit != 0u) a0 = 0.0; if(bit != 1u) a1 = 0.0; }
            else { if(bit != 0u) b0 = 0.0; if(bit != 1u) b1 = 0.0; }
        }
        for(std::size_t k = 0; k < term.indices.size(); k++)
        {
            const unsigned char v = term.values[k];
            if(term.indices[k] < num_alpha)
            {
                a0 *= REAL_OPER_ELEMENTS[4u * v + 3u * 0u];
                a1 *= REAL_OPER_ELEMENTS[4u * v + 3u * 1u];
            }
            else
            {
                b0 *= REAL_OPER_ELEMENTS[4u * v + 3u * 0u];
                b1 *= REAL_OPER_ELEMENTS[4u * v + 3u * 1u];
            }
        }
        const double w = term.coeff.real() * static_cast<double>(term.real_phase);
        const double A_a = 0.5 * (a0 + a1), B_a = 0.5 * (a0 - a1);
        const double A_b = 0.5 * (b0 + b1), B_b = 0.5 * (b0 - b1);
        V_out[(std::size_t)a_orb * num_beta + (unsigned int)b_orb] += static_cast<T>(w * B_a * B_b);
        da_lin_out[a_orb] += static_cast<T>(w * B_a * A_b);
        db_lin_out[b_orb] += static_cast<T>(w * A_a * B_b);
        c_accum += w * A_a * A_b;
    }
    *const_out = static_cast<T>(c_accum);
    return true;
}
