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
#include <boost/sort/pdqsort/pdqsort.hpp>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.hpp"
#include "io.hpp"
#include "oper_utils.hpp"
#include "qubit_term.hpp"

struct QubitOperator;

// forward definitions
void set_sorting_flags(QubitOperator& oper, std::string kind);
inline void term_offdiag_sort(QubitOperator& oper);

/**
 * Comparator for off-diagonal weight grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int offweight_comp(OperatorTerm_t& term1, OperatorTerm_t& term2)
{
    return term1.offdiag_weight < term2.offdiag_weight;
}

inline void set_group_ptrs(const std::vector<OperatorTerm>& __restrict terms,
                           std::vector<std::size_t>& vec)
{
    vec.clear();
    vec.push_back(0);
    std::size_t kk;
    int val = terms[0].group;
    for(kk = 1; kk < terms.size(); kk++)
    {
        if(terms[kk].group > val)
        {
            vec.push_back(kk);
            val = terms[kk].group;
        }
    }
    vec.push_back(terms.size());
}

/**
 * Set the pointers for the off-diagonal weights
 *
 * @param terms Operator terms
 * @param vec Vector to add pointers to
 *
 */
inline void set_offdiag_weight_ptrs(const std::vector<OperatorTerm>& __restrict terms,
                                    std::vector<std::size_t>& vec)
{
    vec.clear();
    std::size_t kk;
    width_t val = terms[0].offdiag_weight;
    vec.push_back(0);
    for(kk = 1; kk < terms.size(); kk++)
    {
        if(terms[kk].offdiag_weight > val)
        {
            vec.push_back(kk);
            val = terms[kk].offdiag_weight;
        }
    }
    vec.push_back(terms.size());
}

/**
 * Find max. number of elements with same off-diag weight
 *
 * Used for offsetting the group counter for parallel execution
 *
 * @param[in] vec Vector of off-diagonal pointers
 * @param[in] size Number of elements in vec
 * 
 * @returns size_t for max. number of terms
 *
 */
inline std::size_t max_offdiag_ptr_size(std::vector<std::size_t>& vec)
{
    std::size_t kk;
    std::size_t temp, max = 0;
    if(!vec.size()) // This is the case for all diagonals operator
    {
        max = 0;
    }
    else
    {
        for(kk = 0; kk < vec.size() - 1; kk++)
        {
            temp = vec[kk + 1] - vec[kk];
            if(temp > max)
            {
                max = temp;
            }
        }
    }
    return max;
}

// Reverse mask for marking terms as extended or not
// Z, 0, 1, X, Y, -, +
const int REV_EXT_MASK[7] = {1, 0, 0, 1, 1, 0, 0};


/**
 * Comparator for term grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int offdiag_group_comp(OperatorTerm_t& term1, OperatorTerm_t& term2)
{
    return term1.group < term2.group;
}

/**
 * Sort terms with same off-diagonal weight into groups that share the
 * same off-diagonal structure
 *
 */
inline void term_group_sort(std::vector<OperatorTerm_t>& terms,
                            std::size_t* __restrict struct_ptrs,
                            std::size_t len_ptrs,
                            std::size_t max_group_size)
{
    std::size_t ii;
#pragma omp parallel if(terms.size() > 1024)
    {
// Reset all groupings
#pragma omp for schedule(dynamic)
        for(ii = 0; ii < terms.size(); ii++)
        {
            terms[ii].group = 0; // diagonals are group 0 by convention
            if(terms[ii].offdiag_weight > 0)
            {
                terms[ii].group = -1;
            }
        } // end reset

#pragma omp for schedule(dynamic)
        for(ii = 0; ii < len_ptrs - 1; ii++)
        {
            std::size_t start = struct_ptrs[ii];
            std::size_t stop = struct_ptrs[ii + 1];
            int group_idx = static_cast<int>(ii * max_group_size);

            if(terms[start].group == 0) // group is the diagonal group
            {
                continue;
            }

            std::map<std::vector<width_t>, int> pattern_to_group;

            for(std::size_t kk = start; kk < stop; kk++)
            {
                OperatorTerm_t* term = &terms[kk];

                // Build canonical key: sorted off-diagonal qubit indices
                // (indices are already stored in ascending order, so no sort needed)
                std::vector<width_t> key;
                key.reserve(term->offdiag_weight);
                for(std::size_t idx = 0; idx < term->values.size(); idx++)
                    if(term->values[idx] > 2)
                        key.push_back(term->indices[idx]);

                auto result = pattern_to_group.emplace(key, 0);
                if(result.second) // new pattern: allocate a new group
                {
                    group_idx += 1;
                    result.first->second = group_idx;
                }
                term->group = result.first->second;
            } // end kk loop

            // sort by group index within the start and stop indices
            boost::sort::pdqsort(terms.begin() + start, terms.begin() + stop, offdiag_group_comp);
        } // end ii loop

    } // end omp parallel

    // relabel groups into continuous integers
    int current_group = 0;
    int current_idx = 0, next_idx = 1;
    for(ii = 0; ii < terms.size(); ii++)
    {
        if(terms[ii].group != current_group)
        {
            current_group = terms[ii].group;
            current_idx = next_idx;
            terms[ii].group = current_idx;
            next_idx += 1;
        }
        else
        {
            terms[ii].group = current_idx;
        }
    }
}

/**
 * Compute the ladder integer value for a given qubit term
 *
 */
inline width_t term_ladder_int(const OperatorTerm& term, width_t ladder_width)
{
    width_t subset = 0;
    width_t kk, counter = 0;
    for(kk = 0; kk < term.indices.size(); kk++)
    {
        if(term.values[kk] > 4)
        {
            subset = subset | ((width_t)term.values[kk] - (width_t)5) << counter;
            counter += 1;
        }
    }
    if(counter < ladder_width)
    {
        ladder_width = counter;
    }
    if(!counter)
    {
        subset = MAX_WIDTH;
    }
    else
    {
        subset = subset & (((width_t)1 << ladder_width) - (width_t)1);
    }
    return subset;
}

/**
 * Sort terms within each group by their ladder integer values
 *
 */
inline void sort_groups_by_ladder_int(std::vector<OperatorTerm>& terms,
                                      const std::size_t* group_ptrs,
                                      std::size_t num_groups,
                                      width_t ladder_width)
{

    std::size_t kk;
#pragma omp parallel for if(num_groups > 128)
    for(kk = 0; kk < num_groups; kk++)
    {
        std::size_t start, stop;
        start = group_ptrs[kk];
        stop = group_ptrs[kk + 1];
        if(!terms[start].group) // This is true if the group=0 and thus are diagonal terms
        {
            continue;
        }
        std::sort(terms.begin() + start,
                  terms.begin() + stop,
                  [=](const OperatorTerm& a, const OperatorTerm& b) {
                      width_t res_a, res_b;
                      res_a = term_ladder_int(a, ladder_width);
                      res_b = term_ladder_int(b, ladder_width);
                      return res_a < res_b;
                  });
    }
}

/**
 * Compute the offdiag indices for the first term in a group and add it to the group
 * offdiag indices vector
 *
 * @param term Operator term
 * @param ladder_inds Pre-sized array (size=num_inds) to store indices in
 * @param num_inds Number of elements to consider for appending
 *
 */
inline void compute_term_offdiag_inds(const OperatorTerm_t& term, width_t* offdiag_inds)
{
    std::size_t kk;
    unsigned int counter = 0;
    for(kk = 0; kk < term.indices.size(); kk++)
    {
        if(term.values[kk] > 2)
        {
            offdiag_inds[counter] = term.indices[kk];
            counter += 1;
        }
    }
}

/**
 * Set the offdiag indices for each group in a off-diagonal type=2 Hamiltonian
 *
 * @param terms Operator terms
 * @param group_indices Vector of vectors of group_indices
 * @param group_ptrs Pointer of array of group pointers
 * @param num_groups Number of groups = len(group_ptrs) - 1
 * @param ladder_width Target ladder indices width for type=2 operators
 * @param oper_type Type of operator, 1 or 2
 *
 */
inline void set_group_offdiag_indices(const std::vector<OperatorTerm_t>& terms,
                                      std::vector<std::vector<width_t>>& group_indices,
                                      const std::size_t* group_ptrs,
                                      std::size_t num_groups)
{
    std::size_t kk;
    width_t inds_len;
    group_indices.resize(num_groups);
#pragma omp parallel for if(num_groups > 4096)
    for(kk = 0; kk < num_groups; kk++)
    {
        inds_len = terms[group_ptrs[kk]].offdiag_weight;
        group_indices[kk].resize(inds_len);
        compute_term_offdiag_inds(terms[group_ptrs[kk]], &(group_indices[kk])[0]);
    }
}

/**
 * Term indices corresponding to ladder integers for each group
 *
 */
inline void ladder_int_starts(const std::vector<OperatorTerm>& terms,
                              const std::size_t* group_ptrs,
                              unsigned int* group_counts,
                              std::size_t* group_ranges,
                              std::size_t num_groups,
                              width_t num_ladder_bins,
                              width_t ladder_width)
{
    std::size_t start, stop, kk, mm;
    width_t term_int;
    std::size_t total;
    total = 0;
    for(kk = 0; kk < num_groups; kk++)
    {
        start = group_ptrs[kk];
        stop = group_ptrs[kk + 1];

        for(mm = start; mm < stop; mm++)
        {
            term_int = term_ladder_int(terms[mm], ladder_width);
            group_counts[kk * num_ladder_bins + term_int] += 1;
        }
        group_ranges[kk * num_ladder_bins] = total;
        for(mm = 1; mm < static_cast<std::size_t>(num_ladder_bins + 1); mm++)
        {
            total += group_counts[kk * num_ladder_bins + mm - 1];
            group_ranges[kk * num_ladder_bins + mm] = total;
        }
    }
}

inline void check_ladder_bins(const std::vector<OperatorTerm>& terms,
                              const std::size_t* group_ptrs,
                              const std::size_t num_groups,
                              const std::size_t* ladder_ptrs,
                              const std::size_t num_ladder_ptrs,
                              const std::size_t ladder_offset)
{
    const std::string fix = " Sort the terms with group_term_sort_by_ladder_int and rebuild the bins.";

    if(ladder_offset == 0 || (ladder_offset & (ladder_offset - 1)) != 0 ||
       num_ladder_ptrs != num_groups * ladder_offset + 1 || ladder_ptrs[num_groups * ladder_offset] != terms.size())
    {
        throw std::invalid_argument("The ladder bins do not have the size of the operator's groups and terms." + fix);
    }

    const width_t ladder_width = static_cast<width_t>(__builtin_ctzll(ladder_offset));

    for(std::size_t group = 0; group < num_groups; group++)
    {
        if(group_ptrs[group] > group_ptrs[group + 1] || group_ptrs[group + 1] > terms.size() ||
           ladder_ptrs[(group + 1) * ladder_offset] != group_ptrs[group + 1] ||
           ladder_ptrs[group * ladder_offset] != group_ptrs[group])
        {
            throw std::invalid_argument("The ladder bins do not start at the operator's groups." + fix);
        }

        for(std::size_t bin = 0; bin < ladder_offset; bin++)
        {
            const std::size_t first = ladder_ptrs[group * ladder_offset + bin];
            const std::size_t last = ladder_ptrs[group * ladder_offset + bin + 1];
            if(first > last || last > group_ptrs[group + 1])
                throw std::invalid_argument("The ladder bins are not ordered within the operator's groups." + fix);

            for(std::size_t term = first; term < last; term++)
            {
                if(term >= terms.size() || term_ladder_int(terms[term], ladder_width) != bin)
                {
                    throw std::invalid_argument("A term is not in the bin of its ladder integer." + fix);
                }
            }
        }
    }
}

/** @struct QubitOperator
 * @brief Data structure for each a qubit operator, i.e. a collection of 'words'
 *
 * @var width is the number of qubits
 * @var terms is a vector of OperatorTerms that make up the operator
 * @var sorted is a flag that indicates the term is sorted (NOT USED AT PRESENT)
 */
typedef struct QubitOperator
{
    width_t width;
    std::vector<OperatorTerm_t> terms;
    int type{1};
    width_t ladder_width{DEFAULT_LADDER_WIDTH};
    int sorted{0}; // Are the operator terms group sorted
    int weight_sorted{0}; // Are the operator terms weight sorted
    int off_weight_sorted{0}; // Are the operator terms off-diagonal weight sorted
    int ladder_sorted{0}; // Are the operator terms ladder int sorted within their groups?
    int structure_sorted{
        0}; // Are the operator terms sorted by (non-unique) off-diagonal structure?

    QubitOperator() = default;
    QubitOperator(const QubitOperator&) = default;
    QubitOperator(QubitOperator&&) = default;
    QubitOperator& operator=(const QubitOperator&) = default;
    QubitOperator& operator=(QubitOperator&&) = default;
    // destructor
    ~QubitOperator() = default;

    /**
     * Constructor building an empty operator with a given width
     *
     * @param[in] width The width (number of qubits) of the operator
     */
    QubitOperator(width_t x)
    {
        width = x;
    }

    QubitOperator(width_t x, std::vector<TermData> data)
        : width(x)
    {
        std::size_t num_terms = data.size();
        std::size_t kk;
        std::complex<double> coeff = 1.0;
        terms.reserve(num_terms);
        for(kk = 0; kk < num_terms; kk++)
        {
            const TermData& tdata = data[kk];
            _validate_indices(std::get<1>(tdata),
                              width); // validate that all indices are less than operator width
            // If there are no indices and the coeff==0 then the term should be an identity term with coeff=1
            if(std::get<1>(tdata).size() == 0 && std::get<2>(tdata) == std::complex<double>(0, 0))
            {
                coeff = 1.0;
            }
            else
            {
                coeff = std::get<2>(tdata);
            }
            OperatorTerm term(std::get<0>(tdata), std::get<1>(tdata), coeff);
            term.set_proj_indices();
            set_offdiag_weight_phase_struct(term);
            terms.push_back(std::move(term));
        }
    }
    /**
     * QubitOperator from string label
     */
    static QubitOperator from_label(std::string label)
    {
        width_t width = static_cast<width_t>(label.size());
        unsigned char val;
        std::size_t counter = 0;
        QubitOperator out = QubitOperator(width);
        OperatorTerm term = OperatorTerm(1.0); // start with term set with coeff = 1.0
        for(auto it = label.rbegin(); it != label.rend(); it++)
        {
            if(*it != 73)
            {
                val = oper_map[*it];
                term.values.push_back(val);
                term.indices.push_back(counter);
            }
            counter += 1;
        }
        set_offdiag_weight_phase_struct(term);
        term.set_proj_indices();
        out.terms.push_back(term);
        return out;
    }
    /**
     * Grab a single term by index
     * 
     * @param[in] Index of term to grab
     * 
     * @return OperatorTerm at the given index
     */
    OperatorTerm_t operator[](std::size_t index) const
    {
        if(index >= this->size())
        {
            throw std::runtime_error("Index is larger than operator size");
        }
        return terms[index];
    }
    /**
     * Inplace multiplication by a complex value
     */
    QubitOperator& operator*=(std::complex<double> c)
    {
        for(auto& term : terms)
            term *= c;
        return *this;
    }
    /**
     * multiplication by a complex value (need one for mult on each side)
     */
    friend QubitOperator operator*(QubitOperator op, std::complex<double> c)
    {
        op *= c;
        return op;
    }
    friend QubitOperator operator*(std::complex<double> c, QubitOperator op)
    {
        op *= c;
        return op;
    }
    /**
     * Inplace addition by another QubitOperator
     *
     * @param[in] other Operator to add to this one
     * @throw Error if operators do not share the same width
     */
    QubitOperator& operator+=(const QubitOperator& other)
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        this->terms.insert(this->terms.end(), other.terms.begin(), other.terms.end());
        this->sorted = 0;
        return *this;
    }
    /**
     * Inplace subtraction by another QubitOperator
     *
     * @param[in] other Operator to subtract from this one
     * @throw Error if operators do not share the same width
     */
    QubitOperator& operator-=(const QubitOperator& other)
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        std::size_t old_size = this->terms.size();
        this->terms.insert(this->terms.end(), other.terms.begin(), other.terms.end());
        for(std::size_t kk = old_size; kk < this->terms.size(); kk++)
            this->terms[kk].coeff *= -1;
        this->sorted = 0;
        return *this;
    }
    /**
     * Subtraction by another QubitOperator
     *
     * @param[in] other Operator to subtract from this one
     * @return New operator
     * @throw Error if operators do not share the same width
     */
    QubitOperator operator-(const QubitOperator& other) const
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        QubitOperator out = *this;
        out -= other;
        return out;
    }
    /**
     * Addition by another QubitOperator
     *
     * @param[in] other Operator to add to this one
     * @return The new operator
     * @throw Error if operators do not share the same width
     */
    QubitOperator operator+(const QubitOperator& other) const
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        QubitOperator out = *this;
        out += other;
        return out;
    }
    /**
     * Print object to standard output stream
     */
    friend auto operator<<(std::ostream& os, const QubitOperator& self) -> std::ostream&
    {
        std::size_t num_terms = self.size();
        std::size_t total_terms = num_terms;
        OperatorTerm_t term;
        int too_many_terms = 0;
        std::size_t kk, jj;

        // restrict to outputting at most 100 terms
        if(num_terms > 100)
        {
            too_many_terms = 1;
            num_terms = 100;
        }
        os << "<QubitOperator["; // start output here
        for(kk = 0; kk < num_terms; kk++)
        {
            term = self.terms[kk];
            os << "{";
            for(jj = 0; jj < term.indices.size(); jj++)
            {
                os << rev_oper_map[term.values[jj]] << ":" << term.indices[jj];
                if(jj != term.indices.size() - 1)
                {
                    os << " ";
                }
            }
            os << ", " << term.coeff;
            os << "}";
            if(kk != num_terms - 1)
            {
                os << ", ";
            }
        }
        if(too_many_terms)
        {
            os << " + " << (total_terms - 100) << "terms";
        }
        return os << ", width=" << self.width << "]>";
    }
    auto begin()
    {
        return terms.begin();
    }
    auto end()
    {
        return terms.end();
    }
    /**
     * The number of terms in the operator
     *
     * @return The number of terms in the operator
     */
    std::size_t size() const
    {
        return terms.size();
    }
    std::size_t num_terms() const
    {
        return terms.size();
    }
    /**
     * The number of terms in the operator
     *
     * @return A copy of the current operator
     */
    QubitOperator copy() const
    {
        return *this;
    }
    /**
     * Is the operator diagonal
     */
    bool is_diagonal() const
    {
        return std::all_of(
            terms.begin(), terms.end(), [](const OperatorTerm_t& t) { return t.is_diagonal(); });
    }
    /**
     * Can operator be described via a symmetric matrix
     */
    bool is_real() const
    {
        bool out = true;
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            if(std::abs(terms[kk].coeff.imag()) > ATOL || !terms[kk].real_phase)
            {
                out = false;
                break;
            }
        }
        return out;
    }
    /**
     * Set operator type inplace
     * 
     * @param[in] x Integer type of operator
     * @throw Error if type is not 1 or 2
     */
    QubitOperator& set_type(int x)
    {
        if(x > 2 || x < 1)
        {
            throw std::runtime_error("Type must be 1 or 2");
        }
        this->type = x;
        return *this;
    }
    /**
    * Return vector of coefficients for each term
    * 
    * @return Vector of coefficients for terms
    */
    std::vector<std::complex<double>> coefficients() const
    {
        std::vector<std::complex<double>> out;
        out.reserve(this->size());
        for(const auto& t : terms)
            out.push_back(t.coeff);
        return out;
    }
    /**
    * Return vector of weights for each term
    *
    * @return Vector of weights for terms
    *
    */
    std::vector<width_t> weights() const
    {
        std::vector<width_t> out;
        out.reserve(this->size());
        for(const auto& t : terms)
            out.push_back(t.weight());
        return out;
    }
    /**
    * Return vector of real-phases for each term
    *
    * @return Vector of real-phases for terms
    */
    std::vector<int8_t> real_phases() const
    {
        std::vector<int8_t> out;
        out.reserve(this->size());
        for(const auto& t : terms)
            out.push_back(t.real_phase);
        return out;
    }
    /**
     * Split operator into diagonal and off-diagonal components
     * 
     * @return Diagonal and off-diagonal operators
     */
    std::pair<QubitOperator, QubitOperator> split_diagonal() const
    {
        QubitOperator diag = QubitOperator(this->width);
        QubitOperator off = QubitOperator(this->width);
        for(auto term : this->terms)
        {
            if(!term.offdiag_weight)
                diag.terms.push_back(term);
            else
                off.terms.push_back(term);
        }
        off.type = this->type;
        diag.type = this->type;
        return {diag, off};
    }
    /**Constant energy of operator
    * 
    */
    double constant_energy() const
    {
        double out = 0;
        for(std::size_t kk = 0; kk < terms.size(); kk++)
        {
            if(!terms[kk].indices.size())
            {
                out += terms[kk].coeff.real();
            }
        }
        return out;
    }
    /**
    * Remove constant terms from operator
    * 
    */
    QubitOperator remove_constant_terms()
    {
        QubitOperator out = QubitOperator(this->width);
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            if(terms[kk].indices.size())
            {
                out.terms.push_back(terms[kk]);
            }
        }
        out.type = this->type;
        return out;
    }
    /**
    * In-place sorting of terms by weight
    * 
    */
    QubitOperator& weight_sort()
    {
// sort by weight
#ifdef FQ_TBB
        tbb::parallel_sort(terms.begin(), terms.end(), [&](OperatorTerm term1, OperatorTerm term2) {
            return term1.indices.size() < term2.indices.size();
        });
#else
        boost::sort::pdqsort(
            terms.begin(), terms.end(), [&](OperatorTerm term1, OperatorTerm term2) {
                return term1.indices.size() < term2.indices.size();
            });
#endif
        set_sorting_flags(*this, "weight");
        return *this;
    }
    /**
    * In-place sorting of terms by off-diagonal weight
    * 
    */
    QubitOperator& offdiag_weight_sort()
    {
        // sort by off-diagonal weight
        boost::sort::pdqsort(terms.begin(), terms.end(), offweight_comp);
        set_sorting_flags(*this, "off_weight");
        return *this;
    }
    /**
    * Pointers to starting indices for off-diagonally sorted operator
    * 
    */
    std::vector<std::size_t> offdiag_weight_ptrs()
    {
        std::vector<std::size_t> ptrs;
        if(!this->off_weight_sorted)
        {
            this->offdiag_weight_sort();
        }
        set_offdiag_weight_ptrs(terms, ptrs);
        return ptrs;
    }
    /**
    * In-place sorting of terms by offdiag structure
    */
    QubitOperator& offdiag_structure_sort()
    {
        if(!(this->structure_sorted))
        {
// sort by weight
#ifdef FQ_TBB
            tbb::parallel_sort(terms.begin(),
                               terms.end(),
                               [](const OperatorTerm& term1, const OperatorTerm& term2) {
                                   return term1.offdiag_structure < term2.offdiag_structure;
                               });
#else
            boost::sort::pdqsort(terms.begin(),
                                 terms.end(),
                                 [](const OperatorTerm& term1, const OperatorTerm& term2) {
                                     return term1.offdiag_structure < term2.offdiag_structure;
                                 });
#endif
            set_sorting_flags(*this, "structure");
        }
        return *this;
    }
    /**
    * Pointers to starting indices for off-diagonally sorted operator
    * 
    */
    std::vector<std::size_t> offdiag_structure_ptrs()
    {
        std::vector<std::size_t> ptrs;
        if(!this->structure_sorted)
        {
            this->offdiag_structure_sort();
        }
        set_offdiag_structure_ptrs(terms, ptrs);
        return ptrs;
    }
    /**
    * In-place sorting of terms into groups (shared off-diagonal structure)
    * 
    */
    QubitOperator& group_sort()
    {
        if(this->size()) // do stuff only if there are terms in the operator
        {
            if(!this->structure_sorted)
            {
                this->offdiag_structure_sort();
            }
            std::vector<std::size_t> ptrs = this->offdiag_structure_ptrs();
            std::size_t max_group_size = max_offdiag_ptr_size(ptrs);
            term_group_sort(this->terms, &ptrs[0], ptrs.size(), max_group_size);
        }
        set_sorting_flags(*this, "group");
        return *this;
    }
    /**
    * Return a vector of all the term group labels
    * 
    */
    std::vector<int> groups() const
    {
        std::vector<int> out;
        out.resize(terms.size());
        for(std::size_t kk = 0; kk < terms.size(); kk++)
        {
            out[kk] = terms[kk].group;
        }
        return out;
    }
    /**
    * Return a vector of pointers to all the groups
    * 
    */
    std::vector<std::size_t> group_ptrs()
    {
        std::vector<std::size_t> out;
        if(!this->size()) // return empty vector if no terms
        {
            return out;
        }
        if(!this->sorted)
        {
            this->group_sort();
        }
        set_group_ptrs(terms, out);
        return out;
    }
    /**
    * Return a vector of pointers to all the groups
    * 
    */
    QubitOperator terms_by_group(int idx)
    {
        if(!this->sorted)
        {
            throw std::runtime_error("Operator must be group sorted first");
        }
        QubitOperator out = QubitOperator(this->width);
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            if(terms[kk].group == idx)
            {
                out.terms.push_back(terms[kk]);
            }
            else if(terms[kk].group > idx)
            {
                break;
            }
        }
        if(!out.size())
        {
            throw std::runtime_error("No terms with given group index found");
        }
        out.sorted = 1;
        out.type = this->type;
        return out;
    }
    /**
    * Off-diagonal indices for each group of terms
    */
    std::vector<std::vector<width_t>> group_offdiag_indices()
    {
        if(!this->sorted)
        {
            throw std::runtime_error("Operator must be group sorted first");
        }

        std::vector<std::vector<width_t>> out;
        std::vector<std::size_t> ptrs = this->group_ptrs();
        set_group_offdiag_indices(this->terms, out, &ptrs[0], ptrs.size() - 1);
        return out;
    }
    /**
    * Combine repeated terms in operator
    * 
    * @param[in] atol Tolerance for determining if a combined coefficient is zero
    * 
    * @return Output QubitOperator with terms combined
    * 
    */
    QubitOperator combine_repeat_terms(double atol = 1e-12)
    {
        QubitOperator out = QubitOperator(this->width);
        if(!this->size())
        {
            return out;
        }
        if(!this->weight_sorted)
        {
            this->weight_sort();
        }
        std::vector<std::size_t> weight_ptrs;
        set_weight_ptrs(terms, weight_ptrs);
        combine_terms(this->terms, out.terms, weight_ptrs, atol);
        out.type = this->type;
        return out;
    }
    /**
    * Return vector of off-diagonal weights for each term
    * 
    * @return Vector of off-diagonal weights for terms
    * 
    */
    std::vector<width_t> offdiag_weights() const
    {
        std::vector<width_t> out;
        out.resize(terms.size());
        for(std::size_t kk = 0; kk < terms.size(); kk++)
            out[kk] = terms[kk].offdiag_weight;
        return out;
    }
    /**
    * In-place sort terms in groups by their ladder integer values
    * 
    */
    QubitOperator& group_term_sort_by_ladder_int(width_t ladder_width = 4)
    {
        if(!(this->type == 2))
        {
            throw std::runtime_error("Operator must be type=2");
        }
        if(!this->sorted)
        {
            this->group_sort();
        }
        std::vector<std::size_t> ptrs = this->group_ptrs();
        sort_groups_by_ladder_int(this->terms, &ptrs[0], ptrs.size() - 1, ladder_width);
        this->ladder_width = ladder_width;
        this->ladder_sorted = 1;
        return *this;
    }
    /**
    * Vector of ladder integer values for terms in operators
    * 
    * If no ladder ops present then default int is max(uint32)
    * 
    */
    std::vector<width_t> ladder_integers()
    {
        std::vector<width_t> out;
        if(!this->ladder_sorted)
            this->group_term_sort_by_ladder_int();
        out.reserve(this->size());
        for(std::size_t kk = 0; kk < this->size(); kk++)
            out.push_back(term_ladder_int(terms[kk], this->ladder_width));
        return out;
    }
    /**
    * Vector of group ladder integer bit lengths
    * 
    */
    std::vector<width_t> group_ladder_int_bit_lengths()
    {
        std::vector<width_t> out;
        if(!this->ladder_sorted)
        {
            this->group_term_sort_by_ladder_int();
        }
        std::vector<std::size_t> ptrs = this->group_ptrs();
        std::size_t num_groups = ptrs.size() - 1;
        out.resize(num_groups);
        for(std::size_t kk = 0; kk < num_groups; kk++)
        {
            out[kk] = std::min(this->ladder_width, terms[ptrs[kk]].offdiag_weight);
        }
        return out;
    }
    /**
    * Flat vector containing pointers to terms within a group with given 
    * ladder integer value
    * 
    */
    std::vector<std::size_t> group_ladder_int_ptrs()
    {
        if(!this->ladder_sorted)
        {
            this->group_term_sort_by_ladder_int();
        }
        std::vector<std::size_t> ptrs = this->group_ptrs();
        width_t num_ladder_ints = static_cast<width_t>(std::pow(2, this->ladder_width));
        std::size_t num_groups = ptrs.size() - 1;
        std::vector<unsigned int> group_counts(num_ladder_ints * num_groups);
        std::vector<std::size_t> group_ranges(num_ladder_ints * num_groups + 1);
        ladder_int_starts(terms,
                          &ptrs[0],
                          &group_counts[0],
                          &group_ranges[0],
                          num_groups,
                          num_ladder_ints,
                          ladder_width);
        return group_ranges;
    }
    /**
     * Convert operator to JSON format, optionally with XZ or ZST compression
     *
     * @param[in] filename The name of the output file, e.g. *.json, *.json.xz, or *.json.zst
     * @param[in] overwrite Allow for overwriting files if they already exist
     *
     * @note One should always use compression as it saves ~10x in file size 
     */
    void to_json(const std::string& filename, bool overwrite = false) const
    {
        operator_to_json(*this, filename, overwrite);
    }
    /**
     * Build operator from a JSON file, optionally with compression
     *
     * @param[in] filename The name of the output file, e.g. *.json, *.json.xz, or *.json.zst
     */
    static QubitOperator from_json(const std::string& filename)
    {

        QubitOperator out;
        json_to_operator(filename, out);
        return out;
    }

} QubitOperator_t;

/**
 * Set the QubitOperator flags when performing sorting of various kinds
 * 
 * @param[in, out] oper The operator whose flags to set
 * @param[in] kind Sting indicating the type of sorting that was performed
 * 
 * @throws Error if sorting type is not a valid kind
 */
inline void set_sorting_flags(QubitOperator& oper, std::string kind)
{
    if(kind == "group")
    {
        oper.sorted = 1;
        oper.weight_sorted = 0;
        oper.off_weight_sorted = 0;
        oper.ladder_sorted = 0; // since group sorting could modify the in-group ordering
        oper.structure_sorted = 0;
    }
    else if(kind == "weight")
    {
        oper.sorted = 0;
        oper.weight_sorted = 1;
        oper.off_weight_sorted = 0;
        oper.ladder_sorted = 0;
        oper.structure_sorted = 0;
    }
    else if(kind == "off_weight")
    {
        oper.sorted = 0;
        oper.weight_sorted = 0;
        oper.off_weight_sorted = 1;
        oper.ladder_sorted = 0;
        oper.structure_sorted = 0;
    }
    else if(kind == "ladder")
    {
        oper.sorted = 1; // since ladder sorting requires group sorting
        oper.weight_sorted = 0;
        oper.off_weight_sorted = 0;
        oper.ladder_sorted = 1;
        oper.structure_sorted = 0;
    }
    else if(kind == "structure")
    {
        oper.sorted = 0;
        oper.weight_sorted = 0;
        oper.off_weight_sorted = 0;
        oper.ladder_sorted = 0;
        oper.structure_sorted = 1;
    }
    else
    {
        throw std::runtime_error("Invalid sorting type.");
    }
}
