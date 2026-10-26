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
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef FQ_TBB
#    include <oneapi/tbb/parallel_sort.h>
#endif

#include "constants.hpp"
#include "fermi_term.hpp"
#include "io.hpp"
#include "oper_utils.hpp"
#include "qubit_oper.hpp"
#include "term_utils.hpp"

// forward definitions
void set_fermi_sorting_flags(FermionicOperator& oper, std::string kind);

/** @struct FermionicOperator
 * @brief Data structure for each a qubit operator, i.e. a collection of 'words'
 *
 * @var width is the number of qubits
 * @var terms is a vector of OperatorTerms that make up the operator
 * @var sorted is a flag that indicates the term is sorted (NOT USED AT PRESENT)
 */
typedef struct FermionicOperator
{
    width_t width;
    unsigned int combined = 0; // have the repeated operators indices been combined?
    unsigned int unique_terms = 0; // are the terms unique? i.e. duplicates removed
    int weight_sorted{0}; // Are the operator terms weight sorted
    int structure_sorted{
        0}; // Are the operator terms sorted by (non-unique) off-diagonal structure?
    std::vector<FermionicTerm_t> terms;
    FermionicOperator() = default;
    FermionicOperator(const FermionicOperator&) = default;
    FermionicOperator(FermionicOperator&&) = default;
    FermionicOperator& operator=(const FermionicOperator&) = default;
    FermionicOperator& operator=(FermionicOperator&&) = default;
    // deallocation
    ~FermionicOperator() = default;
    /**
     * Constructor building an empty operator with a given width
     *
     * @param[in] width The width (number of qubits) of the operator
     */
    FermionicOperator(width_t x)
    {
        width = x;
    }
    FermionicOperator(width_t x, std::vector<TermData> data)
        : width(x)
    {
        std::size_t num_terms = data.size();
        std::size_t kk;
        TermData tdata;
        for(kk = 0; kk < num_terms; kk++)
        {
            tdata = data[kk];
            _validate_indices(std::get<1>(tdata),
                              width); // validate that all indices are less than operator width
            terms.push_back(
                FermionicTerm(std::get<0>(tdata), std::get<1>(tdata), std::get<2>(tdata)));
        }
    }
    static FermionicOperator from_fcidump(std::string filename, double tol);
    /**
     * Print object to standard output stream
     */
    friend auto operator<<(std::ostream& os, const FermionicOperator& self) -> std::ostream&
    {
        std::size_t num_terms = self.size();
        std::size_t total_terms = num_terms;
        FermionicTerm_t term;
        int too_many_terms = 0;
        std::size_t kk, jj;

        // restrict to outputting at most 100 terms
        if(num_terms > 100)
        {
            too_many_terms = 1;
            num_terms = 100;
        }
        os << "<FermionicOperator["; // start output here
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
            os << " + " << (total_terms - 100) << " terms";
        }
        return os << ", width=" << self.width << "]>";
    }
    /**
     * Grab a single term by index
     * 
     * @param[in] Index of term to grab
     * 
     * @return FermionicTerm at the given index
     */
    FermionicTerm_t operator[](std::size_t index) const
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
    FermionicOperator& operator*=(std::complex<double> c)
    {
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            terms[kk] *= c;
        }
        return *this;
    }
    /**
     * multiplication by a complex value (need one for mult on each side)
     */
    friend FermionicOperator operator*(FermionicOperator& op, std::complex<double> c)
    {
        FermionicOperator out = op.copy();
        for(std::size_t kk = 0; kk < out.size(); kk++)
        {
            out.terms[kk] *= c;
        }
        return out;
    }
    friend FermionicOperator operator*(std::complex<double> c, FermionicOperator& op)
    {
        FermionicOperator out = op.copy();
        for(std::size_t kk = 0; kk < out.size(); kk++)
        {
            out.terms[kk] *= c;
        }
        return out;
    }
    /**
     * Inplace addition by another FermionicOperator
     * 
     * @param[in] other Operator to add to this one
     * @throw Error if operators do not share the same width
     */
    FermionicOperator& operator+=(FermionicOperator other)
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        for(std::size_t kk = 0; kk < other.size(); kk++)
        {
            this->terms.push_back(other.terms[kk]);
        }
        this->combined = 0;
        return *this;
    }
    /**
     * Addition by another FermionicOperator
     * 
     * @param[in] other Operator to add to this one
     * @return The new operator
     * @throw Error if operators do not share the same width
     */
    FermionicOperator operator+(FermionicOperator other) const
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        FermionicOperator out = this->copy();
        for(std::size_t kk = 0; kk < other.size(); kk++)
        {
            out.terms.push_back(other.terms[kk]);
        }
        return out;
    }
    /**
     * Inplace subtraction by another FermionicOperator
     * 
     * @param[in] other Operator to add to this one
     * @throw Error if operators do not share the same width
     */
    FermionicOperator& operator-=(FermionicOperator other)
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }

        FermionicTerm term;
        for(std::size_t kk = 0; kk < other.size(); kk++)
        {
            term = other.terms[kk];
            term.coeff *= -1;
            this->terms.push_back(term);
        }
        this->combined = 0;
        return *this;
    }
    /**
     * Subtraction by another FermionicOperator
     * 
     * @param[in] other Operator to subject to this one
     * @return New operator
     * @throw Error if operators do not share the same width
     */
    FermionicOperator operator-(FermionicOperator other)
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }

        FermionicTerm term;
        FermionicOperator out = this->copy();
        for(std::size_t kk = 0; kk < other.size(); kk++)
        {
            term = other.terms[kk];
            term.coeff *= -1;
            out.terms.push_back(term);
        }
        return out;
    }
    /**
     * Return the size of the operator
     */
    std::size_t size() const
    {
        return terms.size();
    }
    /**
     * Make a copy of the operator
     *
     * @return A copy of the current operator
     */
    FermionicOperator copy() const
    {
        FermionicOperator out = FermionicOperator(this->width);
        out.terms = this->terms;
        out.combined = this->combined;
        out.weight_sorted = this->weight_sorted;
        out.structure_sorted = this->structure_sorted;
        return out;
    }
    /**
    * In-place sorting of terms by weight
    * 
    */
    FermionicOperator& weight_sort()
    {
        if(!(this->weight_sorted))
        {
// sort by weight
#ifdef FQ_TBB
            tbb::parallel_sort(terms.begin(),
                               terms.end(),
                               [](const FermionicTerm& term1, const FermionicTerm& term2) {
                                   return term1.indices.size() < term2.indices.size();
                               });
#else
            boost::sort::pdqsort(terms.begin(),
                                 terms.end(),
                                 [](const FermionicTerm& term1, const FermionicTerm& term2) {
                                     return term1.indices.size() < term2.indices.size();
                                 });
#endif
            set_fermi_sorting_flags(*this, "weight");
        }
        return *this;
    }
    /**
    * In-place sorting of terms by weight
    * 
    */
    FermionicOperator& offdiag_structure_sort()
    {
        if(!(this->structure_sorted))
        {
// sort by weight
#ifdef FQ_TBB
            tbb::parallel_sort(terms.begin(),
                               terms.end(),
                               [](const FermionicTerm& term1, const FermionicTerm& term2) {
                                   return term1.offdiag_structure < term2.offdiag_structure;
                               });
#else
            boost::sort::pdqsort(terms.begin(),
                                 terms.end(),
                                 [](const FermionicTerm& term1, const FermionicTerm& term2) {
                                     return term1.offdiag_structure < term2.offdiag_structure;
                                 });
#endif
            set_fermi_sorting_flags(*this, "structure");
        }
        return *this;
    }
    /**
    * Pointers to starting indices for structure sorted operator
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
    * Combine repeated terms in operator
    * 
    * @param[in] atol Tolerance for determining if a combined coefficient is zero
    * 
    * @return Output QubitOperator with terms combined
    * 
    */
    FermionicOperator combine_repeat_terms(double atol = 1e-12)
    {
        if(this->unique_terms)
        {
            return *this;
        }
        FermionicOperator out = FermionicOperator(this->width);
        if(!this->size())
        {
            return out;
        }
        if(!this->structure_sorted)
        {
            this->offdiag_structure_sort();
        }
        std::vector<std::size_t> ptrs;
        set_offdiag_structure_ptrs(terms, ptrs);
        combine_terms(this->terms, out.terms, ptrs, atol);
        this->unique_terms = 1;
        return out;
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
    static FermionicOperator from_json(const std::string& filename)
    {

        FermionicOperator out;
        json_to_operator(filename, out);
        return out;
    }
    FermionicOperator combine_repeat_indices() const
    {
        if(this->combined)
        {
            return *this;
        }
        FermionicOperator out = FermionicOperator(this->width);
        static const std::vector<int> collapsed_values = {
            1, -1, 5, -1, -1, 2, -1, 6, -1, 5, -1, 1, 6, -1, 2, -1};
        std::vector<FermionicTerm_t> temp_terms;
        temp_terms.resize(this->size());
#pragma omp parallel for schedule(dynamic) if(this->size() > 65536)
        for(std::size_t kk = 0; kk < terms.size(); kk++)
        {
            deflate_term_indices(terms[kk], temp_terms[kk], collapsed_values);
        }
        out.terms.reserve(this->size());
        for(std::size_t kk = 0; kk < terms.size(); kk++)
        {
            if(temp_terms[kk].coeff != 0.0)
            {
                out.terms.push_back(std::move(temp_terms[kk]));
            }
        }
        out.terms.shrink_to_fit();
        out.combined = 1;
        return out;
    }

    /**
     * Extended Jordan-Wigner transformation
     * 
     * @note This routine requires combining repeated indices first,
     * and will do so internally if the `combined` flag is not set
     *
     * @return QubitOperator after extended JW transformation
     */
    QubitOperator extended_jw_transformation()
    {
        FermionicOperator& fermi = *this;
        // This requires combining repeated indices
        if(!fermi.combined)
        {
            fermi = fermi.combine_repeat_indices();
        }
        if(!fermi.unique_terms)
        {
            fermi = fermi.combine_repeat_terms();
        }
        QubitOperator_t out = QubitOperator(fermi.width);
        std::size_t kk;
        const std::size_t num_terms = fermi.size();
        out.terms.resize(num_terms);
#pragma omp parallel for schedule(dynamic) if(num_terms > 1024)
        for(kk = 0; kk < num_terms; kk++)
        {
            jw_term(fermi.terms[kk], out.terms[kk]);
            // elements are added high to low, so must reverse to get correct order
            std::reverse(out.terms[kk].indices.begin(), out.terms[kk].indices.end());
            std::reverse(out.terms[kk].values.begin(), out.terms[kk].values.end());
            set_offdiag_weight_phase_struct(out.terms[kk]);
            set_term_proj_indices(out.terms[kk]);
        }
        out.type = 2; // set type=2
        return out;
    }
} FermionicOperator_t;

/**
 * Set the FermionicOperator flags when performing sorting of various kinds
 * 
 * @param[in, out] oper The operator whose flags to set
 * @param[in] kind Sting indicating the type of sorting that was performed
 * 
 * @throws Error if sorting type is not a valid kind
 */
inline void set_fermi_sorting_flags(FermionicOperator& oper, std::string kind)
{
    if(kind == "weight")
    {
        oper.weight_sorted = 1;
        oper.structure_sorted = 0;
    }
    else if(kind == "structure")
    {
        oper.weight_sorted = 0;
        oper.structure_sorted = 1;
    }
    else
    {
        throw std::runtime_error("Invalid sorting type.");
    }
}

#include "../../convert/src/fcidump.hpp"
#include "../../convert/src/integrals.hpp"
inline FermionicOperator FermionicOperator::from_fcidump(std::string filename, double tol = 1e-12)
{
    FCIDumpData_t data = parse_fcidump(filename);
    int norb = data.NORB;
    int norb2 = norb * norb;
    std::vector<double> two_body_ints = data.two_body_integrals(true);
    FermionicOperator out = pyscf_integrals_to_fermionic(
        &data.H1[0], &two_body_ints[0], norb2, norb2 * norb2, data.ECORE, tol);
    return out;
}