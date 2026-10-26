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
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../core/src/constants.hpp"
#include "../../core/src/fermi_oper.hpp"

inline std::size_t _flat_index2d(width_t i, width_t j, width_t dim)
{
    return i * dim + j;
}

inline std::size_t _flat_index4d(width_t i, width_t j, width_t k, width_t l, width_t dim)
{
    return i + j * dim + k * dim * dim + l * dim * dim * dim;
}

template <typename T>
inline FermionicOperator& pyscf_integrals_to_fermionic(T* __restrict fobi,
                                                       T* __restrict ftbi,
                                                       unsigned int ob_arr_len,
                                                       unsigned int tb_arr_len,
                                                       std::complex<double> constant = 0,
                                                       double EQ_TOLERANCE = 1e-12)
{
    width_t half_num_qubits = std::sqrt(ob_arr_len);
    width_t num_qubits = 2 * half_num_qubits;

    if(tb_arr_len != half_num_qubits * half_num_qubits * half_num_qubits * half_num_qubits)
    {
        throw std::runtime_error(
            "Input flat_two_body_integral array does not match expected length");
    }

    static const std::string ob_str = "+-"; // One-body operator string (normal ordered)
    static const std::string tb_str = "++--"; // Two-body operator string (normal ordered)

    std::vector<width_t> qubit_mapping(num_qubits);
    width_t p, kk;

    for(kk = 0; kk < num_qubits; kk++)
    {
        qubit_mapping[kk] = ((!(kk % 2)) * kk / 2) + ((kk % 2) * (kk / 2 + half_num_qubits));
    }

    FermionicOperator* fop = new FermionicOperator(num_qubits);
    if(std::abs(constant) > EQ_TOLERANCE)
    {
        fop->terms.emplace_back(constant);
    }
    std::vector<std::vector<FermionicTerm>> temp_terms;
    temp_terms.resize(half_num_qubits);

#pragma omp parallel for schedule(dynamic) if(half_num_qubits > 8)
    for(p = 0; p < half_num_qubits; p++)
    {
        width_t q, r, s, ii, jj, ll;
        bool do0, do1, do2, do3;
        std::complex<double> valob, val01, val23;
        for(q = 0; q < half_num_qubits; q++)
        {
            valob = fobi[_flat_index2d(p, q, half_num_qubits)];
            if(std::abs(valob) > EQ_TOLERANCE)
            {
                // Populate 1-body coefficients. Require p and q have same spin.
                ii = 2 * p;
                jj = 2 * q;
                temp_terms[p].emplace_back(
                    ob_str, std::vector<width_t>{qubit_mapping[ii], qubit_mapping[jj]}, valob);

                ii = 2 * p + 1;
                jj = 2 * q + 1;
                temp_terms[p].emplace_back(
                    ob_str, std::vector<width_t>{qubit_mapping[ii], qubit_mapping[jj]}, valob);
            }
            if(q < p)
            {
                continue;
            }
            // Continue looping to prepare 2-body coefficients.
            for(r = 0; r < half_num_qubits; r++)
            {
                for(s = 0; s < half_num_qubits; s++)
                {
                    do0 = false;
                    do1 = false;
                    do2 = false;
                    do3 = false;

                    if(p == q)
                    {
                        if(s == r)
                        {
                            val01 = ftbi[_flat_index4d(p, q, r, s, half_num_qubits)];
                            do0 = std::abs(val01) > EQ_TOLERANCE;
                        }
                        else if(s > r)
                        {
                            val01 = 0.5 * (ftbi[_flat_index4d(p, q, r, s, half_num_qubits)] +
                                           ftbi[_flat_index4d(p, q, s, r, half_num_qubits)]);
                            do0 = std::abs(val01) > EQ_TOLERANCE;
                            do1 = do0;
                        }
                    }
                    else
                    {
                        if(s == r)
                        {
                            val01 = 0.5 * (ftbi[_flat_index4d(p, q, r, s, half_num_qubits)] +
                                           ftbi[_flat_index4d(q, p, r, s, half_num_qubits)]);
                            do0 = std::abs(val01) > EQ_TOLERANCE;
                            do1 = do0;
                        }
                        else
                        {
                            val01 = 0.5 * (ftbi[_flat_index4d(p, q, r, s, half_num_qubits)] +
                                           ftbi[_flat_index4d(q, p, s, r, half_num_qubits)]);
                            do0 = std::abs(val01) > EQ_TOLERANCE;
                            do1 = do0;
                            if(s > r)
                            {
                                val23 = val01 -
                                        0.5 * (ftbi[_flat_index4d(p, q, s, r, half_num_qubits)] +
                                               ftbi[_flat_index4d(q, p, r, s, half_num_qubits)]);
                                do2 = std::abs(val23) > EQ_TOLERANCE;
                                do3 = do2;
                            }
                        }
                    }

                    if(do0)
                    {
                        ii = 2 * p;
                        jj = 2 * q + 1;
                        kk = 2 * r + 1;
                        ll = 2 * s;
                        temp_terms[p].emplace_back(tb_str,
                                                   std::vector<width_t>{qubit_mapping[ii],
                                                                        qubit_mapping[jj],
                                                                        qubit_mapping[kk],
                                                                        qubit_mapping[ll]},
                                                   val01);
                    }
                    if(do1)
                    {
                        ii = 2 * p + 1;
                        jj = 2 * q;
                        kk = 2 * r;
                        ll = 2 * s + 1;
                        temp_terms[p].emplace_back(tb_str,
                                                   std::vector<width_t>{qubit_mapping[ii],
                                                                        qubit_mapping[jj],
                                                                        qubit_mapping[kk],
                                                                        qubit_mapping[ll]},
                                                   val01);
                    }
                    if(do2)
                    {
                        ii = 2 * p;
                        jj = 2 * q;
                        kk = 2 * r;
                        ll = 2 * s;
                        temp_terms[p].emplace_back(tb_str,
                                                   std::vector<width_t>{qubit_mapping[ii],
                                                                        qubit_mapping[jj],
                                                                        qubit_mapping[kk],
                                                                        qubit_mapping[ll]},
                                                   val23);
                    }
                    if(do3)
                    {
                        ii = 2 * p + 1;
                        jj = 2 * q + 1;
                        kk = 2 * r + 1;
                        ll = 2 * s + 1;
                        temp_terms[p].emplace_back(tb_str,
                                                   std::vector<width_t>{qubit_mapping[ii],
                                                                        qubit_mapping[jj],
                                                                        qubit_mapping[kk],
                                                                        qubit_mapping[ll]},
                                                   val23);
                    }
                } // end s-loop
            } // end r-loop
        } // end q-loop
    } // end p-loop
    for(auto& item : temp_terms)
    {
        fop->terms.insert(fop->terms.end(),
                          std::make_move_iterator(item.begin()),
                          std::make_move_iterator(item.end()));
    }
    fop->terms.shrink_to_fit();
    fop->unique_terms = 1; // all the terms are unique by construction
    return *fop;
}
