/**
 * This code is a part of Fulqrum.
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
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/* These functions and definintions (NP*) are originally from PySCF and used under the Apache 2 license
    
    Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
    Author: Qiming Sun <osirpt.sun@gmail.com>
*/
#define NPBLOCK_DIM 104
#define NPMIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define NPMAX(X, Y) ((X) > (Y) ? (X) : (Y))

#define NPTRIU_LOOP(I, J)                                                                          \
    for(j0 = 0; j0 < n; j0 += NPBLOCK_DIM)                                                         \
        for(I = 0, j1 = NPMIN(j0 + NPBLOCK_DIM, n); I < j1; I++)                                   \
            for(J = NPMAX(I, j0); J < j1; J++)

inline void NPdcopy(double* out, const double* in, const int n)
{
    std::memcpy(out, in, (size_t)n * sizeof(double));
}

inline void NPdsymm_triu(int n, double* mat)
{
    int i, j, j0, j1;
    NPTRIU_LOOP(i, j)
    {
        mat[i * n + j] = mat[j * n + i];
    }
}

inline void NPdunpack_tril(int norb, double* tril, double* mat)
{
    int i, j, ij;
    for(ij = 0, i = 0; i < norb; i++)
    {
        for(j = 0; j <= i; j++, ij++)
        {
            mat[i * norb + j] = tril[ij];
        }
    }
    NPdsymm_triu(norb, mat);
}

// unpack one row from the compact matrix-tril coefficients
inline void NPdunpack_row(int ndim, int row_id, double* tril, double* row)
{
    int i;
    size_t idx = ((size_t)row_id) * (row_id + 1) / 2;
    NPdcopy(row, tril + idx, row_id);
    for(i = row_id; i < ndim; i++)
    {
        idx += i;
        row[i] = tril[idx];
    }
}

/** Permute two body integrals 
 *
 * Applies the in-place transposition array4D.transpose(0, 2, 3, 1),
 * (i,j,k,l) -> (i,k,l,j)
 *
 * @param vec Pointer to vector to be permuted
 * @param norb Number of orbitals
 */
inline void two_body_permute(double* vec, const int norb)
{
    const int norb2 = norb * norb;
    const int norb3 = norb2 * norb;
    std::vector<double> buf(norb3);

    for(int i = 0; i < norb; i++)
    {
        double* const slice = vec + i * norb3;
        std::copy(slice, slice + norb3, buf.begin());
        for(int j = 0; j < norb; j++)
            for(int k = 0; k < norb; k++)
                for(int l = 0; l < norb; l++)
                    slice[k * norb2 + l * norb + j] = buf[j * norb2 + k * norb + l];
    }
}

typedef struct FCIDumpData
{
    std::vector<double> H1;
    std::vector<double> H2;
    std::vector<int> ORBSYM;
    double ECORE{0};
    int NORB{0}; // Number of orbitals
    int NELEC{0}; // Number of electrons
    int MS2{0}; // spin polarization (num alpha electrons - num beta electrons)
    int ISYM{1}; // symmetry of state
    bool UHF{false}; // unrestricted HF

    /** Store two-body integrals into given pointer
     *  
     * @param out Pointer for storing integrals
     * @param norb Number of orbitals
     * @param permute Permute ordering, default is false
    */
    void two_body_integrals_to_ptr(double* out, const int norb, const bool permute = 0)
    {
        int norb2 = norb * norb;
        int norb3 = norb2 * norb;
        int npair = norb * (norb + 1) / 2;
        int i, j, ij;
        std::vector<double> buffer(npair);
        for(ij = 0, i = 0; i < norb; i++)
        {
            for(j = 0; j < i + 1; j++, ij++)
            {
                NPdunpack_row(npair, ij, &(this->H2)[0], &buffer[0]);
                NPdunpack_tril(norb, &buffer[0], out + i * norb3 + j * norb2);
                if(i > j)
                {
                    NPdcopy(out + j * norb3 + i * norb2, out + i * norb3 + j * norb2, norb2);
                }
            }
        }
        // permute the elements of out to match array4D.transpose(0, 2, 3, 1).ravel()
        if(permute)
        {
            two_body_permute(out, norb);
        }
    }
    /** Return vector of two body integrals
     *  
     * @param permute Permute ordering, default is false
    */
    std::vector<double> two_body_integrals(bool permute = 0)
    {
        int norb = this->NORB;
        std::vector<double> out(norb * norb * norb * norb);
        two_body_integrals_to_ptr(&out[0], norb, permute);
        return out;
    }

} FCIDumpData_t;

inline FCIDumpData_t parse_fcidump(const std::string& filename)
{
    std::ifstream file(filename);
    std::string line, token, temp_line, val, val2;
    std::vector<std::string> header;
    int ii, jj, kk, start;
    FCIDumpData_t output;

    if(file.is_open())
    {

        // Get header information
        for(kk = 0; kk < 10; kk++)
        {
            if(std::getline(file, line))
            {
                // go to uppercase here even though pyscf should write the file that way
                std::transform(line.begin(), line.end(), line.begin(), ::toupper);
                header.push_back(line);
                if(line.find("&END") != std::string::npos || line.find("/") != std::string::npos)
                {
                    break;
                }
            }
        }
        if(kk >= 9)
        {
            file.close();
            throw std::runtime_error("Invalid fcidump header");
        }
        // process header lines
        std::regex del(",");
        for(auto line : header)
        {
            line = std::regex_replace(line, std::regex("&FCI"), "");
            line = std::regex_replace(line, std::regex("&END"), "");
            line = std::regex_replace(line, std::regex(" "), "");
            line = std::regex_replace(line, std::regex("\\n"), "");
            line = std::regex_replace(line, std::regex(",,"), "");
            //std::cout << line << std::endl;
            // IF ORBSYM in line then process entire line by comma sep values
            if((line.find("ORBSYM") != std::string::npos))
            {
                start = line.find("=");
                val = line.substr(start + 1);
                std::sregex_token_iterator it(val.begin(), val.end(), del, -1);
                std::sregex_token_iterator end;
                while(it != end)
                {
                    val2 = *it;
                    output.ORBSYM.push_back(std::stoi(val2));
                    ++it;
                }
            }
            else // look over all elements split by commas for the needed values
            {
                std::sregex_token_iterator it(line.begin(), line.end(), del, -1);
                std::sregex_token_iterator end;
                while(it != end)
                {
                    temp_line = *it;
                    if(temp_line.find("NORB") != std::string::npos)
                    {
                        start = temp_line.find("=");
                        val = temp_line.substr(start + 1);
                        output.NORB = std::stoi(val);
                    }
                    else if((temp_line.find("NELEC") != std::string::npos))
                    {
                        start = temp_line.find("=");
                        val = temp_line.substr(start + 1);
                        output.NELEC = std::stoi(val);
                    }
                    else if((temp_line.find("MS2") != std::string::npos))
                    {
                        start = temp_line.find("=");
                        val = temp_line.substr(start + 1);
                        output.MS2 = std::stoi(val);
                    }
                    else if((temp_line.find("ISYM") != std::string::npos))
                    {
                        start = temp_line.find("=");
                        val = temp_line.substr(start + 1);
                        output.ISYM = std::stoi(val);
                    }
                    else if((temp_line.find("UHF") != std::string::npos))
                    {
                        start = temp_line.find("=");
                        val = temp_line.substr(start + 1);
                        if(val == "TRUE")
                        {
                            output.UHF = true;
                        }
                    }
                    ++it;
                }
            }
        }
        // check that ORBSYM size matches NORB
        if(static_cast<int>(output.ORBSYM.size()) != output.NORB)
        {
            file.close();
            throw std::runtime_error("ORBSYM size does not equal NORB");
        }
        if(output.ISYM != 1)
        {
            file.close();
            throw std::runtime_error("Only ISYM = 1 is currently supported.");
        }

        int norb = output.NORB;
        int norb_pair = norb * (norb + 1) / 2;
        int i, j, k, l, ij, kl;
        output.H1.resize(norb * norb);
        output.H2.resize(norb_pair * (norb_pair + 1) / 2);
        double coeff;
        // Continue with rest of file to get data
        while(std::getline(file, line))
        {
            std::istringstream iss(line);
            iss >> coeff >> i >> j >> k >> l;
            if(k != 0)
            {
                if(i >= j)
                {
                    ij = i * (i - 1) / 2 + j - 1;
                }
                else
                {
                    ij = j * (j - 1) / 2 + i - 1;
                }
                if(k >= l)
                {
                    kl = k * (k - 1) / 2 + l - 1;
                }
                else
                {
                    kl = l * (l - 1) / 2 + k - 1;
                }
                if(ij >= kl)
                {
                    output.H2[ij * (ij + 1) / 2 + kl] = coeff;
                }
                else
                {
                    output.H2[kl * (kl + 1) / 2 + ij] = coeff;
                }
            }
            else
            {
                if(j != 0)
                {
                    output.H1[norb * (i - 1) + (j - 1)] = coeff;
                }
                else
                {
                    output.ECORE = coeff;
                }
            }
        }
        // For H1, look to see if all upper-triangle elements are zero, if so copy
        // values from lower-triangle
        // from https://github.com/pyscf/pyscf/blob/34e5aa023afaf42157932b2b0e35e716f7af0b14/pyscf/tools/fcidump.py#L328
        // where is it presented as an L2-norm equation
        double sum = 0;
        for(ii = 0; ii < (norb - 1); ii++)
        {
            for(jj = ii + 1; jj < norb; jj++)
            {
                sum += std::abs(output.H1[ii * norb + jj]);
            }
        }
        if(sum == 0)
        {
            for(ii = 0; ii < (norb - 1); ii++)
            {
                for(jj = ii + 1; jj < norb; jj++)
                {
                    output.H1[ii * norb + jj] = output.H1[jj * norb + ii];
                }
            }
        }
        else // look to see if all lower triangle elements are zero, if so copy from upper triangle
        {
            sum = 0;
            for(ii = 1; ii < norb; ii++)
            {
                for(jj = 0; jj < ii; jj++)
                {
                    sum += std::abs(output.H1[ii * norb + jj]);
                }
            }
            if(sum == 0)
            {
                for(ii = 1; ii < norb; ii++)
                {
                    for(jj = 0; jj < ii; jj++)
                    {
                        output.H1[ii * norb + jj] = output.H1[jj * norb + ii];
                    }
                }
            }
        }

        // Close the file stream once all lines have been read.
        file.close();
    }

    else
    {
        throw std::runtime_error("Unable to open fcidump file: " + filename);
    }
    return output;
}
