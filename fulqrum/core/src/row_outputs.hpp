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

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.hpp"
#include "checked_size.hpp"
#include "csr_utils.hpp"

/**
 * Builds the CSR arrays of a matrix from its rows, in two calls.
 *
 * make_walk() returns a row walk for one thread; walk(row, emit) calls
 * emit(column, value) for each off-diagonal element of the row, in the same order on
 * every call.
 *
 * @param num_rows The number of rows.
 * @param diag The diagonal, or nullptr for zero.
 * @param make_walk The row-walk factory.
 * @param indptr The row starts.
 * @param indices The column indices.
 * @param data The values.
 * @param compute_values 0 for the count call, 1 for the fill call.
 * @throws std::runtime_error If the fill call finds a row with a different number of
 *     elements than the count call.
 */
template <typename I, typename T, typename MakeWalk>
void csr_from_rows(const std::size_t num_rows,
                   const T* diag,
                   MakeWalk&& make_walk,
                   I* indptr,
                   I* indices,
                   T* data,
                   const int compute_values)
{
    checked_add(num_rows, 1);
    if(num_rows) checked_index<I>(num_rows - 1);
    if(!compute_values)
    {
        bool overflow = false;
#pragma omp parallel reduction(|| : overflow)
        {
            auto walk = make_walk();
#pragma omp for schedule(dynamic, 256)
            for(std::size_t row = 0; row < num_rows; row++)
            {
                std::size_t count = (diag != nullptr && diag[row] != T(0)) ? 1 : 0;
                walk(row, [&](const std::size_t, const T& value) {
                    if(std::abs(value) > ATOL)
                    {
                        if(count == static_cast<std::size_t>(std::numeric_limits<I>::max()))
                            overflow = true;
                        else
                            count++;
                    }
                });
                indptr[row + 1] = static_cast<I>(count);
            }
        }
        if(overflow) throw std::overflow_error("CSR row count exceeds the index type.");
        indptr[0] = 0;
        for(std::size_t row = 0; row < num_rows; row++)
        {
            indptr[row + 1] = checked_index<I>(checked_add(static_cast<std::size_t>(indptr[row + 1]), static_cast<std::size_t>(indptr[row])));
        }
        return;
    }
    bool mismatch = false;
#pragma omp parallel reduction(|| : mismatch)
    {
        auto walk = make_walk();
#pragma omp for schedule(dynamic, 256)
        for(std::size_t row = 0; row < num_rows; row++)
        {
            I position = indptr[row];
            const I end = indptr[row + 1];
            if(diag != nullptr && diag[row] != T(0) && position < end)
            {
                indices[position] = static_cast<I>(row);
                data[position] = diag[row];
                position++;
            }
            walk(row, [&](const std::size_t column, const T& value) {
                if(std::abs(value) > ATOL)
                {
                    if(position < end)
                    {
                        indices[position] = static_cast<I>(column);
                        data[position] = value;
                        position++;
                    }
                    else
                        mismatch = true;
                }
            });
            mismatch = mismatch || position != end;
        }
    }
    if(mismatch)
    {
        throw std::runtime_error("The CSR fill found a different number of elements than the count.");
    }
}

/**
 * Builds the csrlike rows of a matrix from its rows: one column list and one value
 * list per row, with the columns of each row in increasing order.
 *
 * make_walk() returns a row walk for one thread; walk(row, emit) calls
 * emit(column, value) for each off-diagonal element of the row.
 *
 * @param num_rows The number of rows.
 * @param diag The diagonal, or nullptr for zero.
 * @param make_walk The row-walk factory.
 * @param cols The column lists. The function resizes it to num_rows.
 * @param data The value lists. The function resizes it to num_rows.
 */
template <typename T, typename I, typename MakeWalk>
void csrlike_from_rows(const std::size_t num_rows,
                       const T* diag,
                       MakeWalk&& make_walk,
                       std::vector<std::vector<I>>& cols,
                       std::vector<std::vector<T>>& data)
{
    if(num_rows) checked_index<I>(num_rows - 1);
    checked_multiply(num_rows, sizeof(std::vector<I>) + sizeof(std::vector<T>));
    cols.assign(num_rows, std::vector<I>());
    data.assign(num_rows, std::vector<T>());
#pragma omp parallel
    {
        auto walk = make_walk();
#pragma omp for schedule(dynamic, 256)
        for(std::size_t row = 0; row < num_rows; row++)
        {
            if(diag != nullptr && diag[row] != T(0))
            {
                cols[row].push_back(static_cast<I>(row));
                data[row].push_back(diag[row]);
            }
            walk(row, [&](const std::size_t column, const T& value) {
                if(std::abs(value) > ATOL)
                {
                    cols[row].push_back(static_cast<I>(column));
                    data[row].push_back(value);
                }
            });
        }
    }
    sort_paired(cols, data);
}
