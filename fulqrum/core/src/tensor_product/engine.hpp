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
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "build.hpp"
#include "../tier_utils.hpp"

inline constexpr int NARROW_BLOCK = 8;
inline constexpr int WIDE_BLOCK = 32;
inline constexpr std::uint64_t WIDE_BLOCK_ENTRIES = 64;
inline constexpr std::size_t WIDE_BLOCK_BYTES = std::size_t(4) << 20;

template <typename T>
bool use_wide_blocks(const TensorProductEngine<T>& tier, const int sector)
{
    const SectorConnections<T>& connections = tier.sectors[sector];
    const std::uint64_t num_strings = tier.num_strings[sector];
    const std::size_t block_rows = 2 * num_strings + (tier.cross_formed ? connections.slot_pair.size() : 0);
    const std::uint64_t entries = connections.single_offsets.back() + connections.double_offsets.back();
    return entries >= WIDE_BLOCK_ENTRIES * num_strings && block_rows * WIDE_BLOCK * sizeof(T) <= WIDE_BLOCK_BYTES;
}

/**
 * Applies the diagonal and the alpha same-sector arms to blocks of beta rows.
 *
 * Each task owns R beta rows in a transposed buffer, walks each alpha
 * connection once and writes the mirror element with conj.
 *
 * @param tier The tensor-product engine.
 * @param diag The diagonal, or nullptr for zero.
 * @param x The input vector, index beta * num_alpha + alpha.
 * @param y The output vector. The function sets every entry.
 */
template <int R, typename T>
void apply_alpha_arm_blocks(const TensorProductEngine<T>& tier, const T* diag, const T* x, T* y)
{
    const std::size_t num_alpha = tier.num_strings[0];
    const std::size_t num_beta = tier.num_strings[1];
    const SectorConnections<T>& alpha = tier.sectors[0];
    const UpperConnections<T>& upper = tier.upper[0];
    const std::size_t num_blocks = ceiling_divide(num_beta, R);
    const std::size_t block_cells = checked_multiply(num_alpha, R);
    const std::size_t occupancy_cells = tier.cross_formed ? checked_multiply(checked_multiply(2, tier.num_orbitals[1]), R) : 0;
    const std::size_t cross_cells = tier.cross_formed ? checked_multiply(tier.sectors[0].slot_pair.size(), R) : 0;
    checked_add(checked_add(checked_multiply(block_cells, 2 * sizeof(T)), checked_multiply(occupancy_cells, sizeof(double))), checked_multiply(cross_cells, sizeof(T)));
#pragma omp parallel
    {
        std::vector<T> x_block(block_cells);
        std::vector<T> y_block(block_cells);
        std::vector<double> occupancy(occupancy_cells);
        std::vector<T> cross_block(cross_cells);
#pragma omp for schedule(dynamic, 1)
        for(std::size_t block = 0; block < num_blocks; block++)
        {
            const std::size_t first_beta = block * R;
            const int rows = static_cast<int>(std::min<std::size_t>(R, num_beta - first_beta));
            const T* cross_rows = cross_block.data();
            std::size_t cross_stride = R;

            if(tier.cross_formed)
            {
                form_cross_block<R>(tier.cross[0], tier.half_keys[1].data(), tier.half_words[1], tier.num_orbitals[1],
                                    first_beta, rows, occupancy.data(), cross_block.data());
            }
            else
            {
                cross_rows = tier.cross[0].table.data() + first_beta;
                cross_stride = num_beta;
            }

            for(std::size_t a = 0; a < num_alpha; a++)
            {
                for(int r = 0; r < R; r++)
                {
                    const std::size_t index = (first_beta + r) * num_alpha + a;
                    const T value = r < rows ? x[index] : T(0);
                    x_block[a * R + r] = value;
                    y_block[a * R + r] = (r < rows && diag != nullptr) ? diag[index] * value : T(0);
                }
            }

            for(std::size_t a = 0; a < num_alpha; a++)
            {
                const T* x_row = x_block.data() + a * R;
                T* y_row = y_block.data() + a * R;
                T sum[R] = {};

                for(std::uint64_t k = upper.single_offsets[a]; k < upper.single_offsets[a + 1]; k++)
                {
                    const std::uint64_t packed_entry = upper.single_packed_entries[k];
                    const std::uint32_t column = single_column(packed_entry);
                    const T base = upper.single_values[k];
                    const double own_sign = single_own_sign(packed_entry);
                    const T* cross = cross_rows + alpha.pair_slot[single_directed_pair(packed_entry)] * cross_stride;
                    const T* x_column = x_block.data() + std::size_t(column) * R;
                    T* y_column = y_block.data() + std::size_t(column) * R;

                    for(int r = 0; r < R; r++)
                    {
                        const T value = base + own_sign * (r < rows ? cross[r] : T(0));
                        sum[r] += value * x_column[r];
                        y_column[r] += conjugate(value) * x_row[r];
                    }
                }

                for(std::uint64_t k = upper.double_offsets[a]; k < upper.double_offsets[a + 1]; k++)
                {
                    const std::uint32_t column = upper.double_columns[k];
                    const T value = upper.double_values[k];
                    const T mirror = conjugate(value);
                    const T* x_column = x_block.data() + std::size_t(column) * R;
                    T* y_column = y_block.data() + std::size_t(column) * R;

                    for(int r = 0; r < R; r++)
                    {
                        sum[r] += value * x_column[r];
                        y_column[r] += mirror * x_row[r];
                    }
                }

                for(int r = 0; r < R; r++)
                {
                    y_row[r] += sum[r];
                }
            }

            for(int r = 0; r < rows; r++)
            {
                T* y_out = y + (first_beta + r) * num_alpha;

                for(std::size_t a = 0; a < num_alpha; a++)
                {
                    y_out[a] = y_block[a * R + r];
                }
            }
        }
    }
}

/**
 * Adds the beta same-sector arms, over blocks of alpha columns.
 *
 * Each task owns R alpha columns of every beta row in a transposed
 * buffer, walks each beta connection once and writes the mirror element with conj.
 *
 * @param tier The tensor-product engine.
 * @param x The input vector.
 * @param y The output vector.
 */
template <int R, typename T>
void add_beta_arm_blocks(const TensorProductEngine<T>& tier, const T* x, T* y)
{
    const std::size_t num_alpha = tier.num_strings[0];
    const std::size_t num_beta = tier.num_strings[1];
    const SectorConnections<T>& beta = tier.sectors[1];
    const UpperConnections<T>& upper = tier.upper[1];
    const std::size_t num_blocks = ceiling_divide(num_alpha, R);
    const std::size_t block_cells = checked_multiply(num_beta, R);
    const std::size_t occupancy_cells = tier.cross_formed ? checked_multiply(checked_multiply(2, tier.num_orbitals[0]), R) : 0;
    const std::size_t cross_cells = tier.cross_formed ? checked_multiply(tier.sectors[1].slot_pair.size(), R) : 0;
    checked_add(checked_add(checked_multiply(block_cells, 2 * sizeof(T)), checked_multiply(occupancy_cells, sizeof(double))), checked_multiply(cross_cells, sizeof(T)));
#pragma omp parallel
    {
        std::vector<T> x_block(block_cells);
        std::vector<T> y_block(block_cells);
        std::vector<double> occupancy(occupancy_cells);
        std::vector<T> formed_block(cross_cells);
#pragma omp for schedule(dynamic, 1)
        for(std::size_t block = 0; block < num_blocks; block++)
        {
            const std::size_t first_alpha = block * R;
            const int columns = static_cast<int>(std::min<std::size_t>(R, num_alpha - first_alpha));
            const T* cross_rows = formed_block.data();
            std::size_t cross_stride = R;

            if(tier.cross_formed)
            {
                form_cross_block<R>(tier.cross[1], tier.half_keys[0].data(), tier.half_words[0], tier.num_orbitals[0],
                                    first_alpha, columns, occupancy.data(), formed_block.data());
            }
            else
            {
                cross_rows = tier.cross[1].table.data() + first_alpha;
                cross_stride = num_alpha;
            }

            for(std::size_t b = 0; b < num_beta; b++)
            {
                for(int r = 0; r < R; r++)
                {
                    x_block[b * R + r] = r < columns ? x[b * num_alpha + first_alpha + r] : T(0);
                    y_block[b * R + r] = T(0);
                }
            }

            T cross_block[R];

            for(std::size_t b = 0; b < num_beta; b++)
            {
                const T* x_row = x_block.data() + b * R;
                T* y_row = y_block.data() + b * R;
                T sum[R] = {};

                for(std::uint64_t k = upper.single_offsets[b]; k < upper.single_offsets[b + 1]; k++)
                {
                    const std::uint64_t packed_entry = upper.single_packed_entries[k];
                    const std::uint32_t column = single_column(packed_entry);
                    const T base = upper.single_values[k];
                    const double own_sign = single_own_sign(packed_entry);
                    const T* cross = cross_rows + beta.pair_slot[single_directed_pair(packed_entry)] * cross_stride;

                    for(int r = 0; r < R; r++)
                    {
                        cross_block[r] = r < columns ? cross[r] : T(0);
                    }

                    const T* x_column = x_block.data() + std::size_t(column) * R;
                    T* y_column = y_block.data() + std::size_t(column) * R;

                    for(int r = 0; r < R; r++)
                    {
                        const T value = base + own_sign * cross_block[r];
                        sum[r] += value * x_column[r];
                        y_column[r] += conjugate(value) * x_row[r];
                    }
                }

                for(std::uint64_t k = upper.double_offsets[b]; k < upper.double_offsets[b + 1]; k++)
                {
                    const std::uint32_t column = upper.double_columns[k];
                    const T value = upper.double_values[k];
                    const T mirror = conjugate(value);
                    const T* x_column = x_block.data() + std::size_t(column) * R;
                    T* y_column = y_block.data() + std::size_t(column) * R;

                    for(int r = 0; r < R; r++)
                    {
                        sum[r] += value * x_column[r];
                        y_column[r] += mirror * x_row[r];
                    }
                }

                for(int r = 0; r < R; r++)
                {
                    y_row[r] += sum[r];
                }
            }

            for(std::size_t b = 0; b < num_beta; b++)
            {
                for(int r = 0; r < columns; r++)
                {
                    y[b * num_alpha + first_alpha + r] += y_block[b * R + r];
                }
            }
        }
    }
}

template <typename T>
void apply_alpha_arms(const TensorProductEngine<T>& tier, const T* diag, const T* x, T* y)
{
    if(use_wide_blocks(tier, 0))
    {
        apply_alpha_arm_blocks<WIDE_BLOCK>(tier, diag, x, y);
    }
    else
    {
        apply_alpha_arm_blocks<NARROW_BLOCK>(tier, diag, x, y);
    }
}

template <typename T>
void add_beta_arms(const TensorProductEngine<T>& tier, const T* x, T* y)
{
    if(use_wide_blocks(tier, 1))
    {
        add_beta_arm_blocks<WIDE_BLOCK>(tier, x, y);
    }
    else
    {
        add_beta_arm_blocks<NARROW_BLOCK>(tier, x, y);
    }
}

/**
 * Returns the sum of coefficient times input over a range of alpha connections.
 *
 * @param coefficients The cross-sector table row of the slot of one beta connection.
 * @param index The column and signed slot of each alpha connection.
 * @param x_row The input row of the target beta string.
 * @param first The first alpha connection.
 * @param last One past the last alpha connection.
 */
template <typename T>
inline T gather_cross_sum(const T* __restrict coefficients,
                          const std::uint64_t* __restrict index,
                          const T* __restrict x_row,
                          const std::uint64_t first,
                          const std::uint64_t last)
{
    T sum = T(0);

    if constexpr(std::is_same_v<T, double>)
    {
        T partial[4] = {};
        std::uint64_t k = first;

        for(; k + 4 <= last; k += 4)
        {
            for(int u = 0; u < 4; u++)
            {
                partial[u] += coefficients[index[k + u] >> 32] * x_row[static_cast<std::uint32_t>(index[k + u])];
            }
        }

        for(; k < last; k++)
        {
            partial[0] += coefficients[index[k] >> 32] * x_row[static_cast<std::uint32_t>(index[k])];
        }

        sum = partial[0] + partial[1] + partial[2] + partial[3];
    }
    else
    {
        for(std::uint64_t k = first; k < last; k++)
        {
            sum += coefficients[index[k] >> 32] * x_row[static_cast<std::uint32_t>(index[k])];
        }
    }

    return sum;
}

inline void add_cross_slot_batches(const TensorProductEngine<double>& tier, const double* x, double* y)
{
    constexpr int L = CROSS_BATCH_LANES;
    const std::size_t num_alpha = tier.num_strings[0];
    const std::size_t alpha_columns = 2 * tier.sectors[0].slot_pair.size();
    const std::uint64_t* __restrict offsets = tier.sectors[0].single_offsets.data();
    const std::uint64_t* __restrict index = tier.sectors[0].single_cross_index.data();
    const CrossSlotBatches& batches = tier.cross_batches;
    const std::size_t num_tiles = batches.tile_block.size();
#pragma omp parallel
    {
        std::vector<CrossLanes> lanes(num_alpha);
        CrossLanes* __restrict xt = lanes.data();
#pragma omp for schedule(dynamic, 1)
        for(std::size_t tile = 0; tile < num_tiles; tile++)
        {
            const std::uint32_t block = batches.tile_block[tile];
            const std::size_t first_alpha = batches.tile_first_alpha[tile];
            const std::size_t last_alpha = batches.tile_last_alpha[tile];

            for(std::uint64_t batch = batches.block_first_batch[block]; batch < batches.block_first_batch[block + 1];
                batch++)
            {
                const int count = static_cast<int>(batches.batch_count[batch]);
                const std::uint64_t first = batches.batch_first[batch];
                const double* __restrict coefficients =
                    tier.cross_sector_table.data() + std::size_t(batches.batch_slot[batch]) * alpha_columns;
                const double* source[L];
                double* target[L];
                double sign[L];

                for(int t = 0; t < L; t++)
                {
                    const std::uint64_t member = first + (t < count ? t : 0);
                    source[t] = x + std::size_t(batches.member_source[member]) * num_alpha;
                    target[t] = y + std::size_t(batches.member_row[member]) * num_alpha;
                    sign[t] = batches.member_sign[member];
                }

                for(std::size_t column = 0; column < num_alpha; column++)
                {
                    for(int t = 0; t < L; t++)
                    {
                        xt[column].value[t] = source[t][column];
                    }
                }

                for(std::size_t a = first_alpha; a < last_alpha; a++)
                {
                    double partial[4][L] = {};
                    std::uint64_t k = offsets[a];
                    const std::uint64_t end = offsets[a + 1];

                    for(; k + 4 <= end; k += 4)
                    {
                        for(int u = 0; u < 4; u++)
                        {
                            const double coefficient = coefficients[index[k + u] >> 32];
                            const double* lane = xt[static_cast<std::uint32_t>(index[k + u])].value;

                            for(int t = 0; t < L; t++)
                            {
                                partial[u][t] += coefficient * lane[t];
                            }
                        }
                    }

                    for(; k < end; k++)
                    {
                        const double coefficient = coefficients[index[k] >> 32];
                        const double* lane = xt[static_cast<std::uint32_t>(index[k])].value;

                        for(int t = 0; t < L; t++)
                        {
                            partial[0][t] += coefficient * lane[t];
                        }
                    }

                    double sum[L];

                    for(int t = 0; t < L; t++)
                    {
                        sum[t] = ((partial[0][t] + partial[1][t]) + partial[2][t]) + partial[3][t];
                    }

                    for(int t = 0; t < count; t++)
                    {
                        target[t][a] += sign[t] * sum[t];
                    }
                }
            }
        }
    }
}

template <typename T>
void add_sparse_cross_arm(const TensorProductEngine<T>& tier, const T* x, T* y)
{
    const std::size_t num_alpha = tier.num_strings[0];
    const std::size_t num_beta = tier.num_strings[1];
    const SectorConnections<T>& beta = tier.sectors[1];
    const SparseCrossTable<T>& sparse = tier.sparse_cross;
#pragma omp parallel for schedule(dynamic, 4)
    for(std::size_t b = 0; b < num_beta; b++)
    {
        T* __restrict y_row = y + b * num_alpha;

        for(std::uint64_t kb = beta.single_offsets[b]; kb < beta.single_offsets[b + 1]; kb++)
        {
            const std::uint64_t packed_entry = beta.single_packed_entries[kb];
            const T* __restrict x_row = x + std::size_t(single_column(packed_entry)) * num_alpha;
            const std::uint32_t beta_slot = beta.pair_slot[single_directed_pair(packed_entry)];
            const double beta_sign = single_jw_sign(packed_entry);

            for(std::uint64_t entry = sparse.offsets[beta_slot]; entry < sparse.offsets[beta_slot + 1]; entry++)
            {
                const std::uint32_t alpha_slot = sparse.alpha_slot[entry];
                const T value = beta_sign * sparse.value[entry];

                for(std::uint64_t k = sparse.single_offsets[alpha_slot]; k < sparse.single_offsets[alpha_slot + 1]; k++)
                {
                    y_row[sparse.single_row[k]] += value * sparse.single_sign[k] * x_row[sparse.single_column[k]];
                }
            }
        }
    }
}

/**
 * Adds the cross-sector (alpha-beta) arm. Each task owns one beta row.
 *
 * @param tier The tensor-product engine.
 * @param x The input vector.
 * @param y The output vector.
 */
template <typename T>
void add_cross_sector_arm(const TensorProductEngine<T>& tier, const T* x, T* y)
{
    const std::size_t num_alpha = tier.num_strings[0];
    const std::size_t num_beta = tier.num_strings[1];
    const SectorConnections<T>& alpha = tier.sectors[0];
    const SectorConnections<T>& beta = tier.sectors[1];
    const std::size_t alpha_columns = 2 * alpha.slot_pair.size();
    const std::uint64_t* __restrict index = alpha.single_cross_index.data();

    if(!tier.sparse_cross.offsets.empty())
    {
        add_sparse_cross_arm(tier, x, y);
        return;
    }

    if constexpr(std::is_same_v<T, double>)
    {
        if(!tier.cross_batches.tile_block.empty())
        {
            add_cross_slot_batches(tier, x, y);
            return;
        }
    }

#pragma omp parallel for schedule(dynamic, 1)
    for(std::size_t b = 0; b < num_beta; b++)
    {
        T* __restrict y_row = y + b * num_alpha;

        for(std::uint64_t kb = beta.single_offsets[b]; kb < beta.single_offsets[b + 1]; kb++)
        {
            const std::uint64_t packed_entry = beta.single_packed_entries[kb];
            const T* __restrict x_row = x + std::size_t(single_column(packed_entry)) * num_alpha;
            const T* __restrict coefficients =
                tier.cross_sector_table.data() + beta.pair_slot[single_directed_pair(packed_entry)] * alpha_columns;
            const double beta_sign = single_jw_sign(packed_entry);

            for(std::size_t a = 0; a < num_alpha; a++)
            {
                const T sum = gather_cross_sum(coefficients,
                                               index,
                                               x_row,
                                               alpha.single_offsets[a],
                                               alpha.single_offsets[a + 1]);
                y_row[a] += beta_sign * sum;
            }
        }
    }
}

/**
 * Computes y = H x over a tensor-product subspace for the groups of the sector tier.
 *
 * @param tier The tensor-product engine. It must be ready.
 * @param diag The diagonal, or nullptr for zero.
 * @param x The input vector, index beta * num_alpha + alpha.
 * @param y The output vector. The function sets every entry.
 */
template <typename T>
void tensor_product_apply(const TensorProductEngine<T>& tier, const T* diag, const T* x, T* y)
{
    const std::size_t size = checked_multiply(tier.num_strings[0], tier.num_strings[1]);
    checked_multiply(size, sizeof(T));

    apply_alpha_arms(tier, diag, x, y);
    add_beta_arms(tier, x, y);
    add_cross_sector_arm(tier, x, y);
}

/**
 * Calls emit(column, value) for every element of one row with |value| > ATOL.
 *
 * The diagonal comes first, then the alpha arms, the beta arms and the cross arm.
 *
 * @param tier The tensor-product engine. It must be ready.
 * @param diag The diagonal, or nullptr for zero.
 * @param row The row index, beta * num_alpha + alpha.
 * @param emit The callback.
 */
template <typename T, typename Emit>
void tensor_product_row_elements(const TensorProductEngine<T>& tier, const T* diag, const std::size_t row, Emit&& emit)
{
    const std::size_t num_alpha = tier.num_strings[0];
    const std::size_t b = row / num_alpha;
    const std::size_t a = row % num_alpha;
    const SectorConnections<T>& alpha = tier.sectors[0];
    const SectorConnections<T>& beta = tier.sectors[1];
    auto emit_nonzero = [&emit](const std::size_t column, const T& value) {
        if(std::abs(value) > ATOL)
        {
            emit(column, value);
        }
    };

    if(diag != nullptr && diag[row] != T(0))
    {
        emit(row, diag[row]);
    }

    for(std::uint64_t k = alpha.single_offsets[a]; k < alpha.single_offsets[a + 1]; k++)
    {
        const std::uint64_t packed_entry = alpha.single_packed_entries[k];
        const T cross = cross_value(tier, 0, alpha.pair_slot[single_directed_pair(packed_entry)], b);
        emit_nonzero(b * num_alpha + single_column(packed_entry),
                     alpha.single_values[k] + single_own_sign(packed_entry) * cross);
    }

    for(std::uint64_t k = alpha.double_offsets[a]; k < alpha.double_offsets[a + 1]; k++)
    {
        emit_nonzero(b * num_alpha + alpha.double_columns[k], alpha.double_values[k]);
    }

    for(std::uint64_t k = beta.single_offsets[b]; k < beta.single_offsets[b + 1]; k++)
    {
        const std::uint64_t packed_entry = beta.single_packed_entries[k];
        const T cross = cross_value(tier, 1, beta.pair_slot[single_directed_pair(packed_entry)], a);
        emit_nonzero(std::size_t(single_column(packed_entry)) * num_alpha + a,
                     beta.single_values[k] + single_own_sign(packed_entry) * cross);
    }

    for(std::uint64_t k = beta.double_offsets[b]; k < beta.double_offsets[b + 1]; k++)
    {
        emit_nonzero(std::size_t(beta.double_columns[k]) * num_alpha + a, beta.double_values[k]);
    }

    const std::size_t alpha_columns = 2 * alpha.slot_pair.size();

    for(std::uint64_t kb = beta.single_offsets[b]; kb < beta.single_offsets[b + 1]; kb++)
    {
        const std::uint64_t packed_entry = beta.single_packed_entries[kb];
        const std::uint32_t beta_slot = beta.pair_slot[single_directed_pair(packed_entry)];
        const bool sparse = !tier.sparse_cross.offsets.empty();
        const T* coefficients = sparse ? nullptr : tier.cross_sector_table.data() + beta_slot * alpha_columns;
        const std::size_t target_row = std::size_t(single_column(packed_entry)) * num_alpha;
        const double beta_sign = single_jw_sign(packed_entry);

        for(std::uint64_t ka = alpha.single_offsets[a]; ka < alpha.single_offsets[a + 1]; ka++)
        {
            const std::uint64_t index = alpha.single_cross_index[ka];
            const std::uint32_t signed_slot = static_cast<std::uint32_t>(index >> 32);
            const T coefficient =
                sparse ? sparse_cross_value(tier.sparse_cross, beta_slot, signed_slot) : coefficients[signed_slot];
            emit_nonzero(target_row + static_cast<std::uint32_t>(index), beta_sign * coefficient);
        }
    }
}
