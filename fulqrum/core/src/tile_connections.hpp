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
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "operator_tables.hpp"
#include "subspace.hpp"
#include "tensor_product/connections.hpp"
#include "tier_utils.hpp"

/**
 * Connections from output half strings in one spin sector to a block of source half strings.
 *
 * A half string describes occupations in one sector: 0 for alpha, 1 for beta.
 * Output strings identify matrix rows; source strings identify matrix columns.
 * A tile covers a consecutive output range and one source block in that sector.
 * It stores same-sector single and double excitations, without a diagonal list.
 * Excitations that change both sectors use the separate cross-sector table.
 *
 * single_offsets and double_offsets are CSR-style row boundaries: entries for
 * local output row r lie in [offsets[r], offsets[r + 1]). Both arrays have one
 * more element than the output row count and start at zero in a built tile. The matching value
 * arrays have one value per packed entry or column; empty rows have equal boundaries.
 *
 * single_packed_entries uses pack_single: a source column, directed orbital pair,
 * own-sector sign and Jordan-Wigner (JW) sign in one 64-bit packed entry. The source
 * column is relative to the start of the tile's source block. For other-sector
 * output string b, its same-sector matrix contribution is
 * single_values[e] + own_sign * cross_table[b][directed_pair]. The packed entry's JW
 * sign is stored separately for consumers that need it; it is not an extra
 * multiplier in this expression. A directed pair is 2 * pair + direction,
 * with direction 0 when the lower orbital is occupied in the output string.
 *
 * double_columns is also block-relative, and double_values holds the matching
 * same-sector double-excitation contributions. All vectors own their storage;
 * the TileConnections structure retains no pointers to the input subspace or operator tables.
 */
template <typename T>
struct TileConnections
{
    std::vector<std::uint64_t> single_offsets;
    std::vector<std::uint64_t> single_packed_entries;
    std::vector<T> single_values;
    std::vector<std::uint64_t> double_offsets;
    std::vector<std::uint32_t> double_columns;
    std::vector<T> double_values;
};

/**
 * Find same-sector connections for a consecutive range of output half strings.
 *
 * Each result describes one output matrix row. Source columns are indices in
 * the complete half-string array of that sector; they are not yet block-relative.
 * Singles and doubles are discovered separately, using occupied and empty
 * orbitals to generate candidates and the half-key lookup to find source strings
 * present in the subspace. No full Cartesian determinant map is constructed.
 * The returned vector preserves the discovery order within each output row.
 *
 * Rows are processed with OpenMP. Each thread owns its temporary key and orbital
 * buffers. The result owns its connection/value arrays; inputs are read-only
 * and need to remain valid only for the duration of this call.
 *
 * @tparam T Value type, double or std::complex<double>.
 * @param tables Operator tables matching the subspace and sector layout.
 * @param store Non-owning view of a half-string product subspace.
 * @param groups Index from sector/orbital pairs to operator groups.
 * @param sector Excitation sector: 0 for alpha, 1 for beta.
 * @param num_pairs Number of unordered orbital pairs in each sector.
 * @param output_begin First output half-string index, inclusive.
 * @param output_end Last output half-string index, exclusive.
 * @return One StringConnections per output row; result[r] corresponds to
 *     output index output_begin + r. Source indices are sector-wide.
 * @throws std::invalid_argument If the sector or output range is invalid.
 *     Size arithmetic is checked before allocating the result vector.
 */
template <typename T>
std::vector<StringConnections<T>> find_output_connections(const OperatorTables& tables,
                                                          const HalfKeyView& store,
                                                          const SectorGroupIndex& groups,
                                                          const int sector,
                                                          const std::array<std::size_t, 2>& num_pairs,
                                                          const std::size_t output_begin,
                                                          const std::size_t output_end)
{
    const std::size_t half_width = sector == 0 ? store.split_bit() : store.width() - store.split_bit();
    if(sector < 0 || sector > 1 || output_begin > output_end || output_end > store.num_half_keys(sector))
        throw std::invalid_argument("Invalid tile sector or output range.");
    checked_multiply(output_end - output_begin, sizeof(StringConnections<T>));
    std::vector<StringConnections<T>> found(output_end - output_begin);
#pragma omp parallel
    {
        std::vector<KeyWord> row(store.num_words());
        std::vector<KeyWord> target(store.half_words(sector));
        std::vector<std::size_t> occupied;
        std::vector<std::size_t> empty;
#pragma omp for schedule(dynamic, 16)
        for(std::size_t string = output_begin; string < output_end; string++)
        {
            const KeyWord* key = store.half_key(sector, string);
            embed_half_string(store, sector, string, row.data());
            occupied.clear();
            empty.clear();
            for(std::size_t rank = 0; rank < half_width; rank++)
            {
                (half_bit(key, rank) ? occupied : empty).push_back(rank);
            }
            StringConnections<T>& string_found = found[string - output_begin];
            find_string_singles(tables, store, groups, sector, key, row.data(), occupied, empty, target, string_found);
            find_string_doubles(tables, store, groups, sector, num_pairs, key, row.data(), occupied, empty, target,
                                string_found);
        }
    }
    return found;
}

/**
 * Find the source block containing a sector-wide column index.
 *
 * Block k covers [block_bounds[k], block_bounds[k + 1]). An index equal to a
 * boundary belongs to the block starting there. Repeated boundaries describe
 * empty blocks. This helper performs no validation: callers must supply at least
 * two nondecreasing boundaries.
 *
 * @param block_bounds Source-block boundaries in sector-wide string indices.
 * @param column Source half-string index in the complete sector array.
 * @return Containing block index, or block_bounds.size() - 1 if the column is
 *     outside the covered range. This sentinel is the number of blocks.
 */
inline std::size_t block_of(const std::vector<std::size_t>& block_bounds, const std::size_t column)
{
    const std::size_t num_blocks = block_bounds.size() - 1;
    if(column < block_bounds.front() || column >= block_bounds.back())
    {
        return num_blocks;
    }
    const auto after = std::upper_bound(block_bounds.begin(), block_bounds.end(), column);
    return static_cast<std::size_t>(after - block_bounds.begin()) - 1;
}

/**
 * Convert a packed single-excitation column from sector-wide to block-relative indexing.
 *
 * Subtracts offset from the column stored in the low 32 bits. The directed pair,
 * own-sector sign and JW sign in the remaining bits are preserved.
 *
 * @param packed_entry Single-excitation packed entry produced by pack_single.
 * @param offset First sector-wide source index of the destination block.
 * @return Updated packed entry. The caller must ensure offset <= single_column(packed_entry);
 *     subtraction bounds are not checked here.
 */
inline std::uint64_t shift_single_column(const std::uint64_t packed_entry, const std::size_t offset)
{
    const std::uint64_t column = single_column(packed_entry) - offset;
    return (packed_entry & ~std::uint64_t(0xffffffffu)) | column;
}

/**
 * Partition output-row connections by source block and flatten them into tiles.
 *
 * Every returned tile has found.size() output rows, in the same order as found.
 * A connection is placed in the tile containing its source column; that column
 * is then made relative to the block's first index. Connections outside the
 * covered source range are omitted. Order within each row/block is preserved;
 * the function does not sort columns or combine duplicate connections.
 *
 * Singles and doubles get separate CSR-style offsets and matching value arrays.
 * Each offset array contains found.size() + 1 entries, even for blocks with no
 * connections. The input arrays are not changed, and the returned tiles own
 * their storage.
 *
 * @tparam T Connection value type.
 * @param found Output-row connections with sector-wide source columns and
 *     matching packed entry/column and value array lengths.
 * @param block_bounds Nondecreasing source boundaries; block k is
 *     [block_bounds[k], block_bounds[k + 1]). At least two entries are required.
 * @return One TileConnections per source block, including empty blocks.
 * @throws std::invalid_argument If boundaries are unsorted or too short.
 *     Allocation failures propagate; valid packed indices and input array
 *     lengths are caller obligations.
 */
template <typename T>
std::vector<TileConnections<T>> split_by_blocks(const std::vector<StringConnections<T>>& found,
                                                const std::vector<std::size_t>& block_bounds)
{
    if(block_bounds.size() < 2 || !std::is_sorted(block_bounds.begin(), block_bounds.end()))
        throw std::invalid_argument("Tile block boundaries must be sorted and contain at least two entries.");
    const std::size_t num_blocks = block_bounds.size() - 1;
    std::vector<TileConnections<T>> tiles(num_blocks);
    for(TileConnections<T>& tile : tiles)
    {
        tile.single_offsets.assign(1, 0);
        tile.double_offsets.assign(1, 0);
    }
    for(const StringConnections<T>& string : found)
    {
        for(std::size_t edge = 0; edge < string.single_packed_entries.size(); edge++)
        {
            const std::uint64_t packed_entry = string.single_packed_entries[edge];
            const std::size_t block = block_of(block_bounds, single_column(packed_entry));
            if(block < num_blocks)
            {
                tiles[block].single_packed_entries.push_back(shift_single_column(packed_entry, block_bounds[block]));
                tiles[block].single_values.push_back(string.single_values[edge]);
            }
        }
        for(std::size_t edge = 0; edge < string.double_columns.size(); edge++)
        {
            const std::size_t column = string.double_columns[edge];
            const std::size_t block = block_of(block_bounds, column);
            if(block < num_blocks)
            {
                tiles[block].double_columns.push_back(static_cast<std::uint32_t>(column - block_bounds[block]));
                tiles[block].double_values.push_back(string.double_values[edge]);
            }
        }
        for(TileConnections<T>& tile : tiles)
        {
            tile.single_offsets.push_back(tile.single_packed_entries.size());
            tile.double_offsets.push_back(tile.double_columns.size());
        }
    }
    return tiles;
}

/**
 * Build block-local connections for an output range in one spin sector.
 *
 * First finds connections with sector-wide columns, then partitions them with
 * split_by_blocks. A returned tile contains every output row in the requested
 * range and only source columns within its block. Output row r represents
 * half-string index output_begin + r; source column c represents
 * half-string index block_bounds[block] + c.
 *
 * @tparam T Value type, double or std::complex<double>.
 * @param tables Operator tables matching the product subspace.
 * @param store Non-owning view of the product subspace; its owner must stay alive.
 * @param groups Sector/orbital-pair group index derived from tables.
 * @param sector Excitation sector: 0 for alpha, 1 for beta.
 * @param num_pairs Number of unordered orbital pairs in each sector.
 * @param output_begin First output half-string index, inclusive.
 * @param output_end Last output half-string index, exclusive.
 * @param block_bounds Nondecreasing sector-wide source boundaries. They may
 *     cover part of the sector but must not extend beyond the sector's half-string array.
 * @return One owning TileConnections per source block. No references to inputs
 *     are retained, and no full Cartesian determinant map is constructed.
 * @throws std::invalid_argument If the sector, output range or block boundaries
 *     are invalid. Size arithmetic is checked before allocating result arrays.
 */
template <typename T>
std::vector<TileConnections<T>> build_tile_connections(const OperatorTables& tables,
                                                       const HalfKeyView& store,
                                                       const SectorGroupIndex& groups,
                                                       const int sector,
                                                       const std::array<std::size_t, 2>& num_pairs,
                                                       const std::size_t output_begin,
                                                       const std::size_t output_end,
                                                       const std::vector<std::size_t>& block_bounds)
{
    if(sector < 0 || sector > 1 || block_bounds.size() < 2 ||
       !std::is_sorted(block_bounds.begin(), block_bounds.end()) || block_bounds.back() > store.num_half_keys(sector))
        throw std::invalid_argument("Invalid tile block boundaries.");
    return split_by_blocks(find_output_connections<T>(tables, store, groups, sector, num_pairs, output_begin, output_end),
                           block_bounds);
}

/**
 * Cache the other sector's contribution to same-sector single excitations.
 *
 * A single excitation changes two orbitals in sector, but its coefficient can
 * depend on occupations in the other sector. This table stores that dependency
 * separately from the single packed entry's own-sector value and sign. It does not
 * describe excitations that change both sectors.
 *
 * The returned vector is row-major. With stride = 2 * num_pairs[sector], entry
 * table[b * stride + 2 * pair + direction] is the summed cross-factor value for
 * other-sector output string other_begin + b. Direction 0 has the lower orbital
 * occupied in the excitation sector's output string; direction 1 has the higher.
 * Missing groups or empty cross-factor ranges leave zero entries.
 *
 * @tparam T Value type, double or std::complex<double>.
 * @param tables Operator tables containing the single-excitation factor ranges.
 * @param store Non-owning view of a half-string product subspace.
 * @param groups Sector/orbital-pair group index matching tables.
 * @param sector Sector of the single excitation: 0 for alpha, 1 for beta.
 * @param num_pairs Number of unordered orbital pairs in each sector.
 * @param other_begin First output string in sector 1 - sector, inclusive.
 * @param other_end Last output string in sector 1 - sector, exclusive.
 * @return Owning flat table with other_end - other_begin rows and
 *     2 * num_pairs[sector] columns. Inputs are read-only; rows use OpenMP.
 * @throws std::invalid_argument If the sector or other-sector range is invalid.
 *     Table-size overflow is rejected before allocating the table.
 */
template <typename T>
std::vector<T> build_tile_cross_table(const OperatorTables& tables,
                                      const HalfKeyView& store,
                                      const SectorGroupIndex& groups,
                                      const int sector,
                                      const std::array<std::size_t, 2>& num_pairs,
                                      const std::size_t other_begin,
                                      const std::size_t other_end)
{
    const FactorList& factors = tables.factors();
    const int other = 1 - sector;
    if(sector < 0 || sector > 1 || other_begin > other_end || other_end > store.num_half_keys(1 - sector))
        throw std::invalid_argument("Invalid tile cross-table range.");
    const std::size_t directed_pairs = checked_multiply(2, num_pairs[sector]);
    const std::size_t cells = checked_multiply(other_end - other_begin, directed_pairs);
    checked_multiply(cells, sizeof(T));
    std::vector<T> table(cells, T(0));
#pragma omp parallel
    {
        std::vector<KeyWord> row(store.num_words());
#pragma omp for schedule(dynamic, 16)
        for(std::size_t string = other_begin; string < other_end; string++)
        {
            embed_half_string(store, other, string, row.data());
            T* values = table.data() + (string - other_begin) * directed_pairs;
            for(std::size_t pair = 0; pair < num_pairs[sector]; pair++)
            {
                const std::uint32_t group = groups.single[sector][pair];
                if(group == NO_GROUP)
                {
                    continue;
                }
                for(std::size_t direction = 0; direction < 2; direction++)
                {
                    const EntryRange cross = tables.group(group).single[direction].cross;
                    if(cross.end > cross.begin)
                    {
                        values[2 * pair + direction] = to_value<T>(factors.sum(cross.begin, cross.end, row.data()));
                    }
                }
            }
        }
    }
    return table;
}

/**
 * Build coefficients for excitations that change one orbital pair in each spin sector.
 *
 * This is a pair-coefficient table, not a matrix indexed by determinant strings.
 * It uses the cross_sector_double groups in tables; other groups are ignored.
 * Both sectors have directed pairs: directed = 2 * pair + direction, where
 * direction 0 has the lower orbital occupied in the output string.
 *
 * The flat row-major table has 2 * num_pairs[1] beta-directed rows and
 * 4 * num_pairs[0] columns. For beta-directed index b and alpha-directed index a,
 * table[b * stride + 2 * a] holds the coefficient for alpha JW sign +1, and
 * table[b * stride + 2 * a + 1] holds its negative for alpha JW sign -1,
 * where stride = 4 * num_pairs[0]. JW means Jordan-Wigner: the occupation-parity
 * sign associated with the orbital transfer. Unused pair combinations remain zero.
 *
 * @tparam T Value type, double or std::complex<double>.
 * @param tables Classified operator tables with valid sector/rank metadata and
 *     factored cross-sector coefficients.
 * @param num_pairs Number of unordered orbital pairs in alpha (index 0) and
 *     beta (index 1), matching tables.layout().
 * @return Owning coefficient vector; no references to tables are retained.
 *     Checked size arithmetic rejects overflow; allocation failures propagate.
 */
template <typename T>
std::vector<T> build_cross_sector_table(const OperatorTables& tables,
                                        const std::array<std::size_t, 2>& num_pairs)
{
    const std::size_t alpha_columns = checked_multiply(4, num_pairs[0]);
    const std::size_t cells = checked_multiply(checked_multiply(2, num_pairs[1]), alpha_columns);
    checked_multiply(cells, sizeof(T));
    std::vector<T> table(cells, T(0));
    const std::vector<width_t>& rank = tables.layout().rank_in_sector;
    const std::vector<std::uint8_t>& sector_of = tables.layout().sector_of;

    for(std::size_t group = 0; group < tables.num_groups(); group++)
    {
        const GroupData& plan = tables.group(group);

        if(plan.tier != GroupTier::cross_sector_double)
        {
            continue;
        }

        std::array<std::vector<width_t>, 2> ranks;

        for(const width_t qubit : plan.flips)
        {
            ranks[sector_of[qubit]].push_back(rank[qubit]);
        }

        const std::size_t alpha_pair = rank_pair_index(ranks[0][0], ranks[0][1]);
        const std::size_t beta_pair = rank_pair_index(ranks[1][0], ranks[1][1]);

        for(std::size_t pattern = 0; pattern < 4; pattern++)
        {
            const std::size_t alpha_directed = 2 * alpha_pair + (pattern & 1);
            const std::size_t beta_directed = 2 * beta_pair + (pattern >> 1);
            const T value = to_value<T>(plan.cross_values[pattern]);
            T* row = table.data() + beta_directed * alpha_columns;
            row[2 * alpha_directed] = value;
            row[2 * alpha_directed + 1] = -value;
        }
    }

    return table;
}
