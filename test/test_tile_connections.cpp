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
#include "doctest.h"
#include "fulqrum.hpp"
#include "subspace_fixture.hpp"
#include "../fulqrum/core/src/tensor_product/cpu.hpp"
#include "../fulqrum/core/src/tile_connections.hpp"
#include "../fulqrum/core/src/tile_connections.hpp"
#include <algorithm>
#include <complex>
#include <random>
#include <string>
#include <tuple>
#include <vector>

typedef std::complex<double> complex;

namespace
{

/**
 * Makes a Hermitian molecular-like fermionic operator on block-ordered sectors.
 *
 * @param norb The number of orbitals per sector.
 * @param complex_values When true, the coefficients are complex.
 */
FermionicOperator tile_test_operator(const width_t norb, const bool complex_values)
{
    std::mt19937_64 engine(31);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    auto coefficient = [&]() { return complex(uniform(engine), complex_values ? uniform(engine) : 0.0); };
    std::vector<TermData> terms;
    auto add_with_adjoint = [&](const std::string& ops, std::vector<width_t> inds, complex value) {
        terms.emplace_back(ops, inds, value);
        std::string adjoint_ops(ops.rbegin(), ops.rend());
        for(char& op : adjoint_ops)
        {
            op = op == '+' ? '-' : (op == '-' ? '+' : op);
        }
        std::reverse(inds.begin(), inds.end());
        terms.emplace_back(adjoint_ops, inds, std::conj(value));
    };
    for(width_t own = 0; own < 2; own++)
    {
        const width_t base = own * norb;
        const width_t other = (1 - own) * norb;
        for(width_t i = 0; i < norb; i++)
        {
            for(width_t j = i + 1; j < norb; j++)
            {
                add_with_adjoint("+-", {width_t(base + i), width_t(base + j)}, coefficient());
                for(width_t r = 0; r < norb; r++)
                {
                    add_with_adjoint("+-+-", {width_t(base + i), width_t(base + j), width_t(other + r), width_t(other + r)}, coefficient());
                }
                for(width_t k = 0; k < norb; k++)
                {
                    for(width_t l = k + 1; l < norb; l++)
                    {
                        if(k != i && k != j && l != i && l != j)
                        {
                            add_with_adjoint("++--", {width_t(base + i), width_t(base + j), width_t(base + k), width_t(base + l)}, coefficient());
                        }
                    }
                }
            }
        }
    }
    for(width_t i = 0; i < norb; i++)
    {
        for(width_t j = 0; j < norb; j++)
        {
            for(width_t k = 0; k < norb; k++)
            {
                for(width_t l = 0; l < norb; l++)
                {
                    if(i != j && k != l)
                    {
                        add_with_adjoint("+-+-", {i, j, width_t(norb + k), width_t(norb + l)}, coefficient());
                    }
                }
            }
        }
    }
    return FermionicOperator(static_cast<width_t>(2 * norb), terms);
}

/**
 * Returns the half strings of a width with a given number of set bits.
 *
 * @param width The number of bits.
 * @param electrons The number of set bits.
 */
std::vector<KeyWord> tile_half_strings(const width_t width, const unsigned int electrons)
{
    std::vector<KeyWord> keys;
    for(KeyWord value = 0; value < (KeyWord(1) << width); value++)
    {
        if(static_cast<unsigned int>(__builtin_popcountll(value)) == electrons)
        {
            keys.push_back(value);
        }
    }
    return keys;
}

/**
 * Returns the single and double connections of one row as sorted lists with sector columns.
 *
 * @param packed_entries The single packed entries of the row.
 * @param single_values The single values of the row.
 * @param columns The double columns of the row.
 * @param double_values The double values of the row.
 * @param offset The value to add to each column.
 */
template <typename T>
std::pair<std::vector<std::tuple<std::uint64_t, T>>, std::vector<std::tuple<std::uint32_t, T>>>
row_entries(const std::uint64_t* packed_entries,
            const T* single_values,
            const std::size_t num_singles,
            const std::uint32_t* columns,
            const T* double_values,
            const std::size_t num_doubles,
            const std::size_t offset)
{
    std::vector<std::tuple<std::uint64_t, T>> singles;
    std::vector<std::tuple<std::uint32_t, T>> doubles;
    for(std::size_t edge = 0; edge < num_singles; edge++)
    {
        const std::uint64_t packed_entry = packed_entries[edge];
        const std::uint64_t moved = (packed_entry & ~std::uint64_t(0xffffffffu)) | (single_column(packed_entry) + offset);
        singles.emplace_back(moved, single_values[edge]);
    }
    for(std::size_t edge = 0; edge < num_doubles; edge++)
    {
        doubles.emplace_back(static_cast<std::uint32_t>(columns[edge] + offset), double_values[edge]);
    }
    return {singles, doubles};
}

/**
 * Orders tuples by their first field, then by the real and imaginary parts of the second.
 */
template <typename Tuple>
bool entry_less(const Tuple& left, const Tuple& right)
{
    if(std::get<0>(left) != std::get<0>(right))
    {
        return std::get<0>(left) < std::get<0>(right);
    }
    const std::complex<double> a(std::get<1>(left));
    const std::complex<double> b(std::get<1>(right));
    return a.real() != b.real() ? a.real() < b.real() : a.imag() < b.imag();
}

/**
 * Checks that the tiles of a output range and blocks hold the same connections as the sector tier.
 *
 * @param norb The number of orbitals per sector.
 * @param electrons The number of electrons per sector.
 */
template <typename T>
void check_tiles_match_sector(const width_t norb, const unsigned int electrons)
{
    const bool complex_values = !std::is_same_v<T, double>;
    QubitOperator off = tile_test_operator(norb, complex_values).extended_jw_transformation().split_diagonal().second;
    off.group_sort();
    const std::vector<std::size_t> ptrs = off.group_ptrs();
    std::vector<std::vector<width_t>> flips;
    set_group_offdiag_indices(off.terms, flips, ptrs.data(), ptrs.size() - 1);
    const SubspaceFixture store = SubspaceFixture::from_halves(
        tile_half_strings(norb, electrons), norb, tile_half_strings(norb, electrons), norb);
    const SectorLayout layout = choose_sector_layout(SpinLayout::automatic, store, flips, off.width);
    REQUIRE(layout.valid);
    const OperatorTables tables = OperatorTables::build(off, ptrs.data(), flips, layout, false);
    REQUIRE(tables.count(GroupTier::generic) == 0);
    const TensorProductEngine<T> tier = build_tensor_product_engine<T>(tables, store);
    REQUIRE(tier.ready);
    const SectorGroupIndex groups = index_sector_groups(tables, tier.num_pairs);

    for(int sector = 0; sector < 2; sector++)
    {
        const std::size_t num_strings = tier.num_strings[sector];
        const SectorConnections<T>& full = tier.sectors[sector];
        const std::vector<std::size_t> bounds = {0, num_strings / 3, num_strings / 2 + 1, num_strings};
        const std::vector<std::size_t> output_bounds = {0, num_strings / 4, num_strings};
        for(std::size_t range = 0; range + 1 < output_bounds.size(); range++)
        {
            const std::size_t output_begin = output_bounds[range];
            const std::size_t output_end = output_bounds[range + 1];
            const std::vector<TileConnections<T>> tiles =
                build_tile_connections<T>(tables, store, groups, sector, tier.num_pairs, output_begin, output_end, bounds);
            REQUIRE(tiles.size() == bounds.size() - 1);
            for(std::size_t string = output_begin; string < output_end; string++)
            {
                const std::size_t row = string - output_begin;
                std::vector<std::tuple<std::uint64_t, T>> tile_singles;
                std::vector<std::tuple<std::uint32_t, T>> tile_doubles;
                for(std::size_t block = 0; block < tiles.size(); block++)
                {
                    const TileConnections<T>& tile = tiles[block];
                    REQUIRE(tile.single_offsets.size() == output_end - output_begin + 1);
                    const std::size_t s0 = tile.single_offsets[row];
                    const std::size_t d0 = tile.double_offsets[row];
                    auto entries = row_entries<T>(tile.single_packed_entries.data() + s0, tile.single_values.data() + s0,
                                                  tile.single_offsets[row + 1] - s0, tile.double_columns.data() + d0,
                                                  tile.double_values.data() + d0, tile.double_offsets[row + 1] - d0,
                                                  bounds[block]);
                    tile_singles.insert(tile_singles.end(), entries.first.begin(), entries.first.end());
                    tile_doubles.insert(tile_doubles.end(), entries.second.begin(), entries.second.end());
                }
                const std::size_t s0 = full.single_offsets[string];
                const std::size_t d0 = full.double_offsets[string];
                auto expected = row_entries<T>(full.single_packed_entries.data() + s0, full.single_values.data() + s0,
                                               full.single_offsets[string + 1] - s0, full.double_columns.data() + d0,
                                               full.double_values.data() + d0, full.double_offsets[string + 1] - d0, 0);
                std::sort(tile_singles.begin(), tile_singles.end(), entry_less<std::tuple<std::uint64_t, T>>);
                std::sort(expected.first.begin(), expected.first.end(), entry_less<std::tuple<std::uint64_t, T>>);
                std::sort(tile_doubles.begin(), tile_doubles.end(), entry_less<std::tuple<std::uint32_t, T>>);
                std::sort(expected.second.begin(), expected.second.end(), entry_less<std::tuple<std::uint32_t, T>>);
                CHECK(tile_singles == expected.first);
                CHECK(tile_doubles == expected.second);
            }
        }

        const std::size_t num_other = tier.num_strings[1 - sector];
        const std::size_t other_begin = num_other / 3;
        const std::size_t other_end = num_other - 1;
        const std::vector<T> cross =
            build_tile_cross_table<T>(tables, store, groups, sector, tier.num_pairs, other_begin, other_end);
        const std::size_t directed_pairs = 2 * tier.num_pairs[sector];
        REQUIRE(cross.size() == (other_end - other_begin) * directed_pairs);
        for(std::size_t b = other_begin; b < other_end; b++)
        {
            for(std::size_t pair = 0; pair < directed_pairs; pair++)
            {
                if(full.pair_slot[pair] == NO_GROUP)
                {
                    continue;
                }
                CHECK(cross[(b - other_begin) * directed_pairs + pair] == cross_value(tier, sector, full.pair_slot[pair], b));
            }
        }
    }
}

}

TEST_CASE("Tile connections of a output range and source blocks equal the sector tier, real")
{
    check_tiles_match_sector<double>(5, 2);
    check_tiles_match_sector<double>(6, 3);
}

TEST_CASE("Tile connections of a output range and source blocks equal the sector tier, complex")
{
    check_tiles_match_sector<complex>(5, 2);
}

TEST_CASE("Columns outside every source block are dropped and the others are block relative")
{
    const std::vector<std::size_t> bounds = {2, 5, 9};
    CHECK(block_of(bounds, 0) == 2);
    CHECK(block_of(bounds, 2) == 0);
    CHECK(block_of(bounds, 4) == 0);
    CHECK(block_of(bounds, 5) == 1);
    CHECK(block_of(bounds, 8) == 1);
    CHECK(block_of(bounds, 9) == 2);
    StringConnections<double> string;
    string.single_packed_entries = {pack_single(1, 4, 1, -1), pack_single(6, 3, -1, 1)};
    string.single_values = {0.5, 0.25};
    string.double_columns = {3, 9};
    string.double_values = {2.0, 3.0};
    const std::vector<TileConnections<double>> tiles = split_by_blocks<double>({string}, bounds);
    REQUIRE(tiles.size() == 2);
    CHECK(tiles[0].single_packed_entries.empty());
    CHECK(tiles[0].double_columns == std::vector<std::uint32_t>{1});
    REQUIRE(tiles[1].single_packed_entries.size() == 1);
    CHECK(single_column(tiles[1].single_packed_entries[0]) == 1);
    CHECK(single_directed_pair(tiles[1].single_packed_entries[0]) == 3);
    CHECK(single_own_sign(tiles[1].single_packed_entries[0]) == -1.0);
    CHECK(single_jw_sign(tiles[1].single_packed_entries[0]) == 1.0);
    CHECK(tiles[1].single_values == std::vector<double>{0.25});
    CHECK(tiles[1].double_columns.empty());
    CHECK(tiles[0].single_offsets == std::vector<std::uint64_t>{0, 0});
    CHECK(tiles[1].single_offsets == std::vector<std::uint64_t>{0, 1});
}

TEST_CASE("Tile table allocation overflow is rejected before traversal")
{
    Subspace_t space;
    space.initialize_product({"1"}, {"1"});
    HalfKeyView view(space);
    OperatorTables tables;
    SectorGroupIndex groups;
    const std::array<std::size_t, 2> pairs = {std::numeric_limits<std::size_t>::max(), 1};
    CHECK_THROWS_AS(build_tile_cross_table<double>(tables, view, groups, 0, pairs, 0, 1), std::overflow_error);
    CHECK_THROWS_AS(build_cross_sector_table<double>(tables, pairs), std::overflow_error);
}
