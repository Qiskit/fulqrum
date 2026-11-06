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
#include "doctest.h"
#include "fulqrum.hpp"
#include "subspace_fixture.hpp"
#include "../fulqrum/core/src/tensor_product/cpu.hpp"
#include "../fulqrum/core/src/tile_connections.hpp"
#include <algorithm>
#include <array>
#include <complex>
#include <random>
#include <string>
#include <vector>

typedef std::complex<double> complex;

namespace
{

/**
 * Makes a Hermitian molecular-like fermionic operator on block-ordered sectors.
 *
 * The excitations use only the window orbitals of each sector.
 *
 * @param norb The number of orbitals per sector.
 * @param window The orbitals that the terms act on.
 * @param complex_values When true, the coefficients are complex.
 * @param two_density_cross When true, each same-sector single also gets a term with
 *     two other-sector densities.
 * @param local_cross When true, the alpha-beta doubles couple only neighbour hops that
 *     are at most one orbital apart.
 */
FermionicOperator window_molecular_operator(const width_t norb,
                                            const std::vector<width_t>& window,
                                            const bool complex_values,
                                            const bool two_density_cross,
                                            const bool local_cross = false)
{
    std::mt19937_64 engine(23);
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
        for(std::size_t wi = 0; wi < window.size(); wi++)
        {
            for(std::size_t wj = wi + 1; wj < window.size(); wj++)
            {
                const width_t i = window[wi];
                const width_t j = window[wj];
                add_with_adjoint("+-", {width_t(base + i), width_t(base + j)}, coefficient());
                if(two_density_cross)
                {
                    add_with_adjoint("+-+-+-",
                                     {width_t(base + i), width_t(base + j), width_t(other + window[0]),
                                      width_t(other + window[0]), width_t(other + window[1]), width_t(other + window[1])},
                                     coefficient());
                }
                for(const width_t r : window)
                {
                    add_with_adjoint("+-+-", {width_t(base + i), width_t(base + j), width_t(other + r), width_t(other + r)}, coefficient());
                    if(r != i && r != j)
                    {
                        add_with_adjoint("+-+-", {width_t(base + i), width_t(base + j), width_t(base + r), width_t(base + r)}, coefficient());
                    }
                }
                for(std::size_t wk = 0; wk < window.size(); wk++)
                {
                    for(std::size_t wl = wk + 1; wl < window.size(); wl++)
                    {
                        const width_t k = window[wk];
                        const width_t l = window[wl];
                        if(k != i && k != j && l != i && l != j)
                        {
                            add_with_adjoint("++--", {width_t(base + i), width_t(base + j), width_t(base + k), width_t(base + l)}, coefficient());
                        }
                    }
                }
            }
        }
    }
    for(const width_t i : window)
    {
        for(const width_t j : window)
        {
            for(const width_t k : window)
            {
                for(const width_t l : window)
                {
                    const bool near = j == i + 1 && l == k + 1 && (i > k ? i - k : k - i) <= 1;
                    if(i != j && k != l && (!local_cross || near))
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
 * Makes a Hermitian molecular-like fermionic operator on all orbitals of both sectors.
 *
 * @param norb The number of orbitals per sector.
 * @param complex_values When true, the coefficients are complex.
 */
FermionicOperator block_molecular_operator(const width_t norb, const bool complex_values)
{
    std::vector<width_t> window(norb);
    for(width_t orbital = 0; orbital < norb; orbital++)
    {
        window[orbital] = orbital;
    }
    return window_molecular_operator(norb, window, complex_values, false);
}

/**
 * Returns the half strings with a number of set bits inside a window of orbitals.
 *
 * @param norb The number of orbitals; each string has words_for_width(norb) words.
 * @param window The orbitals that may be set.
 * @param electrons The number of set bits.
 */
std::vector<KeyWord> window_strings(const width_t norb, const std::vector<width_t>& window, const unsigned int electrons)
{
    const std::size_t words = words_for_width(norb);
    std::vector<KeyWord> keys;
    for(KeyWord choice = 0; choice < (KeyWord(1) << window.size()); choice++)
    {
        if(static_cast<unsigned int>(__builtin_popcountll(choice)) != electrons)
        {
            continue;
        }
        std::vector<KeyWord> key(words, 0);
        for(std::size_t position = 0; position < window.size(); position++)
        {
            if((choice >> position) & 1u)
            {
                key[window[position] / KEY_WORD_BITS] |= KeyWord(1) << (window[position] % KEY_WORD_BITS);
            }
        }
        keys.insert(keys.end(), key.begin(), key.end());
    }
    return keys;
}

/**
 * Returns the half strings of a width with a given number of set bits.
 *
 * @param width The number of bits.
 * @param electrons The number of set bits.
 */
std::vector<KeyWord> half_strings(const width_t width, const unsigned int electrons)
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
 * Computes y = H x by evaluating every group's terms on every row of a store.
 *
 * @param off The grouped off-diagonal operator.
 * @param ptrs The group boundaries.
 * @param flips The flip qubits of each group.
 * @param store The subspace store.
 * @param x The input vector.
 */
template <typename T>
std::vector<T> reference_matvec(const QubitOperator& off,
                                const std::vector<std::size_t>& ptrs,
                                const std::vector<std::vector<width_t>>& flips,
                                const SubspaceFixture& store,
                                const std::vector<T>& x)
{
    std::vector<T> y(store.size(), T(0));
    for(std::size_t row_index = 0; row_index < store.size(); row_index++)
    {
        const KeyBitset row = store.key(row_index);
        for(std::size_t group = 0; group + 1 < ptrs.size(); group++)
        {
            KeyBitset col = row;
            for(const width_t qubit : flips[group])
            {
                col.flip(qubit);
            }
            const std::size_t col_index = store.find(col);
            if(col_index == KEY_NOT_FOUND)
            {
                continue;
            }
            T value = 0;
            for(std::size_t term = ptrs[group]; term < ptrs[group + 1]; term++)
            {
                const OperatorTerm_t& t = off.terms[term];
                accum_element(row, col, t.indices, t.values, t.coeff, t.real_phase, t.indices.size(), value);
            }
            y[row_index] += value * x[col_index];
        }
    }
    return y;
}

/**
 * Checks a tensor-product engine's matvec, and the production CSR over its row walk,
 * against the term reference.
 *
 * @param tier The engine.
 * @param off The grouped off-diagonal operator.
 * @param ptrs The group boundaries.
 * @param flips The flip qubits of each group.
 * @param store The subspace store.
 * @param x The input vector.
 * @return The matvec output.
 */
template <typename T>
std::vector<T> check_against_reference(const TensorProductEngine<T>& tier,
                                       const QubitOperator& off,
                                       const std::vector<std::size_t>& ptrs,
                                       const std::vector<std::vector<width_t>>& flips,
                                       const SubspaceFixture& store,
                                       const std::vector<T>& x)
{
    std::vector<T> y(store.size());
    tensor_product_apply<T>(tier, nullptr, x.data(), y.data());
    const std::vector<T> expected = reference_matvec<T>(off, ptrs, flips, store, x);
    double largest = 0;
    double error = 0;
    for(std::size_t index = 0; index < y.size(); index++)
    {
        largest = std::max(largest, std::abs(expected[index]));
        error = std::max(error, std::abs(y[index] - expected[index]));
    }
    CHECK(largest > 0);
    CHECK(error <= 1e-12 * largest);

    auto make_walk = [&]() {
        return [&](std::size_t row, auto&& emit) {
            tensor_product_row_elements(tier, static_cast<const T*>(nullptr), row, emit);
        };
    };
    std::vector<std::int64_t> indptr(store.size() + 1);
    csr_from_rows<std::int64_t, T>(store.size(), nullptr, make_walk, indptr.data(), nullptr, nullptr, 0);
    std::vector<std::int64_t> indices(indptr.back());
    std::vector<T> data(indptr.back());
    csr_from_rows<std::int64_t, T>(store.size(), nullptr, make_walk, indptr.data(), indices.data(), data.data(), 1);
    double csr_error = 0;
    for(std::size_t row = 0; row < store.size(); row++)
    {
        T sum = T(0);
        for(std::int64_t k = indptr[row]; k < indptr[row + 1]; k++)
        {
            sum += data[k] * x[indices[k]];
        }
        csr_error = std::max(csr_error, std::abs(sum - expected[row]));
    }
    CHECK(csr_error <= 1e-12 * largest);
    return y;
}

/**
 * Checks both cross modes of the tensor-product engine on one operator and store.
 *
 * Both modes must match the term reference. When the formed mode applies, its cross
 * values and matvec output must equal the stored ones exactly.
 *
 * @param fermionic The fermionic operator.
 * @param store The tensor-product store.
 * @param formed_applies Whether every cross entry has the formed form.
 */
struct EngineChoices
{
    std::array<bool, 2> wide;
    bool batched;
    bool sparse;
};

template <typename T>
EngineChoices check_cross_modes(FermionicOperator fermionic, const SubspaceFixture& store, const bool formed_applies)
{
    QubitOperator off = fermionic.extended_jw_transformation().split_diagonal().second;
    off.group_sort();
    const std::vector<std::size_t> ptrs = off.group_ptrs();
    std::vector<std::vector<width_t>> flips;
    set_group_offdiag_indices(off.terms, flips, ptrs.data(), ptrs.size() - 1);
    const SectorLayout layout = choose_sector_layout(SpinLayout::automatic, store, flips, off.width);
    REQUIRE(layout.valid);
    const OperatorTables tables = OperatorTables::build(off, ptrs.data(), flips, layout, false);
    REQUIRE(tables.count(GroupTier::generic) == 0);
    const TensorProductEngine<T> stored = build_tensor_product_engine<T>(tables, store, CrossMode::stored);
    const TensorProductEngine<T> formed = build_tensor_product_engine<T>(tables, store, CrossMode::formed);
    REQUIRE(stored.ready);
    REQUIRE(formed.ready);
    CHECK(!stored.cross_formed);
    CHECK(formed.cross_formed == formed_applies);
    CHECK(build_tensor_product_engine<T>(tables, store).cross_formed == formed_applies);

    std::mt19937_64 engine(5);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<T> x(store.size());
    for(T& value : x)
    {
        if constexpr(std::is_same_v<T, double>)
        {
            value = uniform(engine);
        }
        else
        {
            value = T(uniform(engine), uniform(engine));
        }
    }
    const std::vector<T> y_stored = check_against_reference<T>(stored, off, ptrs, flips, store, x);
    const std::vector<T> y_formed = check_against_reference<T>(formed, off, ptrs, flips, store, x);
    const EngineChoices choices = {{use_wide_blocks(stored, 0), use_wide_blocks(stored, 1)},
                                   !stored.cross_batches.tile_block.empty(),
                                   !stored.sparse_cross.offsets.empty()};
    CHECK(choices.batched == !formed.cross_batches.tile_block.empty());
    CHECK(choices.sparse == !formed.sparse_cross.offsets.empty());
    CHECK(!(choices.batched && choices.sparse));
    if(!formed_applies)
    {
        return choices;
    }
    CHECK(y_formed == y_stored);
    for(int sector = 0; sector < 2; sector++)
    {
        REQUIRE(formed.sectors[sector].slot_pair == stored.sectors[sector].slot_pair);
        std::size_t mismatches = 0;
        for(std::uint32_t slot = 0; slot < stored.sectors[sector].slot_pair.size(); slot++)
        {
            for(std::size_t other = 0; other < store.num_half_keys(1 - sector); other++)
            {
                mismatches += cross_value(formed, sector, slot, other) != cross_value(stored, sector, slot, other);
            }
        }
        CHECK(mismatches == 0);
    }
    return choices;
}

/**
 * Checks both cross modes on all strings of one shape.
 *
 * @param norb The number of orbitals per sector.
 * @param alpha_electrons The number of alpha electrons.
 * @param beta_electrons The number of beta electrons.
 */
template <typename T>
EngineChoices check_tensor_product_apply(const width_t norb,
                                         const unsigned int alpha_electrons,
                                         const unsigned int beta_electrons)
{
    const bool complex_values = !std::is_same_v<T, double>;
    const SubspaceFixture store = SubspaceFixture::from_halves(
        half_strings(norb, alpha_electrons), norb, half_strings(norb, beta_electrons), norb);
    return check_cross_modes<T>(block_molecular_operator(norb, complex_values), store, true);
}

} // namespace

TEST_CASE("tensor-product engine matches the term reference")
{
    check_tensor_product_apply<double>(5, 2, 2);
    check_tensor_product_apply<double>(6, 3, 2);
    check_tensor_product_apply<complex>(5, 2, 2);
    check_tensor_product_apply<complex>(6, 2, 3);
}

TEST_CASE("tensor-product engine uses wide blocks on dense sectors")
{
    CHECK(check_tensor_product_apply<double>(9, 4, 4).wide == std::array<bool, 2>{true, true});
    CHECK(check_tensor_product_apply<complex>(9, 4, 4).wide == std::array<bool, 2>{true, true});
    CHECK(check_tensor_product_apply<double>(5, 2, 2).wide == std::array<bool, 2>{false, false});
}

TEST_CASE("tensor-product engine uses the sparse cross table on local cross couplings")
{
    const std::vector<width_t> window = {0, 1, 2, 3, 4, 5, 6, 7};
    const SubspaceFixture store =
        SubspaceFixture::from_halves(window_strings(8, window, 3), 8, window_strings(8, window, 4), 8);
    const EngineChoices real =
        check_cross_modes<double>(window_molecular_operator(8, window, false, false, true), store, true);
    CHECK(real.sparse);
    CHECK(!real.batched);
    const EngineChoices dense =
        check_cross_modes<double>(window_molecular_operator(8, window, false, false), store, true);
    CHECK(!dense.sparse);
    CHECK(dense.batched);
    CHECK(!check_cross_modes<complex>(window_molecular_operator(8, window, true, false, true), store, true).sparse);
}

TEST_CASE("tensor-product engine keeps narrow blocks when wide blocks do not fit")
{
    TensorProductEngine<double> tier;
    tier.num_strings = {20000, 2000};
    for(int sector = 0; sector < 2; sector++)
    {
        const std::size_t num_strings = tier.num_strings[sector];
        tier.sectors[sector].single_offsets.assign(num_strings + 1, 0);
        tier.sectors[sector].double_offsets.assign(num_strings + 1, 0);
        tier.sectors[sector].double_offsets.back() = 100 * num_strings;
    }
    CHECK(!use_wide_blocks(tier, 0));
    CHECK(use_wide_blocks(tier, 1));
    tier.sectors[1].double_offsets.back() = 10 * tier.num_strings[1];
    CHECK(!use_wide_blocks(tier, 1));
}

TEST_CASE("tensor-product engine groups beta singles by slot on dense cross tables")
{
    CHECK(check_tensor_product_apply<double>(9, 4, 4).batched);
    CHECK(check_tensor_product_apply<double>(6, 3, 1).batched);
    CHECK(check_tensor_product_apply<double>(6, 1, 5).batched);
    CHECK(!check_tensor_product_apply<complex>(6, 2, 3).batched);
}

TEST_CASE("tensor-product engine handles few beta strings")
{
    check_tensor_product_apply<double>(6, 3, 1);
    check_tensor_product_apply<double>(6, 1, 5);
}

TEST_CASE("tensor-product engine forms cross values on two-word sectors")
{
    const std::vector<width_t> window = {60, 61, 62, 63, 64, 65, 66, 67};
    const SubspaceFixture store =
        SubspaceFixture::from_halves(window_strings(70, window, 3), 70, window_strings(70, window, 2), 70);
    REQUIRE(store.half_words(0) == 2);
    check_cross_modes<double>(window_molecular_operator(70, window, false, false), store, true);
    check_cross_modes<complex>(window_molecular_operator(70, window, true, false), store, true);
}

TEST_CASE("tensor-product engine handles sampled half-det lists")
{
    const std::vector<width_t> all = {0, 1, 2, 3, 4, 5, 6};
    const std::vector<width_t> window = {0, 1, 2, 3, 4};
    std::mt19937_64 engine(19);
    auto sample = [&engine](std::vector<KeyWord> keys, const std::size_t count) {
        std::shuffle(keys.begin(), keys.end(), engine);
        keys.resize(count);
        return keys;
    };
    const SubspaceFixture store =
        SubspaceFixture::from_halves(sample(window_strings(7, all, 3), 6), 7, sample(window_strings(7, all, 2), 5), 7);
    check_cross_modes<double>(window_molecular_operator(7, window, false, false), store, true);
    check_cross_modes<complex>(window_molecular_operator(7, window, true, false), store, true);
}

TEST_CASE("tensor-product engine stores cross values that need two densities")
{
    const std::vector<width_t> window = {0, 1, 2, 3, 4, 5};
    const SubspaceFixture store =
        SubspaceFixture::from_halves(window_strings(6, window, 3), 6, window_strings(6, window, 2), 6);
    check_cross_modes<double>(window_molecular_operator(6, window, false, true), store, false);
}

TEST_CASE("group hash finds every key and refuses others")
{
    GroupHash empty;
    empty.reserve(0);
    CHECK(empty.find(7) == NO_GROUP);
    CHECK(GroupHash().find(7) == NO_GROUP);

    std::mt19937_64 engine(3);
    std::vector<std::uint64_t> keys;
    for(std::uint64_t k = 0; k < 5000; k++)
    {
        keys.push_back(engine() >> 20);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    GroupHash hash;
    hash.reserve(keys.size());
    for(std::size_t index = 0; index < keys.size(); index++)
    {
        hash.insert(keys[index], static_cast<std::uint32_t>(index));
    }
    std::size_t wrong = 0;
    for(std::size_t index = 0; index < keys.size(); index++)
    {
        wrong += hash.find(keys[index]) != index;
        wrong += std::binary_search(keys.begin(), keys.end(), keys[index] + 1) ? 0 : hash.find(keys[index] + 1) != NO_GROUP;
    }
    CHECK(wrong == 0);
    CHECK(hash.keys.size() >= 2 * keys.size());
}
