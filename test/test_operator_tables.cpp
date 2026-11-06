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
#include <complex>
#include <random>
#include <string>
#include <vector>

typedef std::complex<double> complex;

namespace
{

/**
 * Tells whether every term in every group of an operator is Hermitian.
 *
 * @param width The number of qubits.
 * @param data The terms: operator letters, qubit indices and coefficient.
 */
bool is_hermitian(const width_t width, const std::vector<TermData>& data)
{
    QubitOperator oper(width, data);
    oper.group_sort();
    const std::vector<std::size_t> ptrs = oper.group_ptrs();
    const std::size_t num_groups = ptrs.size() - 1;
    return first_non_hermitian_group(oper, ptrs.data(), num_groups) == num_groups;
}

/**
 * A grouped off-diagonal operator with the flip qubits of each group.
 */
struct GroupedOperator
{
    QubitOperator off;
    std::vector<std::size_t> ptrs;
    std::vector<std::vector<width_t>> flips;
};

/**
 * Transforms a fermionic operator and groups its off-diagonal terms by flip indices.
 *
 * @param fermionic The fermionic operator.
 */
GroupedOperator group_operator(FermionicOperator fermionic)
{
    GroupedOperator grouped;
    grouped.off = fermionic.extended_jw_transformation().split_diagonal().second;
    grouped.off.group_sort();
    grouped.ptrs = grouped.off.group_ptrs();
    set_group_offdiag_indices(
        grouped.off.terms, grouped.flips, grouped.ptrs.data(), grouped.ptrs.size() - 1);
    return grouped;
}

/**
 * Makes a Hermitian molecular-like operator: singles and density-weighted singles in
 * each sector, same-sector doubles and cross-sector doubles.
 *
 * @param alpha The qubits of the alpha orbitals.
 * @param beta The qubits of the beta orbitals.
 * @param complex_values When true, the coefficients are complex.
 */
FermionicOperator molecular_operator(const std::vector<width_t>& alpha,
                                     const std::vector<width_t>& beta,
                                     const bool complex_values)
{
    std::mt19937_64 engine(17);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    auto coefficient = [&]() {
        return complex(uniform(engine), complex_values ? uniform(engine) : 0.0);
    };
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
    const std::size_t norb = alpha.size();
    for(const auto* own : {&alpha, &beta})
    {
        const auto* other = own == &alpha ? &beta : &alpha;
        for(std::size_t i = 0; i < norb; i++)
        {
            for(std::size_t j = i + 1; j < norb; j++)
            {
                add_with_adjoint("+-", {(*own)[i], (*own)[j]}, coefficient());
                for(std::size_t r = 0; r < norb; r++)
                {
                    add_with_adjoint("+-+-", {(*own)[i], (*own)[j], (*other)[r], (*other)[r]}, coefficient());
                }
                for(std::size_t k = 0; k < norb; k++)
                {
                    for(std::size_t l = k + 1; l < norb; l++)
                    {
                        if(k != i && k != j && l != i && l != j)
                        {
                            add_with_adjoint("++--", {(*own)[i], (*own)[j], (*own)[k], (*own)[l]}, coefficient());
                        }
                    }
                }
            }
        }
    }
    for(std::size_t i = 0; i < norb; i++)
    {
        for(std::size_t j = 0; j < norb; j++)
        {
            for(std::size_t k = 0; k < norb; k++)
            {
                for(std::size_t l = 0; l < norb; l++)
                {
                    if(i != j && k != l)
                    {
                        add_with_adjoint("+-+-", {alpha[i], alpha[j], beta[k], beta[l]}, coefficient());
                    }
                }
            }
        }
    }
    return FermionicOperator(static_cast<width_t>(2 * norb), terms);
}

/**
 * Makes a flat store of every key with given electron counts on two qubit sets.
 *
 * @param alpha The alpha qubits.
 * @param beta The beta qubits.
 * @param alpha_electrons The number of alpha electrons.
 * @param beta_electrons The number of beta electrons.
 */
SubspaceFixture determinant_store(const std::vector<width_t>& alpha,
                                const std::vector<width_t>& beta,
                                const unsigned int alpha_electrons,
                                const unsigned int beta_electrons)
{
    const width_t width = static_cast<width_t>(alpha.size() + beta.size());
    std::vector<KeyWord> keys;
    for(unsigned int value = 0; value < (1u << width); value++)
    {
        unsigned int in_alpha = 0;
        unsigned int in_beta = 0;
        for(const width_t qubit : alpha)
        {
            in_alpha += (value >> qubit) & 1u;
        }
        for(const width_t qubit : beta)
        {
            in_beta += (value >> qubit) & 1u;
        }
        if(in_alpha == alpha_electrons && in_beta == beta_electrons)
        {
            keys.push_back(value);
        }
    }
    return SubspaceFixture::from_keys(std::move(keys), width, width / 2);
}

/**
 * Checks the element of every sector-tier group against the term evaluation,
 * for every pair of connected keys in a store.
 *
 * @param grouped The grouped operator.
 * @param tables The tables of the operator.
 * @param store The subspace store.
 */
void check_plan_elements(const GroupedOperator& grouped,
                         const OperatorTables& tables,
                         const SubspaceFixture& store)
{
    std::size_t checked = 0;
    for(std::size_t group = 0; group < tables.num_groups(); group++)
    {
        if(tables.group(group).tier == GroupTier::generic)
        {
            continue;
        }
        for(std::size_t index = 0; index < store.size(); index++)
        {
            const KeyBitset row = store.key(index);
            KeyBitset col = row;
            for(const width_t qubit : grouped.flips[group])
            {
                col.flip(qubit);
            }
            complex expected = 0;
            for(std::size_t term = grouped.ptrs[group]; term < grouped.ptrs[group + 1]; term++)
            {
                const OperatorTerm_t& t = grouped.off.terms[term];
                accum_element(row, col, t.indices, t.values, t.coeff, t.real_phase, t.indices.size(), expected);
            }
            const complex planned = tables.element(group, row.m_bits.data());
            CHECK(std::abs(planned - expected) < 1e-12);
            checked++;
        }
    }
    CHECK(checked > 0);
}

} // namespace

TEST_CASE("Hermiticity check accepts an excitation with its adjoint")
{
    CHECK(is_hermitian(2, {{"+-", {1, 0}, 1.0}, {"-+", {1, 0}, 1.0}}));
    CHECK(is_hermitian(2, {{"+-", {1, 0}, complex(1, 2)}, {"-+", {1, 0}, complex(1, -2)}}));
}

TEST_CASE("Hermiticity check rejects an excitation without a matching adjoint")
{
    CHECK_FALSE(is_hermitian(2, {{"+-", {1, 0}, 1.0}}));
    CHECK_FALSE(is_hermitian(2, {{"+-", {1, 0}, 1.0}, {"-+", {1, 0}, 0.5}}));
    CHECK_FALSE(is_hermitian(2, {{"+-", {1, 0}, complex(1, 2)}, {"-+", {1, 0}, complex(1, 2)}}));
}

TEST_CASE("Hermiticity check accepts an adjoint written with other operators")
{
    CHECK(is_hermitian(2, {{"+Z", {0, 1}, 1.0}, {"-0", {0, 1}, 1.0}, {"-1", {0, 1}, -1.0}}));
    CHECK(is_hermitian(1, {{"Y", {0}, 1.0}}));
    CHECK(is_hermitian(1, {{"+", {0}, complex(0, 1)}, {"-", {0}, complex(0, -1)}}));
    CHECK(is_hermitian(2, {{"XY", {0, 1}, 2.0}, {"Z0", {0, 1}, -1.0}}));
}

TEST_CASE("Hermiticity check rejects anti-Hermitian and complex diagonal terms")
{
    CHECK_FALSE(is_hermitian(1, {{"Y", {0}, complex(0, 1)}}));
    CHECK_FALSE(is_hermitian(1, {{"Z", {0}, complex(0, 2)}}));
    CHECK_FALSE(is_hermitian(2, {{"+Z", {0, 1}, 1.0}, {"-0", {0, 1}, 1.0}, {"-1", {0, 1}, 1.0}}));
}

TEST_CASE("Sector layout is derived for block and interleaved orderings")
{
    const std::vector<width_t> block_alpha = {0, 1, 2, 3};
    const std::vector<width_t> block_beta = {4, 5, 6, 7};
    const std::vector<width_t> even = {0, 2, 4, 6};
    const std::vector<width_t> odd = {1, 3, 5, 7};
    for(const auto& [alpha, beta] : {std::make_pair(block_alpha, block_beta), std::make_pair(even, odd)})
    {
        const GroupedOperator grouped = group_operator(molecular_operator(alpha, beta, false));
        const SubspaceFixture store = determinant_store(alpha, beta, 2, 2);
        const SectorLayout layout = choose_sector_layout(SpinLayout::automatic, store, grouped.flips, 8);
        REQUIRE(layout.valid);
        for(const width_t qubit : alpha)
        {
            CHECK(layout.sector_of[qubit] == 0);
        }
        for(const width_t qubit : beta)
        {
            CHECK(layout.sector_of[qubit] == 1);
        }
    }
}

TEST_CASE("OperatorTables elements match the terms for real and complex operators")
{
    for(const bool complex_values : {false, true})
    {
        CAPTURE(complex_values);
        const std::vector<width_t> alpha = {0, 1, 2, 3};
        const std::vector<width_t> beta = {4, 5, 6, 7};
        const GroupedOperator grouped = group_operator(molecular_operator(alpha, beta, complex_values));
        const SubspaceFixture store = determinant_store(alpha, beta, 2, 2);
        const SectorLayout layout = choose_sector_layout(SpinLayout::block, store, grouped.flips, 8);
        const OperatorTables tables = OperatorTables::build(grouped.off, grouped.ptrs.data(), grouped.flips, layout, false);
        CHECK(tables.count(GroupTier::generic) == 0);
        CHECK(tables.count(GroupTier::same_sector_single) > 0);
        CHECK(tables.count(GroupTier::same_sector_double) > 0);
        CHECK(tables.count(GroupTier::cross_sector_double) > 0);
        check_plan_elements(grouped, tables, store);
    }
}

TEST_CASE("OperatorTables refuses cross-sector groups whose sign does not separate")
{
    const std::vector<width_t> alpha = {0, 2, 4, 6};
    const std::vector<width_t> beta = {1, 3, 5, 7};
    const GroupedOperator grouped = group_operator(molecular_operator(alpha, beta, false));
    const SubspaceFixture store = determinant_store(alpha, beta, 2, 2);
    const SectorLayout layout = choose_sector_layout(SpinLayout::interleaved, store, grouped.flips, 8);
    REQUIRE(layout.valid);
    const OperatorTables tables = OperatorTables::build(grouped.off, grouped.ptrs.data(), grouped.flips, layout, false);
    CHECK(tables.count(GroupTier::same_sector_single) > 0);
    std::size_t not_separable = 0;
    for(std::size_t group = 0; group < tables.num_groups(); group++)
    {
        const GroupData& plan = tables.group(group);
        if(plan.tier == GroupTier::generic)
        {
            const bool expected = plan.reason == GenericReason::cross_sign_not_separable ||
                                  plan.reason == GenericReason::double_with_other_factor;
            CHECK(expected);
            not_separable += plan.reason == GenericReason::cross_sign_not_separable ? 1 : 0;
        }
    }
    CHECK(not_separable > 0);
    check_plan_elements(grouped, tables, store);
}

TEST_CASE("Sector layout is not valid for spinless operators or mixed electron counts")
{
    const width_t width = 7;
    std::vector<TermData> terms;
    for(width_t p = 0; p < width; p++)
    {
        for(width_t q = p + 1; q < width; q++)
        {
            terms.emplace_back("+-", std::vector<width_t>{p, q}, 0.5);
            terms.emplace_back("+-", std::vector<width_t>{q, p}, 0.5);
        }
    }
    const GroupedOperator grouped = group_operator(FermionicOperator(width, terms));
    std::vector<KeyWord> keys = {0b0000011, 0b0000101, 0b1100000, 0b0011000};
    const SubspaceFixture store = SubspaceFixture::from_keys(keys, width, width / 2);
    const SectorLayout layout = choose_sector_layout(SpinLayout::automatic, store, grouped.flips, width);
    CHECK_FALSE(layout.valid);
    const OperatorTables tables = OperatorTables::build(grouped.off, grouped.ptrs.data(), grouped.flips, layout, false);
    CHECK(tables.count(GroupTier::generic) == tables.num_groups());
    CHECK(tables.group(0).reason == GenericReason::no_sector_layout);
    CHECK_THROWS_AS(choose_sector_layout(SpinLayout::block, store, grouped.flips, width), std::invalid_argument);

    const SubspaceFixture mixed = SubspaceFixture::from_halves({0b01, 0b11}, 2, {0b01, 0b10}, 2);
    CHECK_FALSE(choose_sector_layout(SpinLayout::automatic, mixed, {}, 4).valid);
    CHECK_THROWS_AS(choose_sector_layout(SpinLayout::interleaved, mixed, {}, 4), std::invalid_argument);
}

TEST_CASE("OperatorTables can force every group to the generic tier")
{
    const std::vector<width_t> alpha = {0, 1, 2, 3};
    const std::vector<width_t> beta = {4, 5, 6, 7};
    const GroupedOperator grouped = group_operator(molecular_operator(alpha, beta, false));
    const SubspaceFixture store = determinant_store(alpha, beta, 2, 2);
    const SectorLayout layout = choose_sector_layout(SpinLayout::block, store, grouped.flips, 8);
    const OperatorTables tables = OperatorTables::build(grouped.off, grouped.ptrs.data(), grouped.flips, layout, true);
    CHECK(tables.count(GroupTier::generic) == tables.num_groups());
    CHECK(tables.group(0).reason == GenericReason::forced);
}
