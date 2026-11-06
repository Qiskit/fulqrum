#include "doctest.h"
#include "fulqrum.hpp"
#include "../fulqrum/gpu/full_strs/src/coefficients.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace
{

using fq::full_strs::Word;
using State = unsigned __int128;

/**
 * Random 8-fold symmetric integrals in chemist notation.
 */
struct Integrals
{
    int n = 0;
    std::vector<double> h;
    std::vector<double> v;
    std::vector<std::array<int, 4>> nonzero;

    /**
     * Returns (pq|rs).
     */
    double eri(const int p, const int q, const int r, const int s) const
    {
        return v[((static_cast<std::size_t>(p) * n + q) * n + r) * n + s];
    }
};

/**
 * Sets (pq|rs) and its seven symmetric copies.
 */
void set_eri(Integrals& ints, const int p, const int q, const int r, const int s, const double value)
{
    const int n = ints.n;
    const int index[8][4] = {{p, q, r, s}, {q, p, r, s}, {p, q, s, r}, {q, p, s, r},
                             {r, s, p, q}, {s, r, p, q}, {r, s, q, p}, {s, r, q, p}};
    for(const auto& k : index)
    {
        ints.v[((static_cast<std::size_t>(k[0]) * n + k[1]) * n + k[2]) * n + k[3]] = value;
    }
}

void fill_integrals(Integrals& ints, unsigned int seed, const std::vector<int>& active);

/**
 * Makes integrals with dense random values on the active orbitals and zero elsewhere.
 *
 * @param n The number of orbitals.
 * @param seed The random seed.
 * @param active The active orbitals; empty for all orbitals.
 * @return The integrals.
 */
Integrals random_integrals(const int n, const unsigned int seed, const std::vector<int>& active)
{
    Integrals ints;
    ints.n = n;
    ints.h.assign(static_cast<std::size_t>(n) * n, 0.0);
    ints.v.assign(static_cast<std::size_t>(n) * n * n * n, 0.0);
    fill_integrals(ints, seed, active);
    for(int p = 0; p < n; p++)
    {
        for(int q = 0; q < n; q++)
        {
            for(int r = 0; r < n; r++)
            {
                for(int s = 0; s < n; s++)
                {
                    if(ints.eri(p, q, r, s) != 0.0)
                    {
                        ints.nonzero.push_back({p, q, r, s});
                    }
                }
            }
        }
    }
    return ints;
}

/**
 * Fills dense random 8-fold symmetric integrals on the active orbitals (all orbitals when empty).
 */
void fill_integrals(Integrals& ints, const unsigned int seed, const std::vector<int>& active)
{
    const int n = ints.n;
    std::vector<int> orbitals = active;
    if(orbitals.empty())
    {
        for(int k = 0; k < n; k++)
        {
            orbitals.push_back(k);
        }
    }
    const int m = static_cast<int>(orbitals.size());
    std::mt19937_64 engine(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j <= i; j++)
        {
            const double value = 0.5 * gauss(engine);
            const int p = orbitals[static_cast<std::size_t>(i)];
            const int q = orbitals[static_cast<std::size_t>(j)];
            ints.h[static_cast<std::size_t>(p) * n + q] = value;
            ints.h[static_cast<std::size_t>(q) * n + p] = value;
        }
    }
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j <= i; j++)
        {
            for(int k = 0; k < m; k++)
            {
                for(int l = 0; l <= k; l++)
                {
                    if(i * m + j >= k * m + l)
                    {
                        set_eri(ints,
                                orbitals[static_cast<std::size_t>(i)],
                                orbitals[static_cast<std::size_t>(j)],
                                orbitals[static_cast<std::size_t>(k)],
                                orbitals[static_cast<std::size_t>(l)],
                                0.1 * gauss(engine));
                    }
                }
            }
        }
    }
}

/**
 * Builds the type-2 qubit operator of the integrals in block layout through the core Jordan-Wigner path.
 *
 * @param ints The integrals.
 * @param constant The constant term.
 * @return The qubit operator.
 */
QubitOperator_t integrals_operator(const Integrals& ints, const double constant)
{
    const int n = ints.n;
    std::vector<std::tuple<std::string, std::vector<width_t>, std::complex<double>>> data;
    data.push_back({"", {}, constant});
    for(int spin = 0; spin < 2; spin++)
    {
        for(int p = 0; p < n; p++)
        {
            for(int q = 0; q < n; q++)
            {
                const double value = ints.h[static_cast<std::size_t>(p) * n + q];
                if(value != 0.0)
                {
                    data.push_back({"+-", {static_cast<width_t>(spin * n + p), static_cast<width_t>(spin * n + q)}, value});
                }
            }
        }
    }
    for(int s1 = 0; s1 < 2; s1++)
    {
        for(int s2 = 0; s2 < 2; s2++)
        {
            for(const auto& [p, q, r, s] : ints.nonzero)
            {
                const int sp = s1 * n + p;
                const int sq = s1 * n + q;
                const int sr = s2 * n + r;
                const int ss = s2 * n + s;
                if(sp == sr || sq == ss)
                {
                    continue;
                }
                data.push_back({"++--",
                                {static_cast<width_t>(sp), static_cast<width_t>(sr), static_cast<width_t>(ss),
                                 static_cast<width_t>(sq)},
                                0.5 * ints.eri(p, q, r, s)});
            }
        }
    }
    FermionicOperator_t fop(static_cast<width_t>(2 * n), data);
    return fop.extended_jw_transformation();
}

/**
 * Applies a creation or annihilation operator to a state with its Jordan-Wigner sign.
 *
 * @return False when the result is zero.
 */
bool apply(State& state, const int k, const bool create, int& sign)
{
    const State bit = State(1) << k;
    if(((state & bit) != 0) == create)
    {
        return false;
    }
    const State below = state & (bit - 1);
    const int count = __builtin_popcountll(static_cast<std::uint64_t>(below)) +
                      __builtin_popcountll(static_cast<std::uint64_t>(below >> 64));
    if(count & 1)
    {
        sign = -sign;
    }
    state ^= bit;
    return true;
}

/**
 * Returns <row|H|col> by applying every term of H literally to the col.
 */
double literal_element(const Integrals& ints, const State bra, const State ket)
{
    const int n = ints.n;
    double total = 0.0;
    for(int spin = 0; spin < 2; spin++)
    {
        for(int p = 0; p < n; p++)
        {
            for(int q = 0; q < n; q++)
            {
                State state = ket;
                int sign = 1;
                if(apply(state, spin * n + q, false, sign) && apply(state, spin * n + p, true, sign) && state == bra)
                {
                    total += ints.h[static_cast<std::size_t>(p) * n + q] * sign;
                }
            }
        }
    }
    for(int s1 = 0; s1 < 2; s1++)
    {
        for(int s2 = 0; s2 < 2; s2++)
        {
            for(const auto& [p, q, r, s] : ints.nonzero)
            {
                State state = ket;
                int sign = 1;
                if(apply(state, s1 * n + q, false, sign) && apply(state, s2 * n + s, false, sign) &&
                   apply(state, s2 * n + r, true, sign) && apply(state, s1 * n + p, true, sign) && state == bra)
                {
                    total += 0.5 * ints.eri(p, q, r, s) * sign;
                }
            }
        }
    }
    return total;
}

/**
 * Returns the state of a sector key with one word per sector.
 */
State to_state(const Word* key, const int n)
{
    return State(key[0]) | (State(key[1]) << n);
}

/**
 * Returns a random string with count electrons on the orbitals of a mask.
 */
Word random_string(std::mt19937_64& engine, const Word allowed, const int count)
{
    std::vector<int> orbitals;
    for(int k = 0; k < 64; k++)
    {
        if((allowed >> k) & 1)
        {
            orbitals.push_back(k);
        }
    }
    std::shuffle(orbitals.begin(), orbitals.end(), engine);
    Word out = 0;
    for(int k = 0; k < count; k++)
    {
        out |= Word(1) << orbitals[static_cast<std::size_t>(k)];
    }
    return out;
}

/**
 * Applies a random excitation of the given number of electrons inside a mask to a sector string.
 */
Word excite(std::mt19937_64& engine, const Word string, const Word allowed, const int level)
{
    std::vector<int> occupied;
    std::vector<int> empty;
    for(int k = 0; k < 64; k++)
    {
        if((allowed >> k) & 1)
        {
            (((string >> k) & 1) ? occupied : empty).push_back(k);
        }
    }
    if(static_cast<int>(occupied.size()) < level || static_cast<int>(empty.size()) < level)
    {
        return string;
    }
    std::shuffle(occupied.begin(), occupied.end(), engine);
    std::shuffle(empty.begin(), empty.end(), engine);
    Word out = string;
    for(int k = 0; k < level; k++)
    {
        out ^= (Word(1) << occupied[static_cast<std::size_t>(k)]) | (Word(1) << empty[static_cast<std::size_t>(k)]);
    }
    return out;
}

/**
 * Compares table elements and diagonal values with the literal evaluator on random rows.
 *
 * The rows have num_alpha and num_beta electrons on the active orbitals and extra electrons
 * on the other orbitals; the moves stay on the active orbitals.
 *
 * @param n The number of orbitals.
 * @param num_alpha The alpha electron count on the active orbitals.
 * @param num_beta The beta electron count on the active orbitals.
 * @param seed The random seed.
 * @param active The active orbitals; empty for all orbitals.
 * @param extra The electron count per sector on the other orbitals.
 * @param num_rows The number of random output rows.
 */
void check_against_literal(const int n,
                           const int num_alpha,
                           const int num_beta,
                           const unsigned int seed,
                           const std::vector<int>& active,
                           const int extra,
                           const int num_rows)
{
    const Integrals ints = random_integrals(n, seed, active);
    const Word all = n == 64 ? ~Word(0) : (Word(1) << n) - 1;
    Word allowed = active.empty() ? all : 0;
    for(const int orbital : active)
    {
        allowed |= Word(1) << orbital;
    }
    const Word rest = all & ~allowed;
    std::mt19937_64 engine(seed + 1000);
    auto make_string = [&](const int count) {
        const Word inside = random_string(engine, allowed, count);
        return extra > 0 ? inside | random_string(engine, rest, extra) : inside;
    };
    const double constant = 0.75;
    const QubitOperator_t op = integrals_operator(ints, constant);
    const auto terms = fq::full_strs::split_operator<1>(op, true);
    fq::full_strs::TableSpec<1> spec;
    spec.num_alpha = num_alpha + extra;
    spec.num_beta = num_beta + extra;
    spec.reference = {make_string(num_alpha), make_string(num_beta)};
    const auto tables = fq::full_strs::build_coefficients<1, double>(terms, spec);

    std::array<int, 5> compared{};
    std::array<int, 5> nonzero{};
    double max_error = 0.0;
    std::vector<Word> rows;
    for(int row = 0; row < num_rows; row++)
    {
        const Word out[2] = {make_string(num_alpha), make_string(num_beta)};
        rows.push_back(out[0]);
        rows.push_back(out[1]);
        const std::array<std::array<int, 2>, 5> moves = {{{1, 0}, {2, 0}, {0, 1}, {0, 2}, {1, 1}}};
        for(std::size_t m = 0; m < moves.size(); m++)
        {
            const Word source[2] = {excite(engine, out[0], allowed, moves[m][0]),
                                    excite(engine, out[1], allowed, moves[m][1])};
            if(__builtin_popcountll(source[0] ^ out[0]) != 2 * moves[m][0] ||
               __builtin_popcountll(source[1] ^ out[1]) != 2 * moves[m][1])
            {
                continue;
            }
            const double expected = literal_element(ints, to_state(out, n), to_state(source, n));
            const double got = tables.element(out, source);
            max_error = std::max(max_error, std::abs(got - expected));
            CHECK(std::abs(got - expected) <= 1e-12 * std::max(1.0, std::abs(expected)));
            compared[m]++;
            nonzero[m] += expected != 0.0 ? 1 : 0;
        }
    }
    for(int m = 0; m < 5; m++)
    {
        CHECK(nonzero[static_cast<std::size_t>(m)] > num_rows / 4);
    }
    const auto store = fq::full_strs::build_key_store<1>(rows.data(), static_cast<std::int64_t>(rows.size() / 2), n);
    const std::vector<double> diagonal = fq::full_strs::diagonal_values<1, double>(terms, store);
    for(std::int64_t row = 0; row < store.num_rows(); row++)
    {
        const State state = to_state(store.alpha(row), n);
        const double expected = constant + literal_element(ints, state, state);
        CHECK(std::abs(diagonal[static_cast<std::size_t>(row)] - expected) <= 1e-12 * std::max(1.0, std::abs(expected)));
    }
    MESSAGE("n=" << n << " groups=" << terms.num_groups() << " max error=" << max_error);
}

/**
 * Returns the message of the invalid_argument that a build of the operator throws, or an empty string.
 */
std::string build_error(const QubitOperator_t& op, const int num_alpha, const int num_beta)
{
    try
    {
        const auto terms = fq::full_strs::split_operator<1>(op, true);
        fq::full_strs::TableSpec<1> spec;
        spec.num_alpha = num_alpha;
        spec.num_beta = num_beta;
        fq::full_strs::build_coefficients<1, double>(terms, spec);
    }
    catch(const std::invalid_argument& error)
    {
        return error.what();
    }
    return std::string();
}

}

TEST_CASE("Full-string coefficient tables equal the literal Hamiltonian")
{
    check_against_literal(6, 3, 2, 11, {}, 0, 300);
    check_against_literal(8, 4, 4, 12, {}, 0, 200);
    check_against_literal(7, 2, 5, 13, {}, 0, 200);
}

TEST_CASE("Full-string coefficient tables at 64 orbitals per sector")
{
    check_against_literal(64, 4, 3, 14, {0, 2, 17, 31, 32, 33, 47, 58, 61, 62, 63}, 3, 60);
}

TEST_CASE("Full-string coefficient tables: rank helpers")
{
    for(std::int64_t rank = 0; rank < fq::full_strs::binomial(20, 4); rank++)
    {
        int positions[4];
        fq::full_strs::quad_orbitals(rank, positions);
        CHECK(positions[0] < positions[1]);
        CHECK(positions[2] < positions[3]);
        CHECK(fq::full_strs::quad_rank(positions) == rank);
    }
    for(std::int64_t rank = 0; rank < fq::full_strs::binomial(64, 2); rank++)
    {
        const auto [p, q] = fq::full_strs::pair_orbitals(rank);
        CHECK(p < q);
        CHECK(fq::full_strs::pair_rank(p, q) == rank);
    }
}

TEST_CASE("Full-string coefficient tables: refusal cases")
{
    const Integrals ints = random_integrals(6, 21, {});
    QubitOperator_t good = integrals_operator(ints, 0.0);
    CHECK(build_error(good, 3, 3).empty());

    QubitOperator_t projector = good;
    projector += QubitOperator(12, {{"++--1", {0, 1, 2, 3, 5}, 0.3}, {"--++1", {0, 1, 2, 3, 5}, 0.3}});
    projector.set_type(2);
    const std::string projector_error = build_error(projector, 3, 3);
    CHECK(projector_error.find("Operator group") != std::string::npos);
    CHECK(projector_error.find("alpha {0,1,2,3}") != std::string::npos);

    QubitOperator_t two_spectators = good;
    two_spectators += QubitOperator(12, {{"+-11", {0, 2, 7, 8}, 0.2}, {"-+11", {0, 2, 7, 8}, 0.2}});
    two_spectators.set_type(2);
    CHECK(build_error(two_spectators, 3, 3).find("cannot express") != std::string::npos);

    QubitOperator_t not_hermitian = good;
    not_hermitian += QubitOperator(12, {{"+-", {1, 4}, 0.4}});
    not_hermitian.set_type(2);
    CHECK(build_error(not_hermitian, 3, 3).find("Hermitian") != std::string::npos);

    QubitOperator_t spin_flip = good;
    spin_flip += QubitOperator(12, {{"+-", {0, 6}, 0.1}, {"-+", {0, 6}, 0.1}});
    spin_flip.set_type(2);
    CHECK(build_error(spin_flip, 3, 3).find("spin-conserving") != std::string::npos);

    QubitOperator_t pauli = good;
    pauli += QubitOperator(12, {{"XX", {0, 1}, 0.1}});
    pauli.set_type(2);
    CHECK(build_error(pauli, 3, 3).find("X or Y") != std::string::npos);

    QubitOperator_t complex_value = good;
    complex_value += QubitOperator(12, {{"+-", {0, 1}, std::complex<double>(0, 0.1)},
                                        {"-+", {0, 1}, std::complex<double>(0, -0.1)}});
    complex_value.set_type(2);
    CHECK(build_error(complex_value, 3, 3).find("complex") != std::string::npos);

    QubitOperator_t type_one = good;
    type_one.set_type(1);
    CHECK(build_error(type_one, 3, 3).find("type 2") != std::string::npos);

    QubitOperator_t odd(5, {{"+-", {0, 1}, 0.1}});
    odd.set_type(2);
    CHECK(build_error(odd, 3, 3).find("even") != std::string::npos);
}
