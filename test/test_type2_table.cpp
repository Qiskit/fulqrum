#include "doctest.h"
#include "fulqrum.hpp"
#include <complex>
#include <string>
#include <tuple>
#include <vector>
#include <filesystem>

typedef std::complex<double> complex;

namespace
{

struct TableFixture
{
    QubitOperator off;
    std::vector<std::size_t> ptrs;
    std::vector<std::vector<width_t>> group_inds;
    Type2GroupTable table;

    std::size_t num_groups() const
    {
        return ptrs.size() - 1;
    }

    // Finds the group that has the given flip indices.
    std::size_t group_of(const std::vector<width_t>& flips) const
    {
        for(std::size_t group = 0; group < num_groups(); group++)
        {
            if(group_inds[group] == flips)
            {
                return group;
            }
        }
        return MAX_SIZE_T;
    }

    bool is_standard(std::size_t group) const
    {
        for(const auto& entry : table.standard_groups)
        {
            if(entry == group)
            {
                return true;
            }
        }
        return false;
    }

    bool is_direct_unpaired(std::size_t group) const
    {
        for(const auto& entry : table.direct_unpaired_groups)
        {
            if(entry == group)
            {
                return true;
            }
        }
        return false;
    }

    bool is_direct_paired(std::size_t group) const
    {
        for(const auto& entries : table.paired_by_low_pair)
        {
            for(const auto& entry : entries)
            {
                if(entry.group == group)
                {
                    return true;
                }
            }
        }
        return false;
    }

    // Gives the accepted row bits of the low pair that holds the given indices.
    std::uint8_t low_pair_mask_of(width_t first, width_t second) const
    {
        for(std::size_t idx = 0; idx < table.low_pairs.size(); idx++)
        {
            if(table.low_pairs[idx][0] == first && table.low_pairs[idx][1] == second)
            {
                return table.low_pair_mask[idx];
            }
        }
        return 0;
    }

    std::uint8_t high_pair_mask_of(width_t first, width_t second) const
    {
        for(std::size_t idx = 0; idx < table.high_pairs.size(); idx++)
        {
            if(table.high_pairs[idx][0] == first && table.high_pairs[idx][1] == second)
            {
                return table.high_pair_mask[idx];
            }
        }
        return 0;
    }
};

// Builds the type-2 table of an operator, the same way the type2 paths do.
TableFixture make_table(width_t width, std::vector<TermData> data, width_t ladder_width = 2)
{
    TableFixture fixture;
    QubitOperator op = QubitOperator(width, data);
    op.set_type(2);
    QubitOperator diag;
    std::tie(diag, fixture.off) = op.split_diagonal();
    fixture.off.group_sort();
    fixture.off.group_term_sort_by_ladder_int(ladder_width);
    fixture.ptrs = fixture.off.group_ptrs();
    fixture.group_inds = fixture.off.group_offdiag_indices();
    fixture.table = build_type2_group_table(fixture.off.terms,
                                            fixture.ptrs.data(),
                                            fixture.group_inds,
                                            fixture.ptrs.size() - 1,
                                            fixture.off.width);
    return fixture;
}

constexpr std::uint16_t pattern_bit(unsigned int pattern)
{
    return static_cast<std::uint16_t>(1u << pattern);
}

} // namespace

TEST_CASE("Type-2 table takes a spinless 4-flip group on the direct path")
{
    TableFixture fixture = make_table(8,
                                      {
                                          {"--++", {0, 1, 6, 7}, 1.0},
                                          {"++--", {0, 1, 6, 7}, 1.0},
                                      });

    REQUIRE(fixture.num_groups() == 1);
    const std::size_t group = fixture.group_of({0, 1, 6, 7});
    REQUIRE(group != MAX_SIZE_T);

    // Row pattern 0b1100 comes from '-' at 0 and 1 with '+' at 6 and 7.
    // Row pattern 0b0011 comes from the other term.
    CHECK(fixture.table.pattern_mask[group] == (pattern_bit(0b1100) | pattern_bit(0b0011)));
    CHECK(fixture.table.direct_coeff[group] == complex(1.0, 0.0));
    CHECK(fixture.is_direct_paired(group));
    CHECK_FALSE(fixture.is_standard(group));

    CHECK(fixture.low_pair_mask_of(0, 1) == 0b1001);
    CHECK(fixture.high_pair_mask_of(6, 7) == 0b1001);
}

TEST_CASE("Type-2 table keeps the pair masks of a single-excitation group")
{
    TableFixture fixture = make_table(8,
                                      {
                                          {"+-+-", {0, 1, 4, 5}, 1.0},
                                          {"-+-+", {0, 1, 4, 5}, 1.0},
                                      });

    const std::size_t group = fixture.group_of({0, 1, 4, 5});
    REQUIRE(group != MAX_SIZE_T);
    CHECK(fixture.table.pattern_mask[group] == (pattern_bit(0b0101) | pattern_bit(0b1010)));
    CHECK(fixture.is_direct_paired(group));

    // Each pair accepts unequal row bits only, which is what the older check did.
    CHECK(fixture.low_pair_mask_of(0, 1) == 0b0110);
    CHECK(fixture.high_pair_mask_of(4, 5) == 0b0110);
}

TEST_CASE("Type-2 table accepts a full Jordan-Wigner Z string")
{
    // Flips at 0, 2, 5 and 7 need a Z at 1 and a Z at 6.
    TableFixture fixture = make_table(8, {{"+Z-+Z-", {0, 1, 2, 5, 6, 7}, 1.0}});

    const std::size_t group = fixture.group_of({0, 2, 5, 7});
    REQUIRE(group != MAX_SIZE_T);
    CHECK(fixture.is_direct_paired(group));
    CHECK(fixture.table.pattern_mask[group] == pattern_bit(0b0101));
}

TEST_CASE("Type-2 table rejects a 4-flip group with a missing Z")
{
    // The same flips as the test above, but with no Z string.
    TableFixture fixture = make_table(8, {{"+-+-", {0, 2, 5, 7}, 1.0}});

    const std::size_t group = fixture.group_of({0, 2, 5, 7});
    REQUIRE(group != MAX_SIZE_T);
    CHECK(fixture.is_standard(group));
    // The pattern mask still holds, therefore the standard path keeps its check.
    CHECK(fixture.table.pattern_mask[group] == pattern_bit(0b0101));
}

TEST_CASE("Type-2 table rejects a 4-flip group with a Z outside the ranges")
{
    // Flips at 0, 3, 5 and 7 need a Z at 1, 2 and 6. This term has the right
    // number of operators but puts one Z at 4, which is outside both ranges.
    TableFixture fixture = make_table(8, {{"+ZZ-Z+Z-", {0, 1, 2, 3, 4, 5, 6, 7}, 1.0}});

    const std::size_t group = fixture.group_of({0, 3, 5, 7});
    REQUIRE(group != MAX_SIZE_T);
    CHECK(fixture.is_standard(group));
}

TEST_CASE("Type-2 table rejects a 4-flip group that holds a projector")
{
    TableFixture fixture = make_table(8, {{"--++1", {0, 1, 6, 7, 3}, 1.0}});

    const std::size_t group = fixture.group_of({0, 1, 6, 7});
    REQUIRE(group != MAX_SIZE_T);
    CHECK(fixture.is_standard(group));
}

TEST_CASE("Type-2 table rejects a 4-flip group that repeats a row pattern")
{
    // The same term twice. The direct path would count the pattern once, thus
    // the group has to stay on the standard path.
    TableFixture fixture = make_table(8,
                                      {
                                          {"--++", {0, 1, 6, 7}, 1.0},
                                          {"--++", {0, 1, 6, 7}, 1.0},
                                      });

    const std::size_t group = fixture.group_of({0, 1, 6, 7});
    REQUIRE(group != MAX_SIZE_T);
    CHECK(fixture.is_standard(group));
}

TEST_CASE("Type-2 table rejects a 4-flip group with two different coefficients")
{
    TableFixture fixture = make_table(8,
                                      {
                                          {"--++", {0, 1, 6, 7}, 1.0},
                                          {"++--", {0, 1, 6, 7}, 0.5},
                                      });

    const std::size_t group = fixture.group_of({0, 1, 6, 7});
    REQUIRE(group != MAX_SIZE_T);
    CHECK(fixture.is_standard(group));
}

TEST_CASE("Type-2 table puts a one-sided 4-flip group on the unpaired direct path")
{
    // Each flip sits below the split point, therefore the group cannot use the
    // half-key prefilter. It still evaluates with the direct formula.
    TableFixture fixture = make_table(8, {{"++--", {0, 1, 2, 3}, 1.0}});

    const std::size_t group = fixture.group_of({0, 1, 2, 3});
    REQUIRE(group != MAX_SIZE_T);
    CHECK(fixture.is_direct_unpaired(group));
    CHECK_FALSE(fixture.table.has_paired_groups());
    CHECK(fixture.table.pattern_mask[group] == pattern_bit(0b0011));
}

TEST_CASE("Type-2 table keeps a 2-flip group that does not conserve particles")
{
    // Both operators create a particle. The older Hamming weight check dropped
    // every row of this group, because it accepted unequal row bits only.
    TableFixture fixture = make_table(4, {{"++", {0, 1}, 1.0}});

    const std::size_t group = fixture.group_of({0, 1});
    REQUIRE(group != MAX_SIZE_T);
    CHECK(fixture.is_standard(group));
    CHECK(fixture.table.pattern_mask[group] == pattern_bit(0b11));
}

TEST_CASE("Type-2 table accepts every row pattern of a wide group")
{
    // The table tracks patterns for up to four flips.
    // A wider group gets a mask that accepts everything.
    TableFixture fixture = make_table(8, {{"++----", {0, 1, 2, 3, 4, 5}, 1.0}});

    const std::size_t group = fixture.group_of({0, 1, 2, 3, 4, 5});
    REQUIRE(group != MAX_SIZE_T);
    CHECK(fixture.is_standard(group));
    CHECK(fixture.table.pattern_mask[group] == 0xFFFF);
}

TEST_CASE("Type-2 table marks an X operator as a don't care row bit")
{
    // An X accepts both 0 and 1 row bits at its flip, therefore the group accepts two
    // patterns. Such a group never reaches the direct path.
    TableFixture fixture = make_table(4, {{"X-", {0, 1}, 1.0}});

    const std::size_t group = fixture.group_of({0, 1});
    REQUIRE(group != MAX_SIZE_T);
    CHECK(fixture.is_standard(group));
    CHECK(fixture.table.pattern_mask[group] == (pattern_bit(0b00) | pattern_bit(0b01)));
}

TEST_CASE("Type-2 table chooses the exact half key for a narrow operator")
{
    TableFixture narrow = make_table(8, {{"--++", {0, 1, 6, 7}, 1.0}});
    CHECK(narrow.table.half_fits_in_word);
    CHECK(narrow.table.split_point == 4);
}

TEST_CASE("Half key reads the requested bit range")
{
    boost::dynamic_bitset<std::size_t> bits(8);
    bits[0] = 1;
    bits[3] = 1;
    bits[5] = 1;

    CHECK(half_bits(bits, 0, 4) == 0b1001);
    CHECK(half_bits(bits, 4, 8) == 0b0010);
    // Flip bit 0 and bit 3 of the low half.
    CHECK(half_bits(bits, 0, 4, 0, 3) == 0b0000);
    // A flip and a hash of the same half agree with themselves.
    CHECK(half_hash(bits, 0, 4) == half_hash(bits, 0, 4));
    CHECK(half_hash(bits, 0, 4, 0, 3) != half_hash(bits, 0, 4));
}

TEST_CASE("Range parity counts the set bits between two flips")
{
    boost::dynamic_bitset<std::size_t> bits(8);
    bits[2] = 1;
    bits[4] = 1;

    CHECK(range_parity(bits, 0, 8) == 0); // two set bits
    CHECK(range_parity(bits, 0, 3) == 1); // one set bit
    CHECK(range_parity(bits, 3, 3) == 0); // empty range

    // Flips at 1 and 6 enclose both set bits, therefore the sign is positive.
    const width_t flips[4] = {1, 6, 6, 7};
    CHECK(jw_4flip_sign(bits, flips) == 1.0);
    const width_t odd_flips[4] = {1, 3, 6, 7};
    CHECK(jw_4flip_sign(bits, odd_flips) == -1.0);
}

TEST_CASE("Direct groups of a molecular operator hold only ladder operators and Z")
{
    // The direct path evaluates an element as coeff * sign, where the sign is the
    // parity of the row bits in the two Jordan-Wigner Z ranges. That formula holds
    // only for a term that has a ladder operator at each flip, a Z at each index
    // inside the two ranges, and nothing else. This test reads real data and confirms
    // the property, instead of only confirming that synthetic bad groups get rejected.
    //
    // The path is relative, therefore run the test binary from the repository root.
    std::filesystem::path cwd = std::filesystem::current_path();
    FermionicOperator fop = FermionicOperator::from_json(cwd.parent_path() / "fulqrum/fulqrum/test/data/lih.json");
    QubitOperator op = fop.extended_jw_transformation();
    QubitOperator diag, off;
    std::tie(diag, off) = op.split_diagonal();
    off.group_sort();
    off.group_term_sort_by_ladder_int(2);

    std::vector<std::size_t> ptrs = off.group_ptrs();
    std::vector<std::vector<width_t>> group_inds = off.group_offdiag_indices();
    const std::size_t num_groups = ptrs.size() - 1;
    Type2GroupTable table =
        build_type2_group_table(off.terms, ptrs.data(), group_inds, num_groups, off.width);

    std::vector<std::size_t> direct = table.direct_unpaired_groups;
    for(const auto& entries : table.paired_by_low_pair)
    {
        for(const auto& entry : entries)
        {
            direct.push_back(entry.group);
        }
    }
    REQUIRE(direct.size() > 0);

    std::size_t terms_checked = 0;
    for(const std::size_t group : direct)
    {
        const GroupIndsView inds = table.inds(group);
        REQUIRE(inds.size() == 4);
        for(std::size_t idx = ptrs[group]; idx < ptrs[group + 1]; idx++)
        {
            const OperatorTerm_t& term = off.terms[idx];
            terms_checked++;
            CHECK(term.proj_indices.empty());
            CHECK(term.offdiag_weight == 4);
            CHECK(term.real_phase == 1);
            for(std::size_t k = 0; k < term.indices.size(); k++)
            {
                const width_t position = term.indices[k];
                const unsigned char value = term.values[k];
                const bool is_flip = (position == inds[0] || position == inds[1] ||
                                      position == inds[2] || position == inds[3]);
                if(is_flip)
                {
                    CHECK((value == OPER_VALUE_PLUS || value == OPER_VALUE_MINUS));
                }
                else
                {
                    // Every other operator is a Z, and it sits inside one of the two
                    // ranges that the sign formula reads.
                    CHECK(value == OPER_VALUE_Z);
                    const bool in_range = (position > inds[0] && position < inds[1]) ||
                                          (position > inds[2] && position < inds[3]);
                    CHECK(in_range);
                }
            }
        }
    }
    CHECK(terms_checked > 0);
}
