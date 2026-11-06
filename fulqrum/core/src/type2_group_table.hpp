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
#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#    define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif
#pragma once
// base.hpp brings in bitset_hashmap.hpp, which holds rapidhashMicro. Do not
// include external/rapidhash.h here, because that header has no include guard.
#include "base.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "element_evaluator.hpp"
#include "elements.hpp"
#include "offdiag_grouping.hpp"
#include <algorithm>
#include <array>
#include <boost/dynamic_bitset.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Type-2 group table
// ---------------------------------------------------------------------------
//
// The type-2 paths (csr2.hpp, csrlike_builder2.hpp, matvec2.hpp) sort the
// off-diagonal groups into three buckets and prepare the tables that the main
// loops read. This header holds that work, and the three path share it.
//
// The buckets come from the operator terms only. No bucket makes an assumption
// about spin order, about the subspace, or about particle number.
//
//   (1) standard groups: Each group goes through ``accum_element`` term by term.
//   (2) direct groups: Each group is a set of compatible Jordan-Wigner 4-flip terms
//          with one shared coefficient. Its matrix element is ``coeff * sign``, where
//          the sign is the parity of the row bits in the two Z-string ranges
//          (between ladder ops).
//
// Direct groups split again into "paired" and "unpaired". A paired group has
// two flips below ``split_point`` and two flips at or above it. A paired
// group uses the half-key prefilter for early rejection. The split point is a boundary
// for the prefilter, not a spin boundary. Any split point gives correct results.

inline constexpr std::size_t MAX_PATTERN_FLIPS = 4;
inline constexpr unsigned int NUM_MAX_FLIP_PATTERNS = 1u << MAX_PATTERN_FLIPS;

/**
 * Gives the mask that accepts each row pattern of a group.
 *
 * @param num_flips The number of flip indices of the group.
 *
 * @return A mask with one set bit for each of the 2^num_flips patterns.
 */
inline std::uint16_t all_patterns_mask(std::size_t num_flips)
{
    if(num_flips > MAX_PATTERN_FLIPS)
    {
        num_flips = MAX_PATTERN_FLIPS;
    }
    const std::uint32_t num_patterns = std::uint32_t(1) << num_flips;
    return static_cast<std::uint16_t>((std::uint32_t(1) << num_patterns) - std::uint32_t(1));
}

/**
 * Computes the row pattern at the flip indices of a group.
 *
 * Example: row = b01011011 (qubit 0 is the rightmost character), flip indices
 * [0, 1, 4, 5]. The row bits there are 1, 1, 1, 0 (in index order), so pattern =
 * (1 << 0) | (1 << 1) | (1 << 2) | (0 << 3) = 0b0111 = 7.
 *
 * No more than MAX_PATTERN_FLIPS indices are read.
 *
 * @param row_set_bits The row bits, one byte per qubit.
 * @param inds The flip indices of the group.
 * @param num_inds The number of flip indices.
 *
 * @return The row pattern as an unsigned integer.
 */
inline unsigned int row_pattern(const uint8_t* __restrict row_set_bits,
                                const width_t* __restrict inds,
                                std::size_t num_inds)
{
    if(num_inds > MAX_PATTERN_FLIPS)
    {
        num_inds = MAX_PATTERN_FLIPS;
    }
    unsigned int pattern = 0;
    for(std::size_t kk = 0; kk < num_inds; kk++)
    {
        pattern |= static_cast<unsigned int>(row_set_bits[inds[kk]]) << kk;
    }
    return pattern;
}

/**
 * Computes the valid row patterns for non-zero element for one term.
 *
 * Offdiagonal flip positions can only have +, -, X, and Y as operators. Only row bit=1
 * is valid pattern for a '+' at a flip index (row bit=0 for a '-'). An X or a Y is a
 * don't-care: both 0 and 1 row bits are valid for them.
 *
 * This term's mask is OR'd into the group's mask. We check the group mask once per row,
 * and if the actual row bits are a mismatch, we skip element eval for the group.
 *
 * @param term The operator term.
 * @param inds The flip indices of the group, in ascending order.
 * @param num_inds The number of flip indices.
 *
 * @return A mask with one set bit for each accepted row pattern.
 */
inline std::uint16_t term_row_pattern_mask(const OperatorTerm_t& term,
                                           const width_t* __restrict inds,
                                           const std::size_t num_inds)
{
    if(num_inds > MAX_PATTERN_FLIPS)
    {
        return all_patterns_mask(num_inds);
    }

    unsigned int fixed_positions = 0;
    unsigned int fixed_set_bits = 0;
    std::size_t num_found = 0;

    std::size_t term_pos = 0;
    std::size_t flip_pos = 0;
    const std::size_t term_length = term.indices.size();
    while(term_pos < term_length && flip_pos < num_inds)
    {
        if(term.indices[term_pos] < inds[flip_pos])
        {
            term_pos++;
        }
        else if(term.indices[term_pos] > inds[flip_pos])
        {
            flip_pos++;
        }
        else
        {
            const unsigned char value = term.values[term_pos];
            if(value == OPER_VALUE_PLUS)
            {
                fixed_positions |= (1u << flip_pos);
                fixed_set_bits |= (1u << flip_pos);
            }
            else if(value == OPER_VALUE_MINUS)
            {
                fixed_positions |= (1u << flip_pos);
            }
            num_found++;
            term_pos++;
            flip_pos++;
        }
    }

    // very likely a unreachable case, where
    // we have flip indices but no operator
    if(num_found != num_inds)
    {
        return all_patterns_mask(num_inds);
    }

    std::uint16_t mask = 0;
    const unsigned int num_patterns = 1u << num_inds;
    for(unsigned int pattern = 0; pattern < num_patterns; pattern++)
    {
        if((pattern & fixed_positions) == fixed_set_bits)
        {
            mask |= static_cast<std::uint16_t>(1u << pattern);
        }
    }
    return mask;
}

/**
 * Tells whether a term is a compatible Jordan-Wigner term on four flip indices.
 *
 * A compatible Jordan-Wigner 2-body term only has ladder operator on each flip index
 * i0 < i1 < i2 < i3 and a Z at each index in the middle ranges (i0, i1) and (i2, i3).
 * It has no other operator and no projector. Such a term has the matrix element
 * ``coeff * sign``, where the sign is the parity of the row bits in the two Z ranges.
 *
 * @param term The operator term.
 * @param flips The four flip indices, in ascending order.
 *
 * @return True if the term has the compatible Jordan-Wigner structure.
 */
inline bool term_is_compat_jw_4flip(const OperatorTerm_t& term, const width_t* __restrict flips)
{
    if(!term.proj_indices.empty() || term.offdiag_weight != 4 || term.real_phase != 1)
    {
        return false;
    }

    const width_t first = flips[0];
    const width_t second = flips[1];
    const width_t third = flips[2];
    const width_t fourth = flips[3];

    const std::size_t expected_length = 4 + static_cast<std::size_t>(second - first - 1) +
                                        static_cast<std::size_t>(fourth - third - 1);
    if(term.indices.size() != expected_length)
    {
        return false;
    }

    std::size_t pos = 0;
    const width_t ladder_pair[4] = {first, second, third, fourth};
    for(std::size_t kk = 0; kk < 4; kk++)
    {
        // The Z string that leads up to this ladder operator.
        if(kk == 1 || kk == 3)
        {
            for(width_t idx = ladder_pair[kk - 1] + 1; idx < ladder_pair[kk]; idx++, pos++)
            {
                if(term.indices[pos] != idx || term.values[pos] != OPER_VALUE_Z)
                {
                    return false;
                }
            }
        }
        const unsigned char value = term.values[pos];
        if(term.indices[pos] != ladder_pair[kk] ||
           (value != OPER_VALUE_PLUS && value != OPER_VALUE_MINUS))
        {
            return false;
        }
        pos++;
    }
    return true;
}

/**
 * Computes the parity of the set row bits in the range [lo, hi).
 *
 * @param row The row bitset.
 * @param lo The first bit position of the range.
 * @param hi The bit position after the range.
 *
 * @return 1 if the range holds an odd number of set bits, else 0.
 */
inline std::size_t
range_parity(const boost::dynamic_bitset<std::size_t>& row, const width_t lo, const width_t hi)
{
    if(lo >= hi)
    {
        return 0;
    }
    const width_t last = hi - 1; // inclusive last bit
    const std::size_t lo_block = lo >> BLOCK_EXPONENT;
    const std::size_t hi_block = static_cast<std::size_t>(last) >> BLOCK_EXPONENT;
    const std::size_t lo_mask = ~std::size_t(0) << (lo & BLOCK_SHIFT);
    const std::size_t hi_mask = ~std::size_t(0) >> (BLOCK_SHIFT - (last & BLOCK_SHIFT));
    std::size_t acc;
    if(lo_block == hi_block)
    {
        acc = row.m_bits[lo_block] & lo_mask & hi_mask;
    }
    else
    {
        acc = row.m_bits[lo_block] & lo_mask;
        for(std::size_t block = lo_block + 1; block < hi_block; block++)
        {
            acc ^= row.m_bits[block];
        }
        acc ^= row.m_bits[hi_block] & hi_mask;
    }
    return static_cast<std::size_t>(__builtin_parityll(static_cast<unsigned long long>(acc)));
}

/**
 * Computes the Jordan-Wigner sign of a 4-flip term for a given row.
 *
 * @param row The row bitset.
 * @param flips The four flip indices, in ascending order.
 *
 * @return -1.0 or 1.0.
 */
inline double jw_4flip_sign(const boost::dynamic_bitset<std::size_t>& row,
                            const width_t* __restrict flips)
{
    const std::size_t parity =
        range_parity(row, flips[0] + 1, flips[1]) ^ range_parity(row, flips[2] + 1, flips[3]);
    return parity ? -1.0 : 1.0;
}

/**
 * Gives the bits of the range [lo, hi) as one word, with flip_a and flip_b
 * bits flipped if given.
 *
 * We have a pre-filter based on half-bitsets. If a half fits in a single word,
 * its own bits already are a unique 64-bit identifier, so we use them directly
 * instead of computing a separate hash to make them 64-bit.
 *
 * Use this function only when the range holds no more than BITS_PER_BLOCK bits.
 * When half does not fit in a word, use half_hash().
 *
 * @param bitset The source bitset.
 * @param lo The first bit position of the range.
 * @param hi The bit position after the range.
 * @param flip_a A bit position to flip, or MAX_WIDTH for no flip.
 * @param flip_b A bit position to flip, or MAX_WIDTH for no flip.
 *
 * @return The bits of the range, in the low bits of the result.
 */
inline std::uint64_t half_bits(const boost::dynamic_bitset<std::size_t>& bitset,
                               const width_t lo,
                               const width_t hi,
                               const width_t flip_a = MAX_WIDTH,
                               const width_t flip_b = MAX_WIDTH)
{
    const std::size_t lo_block = lo >> BLOCK_EXPONENT;
    const std::size_t lo_offset = lo & BLOCK_SHIFT;
    const width_t num_bits = hi - lo;

    std::size_t word = bitset.m_bits[lo_block] >> lo_offset;
    if(lo_offset != 0 && (lo_block + 1) < bitset.num_blocks())
    {
        word |= bitset.m_bits[lo_block + 1] << (BITS_PER_BLOCK - lo_offset);
    }
    if(num_bits < BITS_PER_BLOCK)
    {
        word &= (std::size_t(1) << num_bits) - std::size_t(1);
    }
    if(flip_a != MAX_WIDTH)
    {
        word ^= std::size_t(1) << (flip_a - lo);
    }
    if(flip_b != MAX_WIDTH)
    {
        word ^= std::size_t(1) << (flip_b - lo);
    }
    return static_cast<std::uint64_t>(word);
}

/**
 * Gives a hash of the bits in the range [lo, hi).
 *
 * This is the fallback of ``half_bits`` for a range that is wider than
 * BITS_PER_BLOCK bits. The parameters have the same meaning. Two different halves
 * can give the same hash in rare cases, therefore a set of these keys gives false
 * positives. A false positive costs one subspace look-up and never changes a result.
 */
inline std::uint64_t half_hash(const boost::dynamic_bitset<std::size_t>& bitset,
                               const width_t lo,
                               const width_t hi,
                               const width_t flip_a = MAX_WIDTH,
                               const width_t flip_b = MAX_WIDTH)
{
    const std::size_t lo_block = lo >> BLOCK_EXPONENT;
    const std::size_t hi_block = static_cast<std::size_t>(hi - 1) >> BLOCK_EXPONENT;
    const std::size_t num_blocks = hi_block - lo_block + 1;
    static thread_local std::vector<std::size_t> buffer;
    if(buffer.size() < num_blocks)
    {
        buffer.resize(num_blocks);
    }
    for(std::size_t block = lo_block; block <= hi_block; block++)
    {
        std::size_t word = bitset.m_bits[block];
        if(flip_a != MAX_WIDTH && (flip_a >> BLOCK_EXPONENT) == block)
        {
            word ^= (std::size_t(1) << (flip_a & BLOCK_SHIFT));
        }
        if(flip_b != MAX_WIDTH && (flip_b >> BLOCK_EXPONENT) == block)
        {
            word ^= (std::size_t(1) << (flip_b & BLOCK_SHIFT));
        }
        if(block == lo_block)
        {
            word &= (~std::size_t(0) << (lo & BLOCK_SHIFT));
        }
        if(block == hi_block)
        {
            const std::size_t offset = static_cast<std::size_t>(hi - 1) & BLOCK_SHIFT;
            word &= (offset == BLOCK_SHIFT) ? ~std::size_t(0)
                                            : (~std::size_t(0) >> (BLOCK_SHIFT - offset));
        }
        buffer[block - lo_block] = word;
    }
    return rapidhashMicro(buffer.data(), num_blocks * sizeof(std::size_t));
}

struct PairedDirectGroup
{
    std::uint32_t high_pair_id;
    std::size_t group;
};

struct Type2GroupTable
{
    // The flip indices of every group, in one contiguous array.
    std::vector<width_t> flat_inds;
    std::vector<std::size_t> inds_offsets;

    std::vector<std::uint16_t> pattern_mask;
    std::vector<std::complex<double>> direct_coeff;

    // The buckets. Each list holds group numbers in ascending order.
    // direct_paired_groups is not a flat list like below two.
    // It is grouped by low pair in paired_by_low_pair.
    std::vector<std::size_t> standard_groups;
    std::vector<std::size_t> direct_unpaired_groups;

    // The prefilter boundary.
    // It splits the bitset into a low half [0, split_point)
    // and a high half [split_point, width).
    width_t split_point{0};

    bool half_fits_in_word{true};

    std::vector<std::array<width_t, 2>> low_pairs;
    std::vector<std::array<width_t, 2>> high_pairs;
    std::vector<std::uint8_t> low_pair_mask;
    std::vector<std::uint8_t> high_pair_mask;

    // The direct paired groups, held under the identifier of their low pair.
    std::vector<std::vector<PairedDirectGroup>> paired_by_low_pair;

    GroupIndsView inds(std::size_t group) const
    {
        const std::size_t offset = inds_offsets[group];
        return GroupIndsView{flat_inds.data() + offset, inds_offsets[group + 1] - offset};
    }

    bool has_paired_groups() const
    {
        return !low_pairs.empty();
    }

    std::uint64_t half_key(const boost::dynamic_bitset<std::size_t>& bitset,
                           const width_t lo,
                           const width_t hi,
                           const width_t flip_a = MAX_WIDTH,
                           const width_t flip_b = MAX_WIDTH) const
    {
        if(half_fits_in_word)
        {
            return half_bits(bitset, lo, hi, flip_a, flip_b);
        }
        return half_hash(bitset, lo, hi, flip_a, flip_b);
    }
};

/**
 * Converts the shared coefficient of a direct group into the kernel value type.
 */
template <typename T>
inline T direct_coeff_as(const std::complex<double>& coeff)
{
    if constexpr(std::is_same_v<T, double>)
    {
        return coeff.real();
    }
    else
    {
        return static_cast<T>(coeff);
    }
}

/**
 * Sorts the off-diagonal groups into buckets and builds the look-up tables.
 *
 * The function reads the operator terms only. It makes no assumption about spin
 * order, about the subspace, or about particle number.
 *
 * @param terms The off-diagonal operator terms, sorted by group.
 * @param group_ptrs The first term of each group. It has num_groups + 1 entries.
 * @param group_offdiag_inds The flip indices of each group, in ascending order.
 * @param num_groups The number of groups.
 * @param width The operator width.
 *
 * @return The table.
 */
inline Type2GroupTable
build_type2_group_table(const std::vector<OperatorTerm_t>& terms,
                        const std::size_t* __restrict group_ptrs,
                        const std::vector<std::vector<width_t>>& group_offdiag_inds,
                        const std::size_t num_groups,
                        const width_t width)
{
    Type2GroupTable table;
    flatten_offdiag_inds(group_offdiag_inds, table.flat_inds, table.inds_offsets);
    table.split_point = width / 2;
    // The low half holds split_point bits and the high half holds the rest.
    table.half_fits_in_word = (width <= 2 * BITS_PER_BLOCK);
    table.pattern_mask.assign(num_groups, 0);
    table.direct_coeff.assign(num_groups, std::complex<double>(0.0, 0.0));

    std::unordered_map<std::uint32_t, std::uint32_t> low_pair_ids;
    std::unordered_map<std::uint32_t, std::uint32_t> high_pair_ids;

    for(std::size_t group = 0; group < num_groups; group++)
    {
        const GroupIndsView inds = table.inds(group);
        const std::size_t num_flips = inds.size();
        const std::size_t first_term = group_ptrs[group];
        const std::size_t stop_term = group_ptrs[group + 1];

        if(first_term >= stop_term)
        {
            // An empty group gives no element on any path.
            table.pattern_mask[group] = 0;
            continue;
        }

        // The row patterns that the group accepts.
        std::uint16_t pattern_mask = 0;
        bool has_repeat_pattern = false;
        for(std::size_t idx = first_term; idx < stop_term; idx++)
        {
            const std::uint16_t term_mask =
                term_row_pattern_mask(terms[idx], inds.data(), num_flips);
            if(pattern_mask & term_mask)
            {
                has_repeat_pattern = true;
            }
            pattern_mask |= term_mask;
        }
        table.pattern_mask[group] = pattern_mask;

        // The direct path needs four compatible Jordan-Wigner flips, one shared
        // coefficient, and one term per row pattern.
        bool is_direct = (num_flips == 4) && !has_repeat_pattern;
        if(is_direct)
        {
            const std::complex<double> shared_coeff = terms[first_term].coeff;
            if(std::abs(shared_coeff) <= ATOL)
            {
                is_direct = false;
            }
            for(std::size_t idx = first_term; is_direct && idx < stop_term; idx++)
            {
                if(!term_is_compat_jw_4flip(terms[idx], inds.data()) ||
                   std::abs(terms[idx].coeff - shared_coeff) > 1e-14)
                {
                    is_direct = false;
                }
            }
            if(is_direct)
            {
                table.direct_coeff[group] = shared_coeff;
            }
        }

        if(!is_direct)
        {
            table.standard_groups.push_back(group);
            continue;
        }

        const bool is_paired = (inds[1] < table.split_point) && (inds[2] >= table.split_point);
        if(!is_paired)
        {
            table.direct_unpaired_groups.push_back(group);
            continue;
        }

        const width_t low_first = inds[0];
        const width_t low_second = inds[1];
        const width_t high_first = inds[2];
        const width_t high_second = inds[3];
        const std::uint32_t low_key = (std::uint32_t(low_first) << 16) | low_second;
        const std::uint32_t high_key = (std::uint32_t(high_first) << 16) | high_second;

        std::uint32_t low_pair_id;
        auto low_found = low_pair_ids.find(low_key);
        if(low_found == low_pair_ids.end())
        {
            low_pair_id = static_cast<std::uint32_t>(table.low_pairs.size());
            low_pair_ids.emplace(low_key, low_pair_id);
            table.low_pairs.push_back({low_first, low_second});
            table.low_pair_mask.push_back(0);
            table.paired_by_low_pair.emplace_back();
        }
        else
        {
            low_pair_id = low_found->second;
        }

        std::uint32_t high_pair_id;
        auto high_found = high_pair_ids.find(high_key);
        if(high_found == high_pair_ids.end())
        {
            high_pair_id = static_cast<std::uint32_t>(table.high_pairs.size());
            high_pair_ids.emplace(high_key, high_pair_id);
            table.high_pairs.push_back({high_first, high_second});
            table.high_pair_mask.push_back(0);
        }
        else
        {
            high_pair_id = high_found->second;
        }

        // Project the group patterns onto the two pairs. Bits 0 and 1 of a pattern
        // belong to the low pair, and bits 2 and 3 belong to the high pair.
        std::uint8_t low_bits = 0;
        std::uint8_t high_bits = 0;
        for(unsigned int pattern = 0; pattern < NUM_MAX_FLIP_PATTERNS; pattern++)
        {
            if((pattern_mask >> pattern) & 1u)
            {
                low_bits |= static_cast<std::uint8_t>(1u << (pattern & 3u));
                high_bits |= static_cast<std::uint8_t>(1u << ((pattern >> 2) & 3u));
            }
        }
        table.low_pair_mask[low_pair_id] |= low_bits;
        table.high_pair_mask[high_pair_id] |= high_bits;
        table.paired_by_low_pair[low_pair_id].push_back({high_pair_id, group});
    }

    return table;
}

// ---------------------------------------------------------------------------
// Row loop helpers
// ---------------------------------------------------------------------------
//
// The three type-2 paths run the same row loops and differ only in what they do
// with a computed element. These helpers hold the shared work, avoiding duplicate
// code.

using SubspaceEntry = bitset_map_namespace::BitsetMap::value_type;

template <typename ValueT>
struct Type2RowInputs
{
    const std::vector<OperatorTerm_t>& terms;
    const TermMasks& masks;
    const bitset_map_namespace::BitsetHashMapWrapper& subspace;
    const Type2GroupTable& table;
    const SubspaceEntry* __restrict bitsets;
    const ValueT* __restrict direct_value;
    const width_t* __restrict group_rowint_length;
    const std::uint32_t* __restrict ladder32;
    const std::size_t* __restrict group_ladder_ptrs;
    unsigned int ladder_offset;
    bool ladder_fits;

    // The lean uint32 ladder table when the operator fits it, else the original.
    std::size_t ladder(std::size_t idx) const
    {
        return ladder_fits ? static_cast<std::size_t>(ladder32[idx]) : group_ladder_ptrs[idx];
    }
};

/**
 * @param bitsets The subspace bitsets.
 * @param first_row The first row of the block.
 * @param num_rows The number of rows in the block.
 * @param row_stride The bytes per row.
 * @param row_bits The output buffer. The function sizes and clears it.
 */
inline void fill_block_row_bits(const SubspaceEntry* __restrict bitsets,
                                const std::size_t first_row,
                                const std::size_t num_rows,
                                const std::size_t row_stride,
                                std::vector<uint8_t>& row_bits)
{
    row_bits.assign(num_rows * row_stride, 0);
    for(std::size_t row_in_block = 0; row_in_block < num_rows; row_in_block++)
    {
        const boost::dynamic_bitset<std::size_t>& row = bitsets[first_row + row_in_block].first;
        uint8_t* dst = row_bits.data() + row_in_block * row_stride;
        for(std::size_t block = 0; block < row.num_blocks(); block++)
        {
            std::size_t bits = row.m_bits[block];
            while(bits != 0)
            {
                const int position = __builtin_ctzll(bits);
                dst[block * BITS_PER_BLOCK + position] = 1;
                bits &= bits - 1;
            }
        }
    }
}

/**
 * Tells whether some term of a group accepts the row bits at its flip indices.
 *
 * @param table The group table.
 * @param group The group number.
 * @param group_inds The flip indices of the group.
 * @param row_set_bits The bits of the row, one byte per qubit.
 *
 * @return True when the group can give a non-zero element for this row.
 */
inline bool type2_row_pattern_allowed(const Type2GroupTable& table,
                                      const std::size_t group,
                                      const GroupIndsView& group_inds,
                                      const uint8_t* __restrict row_set_bits)
{
    const unsigned int pattern = row_pattern(row_set_bits, group_inds.data(), group_inds.size());
    return ((table.pattern_mask[group] >> pattern) & 1u) != 0;
}

/**
 * Builds the candidate column of a group and looks it up in the subspace.
 *
 * @param inputs The per-row lookup data shared across groups.
 * @param group_inds The flip indices of the group.
 * @param row The row bitset.
 * @param col_vec Scratch bitset. The function overwrites it with the column.
 * @param col_idx Output column, valid when the function returns true.
 *
 * @return True when the column is in the subspace.
 */
template <typename ValueT>
inline bool type2_lookup_column(const Type2RowInputs<ValueT>& inputs,
                                const GroupIndsView& group_inds,
                                const boost::dynamic_bitset<std::size_t>& row,
                                boost::dynamic_bitset<std::size_t>& col_vec,
                                std::size_t& col_idx)
{
    col_vec = row;
    flip_bits(col_vec, group_inds.data(), group_inds.size());

    std::size_t* col_ptr = inputs.subspace.get_ptr(col_vec);
    if(col_ptr == nullptr)
    {
        return false;
    }
    col_idx = *col_ptr;
    return true;
}

/**
 * Evaluates one standard group for one row.
 *
 * ``accum_element`` loops over the terms of the group that the ladder bin selects.
 *
 * @param inputs The per-row lookup data shared across groups.
 * @param row_index The row, as a subspace index.
 * @param row_set_bits The bits of that row, one byte per qubit.
 * @param group The group number.
 * @param col_vec Scratch bitset. The function overwrites it.
 * @param col_idx Output column, valid when the function returns true.
 * @param value Output element, valid when the function returns true.
 *
 * @return True when the group gives a non-zero element for this row.
 */
template <typename ValueT>
inline bool type2_standard_element(const Type2RowInputs<ValueT>& inputs,
                                   const std::size_t row_index,
                                   const uint8_t* __restrict row_set_bits,
                                   const std::size_t group,
                                   boost::dynamic_bitset<std::size_t>& col_vec,
                                   std::size_t& col_idx,
                                   ValueT& value)
{
    const GroupIndsView group_inds = inputs.table.inds(group);
    if(!type2_row_pattern_allowed(inputs.table, group, group_inds, row_set_bits))
    {
        return false;
    }

    const unsigned int row_int =
        bitset_ladder_int(row_set_bits, group_inds.data(), inputs.group_rowint_length[group]);
    const std::size_t first_term = inputs.ladder(group * inputs.ladder_offset + row_int);
    const std::size_t stop_term = inputs.ladder(group * inputs.ladder_offset + row_int + 1);
    if(first_term >= stop_term)
    {
        return false;
    }

    const boost::dynamic_bitset<std::size_t>& row = inputs.bitsets[row_index].first;
    if(!type2_lookup_column(inputs, group_inds, row, col_vec, col_idx))
    {
        return false;
    }

    value = term_sum<ValueT>(inputs.masks, first_term, stop_term, row.m_bits.data());
    return std::abs(value) > ATOL;
}

/**
 * Evaluates one direct group for one row.
 *
 * The element is the shared coefficient times the Jordan-Wigner sign, therefore
 * this path reads neither the ladder table nor the terms. The parameters and the
 * return value match ``type2_standard_element``.
 */
template <typename ValueT>
inline bool type2_direct_element(const Type2RowInputs<ValueT>& inputs,
                                 const std::size_t row_index,
                                 const uint8_t* __restrict row_set_bits,
                                 const std::size_t group,
                                 boost::dynamic_bitset<std::size_t>& col_vec,
                                 std::size_t& col_idx,
                                 ValueT& value)
{
    const GroupIndsView group_inds = inputs.table.inds(group);
    if(!type2_row_pattern_allowed(inputs.table, group, group_inds, row_set_bits))
    {
        return false;
    }

    const boost::dynamic_bitset<std::size_t>& row = inputs.bitsets[row_index].first;
    if(!type2_lookup_column(inputs, group_inds, row, col_vec, col_idx))
    {
        return false;
    }

    value = inputs.direct_value[group] * static_cast<ValueT>(jw_4flip_sign(row, group_inds.data()));
    return std::abs(value) > ATOL;
}

/**
 * Marks the flip pairs that can give a column inside the subspace, for one row.
 *
 * A direct paired group flips two bits in each half of the bitset, therefore the
 * low half of a candidate column must equal the low half of some subspace element,
 * and the high half must equal the high half of some element. Both conditions are
 * necessary, and they hold per flip pair instead of per group. A failed low pair
 * then drops each of its groups without a look-up in the subspace hash map.
 *
 * @param table The group table.
 * @param row The row bitset.
 * @param row_set_bits The bits of that row, one byte per qubit.
 * @param low_half_set Keys of every low half in the subspace.
 * @param high_half_set Keys of every high half in the subspace.
 * @param width The operator width.
 * @param low_pair_ok Output flags, one per low pair.
 * @param high_pair_ok Output flags, one per high pair.
 */
template <typename HashSetT>
inline void type2_prefilter_pairs(const Type2GroupTable& table,
                                  const boost::dynamic_bitset<std::size_t>& row,
                                  const uint8_t* __restrict row_set_bits,
                                  const HashSetT& low_half_set,
                                  const HashSetT& high_half_set,
                                  const width_t width,
                                  char* __restrict low_pair_ok,
                                  char* __restrict high_pair_ok)
{
    const width_t split_point = table.split_point;

    for(std::size_t i = 0; i < table.low_pairs.size(); i++)
    {
        const width_t first = table.low_pairs[i][0];
        const width_t second = table.low_pairs[i][1];
        const unsigned int bits =
            row_set_bits[first] | (static_cast<unsigned int>(row_set_bits[second]) << 1);
        if(!((table.low_pair_mask[i] >> bits) & 1u))
        {
            low_pair_ok[i] = 0;
            continue;
        }
        low_pair_ok[i] = low_half_set.contains(table.half_key(row, 0, split_point, first, second));
    }

    for(std::size_t j = 0; j < table.high_pairs.size(); j++)
    {
        const width_t first = table.high_pairs[j][0];
        const width_t second = table.high_pairs[j][1];
        const unsigned int bits =
            row_set_bits[first] | (static_cast<unsigned int>(row_set_bits[second]) << 1);
        if(!((table.high_pair_mask[j] >> bits) & 1u))
        {
            high_pair_ok[j] = 0;
            continue;
        }
        high_pair_ok[j] =
            high_half_set.contains(table.half_key(row, split_point, width, first, second));
    }
}
