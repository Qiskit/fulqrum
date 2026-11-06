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
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "base.hpp"
#include "constants.hpp"
#include "elements.hpp"
#include "hermiticity.hpp"
#include "sector_layout.hpp"
#include "subspace.hpp"

inline constexpr std::size_t GROUPS_PER_CHUNK = 256;

/**
 * The tier that evaluates a group of terms.
 */
enum class GroupTier : std::uint8_t
{
    generic = 0,
    same_sector_single = 1,
    same_sector_double = 2,
    cross_sector_double = 3,
};

/**
 * Why a group is in the generic tier.
 */
enum class GenericReason : std::uint8_t
{
    none = 0,
    forced = 1,
    no_sector_layout = 2,
    crosses_sectors = 3,
    mixed_own_factors = 4,
    double_with_other_factor = 5,
    cross_with_projector = 6,
    cross_sign_not_separable = 7,
};

/**
 * Stores each term's contribution from qubits it does not flip.
 *
 * Each entry contains a value and three num_words-word masks: projector positions,
 * required projector bits, and Z positions. Its contribution is zero if a projector
 * fails; otherwise, the value changes sign for an odd number of set row bits at Z positions.
 */
struct FactorList
{
    std::size_t num_words = 0;
    std::vector<KeyWord> masks;
    std::vector<std::complex<double>> values;

    /**
     * Returns the number of entries.
     */
    std::size_t size() const
    {
        return values.size();
    }

    /**
     * Returns the projector mask of an entry; the expected and Z masks follow it.
     *
     * @param entry The entry number.
     */
    const KeyWord* entry_masks(const std::size_t entry) const
    {
        return masks.data() + entry * 3 * num_words;
    }

    /**
     * Appends an entry.
     *
     * @param entry_masks The projector, expected and Z masks, 3 * num_words words.
     * @param value The value of the entry.
     */
    void append(const KeyWord* new_masks, const std::complex<double> value)
    {
        masks.insert(masks.end(), new_masks, new_masks + 3 * num_words);
        values.push_back(value);
    }

    /**
     * Appends every entry of another list.
     *
     * @param other The other list. It must have the same num_words.
     */
    void append_list(const FactorList& other)
    {
        masks.insert(masks.end(), other.masks.begin(), other.masks.end());
        values.insert(values.end(), other.values.begin(), other.values.end());
    }

    /**
     * Returns the diagonal factor of an entry on a row: 0, +1 or -1.
     *
     * @param entry The entry number.
     * @param row The row words, num_words words.
     */
    int factor(const std::size_t entry, const KeyWord* row) const
    {
        const KeyWord* projector = entry_masks(entry);
        const KeyWord* expected = projector + num_words;
        const KeyWord* parity = expected + num_words;
        KeyWord mismatch = 0;
        unsigned int set_bits = 0;
        for(std::size_t word = 0; word < num_words; word++)
        {
            mismatch |= (row[word] & projector[word]) ^ expected[word];
            set_bits += static_cast<unsigned int>(__builtin_popcountll(row[word] & parity[word]));
        }
        if(mismatch != 0)
        {
            return 0;
        }
        return (set_bits & 1u) ? -1 : 1;
    }

    /**
     * Returns the sum of value times factor over a range of entries, on a row.
     *
     * @param begin The first entry.
     * @param end One past the last entry.
     * @param row The row words, num_words words.
     */
    std::complex<double> sum(const std::size_t begin, const std::size_t end, const KeyWord* row) const
    {
        std::complex<double> total = 0;
        for(std::size_t entry = begin; entry < end; entry++)
        {
            total += values[entry] * static_cast<double>(factor(entry, row));
        }
        return total;
    }
};

/**
 * A range [begin, end) of entries in a FactorList.
 */
struct EntryRange
{
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
};

/**
 * The cached data for one direction of a same-sector single excitation.
 *
 * The element on a row (own string a, other string b) is
 * own.sum(a) + own_sign(a) * cross.sum(b), where own_sign uses the shared own-sector
 * masks of the cross terms.
 */
struct SingleDirectionData
{
    EntryRange own;
    EntryRange cross;
    std::uint32_t shared_own_entry = 0;
};

/**
 * The data for one group: its tier, its sectors and the ranks of its flips.
 */
struct GroupData
{
    GroupTier tier = GroupTier::generic;
    GenericReason reason = GenericReason::none;
    std::uint8_t sector = 0;
    std::array<width_t, 4> flips = {0, 0, 0, 0};
    std::array<SingleDirectionData, 2> single;
    std::array<EntryRange, 6> double_patterns;
    std::array<std::complex<double>, 4> cross_values = {0, 0, 0, 0};
    std::uint32_t cross_sign_entry = 0;
};

/**
 * Returns the value of a term's flip factors for each of several row patterns.
 *
 * @param term The term.
 * @param flips The flip qubits of its group.
 * @param patterns The row patterns; bit i of a pattern is the row bit of flips[i].
 * @return One value per pattern: the coefficient times the flip entries <row|op|col>.
 */
inline std::vector<std::complex<double>> flip_values(const OperatorTerm_t& term,
                                                     const std::vector<width_t>& flips,
                                                     const std::vector<unsigned int>& patterns)
{
    std::vector<std::complex<double>> values(patterns.size(), term.coeff);
    for(std::size_t factor = 0; factor < term.indices.size(); factor++)
    {
        const auto position = std::find(flips.begin(), flips.end(), term.indices[factor]);
        if(position == flips.end())
        {
            continue;
        }
        const std::size_t flip = static_cast<std::size_t>(position - flips.begin());
        for(std::size_t pattern = 0; pattern < patterns.size(); pattern++)
        {
            const double row_bit = static_cast<double>((patterns[pattern] >> flip) & 1u);
            values[pattern] *= operator_entry(term.values[factor], row_bit, false);
        }
    }
    return values;
}

/**
 * Writes the masks of a term's diagonal factors, restricted to some qubits.
 *
 * @param term The term.
 * @param keep The qubits to keep, as key words.
 * @param num_words The number of key words.
 * @param target The projector, expected and Z masks, 3 * num_words words.
 */
inline void diagonal_masks(const OperatorTerm_t& term,
                           const std::vector<KeyWord>& keep,
                           const std::size_t num_words,
                           KeyWord* target)
{
    std::fill_n(target, 3 * num_words, KeyWord(0));
    for(std::size_t factor = 0; factor < term.indices.size(); factor++)
    {
        const width_t qubit = term.indices[factor];
        const std::size_t word = qubit / KEY_WORD_BITS;
        const KeyWord bit = KeyWord(1) << (qubit % KEY_WORD_BITS);
        if((keep[word] & bit) == 0)
        {
            continue;
        }
        const unsigned char value = term.values[factor];
        if(value == OPER_VALUE_Z)
        {
            target[2 * num_words + word] |= bit;
        }
        else if(value == OPER_VALUE_PROJ_0 || value == OPER_VALUE_PROJ_1)
        {
            target[word] |= bit;
            if(value == OPER_VALUE_PROJ_1)
            {
                target[num_words + word] |= bit;
            }
        }
    }
}

/**
 * Tells whether a mask has no set bit.
 *
 * @param mask The mask words.
 * @param num_words The number of words.
 */
inline bool mask_is_empty(const KeyWord* mask, const std::size_t num_words)
{
    for(std::size_t word = 0; word < num_words; word++)
    {
        if(mask[word] != 0)
        {
            return false;
        }
    }
    return true;
}

/**
 * Returns the index of a 4-bit occupation pattern with two set bits.
 *
 * @param bits The pattern; bit i is the row bit of the i-th flip qubit.
 * @return The index, 0 to 5, in increasing order of bits.
 */
inline std::size_t double_pattern_index(const unsigned int bits)
{
    std::size_t index = 0;
    for(unsigned int smaller = 0; smaller < bits; smaller++)
    {
        index += __builtin_popcount(smaller) == 2 ? 1 : 0;
    }
    return index;
}

/**
 * Returns the qubits of one sector, as key words.
 *
 * @param layout The sector layout.
 * @param sector The sector, 0 or 1.
 * @param num_words The number of key words.
 */
inline std::vector<KeyWord>
sector_qubits(const SectorLayout& layout, const std::uint8_t sector, const std::size_t num_words)
{
    std::vector<KeyWord> qubits(num_words, 0);
    for(std::size_t qubit = 0; qubit < layout.sector_of.size(); qubit++)
    {
        if(layout.sector_of[qubit] == sector)
        {
            qubits[qubit / KEY_WORD_BITS] |= KeyWord(1) << (qubit % KEY_WORD_BITS);
        }
    }
    return qubits;
}

/**
 * Returns the qubits of one sector strictly between two qubits, as key words.
 *
 * @param layout The sector layout.
 * @param low The lower qubit.
 * @param high The higher qubit.
 * @param num_words The number of key words.
 */
inline std::vector<KeyWord> sector_qubits_between(const SectorLayout& layout,
                                                  const width_t low,
                                                  const width_t high,
                                                  const std::size_t num_words)
{
    std::vector<KeyWord> qubits(num_words, 0);
    for(width_t qubit = low + 1; qubit < high; qubit++)
    {
        if(layout.sector_of[qubit] == layout.sector_of[low])
        {
            qubits[qubit / KEY_WORD_BITS] |= KeyWord(1) << (qubit % KEY_WORD_BITS);
        }
    }
    return qubits;
}

/**
 * The operator data that the sector tiers need: the sector layout, the tier of each
 * group, and the factors from which the tiers compute cached values.
 *
 * The tables depend on the operator and the layout only, not on the subspace strings.
 */
class OperatorTables
{
public:
    /**
     * Builds the tables of an operator whose terms are sorted into flip-set groups.
     *
     * @param oper The off-diagonal operator, group sorted.
     * @param group_ptrs The group boundaries, group_flips.size() + 1 values.
     * @param group_flips The flip qubits of each group, in increasing order.
     * @param layout The sector layout.
     * @param force_generic When true, every group goes to the generic tier.
     * @return The tables.
     */
    static OperatorTables build(const QubitOperator& oper,
                                const std::size_t* group_ptrs,
                                const std::vector<std::vector<width_t>>& group_flips,
                                const SectorLayout& layout,
                                const bool force_generic)
    {
        OperatorTables tables;
        tables.layout_ = layout;
        const std::size_t num_words = words_for_width(oper.width);
        tables.factors_.num_words = num_words;
        tables.groups_.resize(group_flips.size());
        if(layout.valid)
        {
            tables.sector_masks_[0] = sector_qubits(layout, 0, num_words);
            tables.sector_masks_[1] = sector_qubits(layout, 1, num_words);
        }
        const std::size_t num_groups = group_flips.size();
        const std::size_t num_chunks = (num_groups + GROUPS_PER_CHUNK - 1) / GROUPS_PER_CHUNK;
        std::vector<FactorList> chunk_factors(num_chunks);
#pragma omp parallel for schedule(dynamic, 1)
        for(std::size_t chunk = 0; chunk < num_chunks; chunk++)
        {
            chunk_factors[chunk].num_words = num_words;
            const std::size_t last = std::min(num_groups, (chunk + 1) * GROUPS_PER_CHUNK);
            for(std::size_t group = chunk * GROUPS_PER_CHUNK; group < last; group++)
            {
                tables.build_group_data(oper,
                                        group_ptrs[group],
                                        group_ptrs[group + 1],
                                        group_flips[group],
                                        force_generic,
                                        chunk_factors[chunk],
                                        tables.groups_[group]);
            }
        }
        for(std::size_t chunk = 0; chunk < num_chunks; chunk++)
        {
            const auto offset = static_cast<std::uint32_t>(tables.factors_.size());
            tables.factors_.append_list(chunk_factors[chunk]);
            const std::size_t last = std::min(num_groups, (chunk + 1) * GROUPS_PER_CHUNK);
            for(std::size_t group = chunk * GROUPS_PER_CHUNK; group < last; group++)
            {
                shift_entries(tables.groups_[group], offset);
            }
            chunk_factors[chunk] = FactorList();
        }
        return tables;
    }

    /**
     * Returns the sector layout.
     */
    const SectorLayout& layout() const
    {
        return layout_;
    }

    /**
     * Returns the number of groups.
     */
    std::size_t num_groups() const
    {
        return groups_.size();
    }

    /**
     * Returns the data of a group.
     *
     * @param group The group number.
     */
    const GroupData& group(const std::size_t group) const
    {
        return groups_[group];
    }

    /**
     * Returns the factor entries that the group data refers to.
     */
    const FactorList& factors() const
    {
        return factors_;
    }

    /**
     * Returns the element <row|H_g|row ^ flips> of one sector-tier group from its data.
     *
     * This is the reference that the sector tiers reproduce. A row whose flip bits do
     * not match a pattern of the group gives zero.
     *
     * @param group The group number. Its tier must not be generic.
     * @param row The row words.
     * @return The element.
     */
    std::complex<double> element(const std::size_t group, const KeyWord* row) const
    {
        const GroupData& data = groups_[group];
        auto row_bit = [row](const width_t qubit) {
            return static_cast<unsigned int>((row[qubit / KEY_WORD_BITS] >> (qubit % KEY_WORD_BITS)) & 1u);
        };
        if(data.tier == GroupTier::same_sector_single)
        {
            const unsigned int low = row_bit(data.flips[0]);
            if(low == row_bit(data.flips[1]))
            {
                return 0.0;
            }
            const SingleDirectionData& single = data.single[low == 1 ? 0 : 1];
            std::complex<double> value = factors_.sum(single.own.begin, single.own.end, row);
            if(single.cross.end > single.cross.begin)
            {
                value += static_cast<double>(factors_.factor(single.shared_own_entry, row)) *
                         factors_.sum(single.cross.begin, single.cross.end, row);
            }
            return value;
        }
        if(data.tier == GroupTier::same_sector_double)
        {
            unsigned int bits = 0;
            for(unsigned int flip = 0; flip < 4; flip++)
            {
                bits |= row_bit(data.flips[flip]) << flip;
            }
            if(__builtin_popcount(bits) != 2)
            {
                return 0.0;
            }
            const EntryRange range = data.double_patterns[double_pattern_index(bits)];
            return factors_.sum(range.begin, range.end, row);
        }
        std::array<width_t, 2> lower_flip = {MAX_WIDTH, MAX_WIDTH};
        std::array<unsigned int, 2> occupied = {0, 0};
        for(const width_t qubit : data.flips)
        {
            const std::uint8_t sector = layout_.sector_of[qubit];
            lower_flip[sector] = std::min(lower_flip[sector], qubit);
            occupied[sector] += row_bit(qubit);
        }
        if(occupied[0] != 1 || occupied[1] != 1)
        {
            return 0.0;
        }
        const int alpha_direction = row_bit(lower_flip[0]) ? 0 : 1;
        const int beta_direction = row_bit(lower_flip[1]) ? 0 : 1;
        return data.cross_values[2 * beta_direction + alpha_direction] *
               static_cast<double>(factors_.factor(data.cross_sign_entry, row));
    }

    /**
     * Returns the number of groups in a tier.
     *
     * @param tier The tier.
     */
    std::size_t count(const GroupTier tier) const
    {
        return static_cast<std::size_t>(std::count_if(
            groups_.begin(), groups_.end(), [tier](const GroupData& data) { return data.tier == tier; }));
    }

private:
    /**
     * Adds an offset to every factor entry number in a GroupData record.
     *
     * @param data The data.
     * @param offset The offset.
     */
    static void shift_entries(GroupData& data, const std::uint32_t offset)
    {
        for(SingleDirectionData& single : data.single)
        {
            single.own.begin += offset;
            single.own.end += offset;
            single.cross.begin += offset;
            single.cross.end += offset;
            single.shared_own_entry += offset;
        }
        for(EntryRange& range : data.double_patterns)
        {
            range.begin += offset;
            range.end += offset;
        }
        data.cross_sign_entry += offset;
    }

    /**
     * Sets the tier of one group and builds its factor entries.
     *
     * @param oper The operator.
     * @param begin The first term of the group.
     * @param end One past the last term of the group.
     * @param flips The flip qubits of the group.
     * @param force_generic When true, the group goes to the generic tier.
     * @param factors The list that receives the factor entries.
     * @param data The data to fill.
     */
    void build_group_data(const QubitOperator& oper,
                          const std::size_t begin,
                          const std::size_t end,
                          const std::vector<width_t>& flips,
                          const bool force_generic,
                          FactorList& factors,
                          GroupData& data) const
    {
        std::copy_n(flips.begin(), std::min<std::size_t>(flips.size(), 4), data.flips.begin());
        if(force_generic)
        {
            data.reason = GenericReason::forced;
            return;
        }
        if(!layout_.valid)
        {
            data.reason = GenericReason::no_sector_layout;
            return;
        }
        if(!group_fits_sectors(flips, layout_.sector_of))
        {
            data.reason = GenericReason::crosses_sectors;
            return;
        }
        std::size_t in_sector_one = 0;
        for(const width_t qubit : flips)
        {
            in_sector_one += layout_.sector_of[qubit];
        }
        data.sector = static_cast<std::uint8_t>(in_sector_one > 0);
        GenericReason reason = GenericReason::none;
        if(flips.size() == 2)
        {
            reason = build_single_data(oper, begin, end, flips, factors, data);
            data.tier = GroupTier::same_sector_single;
        }
        else if(in_sector_one == 0 || in_sector_one == 4)
        {
            reason = build_double_data(oper, begin, end, flips, factors, data);
            data.tier = GroupTier::same_sector_double;
        }
        else
        {
            reason = build_cross_data(oper, begin, end, flips, factors, data);
            data.tier = GroupTier::cross_sector_double;
        }
        if(reason != GenericReason::none)
        {
            data.tier = GroupTier::generic;
            data.reason = reason;
        }
    }

    /**
     * Builds factor data for both directions of a same-sector single excitation.
     *
     * Direction 0 has the lower flip qubit occupied in the row; direction 1 the higher.
     *
     * @return GenericReason::none, or the reason the group goes to the generic tier.
     */
    GenericReason build_single_data(const QubitOperator& oper,
                                    const std::size_t begin,
                                    const std::size_t end,
                                    const std::vector<width_t>& flips,
                                    FactorList& factors,
                                    GroupData& data) const
    {
        const std::size_t num_words = factors.num_words;
        const std::size_t stride = 3 * num_words;
        const std::size_t num_terms = end - begin;
        std::vector<KeyWord> own_masks(num_terms * stride);
        std::vector<KeyWord> other_masks(num_terms * stride);
        std::vector<std::array<std::complex<double>, 2>> values(num_terms);
        for(std::size_t term = 0; term < num_terms; term++)
        {
            const OperatorTerm_t& source = oper.terms[begin + term];
            const std::vector<std::complex<double>> both = flip_values(source, flips, {0b01u, 0b10u});
            values[term] = {both[0], both[1]};
            diagonal_masks(source, sector_masks_[data.sector], num_words, own_masks.data() + term * stride);
            diagonal_masks(source, sector_masks_[1 - data.sector], num_words, other_masks.data() + term * stride);
        }
        for(int direction = 0; direction < 2; direction++)
        {
            SingleDirectionData& single = data.single[direction];
            const KeyWord* shared_own = nullptr;
            std::vector<std::size_t> cross_terms;
            single.own.begin = static_cast<std::uint32_t>(factors.size());
            for(std::size_t term = 0; term < num_terms; term++)
            {
                const std::complex<double> value = values[term][direction];
                if(value == 0.0)
                {
                    continue;
                }
                const KeyWord* own = own_masks.data() + term * stride;
                if(mask_is_empty(other_masks.data() + term * stride, stride))
                {
                    factors.append(own, value);
                    continue;
                }
                if(shared_own == nullptr)
                {
                    shared_own = own;
                }
                else if(!std::equal(own, own + stride, shared_own))
                {
                    return GenericReason::mixed_own_factors;
                }
                cross_terms.push_back(term);
            }
            single.own.end = static_cast<std::uint32_t>(factors.size());
            if(shared_own != nullptr)
            {
                single.shared_own_entry = static_cast<std::uint32_t>(factors.size());
                factors.append(shared_own, 1.0);
            }
            single.cross.begin = static_cast<std::uint32_t>(factors.size());
            for(const std::size_t term : cross_terms)
            {
                factors.append(other_masks.data() + term * stride, values[term][direction]);
            }
            single.cross.end = static_cast<std::uint32_t>(factors.size());
        }
        return GenericReason::none;
    }

    /**
     * Builds the six occupation patterns of a same-sector double excitation.
     *
     * @return GenericReason::none, or the reason the group goes to the generic tier.
     */
    GenericReason build_double_data(const QubitOperator& oper,
                                    const std::size_t begin,
                                    const std::size_t end,
                                    const std::vector<width_t>& flips,
                                    FactorList& factors,
                                    GroupData& data) const
    {
        static const std::vector<unsigned int> patterns = {0b0011u, 0b0101u, 0b0110u, 0b1001u, 0b1010u, 0b1100u};
        const std::size_t num_words = factors.num_words;
        const std::size_t stride = 3 * num_words;
        const std::size_t num_terms = end - begin;
        std::vector<KeyWord> own_masks(num_terms * stride);
        std::vector<KeyWord> other_masks(stride);
        std::vector<std::vector<std::complex<double>>> values(num_terms);
        for(std::size_t term = 0; term < num_terms; term++)
        {
            const OperatorTerm_t& source = oper.terms[begin + term];
            values[term] = flip_values(source, flips, patterns);
            const bool acts = std::any_of(values[term].begin(), values[term].end(),
                                          [](const std::complex<double>& value) { return value != 0.0; });
            diagonal_masks(source, sector_masks_[1 - data.sector], num_words, other_masks.data());
            if(acts && !mask_is_empty(other_masks.data(), stride))
            {
                return GenericReason::double_with_other_factor;
            }
            diagonal_masks(source, sector_masks_[data.sector], num_words, own_masks.data() + term * stride);
        }
        for(std::size_t pattern = 0; pattern < patterns.size(); pattern++)
        {
            EntryRange& range = data.double_patterns[pattern];
            range.begin = static_cast<std::uint32_t>(factors.size());
            for(std::size_t term = 0; term < num_terms; term++)
            {
                if(values[term][pattern] != 0.0)
                {
                    factors.append(own_masks.data() + term * stride, values[term][pattern]);
                }
            }
            range.end = static_cast<std::uint32_t>(factors.size());
        }
        return GenericReason::none;
    }

    /**
     * Builds the four occupation patterns of a cross-sector double excitation.
     *
     * @return GenericReason::none, or the reason the group goes to the generic tier.
     */
    GenericReason build_cross_data(const QubitOperator& oper,
                                   const std::size_t begin,
                                   const std::size_t end,
                                   const std::vector<width_t>& flips,
                                   FactorList& factors,
                                   GroupData& data) const
    {
        const std::size_t num_words = factors.num_words;
        std::array<std::vector<width_t>, 2> sector_flips;
        for(const width_t qubit : flips)
        {
            sector_flips[layout_.sector_of[qubit]].push_back(qubit);
        }
        std::vector<KeyWord> expected_parity =
            sector_qubits_between(layout_, sector_flips[0][0], sector_flips[0][1], num_words);
        const std::vector<KeyWord> beta_between =
            sector_qubits_between(layout_, sector_flips[1][0], sector_flips[1][1], num_words);
        for(std::size_t word = 0; word < num_words; word++)
        {
            expected_parity[word] |= beta_between[word];
        }
        std::vector<unsigned int> patterns(4, 0);
        for(unsigned int pattern = 0; pattern < 4; pattern++)
        {
            const std::array<unsigned int, 2> direction = {pattern & 1u, pattern >> 1};
            for(std::size_t flip = 0; flip < flips.size(); flip++)
            {
                const std::uint8_t sector = layout_.sector_of[flips[flip]];
                const bool lower = flips[flip] == sector_flips[sector][0];
                if(lower == (direction[sector] == 0))
                {
                    patterns[pattern] |= 1u << flip;
                }
            }
        }
        const std::vector<KeyWord> all_qubits(num_words, ~KeyWord(0));
        std::vector<KeyWord> masks(3 * num_words, 0);
        for(std::size_t index = begin; index < end; index++)
        {
            const OperatorTerm_t& term = oper.terms[index];
            const std::vector<std::complex<double>> values = flip_values(term, flips, patterns);
            if(std::all_of(values.begin(), values.end(),
                           [](const std::complex<double>& value) { return value == 0.0; }))
            {
                continue;
            }
            diagonal_masks(term, all_qubits, num_words, masks.data());
            if(!mask_is_empty(masks.data(), num_words))
            {
                return GenericReason::cross_with_projector;
            }
            if(!std::equal(expected_parity.begin(), expected_parity.end(), masks.begin() + 2 * num_words))
            {
                return GenericReason::cross_sign_not_separable;
            }
            for(std::size_t pattern = 0; pattern < 4; pattern++)
            {
                data.cross_values[pattern] += values[pattern];
            }
        }
        std::fill(masks.begin(), masks.end(), KeyWord(0));
        std::copy(expected_parity.begin(), expected_parity.end(), masks.begin() + 2 * num_words);
        data.cross_sign_entry = static_cast<std::uint32_t>(factors.size());
        factors.append(masks.data(), 1.0);
        return GenericReason::none;
    }

    SectorLayout layout_;
    std::array<std::vector<KeyWord>, 2> sector_masks_;
    std::vector<GroupData> groups_;
    FactorList factors_;
};
