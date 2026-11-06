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
#include "../fulqrum/core/src/constants.hpp"
#include "../fulqrum/core/src/elements.hpp"
#include "../fulqrum/gpu/kernels/formats/clique_word.hpp"
#include "../fulqrum/gpu/kernels/formats/group_key.hpp"
#include "../fulqrum/gpu/kernels/formats/term_word.hpp"
#include "fulqrum.hpp"
#include <algorithm>
#include <cstdint>
#include <string>
#include <cstring>
#include <vector>

namespace
{

const int kTerms = 3;
const int kStride = 5;
const std::uint8_t kTermIndices[] = {0, 7, 9, 0, 0, 3, 0, 0, 0, 0, 1, 2, 63, 64, 127};
const std::uint8_t kTermValues[] = {3, 4, 6, 0, 0, 5, 0, 0, 0, 0, 0, 1, 2, 3, 4};
const std::uint32_t kNumOperators[] = {3, 1, 5};
const std::uint32_t kNumProjectors[] = {2, 0, 7};
const double kCoefficients[] = {0.5, -1.25, 3e-7};
const std::uint64_t kTermWords[] = {0x60904070300ull, 0x0ull, 0x503ull, 0x0ull, 0x340023f01020001ull, 0x47full};
const std::uint64_t kTermHeader[] = {
    0x3fe0000000000000ull, 0x203ull, 0xbff4000000000000ull, 0x1ull, 0x3e9421f5f40d8376ull, 0x705ull};

const std::vector<std::vector<std::uint32_t>> kGroups = {{0, 1},
                                                         {1, 3},
                                                         {4, 6},
                                                         {5, 7},
                                                         {0, 1, 2, 3},
                                                         {4, 5, 6, 7},
                                                         {0, 2, 4, 6},
                                                         {1, 2, 5, 6},
                                                         {0, 3, 5, 7},
                                                         {2, 3}};
const std::uint32_t kRowIntLengths[] = {2, 1, 2, 2, 4, 3, 4, 2, 1, 0};
const std::uint32_t kNumAlpha = 4;
const std::uint32_t kG2Keys[] = {0xffffffffu,
                                 0x30001u,
                                 0xffffffffu,
                                 0xffffffffu,
                                 0xffffffffu,
                                 0xffffffffu,
                                 0xffffffffu,
                                 0x10000u,
                                 0xffffffffu,
                                 0x60004u,
                                 0x70005u,
                                 0xffffffffu,
                                 0x30002u,
                                 0xffffffffu,
                                 0xffffffffu,
                                 0xffffffffu};
const std::int32_t kG2Values[] = {-1, 257, -1, -1, -1, -1, -1, 2, -1, 514, 770, -1, 2304, -1, -1, -1};
const std::uint32_t kG2Mask = 15u;
const std::uint32_t kG2Bitset[] = {0xffffe97du};
const std::uint64_t kG4Keys[] = {0xffffffffffffffffull,
                                 0xffffffffffffffffull,
                                 0xffffffffffffffffull,
                                 0xffffffffffffffffull,
                                 0xffffffffffffffffull,
                                 0x3000200010000ull,
                                 0xffffffffffffffffull,
                                 0x7000600050004ull};
const std::int32_t kG4Values[] = {-1, -1, -1, -1, -1, 1028, -1, 1283};
const std::uint64_t kG4Mask = 7ull;
const std::uint64_t kG4Bitset[] = {0xffffffffffffff5full};

std::int32_t find2(const std::uint32_t key)
{
    std::uint64_t slot = fq::group_key::first_slot2(key, kG2Mask);
    if(fq::group_key::is_empty2(kG2Bitset[fq::group_key::bitset_word2(slot)], slot))
    {
        return fq::group_key::kEmptyValue;
    }
    for(;;)
    {
        if(kG2Keys[slot] == key)
        {
            return kG2Values[slot];
        }
        if(kG2Keys[slot] == fq::group_key::kEmptyKey2)
        {
            return fq::group_key::kEmptyValue;
        }
        slot = fq::group_key::next_slot(slot, kG2Mask);
    }
}

std::int32_t find4(const std::uint64_t key)
{
    std::uint64_t slot = fq::group_key::first_slot4(key, kG4Mask);
    if(fq::group_key::is_empty4(kG4Bitset[fq::group_key::bitset_word4(slot)], slot))
    {
        return fq::group_key::kEmptyValue;
    }
    for(;;)
    {
        if(kG4Keys[slot] == key)
        {
            return kG4Values[slot];
        }
        if(kG4Keys[slot] == fq::group_key::kEmptyKey4)
        {
            return fq::group_key::kEmptyValue;
        }
        slot = fq::group_key::next_slot(slot, kG4Mask);
    }
}

}

TEST_CASE("Term headers preserve coefficients and counts")
{
    for(int t = 0; t < kTerms; t++)
    {
        const std::uint64_t* header = kTermHeader + t * fq::term_word::kHeaderWords;
        double coefficient = 0.0;
        std::memcpy(&coefficient, header + fq::term_word::kHeaderCoefficientWord, sizeof(double));
        CHECK(coefficient == kCoefficients[t]);
        const std::uint64_t meta = header[fq::term_word::kHeaderMetaWord];
        CHECK(meta == fq::term_word::pack_meta(kNumProjectors[t], kNumOperators[t]));
        CHECK(fq::term_word::meta_num_operators(meta) == int(kNumOperators[t]));
        CHECK(fq::term_word::meta_num_projectors(meta) == int(kNumProjectors[t]));
    }
    const std::uint64_t legacy = (std::uint64_t(std::uint16_t(7) << 8) | std::uint16_t(200)) | (0xabcdull << 16);
    CHECK(fq::term_word::meta_num_operators(legacy) == 200);
    CHECK(fq::term_word::meta_num_projectors(legacy) == 7);
}

TEST_CASE("Packed operator words preserve positions and values")
{
    const int packed = fq::term_word::packed_stride(kStride);
    CHECK(packed == 2);
    CHECK(fq::term_word::packed_stride(4) == 1);
    CHECK(fq::term_word::packed_stride(1) == 1);
    for(int t = 0; t < kTerms; t++)
    {
        const std::uint64_t* row = kTermWords + t * packed;
        for(int k = 0; k < packed * fq::term_word::kOperatorsPerWord; k++)
        {
            const std::uint16_t word = fq::term_word::operator_in_word(row[fq::term_word::packed_word_of(k)], k);
            if(k >= kStride)
            {
                CHECK(word == 0);
                continue;
            }
            const int slot = t * kStride + k;
            const std::uint16_t legacy = std::uint16_t((std::uint16_t(kTermValues[slot]) << 8) | kTermIndices[slot]);
            CHECK(word == legacy);
            CHECK(word == fq::term_word::pack_operator(kTermIndices[slot], kTermValues[slot]));
            CHECK(fq::term_word::operator_position(word) == kTermIndices[slot]);
            CHECK(fq::term_word::operator_value(word) == kTermValues[slot]);
        }
        for(int w = 0; w < packed; w++)
        {
            std::uint16_t ops[4] = {0, 0, 0, 0};
            for(int i = 0; i < 4; i++)
            {
                const int k = w * 4 + i;
                if(k < kStride)
                {
                    ops[i] = fq::term_word::pack_operator(kTermIndices[t * kStride + k], kTermValues[t * kStride + k]);
                }
            }
            CHECK(fq::term_word::pack_operators(ops[0], ops[1], ops[2], ops[3]) == row[w]);
        }
    }
}

TEST_CASE("GPU operator elements match CPU values")
{
    CHECK(fq::term_word::kOperatorZ == OPER_VALUE_Z);
    CHECK(fq::term_word::kOperatorProjector0 == OPER_VALUE_PROJ_0);
    CHECK(fq::term_word::kOperatorProjector1 == OPER_VALUE_PROJ_1);
    CHECK(fq::term_word::kOperatorX == OPER_VALUE_X);
    CHECK(fq::term_word::kOperatorY == OPER_VALUE_Y);
    CHECK(fq::term_word::kOperatorMinus == OPER_VALUE_MINUS);
    CHECK(fq::term_word::kOperatorPlus == OPER_VALUE_PLUS);
    CHECK(fq::term_word::kOperatorCount == rev_oper_map.size());
    for(std::uint32_t value = 0; value < fq::term_word::kOperatorCount; value++)
    {
        for(std::uint32_t row = 0; row < 2; row++)
        {
            for(std::uint32_t col = 0; col < 2; col++)
            {
                CHECK(double(fq::term_word::operator_element(value, row, col)) ==
                      REAL_OPER_ELEMENTS[4 * value + 2 * row + col]);
            }
        }
    }
}

TEST_CASE("Sector bits and projector offsets are correct")
{
    const std::uint64_t alpha = 0b1010;
    const std::uint64_t beta = 0b0110;
    const std::uint32_t expected[] = {0, 1, 0, 1, 0, 1, 1, 0};
    for(std::uint32_t p = 0; p < 2 * kNumAlpha; p++)
    {
        CHECK(fq::term_word::string_bit(alpha, beta, p, kNumAlpha) == expected[p]);
    }
    CHECK(fq::term_word::string_bit(0, std::uint64_t(1) << 63, 127, 64) == 1);
    CHECK(fq::term_word::string_bit(std::uint64_t(1) << 63, 0, 63, 64) == 1);
    CHECK(fq::term_word::projector_slot(3, 2, 7) == 23);
    CHECK(fq::term_word::projector_slot(std::int64_t(1) << 32, 1, 4) == (std::int64_t(1) << 34) + 1);
}

TEST_CASE("Group keys and values match host tables")
{
    for(std::size_t g = 0; g < kGroups.size(); g++)
    {
        const std::vector<std::uint32_t>& p = kGroups[g];
        if(p.size() == 2)
        {
            const std::int32_t value = find2(fq::group_key::pack_key2(p[0], p[1]));
            CHECK(value == fq::group_key::pack_group_value(std::uint32_t(g), kRowIntLengths[g]));
            CHECK(fq::group_key::value_group(value) == int(g));
            CHECK(fq::group_key::value_row_int_length(value) == int(kRowIntLengths[g]));
            continue;
        }
        int alpha_positions = 0;
        for(const std::uint32_t position : p)
        {
            alpha_positions += position < kNumAlpha ? 1 : 0;
        }
        const std::int32_t value = find4(fq::group_key::pack_key4(p[0], p[1], p[2], p[3]));
        if(alpha_positions == 2)
        {
            CHECK(value == fq::group_key::kEmptyValue);
            continue;
        }
        CHECK(fq::group_key::value_group(value) == int(g));
        CHECK(fq::group_key::value_row_int_length(value) == int(kRowIntLengths[g]));
    }
    CHECK(find2(fq::group_key::pack_key2(0, 2)) == fq::group_key::kEmptyValue);
    CHECK(find4(fq::group_key::pack_key4(0, 1, 2, 4)) == fq::group_key::kEmptyValue);
}

TEST_CASE("Occupancy bitsets identify empty slots")
{
    for(std::uint64_t slot = 0; slot <= kG2Mask; slot++)
    {
        CHECK(fq::group_key::is_empty2(kG2Bitset[fq::group_key::bitset_word2(slot)], slot) ==
              (kG2Keys[slot] == fq::group_key::kEmptyKey2));
    }
    for(std::uint64_t slot = 0; slot <= kG4Mask; slot++)
    {
        CHECK(fq::group_key::is_empty4(kG4Bitset[fq::group_key::bitset_word4(slot)], slot) ==
              (kG4Keys[slot] == fq::group_key::kEmptyKey4));
    }
    CHECK(fq::group_key::bitset_word2(31) == 0);
    CHECK(fq::group_key::bitset_word2(32) == 1);
    CHECK(fq::group_key::bitset_word4(63) == 0);
    CHECK(fq::group_key::bitset_word4(64) == 1);
}

TEST_CASE("Group hashes match host hashes")
{
    CHECK(fq::group_key::mix2(0u) == 0u);
    CHECK(fq::group_key::mix2(1u) == 0x31251ba7u);
    CHECK(fq::group_key::mix2(0x30001u) == 0x9ebf6d41u);
    CHECK(fq::group_key::mix2(0xdeadbeefu) == 0x5353e2e9u);
    CHECK(fq::group_key::mix2(0xffffffffu) == 0x2028884fu);
    CHECK(fq::group_key::mix4(0ull) == 0ull);
    CHECK(fq::group_key::mix4(1ull) == 0x5692161d100b05e5ull);
    CHECK(fq::group_key::mix4(0x3000200010000ull) == 0x9fade41eb0281ddull);
    CHECK(fq::group_key::mix4(0x123456789abcdefull) == 0xb2c058e4ebb5112cull);
    CHECK(fq::group_key::mix4(0xfffffffffffffffeull) == 0xda26e52fa3730902ull);
    CHECK(fq::group_key::next_slot(kG4Mask, kG4Mask) == 0);
    CHECK(fq::group_key::pack_key2(1, 3) == 0x30001u);
    CHECK(fq::group_key::pack_key4(0, 1, 2, 3) == 0x3000200010000ull);
    const std::int32_t last = fq::group_key::pack_group_value(std::uint32_t(fq::group_key::kMaxGroups - 1), 4u);
    CHECK(last != fq::group_key::kEmptyValue);
    CHECK(fq::group_key::value_group(last) == int(fq::group_key::kMaxGroups - 1));
    CHECK(fq::group_key::value_row_int_length(last) == 4);
}

TEST_CASE("Row-bit packing matches the device layout")
{
    for(std::uint32_t pattern = 0; pattern < 16; pattern++)
    {
        const std::uint32_t b[4] = {pattern & 1u, (pattern >> 1) & 1u, (pattern >> 2) & 1u, (pattern >> 3) & 1u};
        for(int length = 0; length <= fq::group_key::kMaxRowIntLength; length++)
        {
            std::uint32_t loop = 0;
            for(int k = 0; k < length; k++)
            {
                loop |= b[k] << k;
            }
            CHECK(fq::group_key::row_int(b[0], b[1], b[2], b[3], length) == loop);
            if(length >= 1)
            {
                std::uint32_t legacy = b[0];
                if(length > 1)
                {
                    legacy |= b[1] << 1u;
                }
                if(length > 2)
                {
                    legacy |= b[2] << 2u;
                }
                if(length > 3)
                {
                    legacy |= b[3] << 3u;
                }
                CHECK(fq::group_key::row_int(b[0], b[1], b[2], b[3], length) == legacy);
            }
        }
    }
    CHECK(fq::group_key::ladder_index(std::int64_t(1) << 28, 16, 15) == (std::int64_t(1) << 32) + 15);
}

TEST_CASE("Ladder bins contain all contributing terms")
{
    const std::vector<std::string> labels = {"I+I-IIII", "I-I+IIII", "Z+Z-IIII", "++--IIII", "+-+-IIII", "--++IIII",
                                             "-+-+IIII", "+--+IIII", "++--ZIII", "IIII+I-I", "IIII-I+I", "+++---II",
                                             "---+++II", "-+-+-+II", "II+-IIII", "II-+IIII"};
    QubitOperator op = QubitOperator::from_label(labels[0]);
    for(std::size_t i = 1; i < labels.size(); i++)
    {
        op += QubitOperator::from_label(labels[i]);
    }
    op.set_type(2);
    op.group_term_sort_by_ladder_int(4);
    const std::vector<std::size_t> groups = op.group_ptrs();
    const std::vector<std::vector<width_t>> positions = op.group_offdiag_indices();
    const std::vector<width_t> lengths = op.group_ladder_int_bit_lengths();
    const std::vector<std::size_t> bins = op.group_ladder_int_ptrs();
    const std::int64_t offset = std::int64_t(1) << op.ladder_width;
    int coupled = 0;
    for(std::size_t g = 0; g + 1 < groups.size(); g++)
    {
        std::uint64_t flips = 0;
        std::uint32_t at[4] = {0, 0, 0, 0};
        for(std::size_t k = 0; k < positions[g].size(); k++)
        {
            flips |= std::uint64_t(1) << positions[g][k];
            if(k < 4)
            {
                at[k] = positions[g][k];
            }
        }
        for(std::uint64_t row = 0; row < 256; row++)
        {
            std::uint32_t bit[4] = {0, 0, 0, 0};
            for(std::size_t k = 0; k < 4 && k < positions[g].size(); k++)
            {
                bit[k] = std::uint32_t((row >> at[k]) & 1u);
            }
            const std::uint32_t ri = fq::group_key::row_int(bit[0], bit[1], bit[2], bit[3], int(lengths[g]));
            const std::int64_t index = fq::group_key::ladder_index(std::int64_t(g), offset, ri);
            const std::size_t first = bins[std::size_t(index)];
            const std::size_t last = bins[std::size_t(index) + 1];
            CHECK(first >= groups[g]);
            CHECK(last <= groups[g + 1]);
            const std::uint64_t col = row ^ flips;
            for(std::size_t t = groups[g]; t < groups[g + 1]; t++)
            {
                int element = 1;
                for(std::size_t k = 0; k < op.terms[t].indices.size(); k++)
                {
                    const width_t p = op.terms[t].indices[k];
                    element *= fq::term_word::operator_element(
                        op.terms[t].values[k], std::uint32_t((row >> p) & 1u), std::uint32_t((col >> p) & 1u));
                }
                if(element != 0)
                {
                    coupled++;
                    CHECK(t >= first);
                    CHECK(t < last);
                }
            }
        }
    }
    CHECK(coupled > 0);
}

namespace
{

const std::uint64_t kCliqueAlpha[] = {0x7ull, 0xbull, 0xeull, 0x13ull, 0x15ull, 0x16ull, 0x19ull, 0x1aull, 0x23ull, 0x25ull, 0x26ull, 0x29ull, 0x2aull, 0x31ull, 0x32ull, 0x34ull, 0x38ull};
const std::int32_t kUnionParentPtr[] = {0, 3, 7, 11, 14, 18, 22, 25, 29, 33, 37, 41, 45};
const std::uint64_t kUnionMembers[] = {0x1000000000002ull, 0x1000200000001ull, 0xffff000300000000ull, 0x1000000000005ull, 0xffff000100000004ull, 0x1000200000003ull, 0xffff000400000000ull, 0x1000000000007ull, 0xffff000100000006ull, 0x1000300000003ull, 0xffff000400000001ull, 0xffff000200000007ull, 0x1000300000005ull, 0xffff000400000002ull, 0x100000000000aull, 0xffff000100000009ull, 0x1000200000008ull, 0xffff000500000000ull, 0x100000000000cull, 0xffff00010000000bull, 0x1000300000008ull, 0xffff000500000001ull, 0xffff00020000000cull, 0x100030000000aull, 0xffff000500000002ull, 0x100000000000eull, 0xffff00010000000dull, 0x1000400000008ull, 0xffff000500000003ull, 0x100000000000full, 0xffff00020000000dull, 0x1000400000009ull, 0xffff000500000004ull, 0x100010000000full, 0xffff00020000000eull, 0x100040000000aull, 0xffff000500000005ull, 0x1000000000010ull, 0xffff00030000000dull, 0x100040000000bull, 0xffff000500000006ull, 0x1000100000010ull, 0xffff00030000000eull, 0x100040000000cull, 0xffff000500000007ull};
const std::int32_t kIntersectionParentPtr[] = {0, 4, 7, 11, 14, 18, 22, 26, 29, 32, 36, 40, 43, 46, 50};
const std::uint64_t kIntersectionMembers[] = {0x1800200000000ull, 0x1800300000001ull, 0x1800400000003ull, 0x1800500000008ull, 0xffff800100000000ull, 0x1800400000004ull, 0x1800500000009ull, 0x1800000000000ull, 0x1800300000002ull, 0x1800400000005ull, 0x180050000000aull, 0xffff800100000001ull, 0x1800400000006ull, 0x180050000000bull, 0x1800000000001ull, 0xffff800200000002ull, 0x1800400000007ull, 0x180050000000cull, 0xffff800100000003ull, 0xffff800200000004ull, 0xffff800300000006ull, 0x180050000000dull, 0x1800000000003ull, 0xffff800200000005ull, 0xffff800300000007ull, 0x180050000000eull, 0x1800000000004ull, 0x1800100000005ull, 0x180050000000full, 0x1800000000006ull, 0x1800100000007ull, 0x1800500000010ull, 0xffff800100000008ull, 0xffff800200000009ull, 0xffff80030000000bull, 0xffff80040000000dull, 0x1800000000008ull, 0xffff80020000000aull, 0xffff80030000000cull, 0xffff80040000000eull, 0x1800000000009ull, 0x180010000000aull, 0xffff80040000000full, 0x180000000000bull, 0x180010000000cull, 0xffff800400000010ull, 0x180000000000dull, 0x180010000000eull, 0x180020000000full, 0x1800300000010ull};
const std::int32_t kTileInColumns[] = {26, 32, 0, 32, 18, 20, 25, 11, 39, 2, 11, 15, 22, 16, 5, 1, 0, 1, 5, 39, 7, 26, 30};
const std::int32_t kTileInPairs[] = {4, 5, 9, 5, 20, 3, 18, 16, 17, 2, 8, 13, 10, 13, 14, 13, 1, 20, 11, 18, 5, 7, 18};
const std::int32_t kTileInSigns[] = {-1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1};
const std::int32_t kTileInRows[] = {2, 5, 3, 3, 8, 1, 3, 1, 1, 3, 7, 6, 4, 1, 8, 8, 2, 0, 3, 1, 8, 5, 5};
const std::uint64_t kTileWords[] = {0x200000000ull, 0x300000002ull, 0x100000014ull, 0x800000020000001aull, 0x300000020ull, 0x8000000300000000ull, 0x8000000300000005ull, 0x8000000100000010ull, 0x800000010000000bull, 0x100000027ull, 0x8000000300000019ull, 0x100000027ull, 0x8000000000000001ull, 0x8000000100000020ull, 0x800000010000001aull, 0x30000000bull, 0x8000000000000016ull, 0x20000000full, 0x10000001eull, 0x8000000000000007ull, 0x1ull, 0x5ull, 0x8000000000000012ull};
const std::int32_t kTileGroupPtr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
const std::int32_t kTileGroupPair[] = {1, 2, 3, 4, 5, 9, 11, 13, 16, 17, 18, 20, 5, 7, 8, 10, 13, 18, 5, 13, 14, 20};
const std::int32_t kTileGroupStart[] = {0, 12, 18, 22};
const std::int32_t kBucketSizes[] = {2, 8, 9, 12, 13, 32, 31, 17, 5, 24, 25, 28};
const std::int32_t kBucketIds[] = {0, 1, 8, 2, 3, 4, 7, 9, 10, 11, 5, 6};
const std::int32_t kBucketStart[] = {0, 3, 5, 6, 7, 8, 10, 12};

int parity(const std::uint64_t value)
{
    return __builtin_popcountll(value) & 1;
}

void check_parents(const std::int32_t* parent_ptr, const std::size_t n_ptr, const std::uint64_t* members,
                   const bool intersection)
{
    for(std::size_t p = 0; p + 1 < n_ptr; p++)
    {
        std::uint64_t parent = 0;
        for(std::int32_t m = parent_ptr[p]; m < parent_ptr[p + 1]; m++)
        {
            const std::uint64_t word = members[m];
            const int alpha = fq::clique_word::member_alpha(word);
            const int orbital = fq::clique_word::member_orbital(word);
            const std::uint64_t bit = std::uint64_t(1) << orbital;
            const std::uint64_t string = kCliqueAlpha[alpha];
            CHECK(fq::clique_word::member_is_intersection(word) == intersection);
            CHECK(((string & bit) != 0) == intersection);
            const std::uint64_t this_parent = intersection ? (string & ~bit) : (string | bit);
            if(m == parent_ptr[p])
            {
                parent = this_parent;
            }
            CHECK(this_parent == parent);
            const int phase = fq::clique_word::member_phase(word);
            CHECK(phase == 1 - 2 * parity(parent & (bit - 1)));
            CHECK(((word & fq::clique_word::kSignBit) != 0) == (phase < 0));
            CHECK(fq::clique_word::pack_member(std::uint32_t(alpha), std::uint32_t(orbital), intersection, phase) == word);
            const bool output_negative = (phase < 0) != !intersection;
            CHECK((fq::clique_word::output_sign_bit(word) != 0) == output_negative);
        }
    }
}

}

TEST_CASE("Parent members decode with correct JW signs")
{
    check_parents(kUnionParentPtr, sizeof(kUnionParentPtr) / sizeof(kUnionParentPtr[0]), kUnionMembers, false);
    check_parents(kIntersectionParentPtr, sizeof(kIntersectionParentPtr) / sizeof(kIntersectionParentPtr[0]),
                  kIntersectionMembers, true);
}

TEST_CASE("Tile decoding preserves columns, rows, pairs and signs")
{
    const int tile_rows = 4;
    const std::size_t n = sizeof(kTileInColumns) / sizeof(kTileInColumns[0]);
    std::vector<std::vector<std::int64_t>> expected;
    for(std::size_t i = 0; i < n; i++)
    {
        expected.push_back({kTileInColumns[i], kTileInRows[i], kTileInSigns[i] < 0 ? 1 : 0, kTileInPairs[i]});
    }
    std::vector<std::vector<std::int64_t>> decoded;
    const std::size_t n_tiles = sizeof(kTileGroupStart) / sizeof(kTileGroupStart[0]) - 1;
    for(std::size_t tile = 0; tile < n_tiles; tile++)
    {
        for(std::int32_t group = kTileGroupStart[tile]; group < kTileGroupStart[tile + 1]; group++)
        {
            for(std::int32_t w = kTileGroupPtr[group]; w < kTileGroupPtr[group + 1]; w++)
            {
                const std::uint64_t word = kTileWords[w];
                const int row = fq::clique_word::tile_word_row(word);
                CHECK(row < tile_rows);
                const bool negative = (word & fq::clique_word::kSignBit) != 0;
                CHECK(fq::clique_word::pack_tile_word(std::uint32_t(fq::clique_word::tile_word_column(word)),
                                                      std::uint32_t(row), negative) == word);
                decoded.push_back({fq::clique_word::tile_word_column(word), std::int64_t(tile) * tile_rows + row,
                                   negative ? 1 : 0, kTileGroupPair[group]});
            }
        }
    }
    std::sort(expected.begin(), expected.end());
    std::sort(decoded.begin(), decoded.end());
    CHECK(decoded == expected);
}

TEST_CASE("Gather and output signs match phase rules")
{
    for(const bool intersection : {false, true})
    {
        for(const int phase : {-1, 1})
        {
            for(const bool negative : {false, true})
            {
                const std::uint64_t member = fq::clique_word::pack_member(3u, 5u, intersection, phase);
                const std::uint64_t word = fq::clique_word::pack_tile_word(7u, 2u, negative);
                CHECK((fq::clique_word::gather_sign_bit(member, word) != 0) == ((phase < 0) != negative));
                CHECK((fq::clique_word::output_sign_bit(member) != 0) == ((phase < 0) != !intersection));
            }
        }
    }
}

TEST_CASE("Clique buckets and shared-memory sizes match host rules")
{
    const std::size_t n_buckets = sizeof(kBucketStart) / sizeof(kBucketStart[0]) - 1;
    CHECK(n_buckets == std::size_t(fq::clique_word::kBuckets));
    CHECK(fq::clique_word::bucket_members(fq::clique_word::kBuckets - 1) == fq::clique_word::kMaxMembers);
    for(std::size_t b = 0; b < n_buckets; b++)
    {
        for(std::int32_t i = kBucketStart[b]; i < kBucketStart[b + 1]; i++)
        {
            const int size = kBucketSizes[kBucketIds[i]];
            CHECK(fq::clique_word::bucket_of(size) == int(b));
            CHECK(size <= fq::clique_word::bucket_members(int(b)));
        }
    }
    for(int members = 1; members <= fq::clique_word::kMaxMembers; members++)
    {
        const int b = fq::clique_word::bucket_of(members);
        CHECK(members <= fq::clique_word::bucket_members(b));
        CHECK((b == 0 || members > fq::clique_word::bucket_members(b - 1)));
    }
    CHECK(fq::clique_word::shared_bytes(32, 8, 219) == 232448);
    CHECK(fq::clique_word::shared_bytes(32, 8, 219) <= fq::clique_word::kMaxSharedBytes);
    CHECK(fq::clique_word::shared_bytes(32, 8, 220) > fq::clique_word::kMaxSharedBytes);
    CHECK(fq::clique_word::shared_bytes(8, 4, 64) == std::size_t(4) * (64 * 8 + 8 * 4) * 8);
}
