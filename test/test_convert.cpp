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
#include <complex>
#include <vector>
#include <filesystem>


typedef std::complex<double> complex;


TEST_CASE("Test basic properites of H2 from fcidump")
{
    std::filesystem::path cwd = std::filesystem::current_path();
    FermionicOperator_t fop = FermionicOperator::from_fcidump( cwd.parent_path() / "fulqrum/fulqrum/test/data/fcidump_h2.txt");
    CHECK(fop.width == 4);
    CHECK(fop.size() == 15);
}

TEST_CASE("Test basic properites of LiH from fcidump")
{
    std::filesystem::path cwd = std::filesystem::current_path();
    FermionicOperator_t fop = FermionicOperator::from_fcidump(cwd.parent_path() / "fulqrum/fulqrum/test/data/fcidump_lih.txt");
    CHECK(fop.width == 12);
    CHECK(fop.size() == 631);
}

TEST_CASE("Test basic properites of N2 from fcidump")
{
    std::filesystem::path cwd = std::filesystem::current_path();
    FermionicOperator_t fop = FermionicOperator::from_fcidump(cwd.parent_path() / "fulqrum/fulqrum/test/data/fcidump_n2.txt");
    CHECK(fop.width == 20);
    CHECK(fop.size() == 2239);
}

TEST_CASE("Test basic properites of N2 from fcidump")
{
    std::filesystem::path cwd = std::filesystem::current_path();
    FermionicOperator_t fop = FermionicOperator::from_fcidump(cwd.parent_path() / "fulqrum/fulqrum/test/data/fcidump_Fe4S4_MO.txt");
    CHECK(fop.width == 72);
    CHECK(fop.size() == 2476008);
}
