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

TEST_CASE("Test diag fast mode compatibility check")
{
    std::filesystem::path cwd = std::filesystem::current_path();
    FermionicOperator_t fop = FermionicOperator::from_json(cwd.parent_path() / "fulqrum/fulqrum/test/data/lih.json");
    QubitOperator_t op = fop.extended_jw_transformation();
    CHECK(!fast_diag_compatible(op));
    auto [diag, off] = op.split_diagonal();
    CHECK(!fast_diag_compatible(diag));
    diag = diag.remove_constant_terms();
    CHECK(fast_diag_compatible(diag));
}

TEST_CASE("Test diag fast mode term sorting")
{
    std::filesystem::path cwd = std::filesystem::current_path();
    FermionicOperator_t fop = FermionicOperator::from_json(cwd.parent_path() / "fulqrum/fulqrum/test/data/lih.json");
    QubitOperator_t op = fop.extended_jw_transformation();
    auto [diag, off] = op.split_diagonal();
    diag = diag.remove_constant_terms();
    fast_diag_term_sort(diag);
    std::size_t counter = 0;
    for(width_t kk = 0; kk < diag.width; kk++)
    {
        for(width_t ll = kk; ll < diag.width; ll++)
        {
            if(kk == ll)
            {
                CHECK(diag[counter].proj_indices.size() == 1);
                CHECK(diag[counter].proj_indices[0] == kk);
            }
            else
            {
                CHECK(diag[counter].proj_indices[0] == kk);
                CHECK(diag[counter].proj_indices[1] == ll);
            }
            counter += 1;
        }
    }
}
