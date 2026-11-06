/**
 * This code is a part of Fulqrum.
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
 * Modified: descriptive GPU flag and clique names.
 */
#pragma once

#include <cstddef>

#include "formats/params.hpp"

extern "C"
{


    int fq_launch_diagonal(const FqTileGeometry* geometry,
                           const FqVectors* vectors,
                           const FqDiagonalTables* tables,
                           void* stream);


    int fq_launch_alpha_singles_pair(const FqTileGeometry* geometry,
                                     const FqVectors* vectors,
                                     const FqAlphaSinglePairTables* tables,
                                     void* stream);


    int fq_launch_alpha_singles_delta(const FqTileGeometry* geometry,
                                      const FqVectors* vectors,
                                      const FqAlphaSingleDeltaTables* tables,
                                      void* stream);


    int fq_launch_beta_singles_pair(const FqTileGeometry* geometry,
                                    const FqVectors* vectors,
                                    const FqBetaSinglePairTables* tables,
                                    void* stream);


    int fq_launch_beta_singles_delta(const FqTileGeometry* geometry,
                                     const FqVectors* vectors,
                                     const FqBetaSingleDeltaTables* tables,
                                     void* stream);


    int fq_launch_alpha_doubles(const FqTileGeometry* geometry,
                                const FqVectors* vectors,
                                const FqAlphaDoubleTables* tables,
                                void* stream);


    int fq_launch_beta_doubles(const FqTileGeometry* geometry,
                               const FqVectors* vectors,
                               const FqBetaDoubleTables* tables,
                               void* stream);


    int fq_launch_beta_doubles_class(const FqTileGeometry* geometry,
                                     const FqVectors* vectors,
                                     const FqBetaDoubleClassTables* tables,
                                     void* stream);


    int fq_launch_alpha_doubles_class(const FqTileGeometry* geometry,
                                      const FqVectors* vectors,
                                      const FqAlphaDoubleClassTables* tables,
                                      void* stream);


    int fq_launch_cross_sell(const FqTileGeometry* geometry,
                             const FqVectors* vectors,
                             const FqCrossSellTables* tables,
                             void* stream);


    int fq_launch_cross_direct(const FqTileGeometry* geometry,
                               const FqVectors* vectors,
                               const FqCrossDirectTables* tables,
                               void* stream);


    int fq_prepare_cross_clique(int max_shared_bytes);


    int fq_launch_cross_clique(const FqTileGeometry* geometry,
                               const FqVectors* vectors,
                               const FqCrossCliqueTables* tables,
                               void* stream);


    int fq_cross_clique_panel_row(int bucket);


    int fq_launch_cross_clique_gated(const FqTileGeometry* geometry,
                                     const FqVectors* vectors,
                                     const FqCrossCliqueTables* tables,
                                     void* stream);


    int fq_launch_cross_clique_pack(const FqTileGeometry* geometry,
                                    const FqVectors* vectors,
                                    const FqCrossCliqueTables* tables,
                                    void* stream);


    int fq_launch_lbg_finite(const double* source, long long n, std::int32_t* nonfinite, void* stream);


    int fq_launch_cross_lbg(const FqTileGeometry* geometry,
                            const FqVectors* vectors,
                            const FqCrossCliqueTables* tables,
                            const FqCrossLbgTables* lbg,
                            void* stream);


    int fq_launch_cross_rsl(const FqTileGeometry* geometry,
                            const FqVectors* vectors,
                            const FqCrossRslTables* rsl,
                            void* stream);


    int fq_launch_build_alpha_doubles(const FqTileGeometry* geometry,
                                      const FqTermTables* terms,
                                      const FqDoublesBuildTables* tables,
                                      void* stream);


    int fq_launch_build_beta_doubles(const FqTileGeometry* geometry,
                                     const FqTermTables* terms,
                                     const FqDoublesBuildTables* tables,
                                     void* stream);


    int fq_launch_build_alpha_singles(const FqTileGeometry* geometry,
                                      const FqTermTables* terms,
                                      const FqSinglesBuildTables* tables,
                                      void* stream);


    int fq_launch_build_beta_singles(const FqTileGeometry* geometry,
                                     const FqTermTables* terms,
                                     const FqSinglesBuildTables* tables,
                                     void* stream);


    const char* fq_error_name(int status);


    std::size_t fq_param_struct_size(int which);


    int fq_probe_fp64_mma(int iterations, float* mma_ms, float* scalar_ms, float* copy_ms, double* flops,
                          double* copy_bytes, int* exact);
}
