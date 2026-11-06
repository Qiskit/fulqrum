# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import numpy as np

from .hash_utils import _build_hash_table_simple
from .tile_discovery import _build_tile_flip_csr_hybrid, _empty_csr_table


def _build_axis_connection_tables(
    bra_vals: np.ndarray,
    src_vals: np.ndarray,
    masks_2: list,
    masks_4: list,
) -> dict:


    n_bra = len(bra_vals)
    if not masks_2 and not masks_4:
        empty = _empty_csr_table(n_bra)
        return {"csr2": empty, "csr4": empty}

    src_ht = _build_hash_table_simple(src_vals.astype(np.uint64))
    return {
        "csr2": _build_tile_flip_csr_hybrid(bra_vals, src_vals, masks_2, 2, src_ht),
        "csr4": _build_tile_flip_csr_hybrid(bra_vals, src_vals, masks_4, 4, src_ht),
    }
