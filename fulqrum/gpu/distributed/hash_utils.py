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

from ..common.hash_utils import (
    _EMPTY_KEY,
    _splitmix64,
    _splitmix64_vec,
)


def _pack_half_strs_to_uint64(alpha, beta):


    alpha_arr = np.array([int(s, 2) for s in alpha], dtype=np.uint64)
    beta_arr = np.array([int(s, 2) for s in beta], dtype=np.uint64)
    return alpha_arr, beta_arr


def _build_hash_table_simple(arr: np.ndarray):

    n = len(arr)
    cap = 1
    while cap < 3 * n:
        cap <<= 1
    mask = int(cap - 1)
    ht_keys = np.full(cap, _EMPTY_KEY, dtype=np.uint64)
    ht_vals = np.full(cap, np.int32(-1), dtype=np.int32)
    for i, key in enumerate(arr):
        slot = _splitmix64(int(key)) & mask
        while ht_keys[slot] != _EMPTY_KEY:
            slot = (slot + 1) & mask
        ht_keys[slot] = np.uint64(key)
        ht_vals[slot] = np.int32(i)
    return ht_keys, ht_vals, np.uint64(mask)


def _ht_lookup_batch(
    ht_keys: np.ndarray, ht_vals: np.ndarray, mask_int: int, keys: np.ndarray
) -> np.ndarray:


    mask = np.uint64(mask_int)
    slots = (_splitmix64_vec(keys) & mask).astype(np.intp)

    stored = ht_keys[slots]
    results = np.where(stored == keys, ht_vals[slots], np.int32(-1))

    collision_idx = np.where((stored != keys) & (stored != _EMPTY_KEY))[0]
    for ci in collision_idx:
        slot = int(slots[ci])
        key = int(keys[ci])
        while True:
            slot = (slot + 1) & mask_int
            s = int(ht_keys[slot])
            if s == (key & 0xFFFFFFFFFFFFFFFF):
                results[ci] = ht_vals[slot]
                break
            if s == int(_EMPTY_KEY):
                break

    return results
