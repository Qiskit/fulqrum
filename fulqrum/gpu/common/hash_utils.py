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


_EMPTY_KEY = np.uint64(0xFFFFFFFFFFFFFFFF)
_SM64_M1 = int(0xBF58476D1CE4E5B9)
_SM64_M2 = int(0x94D049BB133111EB)

_SM64_M1_U64 = np.uint64(0xBF58476D1CE4E5B9)
_SM64_M2_U64 = np.uint64(0x94D049BB133111EB)


def _splitmix64(key: int) -> int:


    k = int(key)
    k = ((k ^ (k >> 30)) * _SM64_M1) & 0xFFFFFFFFFFFFFFFF
    k = ((k ^ (k >> 27)) * _SM64_M2) & 0xFFFFFFFFFFFFFFFF
    return (k ^ (k >> 31)) & 0xFFFFFFFFFFFFFFFF


def _splitmix64_vec(keys: np.ndarray) -> np.ndarray:

    k = keys.astype(np.uint64, copy=True)
    k ^= k >> np.uint64(30)
    k *= _SM64_M1_U64
    k ^= k >> np.uint64(27)
    k *= _SM64_M2_U64
    k ^= k >> np.uint64(31)
    return k


def _popcount_u64(v: np.ndarray, *, dtype=np.uint8) -> np.ndarray:


    x = v.astype(np.uint64, copy=True)
    x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(dtype)
