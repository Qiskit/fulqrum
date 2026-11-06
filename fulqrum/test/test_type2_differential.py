# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import itertools

import numpy as np

import fulqrum as fq

NORB = 4
WIDTH = 2 * NORB

ELEMENTS = {
    "Z": [1, 0, 0, -1],
    "0": [1, 0, 0, 0],
    "1": [0, 0, 0, 1],
    "X": [0, 1, 1, 0],
    "Y": [0, -1j, 1j, 0],
    "-": [0, 1, 0, 0],
    "+": [0, 0, 1, 0],
}
FLIPPING = frozenset("XY-+")

ORDERINGS = {
    "block": (list(range(NORB)), list(range(NORB, WIDTH))),
    "interleaved": ([2 * i for i in range(NORB)], [2 * i + 1 for i in range(NORB)]),
}

OPERATOR_TYPES = (1, 2)


def make_operator(kind, alpha_pos, beta_pos):
    norb = len(alpha_pos)
    rng = np.random.RandomState(11)
    terms = []

    def add(ops, indices, coeff):
        terms.append((ops, list(indices), complex(coeff)))

    if kind in ("singles", "mixed", "aabb_uniform_singles"):
        for positions in (alpha_pos, beta_pos):
            for i in range(norb):
                for j in range(i, norb):
                    coeff = rng.uniform(-1.0, 1.0)
                    add("+-", [positions[i], positions[j]], coeff)
                    if i != j:
                        add("+-", [positions[j], positions[i]], coeff)

    if kind in ("aabb_uniform", "aabb_random", "mixed", "aabb_uniform_singles"):
        uniform = kind in ("aabb_uniform", "aabb_uniform_singles")
        for i, j, k, l in itertools.product(range(norb), repeat=4):
            coeff = 0.25 if uniform else rng.uniform(-0.4, 0.4)
            add("++--", [alpha_pos[i], beta_pos[k], beta_pos[l], alpha_pos[j]], coeff)
            add("++--", [alpha_pos[j], beta_pos[l], beta_pos[k], alpha_pos[i]], coeff)

    if kind in ("same_sector_doubles", "mixed"):
        for positions in (alpha_pos, beta_pos):
            for i, j in itertools.combinations(range(norb), 2):
                for k, l in itertools.combinations(range(norb), 2):
                    coeff = rng.uniform(-0.5, 0.5)
                    add(
                        "++--",
                        [positions[i], positions[j], positions[k], positions[l]],
                        coeff,
                    )
                    add(
                        "++--",
                        [positions[l], positions[k], positions[j], positions[i]],
                        coeff,
                    )

    if kind in ("complex", "complex_doubles"):
        for positions in (alpha_pos, beta_pos):
            for i, j in itertools.combinations(range(norb), 2):
                coeff = complex(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
                add("+-", [positions[i], positions[j]], coeff)
                add("+-", [positions[j], positions[i]], np.conj(coeff))

    if kind == "complex_doubles":
        for i, j, k, l in itertools.product(range(norb), repeat=4):
            coeff = complex(rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4))
            add("++--", [alpha_pos[i], beta_pos[k], beta_pos[l], alpha_pos[j]], coeff)
            add(
                "++--",
                [alpha_pos[j], beta_pos[l], beta_pos[k], alpha_pos[i]],
                np.conj(coeff),
            )
        for positions in (alpha_pos, beta_pos):
            for i, j in itertools.combinations(range(norb), 2):
                for k, l in itertools.combinations(range(norb), 2):
                    coeff = complex(rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5))
                    add(
                        "++--",
                        [positions[i], positions[j], positions[k], positions[l]],
                        coeff,
                    )
                    add(
                        "++--",
                        [positions[l], positions[k], positions[j], positions[i]],
                        np.conj(coeff),
                    )

    if kind in ("sector_transfer", "sector_transfer_mixed", "aabb_partial"):
        for i, j in itertools.combinations(range(norb), 2):
            for k, l in itertools.combinations(range(norb), 2):
                a_i, a_j = alpha_pos[i], alpha_pos[j]
                b_k, b_l = beta_pos[k], beta_pos[l]
                if kind != "sector_transfer":
                    add("+-+-", [a_i, a_j, b_k, b_l], 0.25)
                    add("+-+-", [a_j, a_i, b_l, b_k], 0.25)
                if kind == "sector_transfer_mixed":
                    add("+-+-", [a_i, a_j, b_l, b_k], 0.25)
                    add("+-+-", [a_j, a_i, b_k, b_l], 0.25)
                if kind != "aabb_partial":
                    add("++--", [a_i, a_j, b_k, b_l], 0.25)
                    add("++--", [b_l, b_k, a_j, a_i], 0.25)
        if kind == "sector_transfer":
            for positions in (alpha_pos, beta_pos):
                for i, j in itertools.combinations(range(norb), 2):
                    coeff = rng.uniform(-1.0, 1.0)
                    add("+-", [positions[i], positions[j]], coeff)
                    add("+-", [positions[j], positions[i]], coeff)

    assert terms, f"unknown operator kind {kind}"
    return fq.FermionicOperator(2 * norb, terms)


def make_spinless_operator(width):
    rng = np.random.RandomState(5)
    terms = []
    for i, j in itertools.combinations(range(width), 2):
        coeff = rng.uniform(-1.0, 1.0)
        terms.append(("+-", [i, j], coeff))
        terms.append(("+-", [j, i], coeff))
    for i, j, k, l in itertools.combinations(range(width), 4):
        coeff = rng.uniform(-0.5, 0.5)
        terms.append(("++--", [i, k, l, j], coeff))
        terms.append(("++--", [j, l, k, i], coeff))
    return fq.FermionicOperator(width, terms)


def bitstring(occupied, width):
    bits = ["0"] * width
    for p in occupied:
        bits[width - 1 - p] = "1"
    return "".join(bits)
