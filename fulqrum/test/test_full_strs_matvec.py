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


import itertools
import lzma

import numpy as np
import pytest

import fulqrum as fq
from fulqrum.convert.integrals import fcidump_to_fq_fermionic_op
from fulqrum.gpu.full_strs.host import KINDS, HostOperator
from fulqrum.test.conftest import FE4S4_SAMPLES, fe4s4_fcidump
from fulqrum.test.test_full_strs_coefficients import FE4S4, random_operator



def labels_of(keys, norb):
    return [
        format(int(beta), f"0{norb}b") + format(int(alpha), f"0{norb}b")
        for alpha, beta in keys
    ]


def cpu_hamiltonian(op, keys, norb):
    labels = labels_of(keys, norb)
    hamiltonian = fq.SubspaceHamiltonian(op, fq.Subspace([list(labels)]))
    for index in np.linspace(0, len(labels) - 1, 7).astype(int):
        assert hamiltonian.get_n_th_bitstring(int(index)) == labels[index]
    return hamiltonian


def relative_difference(y, expected):
    return float(np.max(np.abs(y - expected)) / np.max(np.abs(expected)))


def clustered_keys(rng, norb, num_alpha, num_beta, rows, pool):

    def pool_strings(count):
        spread = min(norb, count + 4)
        out = []
        for _ in range(pool):
            out.append(
                sum(1 << int(k) for k in rng.choice(spread, count, replace=False))
            )
        return np.array(out, dtype=np.uint64)

    alpha = pool_strings(num_alpha)[rng.integers(0, pool, rows)]
    beta = pool_strings(num_beta)[rng.integers(0, pool, rows)]
    return np.stack([alpha, beta], axis=1)


@pytest.mark.parametrize(
    "norb, num_alpha, num_beta, seed", [(8, 4, 4, 1), (10, 7, 3, 2), (9, 2, 6, 3)]
)
def test_matvec_equals_the_grid_tier_csr(norb, num_alpha, num_beta, seed):
    rng = np.random.default_rng(seed)
    op = random_operator(norb, seed)
    keys = clustered_keys(rng, norb, num_alpha, num_beta, 1500, 40)
    reference_y = None
    for alpha_kind, beta_kind in itertools.product(KINDS, KINDS):
        host = HostOperator(op, keys, alpha_kind, beta_kind)
        assert host.kinds == (alpha_kind, beta_kind)
        if reference_y is None:
            hamiltonian = cpu_hamiltonian(op, host.sorted_keys(), norb)
            assert hamiltonian.spmv._cpu_diagnostics()["route"] == "retained"
            matrix = hamiltonian.to_csr_linearoperator()
            x = rng.normal(size=host.num_rows)
            reference_y = matrix @ x
        y = host.matvec(x)
        assert relative_difference(y, reference_y) <= 1e-12


@pytest.mark.parametrize("num_slices", [1, 3, 8])
def test_chunk_matvec_equals_the_reference(num_slices):
    rng = np.random.default_rng(11)
    op = random_operator(10, 12)
    keys = clustered_keys(rng, 10, 5, 4, 3000, 60)
    reference = HostOperator(op, keys)
    chunked = HostOperator(op, keys, num_slices=num_slices)
    assert chunked.num_chunks == num_slices
    x = rng.normal(size=reference.num_rows)
    y = chunked.chunk_matvec(x)
    assert relative_difference(y, reference.matvec(x)) <= 1e-14
    assert np.array_equal(y, chunked.chunk_matvec(x))
    with pytest.raises(RuntimeError):
        chunked.matvec(x)
    members = [
        chunked.chunk_views(chunk, cls)[0].size
        for chunk in range(num_slices)
        for cls in range(5)
    ]
    assert sum(members) > 0


def test_matvec_is_deterministic():
    rng = np.random.default_rng(5)
    op = random_operator(10, 6)
    host = HostOperator(op, clustered_keys(rng, 10, 5, 5, 3000, 60))
    x = rng.normal(size=host.num_rows)
    assert np.array_equal(host.matvec(x), host.matvec(x))


@pytest.mark.long
@pytest.mark.parametrize("sample", FE4S4_SAMPLES)
def test_matvec_equals_the_cpu_matvec_on_fe4s4(sample, tmp_path):
    op = fcidump_to_fq_fermionic_op(str(fe4s4_fcidump(tmp_path))).extended_jw_transformation()
    with lzma.open(FE4S4 / f"fe4s4_1000_{sample}.txt.xz", "rt") as handle:
        text = np.array(handle.read().split(), dtype="U72")
    beta = np.array([int(s[:36], 2) for s in text], dtype=np.uint64)
    alpha = np.array([int(s[36:], 2) for s in text], dtype=np.uint64)
    host = HostOperator(op, np.stack([alpha, beta], axis=1))
    hamiltonian = cpu_hamiltonian(op, host.sorted_keys(), 36)
    x = np.random.default_rng(7).normal(size=host.num_rows)
    assert relative_difference(host.matvec(x), hamiltonian.matvec(x)) <= 1e-12
