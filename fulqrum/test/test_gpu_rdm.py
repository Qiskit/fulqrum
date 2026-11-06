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


import gc
from itertools import combinations

import numpy as np
import pytest

from . import rdm_oracle
from .conftest import (
    FE4S4_CLUSTER_PREFIXES,
    fe4s4_alpha_strings,
    fe4s4_fcidump,
    gpu_available,
)

cupy = pytest.importorskip("cupy")
MPI = pytest.importorskip("mpi4py.MPI")

COMM = MPI.COMM_WORLD
MAX_RANKS = 4
TOLERANCE = 1e-12

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not gpu_available(), reason="needs the GPU build and a CUDA device"
    ),
    pytest.mark.skipif(
        COMM.Get_size() > MAX_RANKS, reason="at most 4 ranks on this box"
    ),
]


def grids_for(size):
    return [
        (p_alpha, size // p_alpha)
        for p_alpha in range(1, size + 1)
        if size % p_alpha == 0
    ]


def strings(m, ne):
    out = []
    for ones in combinations(range(m), ne):
        bits = ["0"] * m
        for i in ones:
            bits[m - 1 - i] = "1"
        out.append("".join(bits))
    return sorted(out)


def h_chain(atoms, spin, seed=0):
    payload = None
    if COMM.Get_rank() == 0:
        from pyscf import ao2mo, gto, scf  # noqa: PLC0415

        geometry = "; ".join(f"H 0 0 {0.9 * k}" for k in range(atoms))
        mol = gto.M(atom=geometry, basis="sto-3g", spin=spin, verbose=0)
        mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run()
        norb = mol.nao_nr()
        q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((norb, norb)))
        c = mf.mo_coeff @ q
        h1 = c.T @ mf.get_hcore() @ c
        eri = ao2mo.restore(1, ao2mo.kernel(mol, c), norb)
        payload = (h1, eri, mol.energy_nuc(), mol.nelec)
    return COMM.bcast(payload, root=0)


def chain_hamiltonian(atoms, spin):
    import fulqrum as fq  # noqa: PLC0415
    from fulqrum.convert import integrals_to_fq_fermionic_op  # noqa: PLC0415

    pytest.importorskip("pyscf")
    h1, eri, constant, nelec = h_chain(atoms, spin)
    op = integrals_to_fq_fermionic_op(
        np.ascontiguousarray(h1), np.ascontiguousarray(eri), constant=constant
    ).extended_jw_transformation()
    op.set_type(2)
    m = h1.shape[0]
    alpha, beta = strings(m, nelec[0]), strings(m, nelec[1])
    return (
        fq.SubspaceHamiltonian(op, fq.Subspace([alpha, beta])),
        alpha,
        beta,
        h1,
        eri,
        constant,
        nelec,
    )


@pytest.fixture(scope="module")
def fe4s4(tmp_path_factory):
    from pyscf import ao2mo, tools  # noqa: PLC0415

    from fulqrum.convert import fcidump_to_fq_fermionic_op  # noqa: PLC0415

    path = str(fe4s4_fcidump(tmp_path_factory.mktemp("fe4s4")))
    op = fcidump_to_fq_fermionic_op(path).extended_jw_transformation()
    half = fe4s4_alpha_strings()
    mf = tools.fcidump.to_scf(path)
    h1 = mf.get_hcore()
    return op, half, h1, ao2mo.restore(1, mf._eri, h1.shape[0]), mf.mol.energy_nuc()


def pick(half, rule, count):
    if rule == "clusters":
        prefix_a, prefix_b = FE4S4_CLUSTER_PREFIXES
        group_a = [x for x in half if x.startswith(prefix_a)]
        rest_a = {x[8:] for x in group_a}
        group_b = [x for x in half if x.startswith(prefix_b) and x[8:] not in rest_a]
        return sorted(group_a[: count // 2] + group_b[: count - count // 2])
    return sorted(half[:count])


@pytest.fixture(params=["relay", "pieces", "direct"])
def route(request, monkeypatch):
    from fulqrum.gpu.distributed import rdm_cart  # noqa: PLC0415

    monkeypatch.setattr(
        rdm_cart,
        "_ROUTE_FOR_TESTING",
        "direct" if request.param == "direct" else "relay",
    )
    if request.param == "pieces":
        monkeypatch.setattr(rdm_cart, "_PIECE_FOR_TESTING", 7)
    return request.param


@pytest.fixture(autouse=True)
def released_gpu_memory():
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    COMM.Barrier()
    yield


def distributed_rdms(hsub, alpha, beta, grid, **options):
    from fulqrum.gpu.distributed import _collective  # noqa: PLC0415
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )  # noqa: PLC0415

    _collective.reset_for_testing()
    op = TileGPUResidentHamiltonian(
        hsub,
        list(alpha),
        list(beta),
        comm=COMM,
        process_grid=grid,
        comm_backend="host_mpi",
    )
    state = np.random.RandomState(23).uniform(-1.0, 1.0, op._n_a * op._n_b)
    state /= np.linalg.norm(state)
    tile = np.ascontiguousarray(
        state.reshape(op._n_b, op._n_a)[op._b0 : op._b1, op._a0 : op._a1]
    ).ravel()
    rq = COMM.allreduce(float(tile @ np.asarray(op._matvec(tile.copy()))))
    rdms = op.compute_rdms(tile, **options)
    geometry = (
        np.asarray(op._alpha_uniq, np.uint64),
        np.asarray(op._beta_uniq, np.uint64),
        int(op._num_alpha),
    )
    del op
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()
    return rdms, state, geometry, rq


def oracle_rdms(state, geometry):
    alpha, beta, m = geometry
    a_index = np.tile(np.arange(alpha.size), beta.size)
    b_index = np.repeat(np.arange(beta.size), alpha.size)
    return rdm_oracle.rdm12s(alpha[a_index], beta[b_index], state, m)


def agreed(ok, message):
    failures = [text for text in COMM.allgather(None if ok else message) if text]
    assert not failures, "; ".join(failures)


def largest_error(gpu, ref):
    flat_gpu = [gpu[0][0], gpu[0][1], *gpu[1]]
    flat_ref = [ref[0][0], ref[0][1], *ref[1]]
    return max(
        float(np.abs(np.asarray(g) - np.asarray(r)).max())
        for g, r in zip(flat_gpu, flat_ref)
    )


def test_oracle_matches_pyscf():
    pytest.importorskip("pyscf")
    import scipy.sparse.linalg as spla  # noqa: PLC0415
    from pyscf import fci  # noqa: PLC0415

    hsub, _alpha, _beta, h1, eri, constant, nelec = chain_hamiltonian(5, 1)
    m = h1.shape[0]
    solver = fci.direct_spin1.FCI()
    _e, civec = solver.kernel(h1, eri, m, nelec)
    ref1, ref2 = solver.make_rdm12s(civec, m, nelec, reorder=True)
    _vals, vecs = spla.eigsh(hsub, k=1, which="SA", v0=np.ones(hsub.shape[0]))
    keys = [int(hsub.get_n_th_bitstring(i), 2) for i in range(hsub.shape[0])]
    mask = (1 << m) - 1
    alpha = np.array([k & mask for k in keys], np.uint64)
    beta = np.array([k >> m for k in keys], np.uint64)
    got = rdm_oracle.rdm12s(alpha, beta, vecs[:, 0] / np.linalg.norm(vecs[:, 0]), m)
    assert largest_error(got, (tuple(ref1), tuple(ref2))) < 1e-10


@pytest.mark.parametrize("atoms,spin", [(5, 1), (6, 0)])
def test_chain_rdms_match_the_oracle(atoms, spin, route):
    hsub, alpha, beta, h1, eri, constant, nelec = chain_hamiltonian(atoms, spin)
    for grid in grids_for(COMM.Get_size()):
        rdms, state, geometry, rq = distributed_rdms(hsub, alpha, beta, grid, rank=2)
        ok, message = True, ""
        if COMM.Get_rank() == 0:
            error = largest_error(rdms, oracle_rdms(state, geometry))
            energy = rdm_oracle.rdm_energy(h1, eri, rdms[0], rdms[1], constant) - rq
            ok = error < TOLERANCE and abs(energy) < 1e-10
            message = (
                f"grid {grid}: element error {error:.2e}, energy error {energy:.2e}"
            )
        agreed(ok, message)


@pytest.mark.parametrize("rule", ["first", "clusters"])
def test_fe4s4_rdms_match_the_oracle(fe4s4, rule, route):
    import fulqrum as fq  # noqa: PLC0415

    op, half, h1, eri, constant = fe4s4
    chosen = pick(half, rule, 40)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([list(chosen), list(chosen)]))
    ne = chosen[0].count("1")
    for grid in grids_for(COMM.Get_size()):
        rdms, state, geometry, rq = distributed_rdms(hsub, chosen, chosen, grid, rank=2)
        ok, message = True, ""
        if COMM.Get_rank() == 0:
            error = largest_error(rdms, oracle_rdms(state, geometry))
            energy = rdm_oracle.rdm_energy(h1, eri, rdms[0], rdms[1], constant) - rq
            worst = max(rdm_oracle.invariants(rdms[0], rdms[1], ne, ne).values())
            ok = error < TOLERANCE and abs(energy) < 1e-9 and worst < 1e-11
            message = f"grid {grid}: element error {error:.2e}, energy error {energy:.2e}, invariants {worst:.2e}"
        agreed(ok, message)


def test_rank_one_and_spin_summed_forms_agree_with_the_blocks():
    hsub, alpha, beta, *_ = chain_hamiltonian(5, 1)
    grid = grids_for(COMM.Get_size())[0]
    blocks, *_ = distributed_rdms(hsub, alpha, beta, grid, rank=2)
    one, *_ = distributed_rdms(hsub, alpha, beta, grid, rank=1)
    one_summed, *_ = distributed_rdms(hsub, alpha, beta, grid, rank=1, spin_summed=True)
    gamma, g2 = distributed_rdms(hsub, alpha, beta, grid, rank=2, spin_summed=True)[0]
    (ga, gb), (aa, ab, bb) = blocks
    errors = [
        np.abs(one[0] - ga).max(),
        np.abs(one[1] - gb).max(),
        np.abs(one_summed - (ga + gb)).max(),
        np.abs(gamma - (ga + gb)).max(),
        np.abs(g2 - (aa + bb + ab + ab.transpose(2, 3, 0, 1))).max(),
    ]
    agreed(max(errors) < TOLERANCE, f"form errors {errors}")


def fake_operator(num_alpha, num_beta, alpha, beta):
    from types import SimpleNamespace  # noqa: PLC0415

    return SimpleNamespace(
        _num_alpha=num_alpha,
        _num_beta=num_beta,
        _alpha_uniq=np.asarray(alpha, np.uint64),
        _beta_uniq=np.asarray(beta, np.uint64),
    )


@pytest.mark.parametrize(
    "ham,rank,message",
    [
        (fake_operator(4, 4, [3, 5], [3, 5]), 3, "rank must be 1 or 2"),
        (fake_operator(4, 5, [3, 5], [3, 5]), 2, "equal alpha and beta orbital counts"),
        (fake_operator(4, 4, [3, 7], [3, 5]), 2, "one electron count"),
    ],
)
def test_bad_arguments_and_subspaces_raise(ham, rank, message):
    from fulqrum.gpu.distributed.rdm_cart import compute_rdms_cart  # noqa: PLC0415

    with pytest.raises(ValueError, match=message):
        compute_rdms_cart(ham, np.zeros(1), rank=rank)


def raised_on_every_rank(monkeypatch, patch, expected):
    from fulqrum.gpu.distributed import rdm_cart  # noqa: PLC0415

    hsub, alpha, beta, *_ = chain_hamiltonian(5, 1)
    if COMM.Get_rank() == COMM.Get_size() - 1:
        patch(monkeypatch, rdm_cart)
    with pytest.raises(expected) as caught:
        distributed_rdms(hsub, alpha, beta, grids_for(COMM.Get_size())[-1], rank=2)
    return str(caught.value)


def test_an_error_on_one_rank_raises_on_every_rank(monkeypatch):

    def patch(mp, module):

        def fail(*_args, **_kwargs):
            raise MemoryError("forced allocation failure")

        mp.setattr(module, "_pair_graph", fail)

    text = raised_on_every_rank(monkeypatch, patch, MemoryError)
    agreed(f"rank {COMM.Get_size() - 1}: forced allocation failure" in text, text)


def test_a_memory_refusal_on_one_rank_raises_on_every_rank(monkeypatch):

    def patch(mp, module):
        mp.setattr(module, "FIT_SHARE", 0.0)

    text = raised_on_every_rank(monkeypatch, patch, MemoryError)
    agreed(
        "RDM memory check failed" in text and f"rank {COMM.Get_size() - 1}:" in text,
        text,
    )


IN_PROCESS_GRIDS = [(2, 2), (1, 4), (4, 1), (3, 2)]


def in_process_rdms(hsub, alpha, beta, grid, patch=None):
    from fulqrum.gpu.distributed import _collective  # noqa: PLC0415
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )  # noqa: PLC0415
    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415

    _collective.reset_for_testing()
    n_a, n_b = len(alpha), len(beta)
    state = np.random.RandomState(23).uniform(-1.0, 1.0, n_a * n_b)
    state /= np.linalg.norm(state)

    def rank(comm):
        cupy.cuda.Device(0).use()
        op = TileGPUResidentHamiltonian(
            hsub,
            list(alpha),
            list(beta),
            gpu_ids=(0,),
            comm=comm,
            process_grid=grid,
            comm_backend="inproc",
            build_threads=1,
        )
        tile = np.ascontiguousarray(
            state.reshape(n_b, n_a)[op._b0 : op._b1, op._a0 : op._a1]
        ).ravel()
        return op.compute_rdms(tile, rank=2)

    results = run_ranks(LocalComm.create(grid[0] * grid[1]), rank)
    for other in results[1:]:
        assert largest_error(other, results[0]) == 0.0
    return results[0], state


def string_keys(strings_list):
    return np.array(sorted(int(s, 2) for s in strings_list), dtype=np.uint64)


@pytest.mark.skipif(
    COMM.Get_size() > 1, reason="in-process ranks run in the one-process suite"
)
@pytest.mark.parametrize("grid", IN_PROCESS_GRIDS)
def test_in_process_rdms_match_the_oracle(grid, route):
    hsub, alpha, beta, *_ = chain_hamiltonian(5, 1)
    rdms, state = in_process_rdms(hsub, alpha, beta, grid)
    geometry = (string_keys(alpha), string_keys(beta), len(alpha[0]))
    assert largest_error(rdms, oracle_rdms(state, geometry)) < TOLERANCE


@pytest.mark.skipif(
    COMM.Get_size() > 1, reason="in-process ranks run in the one-process suite"
)
@pytest.mark.parametrize("grid", [(2, 2), (3, 2)])
def test_in_process_fe4s4_rdms_match_the_oracle(fe4s4, grid, route):
    import fulqrum as fq  # noqa: PLC0415

    op, half, *_ = fe4s4
    chosen = pick(half, "clusters", 40)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([list(chosen), list(chosen)]))
    rdms, state = in_process_rdms(hsub, chosen, chosen, grid)
    geometry = (string_keys(chosen), string_keys(chosen), len(chosen[0]))
    assert largest_error(rdms, oracle_rdms(state, geometry)) < TOLERANCE


@pytest.mark.skipif(
    COMM.Get_size() > 1, reason="in-process ranks run in the one-process suite"
)
def test_in_process_error_on_one_rank_raises_on_every_rank(monkeypatch):
    from fulqrum.gpu.distributed import rdm_cart  # noqa: PLC0415

    original = rdm_cart._needed_tiles

    def needed(ham, me, threads):
        if me == 1:
            raise MemoryError("forced failure on rank 1")
        return original(ham, me, threads)

    monkeypatch.setattr(rdm_cart, "_needed_tiles", needed)
    hsub, alpha, beta, *_ = chain_hamiltonian(5, 1)
    with pytest.raises(MemoryError, match="rank 1: forced failure on rank 1"):
        in_process_rdms(hsub, alpha, beta, (2, 2))


@pytest.mark.skipif(
    COMM.Get_size() > 1, reason="in-process ranks run in the one-process suite"
)
@pytest.mark.parametrize("gpu_ids", [(0,), (0, 0), (0, 0, 0, 0)])
def test_gpu_hamiltonian_rdms_match_the_oracle(gpu_ids):
    from fulqrum.gpu import GPUHamiltonian  # noqa: PLC0415

    hsub, alpha, beta, *_ = chain_hamiltonian(5, 1)
    op = GPUHamiltonian(hsub, gpu_ids=gpu_ids)
    state = np.random.RandomState(23).uniform(-1.0, 1.0, op.shape[0])
    state /= np.linalg.norm(state)
    rdms = op.compute_rdms(state, rank=2)
    geometry = (string_keys(alpha), string_keys(beta), len(alpha[0]))
    assert largest_error(rdms, oracle_rdms(state, geometry)) < TOLERANCE
    gamma = op.compute_rdms(cupy.asarray(state), rank=1, spin_summed=True)
    assert np.abs(gamma - (rdms[0][0] + rdms[0][1])).max() < TOLERANCE
    with pytest.raises(ValueError, match="values"):
        op.compute_rdms(state[:-1])
    with pytest.raises(ValueError, match="rank must be 1 or 2"):
        op.compute_rdms(state, rank=3)
    assert largest_error(op.compute_rdms(state, rank=2), rdms) < TOLERANCE


def product_keys(alpha_strings, beta_strings):
    alpha = string_keys(alpha_strings)
    beta = string_keys(beta_strings)
    return np.stack([np.tile(alpha, beta.size), np.repeat(beta, alpha.size)], axis=1)


def keyed_state(keys):
    order = np.lexsort((keys[:, 0], keys[:, 1]))
    values = np.random.RandomState(29).uniform(-1.0, 1.0, keys.shape[0])
    values /= np.linalg.norm(values)
    return {(int(keys[i, 0]), int(keys[i, 1])): values[j] for j, i in enumerate(order)}


def full_strs_rdms(operator, keys, ranks=1, **options):
    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415

    amplitude = keyed_state(keys)
    m = operator.width // 2
    oracle = rdm_oracle.rdm12s(
        keys[:, 0],
        keys[:, 1],
        np.array([amplitude[(int(a), int(b))] for a, b in keys]),
        m,
    )

    def state_of(h):
        return np.array(
            [amplitude[(int(a), int(b))] for a, b in h.keys], dtype=np.float64
        )

    if ranks == 1:
        h = FullStrsHamiltonian(operator, keys, num_slices=3, **options)
        return h.compute_rdms(state_of(h), rank=2), oracle
    parts = np.array_split(
        keys[np.random.default_rng(ranks).permutation(keys.shape[0])], ranks
    )

    def body(comm):
        cupy.cuda.Device(0).use()
        h = FullStrsHamiltonian(
            operator, parts[comm.Get_rank()], comm=comm, num_slices=3, **options
        )
        return h.compute_rdms(state_of(h), rank=2)

    results = run_ranks(LocalComm.create(ranks, timeout=600.0), body)
    for other in results[1:]:
        assert largest_error(other, results[0]) == 0.0
    return results[0], oracle


def chain_qubit_operator(atoms, spin):
    from fulqrum.convert import integrals_to_fq_fermionic_op  # noqa: PLC0415

    pytest.importorskip("pyscf")
    h1, eri, constant, nelec = h_chain(atoms, spin)
    op = integrals_to_fq_fermionic_op(
        np.ascontiguousarray(h1), np.ascontiguousarray(eri), constant=constant
    ).extended_jw_transformation()
    op.set_type(2)
    m = h1.shape[0]
    return op, strings(m, nelec[0]), strings(m, nelec[1])


@pytest.mark.skipif(
    COMM.Get_size() > 1, reason="full-string RDM tests run in the one-process suite"
)
def test_full_strs_rdms_match_the_oracle_on_complete_h5():
    op, alpha, beta = chain_qubit_operator(5, 1)
    rdms, oracle = full_strs_rdms(op, product_keys(alpha, beta))
    assert largest_error(rdms, oracle) < TOLERANCE


@pytest.mark.skipif(
    COMM.Get_size() > 1, reason="full-string RDM tests run in the one-process suite"
)
@pytest.mark.parametrize("residency", ["auto", "streamed"])
def test_full_strs_rdms_match_the_oracle_on_a_fe4s4_subset(fe4s4, residency):
    op, half, *_ = fe4s4
    keys = product_keys(half[:40], half[:40])
    keys = keys[
        np.sort(
            np.random.default_rng(3).choice(
                keys.shape[0], keys.shape[0] // 2, replace=False
            )
        )
    ]
    rdms, oracle = full_strs_rdms(op, keys, residency=residency)
    assert largest_error(rdms, oracle) < TOLERANCE


FULL_STRS_MODES = [
    {},
    {"copies": 2},
    {"groups": "by_key"},
    {"groups": "by_key", "vectors": "full"},
    {"residency": "streamed"},
]


@pytest.mark.skipif(
    COMM.Get_size() > 1, reason="full-string RDM tests run in the one-process suite"
)
@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize(
    "mode", FULL_STRS_MODES, ids=["by_rows", "copies2", "by_key", "full", "streamed"]
)
def test_full_strs_rdms_on_in_process_ranks_match_the_oracle(ranks, mode):
    op, alpha, beta = chain_qubit_operator(5, 1)
    rdms, oracle = full_strs_rdms(op, product_keys(alpha, beta), ranks=ranks, **mode)
    assert largest_error(rdms, oracle) < TOLERANCE


@pytest.mark.skipif(
    COMM.Get_size() > 1, reason="full-string RDM tests run in the one-process suite"
)
def test_full_strs_half_visits_equal_both_visits(fe4s4, monkeypatch):
    from fulqrum.gpu.full_strs import rdm as fs_rdm  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415

    op, half_strings, *_ = fe4s4
    keys = product_keys(half_strings[:40], half_strings[:40])
    keys = keys[
        np.sort(
            np.random.default_rng(5).choice(
                keys.shape[0], keys.shape[0] // 2, replace=False
            )
        )
    ]
    h = FullStrsHamiltonian(op, keys, num_slices=3)
    amplitude = keyed_state(keys)
    x = np.array([amplitude[(int(a), int(b))] for a, b in h.keys], dtype=np.float64)
    assert fs_rdm._half_visits(h) == 2
    half = h.compute_rdms(x, rank=2)
    monkeypatch.setattr(fs_rdm, "_half_visits", lambda ham: 1)
    by_keys = h.compute_rdms(x, rank=2)
    monkeypatch.setattr(fs_rdm, "_half_visits", lambda ham: 0)
    both = h.compute_rdms(x, rank=2)
    assert largest_error(half, both) < TOLERANCE
    assert largest_error(by_keys, both) < TOLERANCE


@pytest.mark.skipif(
    COMM.Get_size() > 1, reason="full-string RDM tests run in the one-process suite"
)
@pytest.mark.parametrize(
    "mode, half",
    [
        ({}, 0),
        ({"copies": 2}, 0),
        ({"groups": "by_key"}, 1),
        ({"groups": "by_key", "vectors": "full"}, 2),
    ],
    ids=["by_rows", "copies2", "by_key", "full"],
)
def test_full_strs_half_visits_only_where_outputs_are_sources(mode, half):
    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks  # noqa: PLC0415
    from fulqrum.gpu.full_strs import rdm as fs_rdm  # noqa: PLC0415
    from fulqrum.gpu.full_strs.hamiltonian import FullStrsHamiltonian  # noqa: PLC0415

    op, alpha, beta = chain_qubit_operator(5, 1)
    keys = product_keys(alpha, beta)
    parts = np.array_split(keys, 2)

    def body(comm):
        cupy.cuda.Device(0).use()
        return fs_rdm._half_visits(
            FullStrsHamiltonian(
                op, parts[comm.Get_rank()], comm=comm, num_slices=3, **mode
            )
        )

    assert run_ranks(LocalComm.create(2, timeout=600.0), body) == [half, half]


@pytest.mark.parametrize("grid", [(1, 1), (1, 4), (4, 1), (2, 3), (4, 8), (8, 8)])
def test_relay_walk_visits_every_tile_once(grid):
    from fulqrum.gpu.distributed import rdm_cart  # noqa: PLC0415

    p_alpha, p_beta = grid
    moves = rdm_cart._relay_moves(p_alpha, p_beta)
    offsets = [(da % p_alpha, db % p_beta) for _step, (da, db) in moves]
    assert len(moves) == p_alpha * p_beta - 1
    assert len(set(offsets)) == len(offsets) and (0, 0) not in offsets
    assert all(abs(step[0]) + abs(step[1]) == 1 for step, _offset in moves)


def test_relay_route_model_and_lent_buffers():
    from fulqrum.gpu.distributed import rdm_cart  # noqa: PLC0415

    class One:

        def Get_rank(self):
            return 5

        def allgather(self, value):
            return [value]

    p_alpha, p_beta = 4, 8
    big = 120_000
    tiles = [(0, big, 0, big // 4, a, b) for a in range(p_alpha) for b in range(p_beta)]
    dense = np.ones(p_alpha * p_beta, np.int32)
    sparse = np.zeros_like(dense)
    sparse[[5, 6, 13]] = 1
    assert rdm_cart._use_relay(One(), tiles, dense, p_alpha, p_beta)
    assert not rdm_cart._use_relay(One(), tiles, sparse, p_alpha, p_beta)
    assert rdm_cart._lent_elems([10, 10, 8, 8], [9, 12, 10]) == 10 + 10 + 8
    assert rdm_cart._lent_elems([10], []) == 0


def test_an_rdm_over_the_eigensolver_peak_raises_on_every_rank(monkeypatch):
    from fulqrum.gpu.distributed import rdm_cart  # noqa: PLC0415

    monkeypatch.setattr(rdm_cart, "BUDGET_FLOOR", 0)
    monkeypatch.setattr(rdm_cart, "BUDGET_VECTORS", 0)  # no plan fits a zero budget
    hsub, alpha, beta, *_ = chain_hamiltonian(5, 1)
    grid = grids_for(COMM.Get_size())[0]
    with pytest.raises(MemoryError, match="RDM memory budget"):
        distributed_rdms(hsub, alpha, beta, grid, rank=2)
