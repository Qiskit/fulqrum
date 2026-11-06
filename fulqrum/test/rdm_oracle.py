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

"""Independent CPU oracle for spin-resolved 1- and 2-RDMs on any determinant list.

Conventions (PySCF ``make_rdm12s``, ``reorder=True``):
    dm1s[p, q]        = <a+_p a_q>
    dm2aa[p, q, r, s] = <a+_p a+_r a_s a_q>   (both alpha; bb the same)
    dm2ab[p, q, r, s] = <a+_p(alpha) a+_r(beta) a_s(beta) a_q(alpha)>
"""

import numpy as np

ONE = np.uint64(1)


def occupied(keys, m):
    """Return the occupied and empty orbitals of each key, ascending."""
    counts = np.bitwise_count(keys)
    if keys.size and counts.min() != counts.max():
        raise ValueError("oracle needs one electron count per sector")
    ne = int(counts[0]) if keys.size else 0
    bits = ((keys[:, None] >> np.arange(m, dtype=np.uint64)[None, :]) & ONE).astype(
        bool
    )
    occ = np.nonzero(bits)[1].reshape(keys.size, ne)
    vir = np.nonzero(~bits)[1].reshape(keys.size, m - ne)
    return occ.astype(np.int64), vir.astype(np.int64)


def ladder(keys, orb, create):
    """Apply one creation or annihilation operator to each key."""
    bit = ONE << orb.astype(np.uint64)
    full = (keys & bit) != 0
    ok = ~full if create else full
    sign = np.where(np.bitwise_count(keys & (bit - ONE)) & 1, -1.0, 1.0)
    return keys ^ bit, sign, ok


class DetIndex:
    """Lookup of (alpha key, beta key) pairs in a determinant list."""

    def __init__(self, alpha, beta):
        """Index the determinant list given by its alpha and beta keys (uint64 arrays)."""
        self.ua = np.unique(alpha)
        self.ub = np.unique(beta)
        ids = self._ids(alpha, beta)[0]
        self.order = np.argsort(ids, kind="stable")
        self.sorted = ids[self.order]
        if np.any(np.diff(self.sorted) == 0):
            raise ValueError("determinant list has duplicates")

    def _ids(self, alpha, beta):
        """Return the combined id and the found mask of each (alpha, beta) pair."""
        ia = np.clip(np.searchsorted(self.ua, alpha), 0, self.ua.size - 1)
        ib = np.clip(np.searchsorted(self.ub, beta), 0, self.ub.size - 1)
        found = (self.ua[ia] == alpha) & (self.ub[ib] == beta)
        return ib.astype(np.int64) * self.ua.size + ia, found

    def find(self, alpha, beta):
        """Return the position of each pair in the list and the found mask."""
        ids, found = self._ids(alpha, beta)
        pos = np.clip(np.searchsorted(self.sorted, ids), 0, self.sorted.size - 1)
        found &= self.sorted[pos] == ids
        return self.order[pos], found


def _moves(keys, m, rank):
    """Return the one- or two-orbital moves of every key in one sector."""
    n = keys.size
    occ, vir = occupied(keys, m)
    ne = occ.shape[1]
    out_k, out_s, out_l = [], [], []
    if rank == 1:
        for k in range(ne):
            q = occ[:, k]
            k1, s1, ok1 = ladder(keys, q, False)
            for p in list(vir.T) + [q]:
                k2, s2, ok2 = ladder(k1, p, True)
                out_k.append(np.where(ok1 & ok2, k2, 0))
                out_s.append(np.where(ok1 & ok2, s1 * s2, 0.0))
                out_l.append(np.stack([p, q], axis=1).astype(np.int16))
    else:
        for k_q in range(ne):
            for k_s in range(k_q + 1, ne):
                q, s = occ[:, k_q], occ[:, k_s]
                k1, s1, _ = ladder(keys, q, False)
                k2, s2, _ = ladder(k1, s, False)
                avail = np.concatenate([vir, q[:, None], s[:, None]], axis=1)
                na = avail.shape[1]
                for x in range(na):
                    for y in range(x + 1, na):
                        r, p = avail[:, y], avail[:, x]
                        k3, s3, ok3 = ladder(k2, r, True)
                        k4, s4, ok4 = ladder(k3, p, True)
                        ok = ok3 & ok4
                        out_k.append(np.where(ok, k4, 0))
                        out_s.append(np.where(ok, s1 * s2 * s3 * s4, 0.0))
                        out_l.append(np.stack([p, q, r, s], axis=1).astype(np.int16))
    return (
        np.stack(out_k, axis=1),
        np.stack(out_s, axis=1),
        np.stack(out_l, axis=1).reshape(n, len(out_k), -1),
    )


def rdm12s(alpha, beta, coeff, m, rank=2):
    """Return the spin-resolved RDMs of the state sum_i coeff[i] |alpha[i], beta[i]>.

    Args:
        alpha: uint64 alpha key of each determinant.
        beta: uint64 beta key of each determinant.
        coeff: float64 amplitude of each determinant (not normalized here).
        m: orbitals per sector.
        rank: 1 returns (dm1a, dm1b); 2 returns ((dm1a, dm1b), (dm2aa, dm2ab, dm2bb)).

    Returns:
        The RDMs in the PySCF ``make_rdm12s`` (reorder=True) convention.
    """
    alpha = np.asarray(alpha, np.uint64)
    beta = np.asarray(beta, np.uint64)
    coeff = np.asarray(coeff, np.float64)
    index = DetIndex(alpha, beta)
    m2 = m * m
    dm1 = []
    for sector in (0, 1):
        own, other = (alpha, beta) if sector == 0 else (beta, alpha)
        keys, sign, lab = _moves(own, m, 1)
        acc = np.zeros(m2)
        for col in range(keys.shape[1]):
            a, b = (keys[:, col], other) if sector == 0 else (other, keys[:, col])
            j, found = index.find(a, b)
            w = np.where(found, coeff[j] * coeff * sign[:, col], 0.0)
            np.add.at(acc, lab[:, col, 0].astype(np.int64) * m + lab[:, col, 1], w)
        dm1.append(acc.reshape(m, m))
    if rank == 1:
        return tuple(dm1)
    dm2 = []
    for sector in (0, 1):
        own, other = (alpha, beta) if sector == 0 else (beta, alpha)
        keys, sign, lab = _moves(own, m, 2)
        acc = np.zeros(m2 * m2)
        for col in range(keys.shape[1]):
            a, b = (keys[:, col], other) if sector == 0 else (other, keys[:, col])
            j, found = index.find(a, b)
            w = np.where(found, coeff[j] * coeff * sign[:, col], 0.0)
            p, q, r, s = (lab[:, col, t].astype(np.int64) for t in range(4))
            for (i0, i1, i2, i3), f in (
                ((p, q, r, s), 1.0),
                ((r, q, p, s), -1.0),
                ((p, s, r, q), -1.0),
                ((r, s, p, q), 1.0),
            ):
                np.add.at(acc, ((i0 * m + i1) * m + i2) * m + i3, f * w)
        dm2.append(acc.reshape(m, m, m, m))
    ka, sa, la = _moves(alpha, m, 1)
    kb, sb, lb = _moves(beta, m, 1)
    acc = np.zeros(m2 * m2)
    for col in range(ka.shape[1]):
        new_a = np.repeat(ka[:, col], kb.shape[1])
        j, found = index.find(new_a, kb.reshape(-1))
        cj = np.where(found, coeff[j], 0.0).reshape(kb.shape)
        w = cj * (coeff * sa[:, col])[:, None] * sb
        la64, lb64 = la[:, col].astype(np.int64), lb.astype(np.int64)
        flat = (
            (la64[:, 0, None] * m + la64[:, 1, None]) * m + lb64[:, :, 0]
        ) * m + lb64[:, :, 1]
        np.add.at(acc, flat.reshape(-1), w.reshape(-1))
    dm2ab = acc.reshape(m, m, m, m)
    return tuple(dm1), (dm2[0], dm2ab, dm2[1])


def rdm_energy(h1, eri, dm1s, dm2s, constant=0.0):
    """Return the energy E = const + h.(dm1a + dm1b) + 1/2 (pq|rs).(dm2aa + dm2bb + 2 dm2ab)."""
    dm1 = dm1s[0] + dm1s[1]
    dm2 = dm2s[0] + dm2s[2] + dm2s[1] + dm2s[1].transpose(2, 3, 0, 1)
    return (
        constant + np.einsum("pq,pq", h1, dm1) + 0.5 * np.einsum("pqrs,pqrs", eri, dm2)
    )


def invariants(dm1s, dm2s, n_alpha, n_beta):
    """Return the largest violation of the trace, partial-trace and symmetry identities.

    Takes the RDMs of a normalized state and the electron counts. Returns a dict of maxima.
    """
    dm1a, dm1b = dm1s
    aa, ab, bb = dm2s
    out = {
        "tr_a": abs(np.trace(dm1a) - n_alpha),
        "tr_b": abs(np.trace(dm1b) - n_beta),
        "ptr_aa": np.abs(np.einsum("pqrr->pq", aa) - (n_alpha - 1) * dm1a).max(),
        "ptr_bb": np.abs(np.einsum("pqrr->pq", bb) - (n_beta - 1) * dm1b).max(),
        "ptr_ab": np.abs(np.einsum("pqrr->pq", ab) - n_beta * dm1a).max(),
        "ptr_ba": np.abs(np.einsum("pprs->rs", ab) - n_alpha * dm1b).max(),
        "herm_aa": np.abs(aa - aa.transpose(1, 0, 3, 2)).max(),
        "herm_ab": np.abs(ab - ab.transpose(1, 0, 3, 2)).max(),
        "anti_aa": np.abs(aa + aa.transpose(2, 1, 0, 3)).max(),
    }
    return out
