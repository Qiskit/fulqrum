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

from .rdm_cart import compute_rdms_cart

__all__ = ["compute_rdms_tile_resident"]


def compute_rdms_tile_resident(ham, evec_local, *, rank=1, spin_summed=False):
    """Compute reduced density matrices from each rank's local state-vector tile.

    All communicator ranks must call with parts of the same real state and the
    same options. Every rank receives the full result as host NumPy arrays after
    GPU work and communication complete. Alpha and beta must have equal orbital
    counts. Conventions follow PySCF ``make_rdm12s(reorder=True)``.

    No normalization is applied. For normalized RDMs, normalize the full state
    before distributing it among ranks.

    Args:
        ham: ``TileGPUResidentHamiltonian`` for the tensor-product subspace.
        evec_local: NumPy-compatible or CuPy array of ``ham.n_local`` real
            amplitudes, or shape ``(ham.n_b_local, ham.n_a_local)``. Uses
            beta-outer, alpha-inner order within this rank's tile.
        rank: RDM order, 1 or 2; defaults to 1. This is not an MPI rank.
        spin_summed: Combine alpha and beta spin blocks; defaults to False.

    Returns:
        For ``rank=1``, ``(gamma_a, gamma_b)`` or spin-summed ``gamma``.
        For ``rank=2``, ``((gamma_a, gamma_b), (G_aa, G_ab, G_bb))`` or
        spin-summed ``(gamma, G)``. Shapes and support requirements match
        ``TileGPUResidentHamiltonian.compute_rdms``.
    """
    return compute_rdms_cart(ham, evec_local, rank=rank, spin_summed=spin_summed)
