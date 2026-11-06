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


import os

import pytest

from fulqrum.gpu.distributed import _collective
from fulqrum.gpu.distributed._collective import choose_auto_backend


def test_one_rank_uses_host_mpi():
    assert choose_auto_backend([(True, "node0", "gpu0")])[0] == "host_mpi"


def test_own_gpu_per_rank_uses_nccl():
    facts = [(True, "node0", "gpu0"), (True, "node0", "gpu1"), (True, "node1", "gpu0")]
    assert choose_auto_backend(facts)[0] == "nccl"


def test_shared_gpu_uses_host_mpi():
    name, reason = choose_auto_backend(
        [(True, "node0", "gpu0"), (True, "node0", "gpu0")]
    )
    assert name == "host_mpi"
    assert "share" in reason


def test_missing_nccl_on_one_rank_uses_host_mpi():
    assert (
        choose_auto_backend([(True, "node0", "gpu0"), (False, "node1", "")])[0]
        == "host_mpi"
    )


def test_mpi_auto_is_the_same_on_every_rank():
    MPI = pytest.importorskip("mpi4py.MPI")
    pytest.importorskip("cupy")
    comm = MPI.COMM_WORLD
    saved = os.environ.pop("FQ_DIST_BACKEND", None)
    try:
        name = _collective._resolve_backend_name(comm)
    finally:
        if saved is not None:
            os.environ["FQ_DIST_BACKEND"] = saved
    names = comm.allgather(name)
    assert len(set(names)) == 1
    assert name in ("host_mpi", "nccl")
    facts = comm.allgather(_collective._auto_backend_facts())
    if len({(host, uuid) for _ok, host, uuid in facts}) < len(facts):
        assert name == "host_mpi"


def test_backend_is_made_once_per_communicator_and_name():
    MPI = pytest.importorskip("mpi4py.MPI")
    pytest.importorskip("cupy")
    _collective.reset_for_testing()
    comm = MPI.COMM_WORLD
    first, _reason = _collective.backend_for(comm, "host_mpi")
    again, _reason = _collective.backend_for(comm, "host_mpi")
    assert again is first
    other_comm = comm.Dup()
    other, _reason = _collective.backend_for(other_comm, "host_mpi")
    assert other is not first
    _collective.reset_for_testing()
    other_comm.Free()
