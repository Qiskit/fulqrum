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


import io

import numpy as np
import pytest

from .conftest import gpu_available, make_lih_hsub
from .test_gpu_logging import logging_state  # noqa: F401

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not gpu_available(), reason="CUDA build/device unavailable"),
]


@pytest.mark.parametrize("table_source", ["host", "kernel"])
def test_default_runs_are_quiet_and_numerically_correct(
    monkeypatch, capsys, table_source
):
    from fulqrum.gpu import GPUHamiltonian

    # Exercise automatically selected optimization paths on a small input.
    monkeypatch.setenv("FQ_FP64_MMA", "1")
    monkeypatch.setenv("FQ_ARM_TABLE_SOURCE", table_source)
    for name in (
        "FQ_HBM_LOG",
        "FQ_RDM_VERBOSE",
        "FQ_COMM_TIMING",
        "_FULQRUM_DEBUG_MATVEC",
    ):
        monkeypatch.delenv(name, raising=False)
    hsub, _, _ = make_lih_hsub()
    op = GPUHamiltonian(hsub, gpu_ids=(0, 0))
    x = np.random.default_rng(21).standard_normal(op.shape[0])
    expected = hsub @ x
    np.testing.assert_allclose(op @ x, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(op @ x, expected, rtol=1e-12, atol=1e-12)
    state = x / np.linalg.norm(x)
    op.compute_rdms(state, rank=1)
    op.compute_rdms(state, rank=2)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_summary_reports_one_construction_and_rdm_completion():
    from fulqrum.gpu import GPUHamiltonian, configure_gpu_logging, write_gpu_profile

    stream = io.StringIO()
    configure_gpu_logging("summary", stream=stream)
    hsub, _, _ = make_lih_hsub()
    op = GPUHamiltonian(hsub, gpu_ids=(0, 0))
    x = np.ones(op.shape[0]) / np.sqrt(op.shape[0])
    op @ x
    op @ x
    op.compute_rdms(x, rank=1)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("GPU Hamiltonian ready:")
    assert lines[1].startswith("RDM calculation completed: order=1")
    profile = io.StringIO()
    record = write_gpu_profile(op, profile)
    assert len(record["ranks"]) == 2
    assert record["ranks"][0]["last_matvec"]["wall"] > 0
    summary = io.StringIO()
    text = op.dump_profile_summary(summary)
    assert "rank=0, device=0" in text and "rank=1, device=0" in text
    assert summary.getvalue() == text + "\n"
