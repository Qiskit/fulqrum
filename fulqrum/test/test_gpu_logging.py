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
import json
import logging
import sys
from types import SimpleNamespace

import pytest

from fulqrum.gpu import configure_gpu_logging, gpu_profile_record, write_gpu_profile
from fulqrum.gpu import config
from fulqrum.gpu.distributed import diagnostics
from fulqrum.gpu.logging import format_memory


@pytest.fixture(autouse=True)
def logging_state():
    names = ("", "fulqrum.gpu", diagnostics.REPORT_LOGGER, "fulqrum.gpu.test_logging")
    loggers = [logging.getLogger(name) for name in names]
    saved = [(log, log.level, log.propagate, list(log.handlers)) for log in loggers]
    package, diagnostic, emitter = loggers[1:]
    package.handlers = [logging.NullHandler()]
    package.setLevel(logging.NOTSET)
    package.propagate = True
    diagnostic.handlers = []
    diagnostic.setLevel(logging.NOTSET)
    diagnostic.propagate = True
    emitter.handlers = []
    emitter.setLevel(logging.NOTSET)
    emitter.propagate = True
    loggers[0].setLevel(logging.WARNING)
    yield
    for log, level, propagate, handlers in saved:
        for handler in log.handlers:
            if getattr(handler, "_fulqrum_gpu_owned", False):
                handler.close()
        log.handlers = handlers
        log.setLevel(level)
        log.propagate = propagate


def test_unconfigured_reports_do_not_install_handlers_or_print(capsys):
    package = logging.getLogger("fulqrum.gpu")
    diagnostic = logging.getLogger(diagnostics.REPORT_LOGGER)
    before = (
        package.level,
        list(package.handlers),
        diagnostic.level,
        list(diagnostic.handlers),
    )
    diagnostics.report("fingerprint")
    logging.getLogger("fulqrum.gpu.test_logging").info("ready")
    assert capsys.readouterr() == ("", "")
    assert before == (
        package.level,
        list(package.handlers),
        diagnostic.level,
        list(diagnostic.handlers),
    )


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("quiet", []),
        ("summary", ["ready"]),
        ("detailed", ["ready", "setup"]),
        ("trace", ["ready", "setup", "fingerprint"]),
    ],
)
def test_tiers_select_summary_detail_and_trace(mode, expected, monkeypatch):
    monkeypatch.setenv("FQ_COMM_TIMING", "0")
    monkeypatch.setenv("_FULQRUM_DEBUG_MATVEC", "tail")
    stream = io.StringIO()
    configure_gpu_logging(mode, stream=stream)
    logger = logging.getLogger("fulqrum.gpu.test_logging")
    logger.info("ready")
    logger.debug("setup")
    diagnostics.report("fingerprint")
    assert stream.getvalue().splitlines() == expected
    assert config.read("FQ_COMM_TIMING") is False
    assert diagnostics._hv_tail_enabled()


def test_application_parent_handler_receives_diagnostics():
    stream = io.StringIO()
    package = logging.getLogger("fulqrum.gpu")
    package.setLevel(logging.DEBUG)
    package.addHandler(logging.StreamHandler(stream))
    diagnostics.report("collected data")
    assert stream.getvalue() == "collected data\n"
    assert not logging.getLogger(diagnostics.REPORT_LOGGER).handlers


def test_reconfiguration_preserves_caller_handlers_and_root_settings():
    caller = logging.NullHandler()
    package = logging.getLogger("fulqrum.gpu")
    package.addHandler(caller)
    root = logging.getLogger()
    before = (root.level, list(root.handlers))
    discarded, stream = io.StringIO(), io.StringIO()
    configure_gpu_logging("trace", stream=discarded)
    configure_gpu_logging("summary", stream=stream)
    logging.getLogger("fulqrum.gpu.test_logging").info("once")
    assert discarded.getvalue() == ""
    assert stream.getvalue() == "once\n"
    assert caller in package.handlers
    assert before == (root.level, list(root.handlers))
    assert (
        sum(
            getattr(handler, "_fulqrum_gpu_owned", False)
            for handler in package.handlers
        )
        == 1
    )


def test_selected_ranks_filter_routine_records_but_preserve_errors():
    stream = io.StringIO()
    configure_gpu_logging("trace", stream=stream, rank=3, ranks=[2])
    diagnostics.report("rank zero", rank=0)
    diagnostics.report("rank two", rank=2)
    logger = logging.getLogger("fulqrum.gpu.test_logging")
    logger.info("unselected process")
    logger.error("failure on rank three", extra={"gpu_rank": 3})
    assert stream.getvalue().splitlines() == ["rank two", "failure on rank three"]


def test_primary_rank_filter_uses_process_rank_for_records_without_context():
    stream = io.StringIO()
    configure_gpu_logging("summary", stream=stream, rank=3)
    logger = logging.getLogger("fulqrum.gpu.test_logging")
    logger.info("rank three")
    logger.info("rank zero", extra={"gpu_rank": 0})
    assert stream.getvalue() == "rank zero\n"


@pytest.mark.parametrize("kwargs", [{"mode": "noisy"}, {"ranks": "unknown"}])
def test_invalid_configuration_does_not_mutate_logging(kwargs):
    package = logging.getLogger("fulqrum.gpu")
    before = (package.level, list(package.handlers), package.propagate)
    with pytest.raises(ValueError):
        configure_gpu_logging(**kwargs)
    assert before == (package.level, list(package.handlers), package.propagate)


def test_console_diagnostic_defaults_are_off():
    for name in ("FQ_HBM_LOG", "FQ_RDM_VERBOSE", "FQ_PRIMME_MANAGED_BASIS_LOG"):
        assert config.read(name, {}) is False


def test_profile_snapshot_uses_existing_data_without_collectives(tmp_path):
    def forbidden(*args):
        raise AssertionError("Profile export must not collect or communicate")

    comm = SimpleNamespace(Get_rank=lambda: 2, allgather=forbidden, allreduce=forbidden)
    operator = SimpleNamespace(
        _comm=comm,
        _gpu_ids=(1,),
        construction_timing={"upload": 0.1},
        last_timing={"wall": 0.2, "comm_ms_total": 7.0, "unknown": float("nan")},
        run_record={"exchange": "ring"},
    )
    record = gpu_profile_record(operator)
    assert record["ranks"][0]["rank"] == 2
    assert record["ranks"][0]["last_matvec"]["unknown"] is None
    assert record["schema_version"] == 1 and record["scope"] == "process"
    stream = io.StringIO()
    assert write_gpu_profile(operator, stream)["ranks"] == record["ranks"]
    assert json.loads(stream.getvalue())["ranks"] == record["ranks"]
    path = tmp_path / "profile.json"
    write_gpu_profile(operator, path)
    assert json.loads(path.read_text())["ranks"] == record["ranks"]


def test_profile_preserves_inprocess_ranks_with_repeated_devices():
    ranks = [
        SimpleNamespace(_rank=index, _gpu_ids=(0,), last_timing={"wall": float(index)})
        for index in range(2)
    ]
    record = gpu_profile_record(SimpleNamespace(_ranks=ranks))
    assert [row["rank"] for row in record["ranks"]] == [0, 1]
    assert [row["devices"] for row in record["ranks"]] == [[0], [0]]


def test_memory_format_uses_binary_units_and_unavailable_values():
    assert format_memory(2**30) == "1.000 GiB"
    assert format_memory(2**20, unit="MiB") == "1.000 MiB"
    assert format_memory(None) == "unavailable"


class _Scalar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _Vector:
    def ravel(self):
        return self

    def sum(self):
        return _Scalar(4.0)

    def min(self):
        return _Scalar(1.0)

    def max(self):
        return _Scalar(2.0)

    def __mul__(self, other):
        return self


def test_logger_tiers_do_not_change_fingerprint_collectives(monkeypatch):
    monkeypatch.setenv("_FULQRUM_DEBUG_MATVEC", "tail")
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(abs=lambda vector: vector))
    monkeypatch.setattr(
        diagnostics, "mpi_namespace", lambda comm: SimpleNamespace(SUM=1, MIN=2, MAX=3)
    )
    calls = []

    def reduce(value, op):
        calls.append(op)
        return value

    comm = SimpleNamespace(allreduce=reduce)
    for mode in ("quiet", "summary", "detailed", "trace"):
        stream = io.StringIO()
        configure_gpu_logging(mode, stream=stream)
        calls.clear()
        diagnostics._hv_tail_fingerprint(_Vector(), comm, 0, 1)
        assert calls == [1, 1, 2, 3, 3]
        assert bool(stream.getvalue()) == (mode == "trace")
    monkeypatch.delenv("_FULQRUM_DEBUG_MATVEC")
    calls.clear()
    diagnostics._hv_tail_fingerprint(_Vector(), comm, 0, 1)
    assert calls == []
