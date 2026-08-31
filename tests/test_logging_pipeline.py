from pathlib import Path

import pytest

from aoi_pipeline.logging_pipeline.capture_logger import CaptureLogger, iter_records
from aoi_pipeline.logging_pipeline.storage import StorageCriticalError, StorageMonitor, StorageStatus


def test_log_capture_assigns_incrementing_sequence_numbers(tmp_path: Path):
    logger = CaptureLogger(log_dir=tmp_path, station_id="station-01")
    r1 = logger.log_capture("img_0.png")
    r2 = logger.log_capture("img_1.png", board_id="BOARD-123")
    assert r1.sequence_number == 0
    assert r2.sequence_number == 1
    assert r2.board_id == "BOARD-123"


def test_log_capture_persists_and_resumes_sequence(tmp_path: Path):
    logger1 = CaptureLogger(log_dir=tmp_path)
    logger1.log_capture("img_0.png")
    logger1.log_capture("img_1.png")

    # Simulate a service restart: a fresh CaptureLogger over the same dir
    # must resume the sequence rather than restarting at 0.
    logger2 = CaptureLogger(log_dir=tmp_path)
    r = logger2.log_capture("img_2.png")
    assert r.sequence_number == 2

    log_file = next(tmp_path.glob("captures_*.jsonl"))
    records = list(iter_records(log_file))
    assert [r.sequence_number for r in records] == [0, 1, 2]


def test_storage_monitor_thresholds():
    def fake_status(percent_used: float) -> StorageStatus:
        total = 100
        used = int(total * percent_used / 100)
        return StorageStatus(path="/fake", total_bytes=total, used_bytes=used, free_bytes=total - used)

    warned = []
    critical = []
    monitor = StorageMonitor(
        path=Path("/tmp"),
        warning_pct=80,
        critical_pct=95,
        on_warning=lambda s: warned.append(s),
        on_critical=lambda s: critical.append(s),
    )
    monitor.get_status = lambda: fake_status(50)  # type: ignore[method-assign]
    monitor.check()
    assert not warned and not critical

    monitor.get_status = lambda: fake_status(85)  # type: ignore[method-assign]
    monitor.check()
    assert len(warned) == 1 and not critical

    monitor.get_status = lambda: fake_status(97)  # type: ignore[method-assign]
    monitor.check()
    assert len(critical) == 1


def test_capture_logger_refuses_to_write_when_storage_critical(tmp_path: Path):
    monitor = StorageMonitor(path=tmp_path, warning_pct=80, critical_pct=95)
    monitor.get_status = lambda: StorageStatus(  # type: ignore[method-assign]
        path=str(tmp_path), total_bytes=100, used_bytes=99, free_bytes=1
    )
    logger = CaptureLogger(log_dir=tmp_path, storage_monitor=monitor)
    with pytest.raises(StorageCriticalError):
        logger.log_capture("img_0.png")
