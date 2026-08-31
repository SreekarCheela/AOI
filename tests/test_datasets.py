import json
from pathlib import Path

from aoi_pipeline.datasets.base import DatasetPreparer, DatasetSpec
from aoi_pipeline.datasets.soldef_ai import SolDefAIPreparer
from aoi_pipeline.schema.labels import DefectType, Verdict


def _write_fake_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Prep/inspect only care about file extension + folder structure, not
    # real image bytes, so a placeholder is enough for these unit tests.
    path.write_bytes(b"fake-image-bytes")


def test_inspect_reports_image_count_and_unmapped_labels(tmp_path: Path):
    raw = tmp_path / "raw" / "fake_dataset"
    _write_fake_image(raw / "good" / "a.png")
    _write_fake_image(raw / "misalignment" / "b.png")
    _write_fake_image(raw / "some_unmapped_label" / "c.png")

    spec = DatasetSpec(
        name="fake_dataset",
        kaggle_slug="someone/fake-dataset",
        expected_image_count=3,
        license="test",
        description="test",
        raw_label_map={"good": DefectType.NONE, "misalignment": DefectType.MISALIGNMENT},
    )
    preparer = DatasetPreparer(spec, raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed")
    report = preparer.inspect(raw_path=raw)

    assert report.image_count == 3
    assert report.unmapped_raw_labels == ["some_unmapped_label"]
    assert report.is_usable(spec)


def test_prep_writes_manifest_with_valid_board_labels(tmp_path: Path):
    raw = tmp_path / "raw" / "soldef_ai"
    _write_fake_image(raw / "good" / "a.png")
    _write_fake_image(raw / "misalignment" / "b.png")
    _write_fake_image(raw / "insufficient_solder" / "c.png")

    preparer = SolDefAIPreparer(raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed")
    manifest_path = preparer.prep(raw_path=raw)

    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(records) == 3

    by_stem = {Path(r["image_path"]).stem: r for r in records}
    assert by_stem["a"]["verdict"] == Verdict.PASS.value
    assert by_stem["a"]["defect_types"] == []
    assert by_stem["b"]["verdict"] == Verdict.FAIL.value
    assert by_stem["b"]["defect_types"] == [DefectType.MISALIGNMENT.value]
    assert by_stem["c"]["defect_types"] == [DefectType.INSUFFICIENT_SOLDER.value]


def test_is_usable_flags_dataset_far_below_expected_count(tmp_path: Path):
    spec = DatasetSpec(
        name="fake_dataset",
        kaggle_slug="someone/fake-dataset",
        expected_image_count=1000,
        license="test",
        description="test",
    )
    from aoi_pipeline.datasets.base import DatasetInspectionReport

    report = DatasetInspectionReport(
        name="fake_dataset", image_count=10, raw_label_values=[], unmapped_raw_labels=[], sample_paths=[]
    )
    assert not report.is_usable(spec)
