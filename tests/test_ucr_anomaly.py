import json

import numpy as np
import pytest

from sarsim_fm.data.ucr_anomaly import (
    detector_interval,
    iter_staged,
    load_values,
    mnemonic_family,
    parse_filename,
    stage_ucr,
)


def test_names_and_single_point():
    meta = parse_filename("001_UCR_Anomaly_name_with_underscores_2_4_4.txt")
    assert meta.name == "name_with_underscores"
    assert detector_interval(meta) == (3, 4)
    assert mnemonic_family("DISTORTED1sddb40") == mnemonic_family("sddb40")
    assert mnemonic_family("NOISEECG3") == mnemonic_family("ECG1")


def test_whitespace_indices_and_finite(tmp_path):
    path = tmp_path / "001_UCR_Anomaly_name_2_4_4.txt"
    path.write_text("1\n2\n3\n4\n5")
    a = load_values(path)
    path.write_text("1\t2 3\t4  5\n")
    np.testing.assert_array_equal(a, load_values(path))
    path.write_text("1 2 3 nan 5")
    with pytest.raises(ValueError, match="finite"):
        load_values(path)
    path.write_text("1 2")
    with pytest.raises(ValueError, match="index ordering"):
        load_values(path)


def test_manifest_ids_hashes_and_ordering(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for i in range(250, 0, -1):
        (source / f"{i:03d}_UCR_Anomaly_demo_2_4_4.txt").write_text("1\t2\n3 4 5")
    out = tmp_path / "manifest.json"
    stage_ucr(source, out)
    original = out.read_bytes()
    stage_ucr(source, out)
    assert original == out.read_bytes()
    manifest = json.loads(original)
    assert [s["dataset_id"] for s in manifest["series"]] == list(range(1, 251))
    row = next(iter_staged(manifest, [1]))
    assert (row["anomaly_start"], row["anomaly_end"]) == (3, 4)
    assert manifest["series"][0]["anomaly_start"] == 4
    (source / "001_UCR_Anomaly_demo_2_4_4.txt").write_text("1 2 3 4 6")
    with pytest.raises(ValueError, match="hash"):
        next(iter_staged(manifest, [1]))
    (source / "001_UCR_Anomaly_demo_2_4_4.txt").rename(source / "251_UCR_Anomaly_demo_2_4_4.txt")
    with pytest.raises(ValueError, match="unique IDs"):
        stage_ucr(source, out)


def test_reject_incomplete_archive(tmp_path):
    with pytest.raises(ValueError, match="250"):
        stage_ucr(tmp_path, tmp_path / "manifest.json")
