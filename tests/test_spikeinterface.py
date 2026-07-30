from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import zarr

from npc_ephys.spikeinterface import (
    SpikeInterfaceData,
    SpikeInterfaceKS25Data,
    get_spikeinterface_data,
)
from npc_ephys.units import (
    add_global_unit_ids,
    make_units_table_from_spike_interface,
    make_units_table_from_spike_interface_ks25,
)

UNIT_IDS = np.array([5, 8], dtype=np.int64)
SAMPLE_INDEXES = np.array([10, 20, 30], dtype=np.int64)
UNIT_INDEXES = np.array([0, 1, 0], dtype=np.int64)
SPIKE_AMPLITUDES = np.array([0.1, 0.2, 0.3], dtype=np.float32)
TEMPLATES_AVERAGE = np.array(
    [
        [[0, 0, 0], [-2, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, -1], [0, 0, 2], [0, 0, 0]],
    ],
    dtype=np.float64,
)
TEMPLATES_STD = np.abs(TEMPLATES_AVERAGE) / 10
LOCATIONS = [[0.0, 0.0], [20.0, 40.0], [40.0, 80.0]]
QUALITY_METRICS = {
    "num_spikes": np.array([2, 1], dtype=np.int64),
    "snr": np.array([4.0, 5.0]),
}
TEMPLATE_METRICS = {"exp_decay": np.array([0.4, 0.5])}
ANALYZER_SORTING_NAME = (
    "experiment1_Record Node 109#Neuropix-PXI-100.ProbeA-1_recording1"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_common_sorting(root: Path, sorting_name: str = "ProbeA") -> Path:
    curated = root / "curated" / sorting_name
    properties = curated / "properties"
    properties.mkdir(parents=True)
    _write_json(curated / "numpysorting_info.json", {"num_segments": 1})
    np.save(
        curated / "spikes.npy",
        np.column_stack((SAMPLE_INDEXES, UNIT_INDEXES, np.zeros_like(SAMPLE_INDEXES))),
    )
    np.save(properties / "original_cluster_id.npy", UNIT_IDS)
    np.save(properties / "default_qc.npy", np.array([True, False]))
    np.save(properties / "decoder_probability.npy", np.array([0.9, 0.2]))
    np.save(properties / "decoder_label.npy", np.array(["good", "noise"]))
    return curated


def _write_legacy_result(root: Path) -> None:
    curated = _write_common_sorting(root)
    (root / "output").write_text("legacy pipeline")
    _write_json(
        curated / "provenance.json",
        {"kwargs": {"parent_sorting": {"version": "0.100.0"}}},
    )

    postprocessed = root / "postprocessed" / "ProbeA"
    _write_json(postprocessed / "params.json", {"legacy": True})
    _write_json(
        postprocessed / "quality_metrics" / "params.json",
        {"metric_names": list(QUALITY_METRICS)},
    )
    _write_json(
        postprocessed / "template_metrics" / "params.json",
        {"metric_names": list(TEMPLATE_METRICS)},
    )
    pd.DataFrame(QUALITY_METRICS, index=UNIT_IDS).to_csv(
        postprocessed / "quality_metrics" / "metrics.csv"
    )
    pd.DataFrame(TEMPLATE_METRICS, index=UNIT_IDS).to_csv(
        postprocessed / "template_metrics" / "metrics.csv"
    )
    np.save(postprocessed / "templates_average.npy", TEMPLATES_AVERAGE)
    np.save(postprocessed / "templates_std.npy", TEMPLATES_STD)
    amplitude_dir = postprocessed / "spike_amplitudes"
    amplitude_dir.mkdir()
    np.save(amplitude_dir / "amplitude_segment_0.npy", SPIKE_AMPLITUDES)
    location_dir = postprocessed / "unit_locations"
    location_dir.mkdir()
    np.save(location_dir / "unit_locations.npy", np.array([[0, 1], [2, 3]]))
    _write_json(
        postprocessed / "recording_info" / "recording_attributes.json",
        {"channel_ids": ["AP1", "AP2", "AP3"]},
    )
    _write_json(
        postprocessed / "sorting.json",
        {
            "annotations": {
                "__sorting_info__": {
                    "recording": {"properties": {"location": LOCATIONS}}
                }
            }
        },
    )
    _write_json(
        postprocessed / "sparsity.json",
        {
            "unit_id_to_channel_ids": {"5": ["AP1"], "8": ["AP3"]},
            "channel_ids": ["AP1", "AP2", "AP3"],
            "unit_ids": UNIT_IDS.tolist(),
        },
    )


def _create_arrays(group: zarr.hierarchy.Group, values: dict[str, np.ndarray]) -> None:
    for name, value in values.items():
        group.create_dataset(name, data=value)


def _write_analyzer_result(root: Path) -> None:
    _write_common_sorting(root, ANALYZER_SORTING_NAME)
    (root / "output").write_text("N E X T F L O W")
    analyzer_path = root / "postprocessed" / f"{ANALYZER_SORTING_NAME}.zarr"
    analyzer_path.parent.mkdir()
    analyzer = zarr.open(analyzer_path, mode="w")
    analyzer.attrs["settings"] = {"return_in_uV": True}
    analyzer.attrs["spikeinterface_info"] = {
        "dev_mode": False,
        "object": "SortingAnalyzer",
        "version": "0.104.7",
    }

    recording_info = analyzer.create_group("recording_info")
    recording_info.attrs["recording_attributes"] = {
        "channel_ids": ["CH0", "CH1", "CH2"]
    }
    sorting = analyzer.create_group("sorting")
    sorting.attrs["num_segments"] = 1
    sorting.attrs["annotations"] = {
        "__sorting_info__": {"recording": {"properties": {"location": LOCATIONS}}}
    }
    sorting.create_dataset("unit_ids", data=UNIT_IDS)
    _create_arrays(
        sorting.create_group("spikes"),
        {
            "sample_index": SAMPLE_INDEXES,
            "unit_index": UNIT_INDEXES,
            "segment_slices": np.array([[0, len(SAMPLE_INDEXES)]]),
        },
    )
    _create_arrays(
        sorting.create_group("properties"),
        {
            "original_cluster_id": UNIT_IDS,
            "default_qc": np.array([True, False]),
            "decoder_probability": np.array([0.9, 0.2]),
            "decoder_label": np.array(["good", "noise"]),
        },
    )
    analyzer.create_dataset(
        "sparsity_mask",
        data=np.array([[True, False, False], [False, False, True]]),
    )

    extensions = analyzer.create_group("extensions")
    quality = extensions.create_group("quality_metrics")
    quality.attrs["params"] = {"metric_names": list(QUALITY_METRICS)}
    quality_metrics = quality.create_group("metrics")
    quality_metrics.create_dataset("index", data=UNIT_IDS)
    _create_arrays(quality_metrics, QUALITY_METRICS)

    template = extensions.create_group("template_metrics")
    template.attrs["params"] = {"metric_names": list(TEMPLATE_METRICS)}
    template_metrics = template.create_group("metrics")
    template_metrics.create_dataset("index", data=UNIT_IDS)
    _create_arrays(template_metrics, TEMPLATE_METRICS)

    templates = extensions.create_group("templates")
    templates.create_dataset("average", data=TEMPLATES_AVERAGE)
    templates.create_dataset("std", data=TEMPLATES_STD)
    spike_amplitudes = extensions.create_group("spike_amplitudes")
    spike_amplitudes.create_dataset("amplitudes", data=SPIKE_AMPLITUDES)
    unit_locations = extensions.create_group("unit_locations")
    unit_locations.create_dataset("unit_locations", data=np.array([[0, 1], [2, 3]]))
    zarr.consolidate_metadata(analyzer.store)


@pytest.fixture(params=["legacy", "analyzer"])
def spikeinterface_data(
    tmp_path: Path, request: pytest.FixtureRequest
) -> SpikeInterfaceData:
    root = tmp_path / request.param
    root.mkdir()
    if request.param == "legacy":
        _write_legacy_result(root)
    else:
        _write_analyzer_result(root)
    return SpikeInterfaceData(root=root)


def test_format_specific_storage_has_common_interface(
    spikeinterface_data: SpikeInterfaceData,
) -> None:
    data = spikeinterface_data
    expected_version = "0.100.0" if data.data_format == "legacy" else "0.104.7"

    assert data.version == expected_version
    assert data.is_analyzer is (data.data_format == "analyzer")
    pd.testing.assert_frame_equal(
        data.quality_metrics_df("probeA"),
        pd.DataFrame(QUALITY_METRICS, index=UNIT_IDS),
        check_names=False,
    )
    pd.testing.assert_frame_equal(
        data.template_metrics_df("probeA"),
        pd.DataFrame(TEMPLATE_METRICS, index=UNIT_IDS),
        check_names=False,
    )
    np.testing.assert_array_equal(data.templates_average("probeA"), TEMPLATES_AVERAGE)
    np.testing.assert_array_equal(data.templates_std("probeA"), TEMPLATES_STD)
    np.testing.assert_array_equal(data.spike_indexes("probeA"), SAMPLE_INDEXES)
    np.testing.assert_array_equal(data.unit_indexes("probeA"), UNIT_INDEXES)
    np.testing.assert_array_equal(data.original_cluster_id("probeA"), UNIT_IDS)
    np.testing.assert_array_equal(data.cluster_indexes("probeA"), [5, 8, 5])
    np.testing.assert_array_equal(data.default_qc("probeA"), [True, False])
    np.testing.assert_array_equal(data.decoder_label("probeA"), ["good", "noise"])
    assert data.sparse_channel_indices("probeA") == (0, 1, 2)
    np.testing.assert_array_equal(data.electrode_locations_xy("probeA"), LOCATIONS)
    amplitudes = data.spike_amplitudes("probeA")
    np.testing.assert_array_equal(amplitudes[0], SPIKE_AMPLITUDES[[0, 2]])
    np.testing.assert_array_equal(amplitudes[1], SPIKE_AMPLITUDES[[1]])


def test_legacy_structured_spikes_array(tmp_path: Path) -> None:
    root = tmp_path / "legacy-structured"
    root.mkdir()
    curated = _write_common_sorting(root)
    (root / "output").write_text("legacy pipeline")
    _write_json(
        curated / "provenance.json",
        {"kwargs": {"parent_sorting": {"version": "0.100.0"}}},
    )
    structured_spikes = np.array(
        list(zip(SAMPLE_INDEXES, UNIT_INDEXES, np.zeros_like(SAMPLE_INDEXES))),
        dtype=[
            ("sample_index", "<i8"),
            ("unit_index", "<i8"),
            ("segment_index", "<i8"),
        ],
    )
    np.save(curated / "spikes.npy", structured_spikes)

    data = SpikeInterfaceData(root=root)

    np.testing.assert_array_equal(data.spike_indexes("probeA"), SAMPLE_INDEXES)
    np.testing.assert_array_equal(data.unit_indexes("probeA"), UNIT_INDEXES)


def test_sorting_names_match_exact_device_and_all_grouped_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "grouped"
    postprocessed = root / "postprocessed"
    postprocessed.mkdir(parents=True)
    sorting_names = tuple(
        f"experiment1_Record Node 109#Neuropix-PXI-100.ProbeC_recording1_group{group}"
        for group in range(4)
    )
    for sorting_name in sorting_names:
        (postprocessed / f"{sorting_name}.zarr").mkdir()

    data = SpikeInterfaceData(root=root)

    assert data.sorting_names_for_device("Neuropix-PXI-100.ProbeC") == sorting_names


def test_units_entry_point_fans_out_grouped_analyzer_outputs(tmp_path: Path) -> None:
    root = tmp_path / "grouped-analyzer"
    root.mkdir()
    _write_analyzer_result(root)
    source = root / "postprocessed" / f"{ANALYZER_SORTING_NAME}.zarr"
    for group in range(2):
        sorting_name = (
            "experiment1_Record Node 109#"
            f"Neuropix-PXI-100.ProbeC_recording1_group{group}.zarr"
        )
        destination = root / "postprocessed" / sorting_name
        shutil.copytree(source, destination)
        analyzer = zarr.open(destination, mode="a")
        analyzer["recording_info"].attrs["recording_attributes"] = {
            "channel_ids": [f"CH{48 * group + offset}" for offset in (0, 2, 4)]
        }
        zarr.consolidate_metadata(analyzer.store)

    timing = SimpleNamespace(
        device=SimpleNamespace(name="Neuropix-PXI-100.ProbeC", is_sync_adjusted=False),
        sampling_rate=10_000.0,
        start_time=1.0,
    )

    units = make_units_table_from_spike_interface(
        SpikeInterfaceData(root=root), [timing]
    )
    add_global_unit_ids(units, "841363_2026-06-12")

    assert units["cluster_id"].tolist() == [5, 8, 5, 8]
    assert units["electrode_group_name"].tolist() == ["probeC"] * 4
    assert units["shank"].tolist() == [0, 0, 1, 1]
    assert units["peak_channel"].tolist() == [0, 4, 48, 52]
    assert units["unit_id"].is_unique


def test_generic_units_entry_point_dispatches_storage_format(
    spikeinterface_data: SpikeInterfaceData,
) -> None:
    device_name = (
        "Neuropix-PXI-100.ProbeA-1" if spikeinterface_data.is_analyzer else "ProbeA-AP"
    )
    timing = SimpleNamespace(
        device=SimpleNamespace(name=device_name, is_sync_adjusted=False),
        sampling_rate=10_000.0,
        start_time=1.0,
    )

    units = make_units_table_from_spike_interface(
        spikeinterface_data, [timing], include_waveform_arrays=True
    )

    assert units["cluster_id"].tolist() == UNIT_IDS.tolist()
    assert units["electrode_group_name"].tolist() == ["probeA", "probeA"]
    np.testing.assert_array_equal(units.iloc[0]["spike_times"], [1.001, 1.003])
    np.testing.assert_array_equal(units.iloc[1]["spike_times"], [1.002])
    assert units["peak_channel"].tolist() == [0, 2]
    assert units["channels"].tolist() == [(0,), (2,)]


def test_legacy_names_are_aliases(spikeinterface_data: SpikeInterfaceData) -> None:
    assert SpikeInterfaceKS25Data is SpikeInterfaceData
    assert (
        make_units_table_from_spike_interface_ks25
        is make_units_table_from_spike_interface
    )
    assert get_spikeinterface_data(spikeinterface_data) is spikeinterface_data


def test_s3_result_path_is_not_parsed_as_session_id() -> None:
    path = (
        "s3://bucket/" "ecephys_841363_2026-06-12_15-30-03_sorted_2026-07-21_09-51-19"
    )

    data = get_spikeinterface_data(path)

    assert data.session is None
    assert str(data.root) == path
