"""Read legacy and ``SortingAnalyzer`` SpikeInterface pipeline output."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import io
import json
import logging
from typing import Any, Literal, Union, cast

import aind_session
import npc_io
import npc_lims
import npc_session
import numpy as np
import numpy.typing as npt
import packaging.version
import pandas as pd
import upath
import zarr
from typing_extensions import TypeAlias

import npc_ephys.settings_xml as _settings_xml

logger = logging.getLogger(__name__)

SpikeInterfaceDataSource: TypeAlias = Union[
    str, npc_session.SessionRecord, npc_io.PathLike, "SpikeInterfaceData"
]


class ProbeNotFoundError(FileNotFoundError):
    pass


def get_spikeinterface_data(
    session_or_root_path: SpikeInterfaceDataSource,
) -> SpikeInterfaceData:
    """Return format-agnostic SpikeInterface data for a session or result path.

    >>> paths = get_spikeinterface_data('668759_20230711')
    >>> paths.root == get_spikeinterface_data('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/83754308-0a91-4b54-af79-3c42f6bc831b').root
    True
    """
    if isinstance(session_or_root_path, SpikeInterfaceData):
        return session_or_root_path

    source = str(session_or_root_path)
    is_explicit_path = (
        not isinstance(session_or_root_path, (str, npc_session.SessionRecord))
        or "://" in source
        or "/" in source
        or "\\" in source
    )
    if is_explicit_path:
        session = None
        root = npc_io.from_pathlike(session_or_root_path)
    else:
        try:
            session = npc_session.SessionRecord(source)
            root = None
        except ValueError:
            session = None
            root = npc_io.from_pathlike(session_or_root_path)
    return SpikeInterfaceData(session=session, root=root)


@dataclasses.dataclass(unsafe_hash=True, eq=True)
class SpikeInterfaceData:
    """Data stored by a SpikeInterface sorting pipeline.

    Both the legacy folder organization and the Zarr ``SortingAnalyzer``
    organization are supported.

    Provide a session ID or a root path:
    >>> si = SpikeInterfaceData('668759_20230711')
    >>> si.root
    S3Path('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/83754308-0a91-4b54-af79-3c42f6bc831b')

    >>> si.template_metrics_dict('probeA')
    {'metric_names': ['exp_decay', 'half_width', 'num_negative_peaks', 'num_positive_peaks', 'peak_to_valley', 'peak_trough_ratio', 'recovery_slope', 'repolarization_slope', 'spread', 'velocity_above', 'velocity_below'], 'sparsity': None, 'peak_sign': 'neg', 'upsampling_factor': 10, 'metrics_kwargs': {'recovery_window_ms': 0.7, 'peak_relative_threshold': 0.2, 'peak_width_ms': 0.1, 'depth_direction': 'y', 'min_channels_for_velocity': 5, 'min_r2_velocity': 0.5, 'exp_peak_function': 'ptp',
    'min_r2_exp_decay': 0.5, 'spread_threshold': 0.2, 'spread_smooth_um': 20, 'column_range': None}}

    >>> si.quality_metrics_df('probeA').columns
    Index(['amplitude_cutoff', 'amplitude_cv_median', 'amplitude_cv_range',
           'amplitude_median', 'drift_ptp', 'drift_std', 'drift_mad',
           'firing_range', 'firing_rate', 'isi_violations_ratio',
           'isi_violations_count', 'num_spikes', 'presence_ratio',
           'rp_contamination', 'rp_violations', 'sliding_rp_violation', 'snr',
           'sync_spike_2', 'sync_spike_4', 'sync_spike_8', 'd_prime',
           'isolation_distance', 'l_ratio', 'silhouette', 'nn_hit_rate',
           'nn_miss_rate'],
          dtype='object')
    >>> si.version
    '0.100.0'
    >>> ''.join(si.probes)
    'ABCDEF'
    >>> si.spike_indexes('probeA')[[0, 1, 2, -3, -2, -1]].tolist()
    [145, 491, 738, 143124925, 143125165, 143125201]
    >>> si.unit_indexes('probeA')[[0, 1, 2, -3, -2, -1]].tolist()
    [36, 50, 55, 52, 132, 53]
    >>> len(si.original_cluster_id('probeA'))
    139
    >>> si = SpikeInterfaceData('712815_2024-05-21')
    >>> si.is_nextflow_pipeline
    True
    """

    session: str | npc_session.SessionRecord | None = None
    root: upath.UPath | None = None

    def __post_init__(self) -> None:
        if self.root is None and self.session is None:
            raise ValueError("Must provide either session or root")
        if self.root is None:
            self.root = npc_lims.get_sorted_data_paths_from_s3(self.session)[0].parent

    @property
    def probes(self) -> tuple[npc_session.ProbeRecord, ...]:
        """Probes available from this SpikeInterface dataset, with full set of
        data (probe is present in "curated" container)."""
        probes = set()
        try:
            dir_path = self.get_path("sorting_precurated")
        except FileNotFoundError:
            dir_path = self.get_path("curated")
        for path in dir_path.iterdir():
            with contextlib.suppress(ValueError):
                probes.add(npc_session.ProbeRecord(path.name))
        return tuple(sorted(probes))

    @property
    def is_nextflow_pipeline(self) -> bool:
        if self.root is not None:
            with contextlib.suppress(FileNotFoundError):
                return "N E X T F L O W" in self.output().read_text()

        return False

    @functools.cache
    def _analyzer_metadata(self, probe: str) -> dict:
        with contextlib.suppress(FileNotFoundError):
            metadata = self.read_json(
                self.get_correct_path(
                    self.postprocessed(probe=probe),
                    ".zattrs",
                )
            )
            if (
                metadata.get("spikeinterface_info", {}).get("object")
                == "SortingAnalyzer"
            ):
                return metadata
        return {}

    @property
    def data_format(self) -> Literal["legacy", "analyzer"]:
        """On-disk organization, independent of the spike sorter used."""
        return "analyzer" if self._analyzer_metadata(str(self.probes[0])) else "legacy"

    @property
    def is_analyzer(self) -> bool:
        return self.data_format == "analyzer"

    @property
    def version(self) -> str:
        if self.is_analyzer:
            return self._analyzer_metadata(str(self.probes[0]))["spikeinterface_info"][
                "version"
            ]
        if not self.is_nextflow_pipeline:
            return self.provenance(self.probes[0])["kwargs"]["parent_sorting"][
                "version"
            ]

        return self.provenance(self.probes[0])["version"]

    @property
    def is_pre_v0_99(self) -> bool:
        return packaging.version.parse(self.version) < packaging.version.parse("0.99")

    @staticmethod
    @functools.cache
    def get_correct_path(*path_components: npc_io.PathLike) -> upath.UPath:
        """SpikeInterface makes paths with '#' in them, which is not allowed in s3
        paths in general - run paths through this function to fix them."""
        if not path_components:
            raise ValueError("Must provide at least one path component")
        path = npc_io.from_pathlike(path_components[0])
        if path.protocol in ("", "file"):
            for path_component in path_components[1:]:
                path /= str(path_component)
        else:
            path = npc_io.from_pathlike(
                "/".join(str(path_component) for path_component in path_components)
            )
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        return path

    @functools.cached_property
    def aind_session(self) -> aind_session.Session:
        if self.session is not None:
            session_record = npc_session.SessionRecord(self.session)
            return aind_session.get_sessions(
                subject_id=session_record.subject, date=session_record.date
            )[0]
        else:
            assert self.root is not None
            data_description = json.loads(
                (self.root / "data_description.json").read_text()
            )
            return aind_session.Session(data_description["name"])

    @functools.cached_property
    def settings_xml(self) -> _settings_xml.SettingsXmlInfo:
        for record_node_dir in self.aind_session.ecephys.clipped_dir.iterdir():
            settings_xml_path = next(record_node_dir.glob("settings*.xml"), None)
            if settings_xml_path is not None and settings_xml_path.exists():
                return _settings_xml.get_settings_xml_data(settings_xml_path)
        raise FileNotFoundError(
            f"No settings XML files found in {self.aind_session.ecephys.clipped_dir}"
        )

    @staticmethod
    def read_json(path: upath.UPath) -> dict:
        return json.loads(path.read_text())

    @staticmethod
    def read_csv(path: upath.UPath) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0)

    def get_json(self, filename: str) -> dict:
        assert self.root is not None
        return self.read_json(self.get_correct_path(self.root, filename))

    def get_path(
        self,
        dirname: str,
        probe: str | None = None,
        excl_name_component: str | None = None,
    ) -> upath.UPath:
        """Return a path to a single dir or file: either `self.root/dirname` or, if `probe` is specified,
        the probe-specific sub-path within `self.root/dirname`."""
        assert self.root is not None
        if not dirname:
            raise ValueError("Must provide a dirname to get path")
        path: upath.UPath | None
        if probe is None:
            path = self.get_correct_path(self.root, dirname)
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist")
        else:
            candidates = tuple(
                path
                for path in sorted(self.get_correct_path(self.root, dirname).iterdir())
                if (excl_name_component is None or excl_name_component not in path.name)
            )
            target_name = str(probe).removesuffix(".zarr")
            path = next(
                (
                    path
                    for path in candidates
                    if path.name.removesuffix(".zarr") == target_name
                ),
                None,
            )
            path = path or next(
                (
                    path
                    for path in candidates
                    if npc_session.ProbeRecord(probe)
                    == npc_session.ProbeRecord(path.as_posix())
                ),
                None,
            )
            if path is None or not path.exists():
                raise ProbeNotFoundError(
                    f"{path} does not exist - sorting likely skipped by SpikeInterface due to fraction of bad channels"
                )
        return path

    @functools.cached_property
    def sorting_names(self) -> tuple[str, ...]:
        """Names of all probe/stream outputs in the postprocessed container."""
        assert self.root is not None
        postprocessed = self.get_correct_path(self.root, "postprocessed")
        return tuple(
            path.name.removesuffix(".zarr")
            for path in sorted(postprocessed.iterdir())
            if "sorting" not in path.name
        )

    def sorting_names_for_device(self, device_name: str) -> tuple[str, ...]:
        """Return sorting outputs associated with an Open Ephys device name."""
        exact_matches = tuple(
            sorting_name
            for sorting_name in self.sorting_names
            if device_name.casefold() in sorting_name.casefold()
        )
        if exact_matches:
            return exact_matches

        probe = npc_session.ProbeRecord(device_name)
        return tuple(
            sorting_name
            for sorting_name in self.sorting_names
            if npc_session.ProbeRecord(sorting_name) == probe
        )

    # json data
    processing_json = functools.partialmethod(get_json, "processing.json")
    subject_json = functools.partialmethod(get_json, "subject.json")
    data_description_json = functools.partialmethod(get_json, "data_description.json")
    procedures_json = functools.partialmethod(get_json, "procedures.json")
    visualization_output_json = functools.partialmethod(
        get_json, "visualization_output.json"
    )

    # dirs
    drift_maps = functools.partialmethod(get_path, dirname="drift_maps")
    output = functools.partialmethod(get_path, dirname="output")
    postprocessed = functools.partialmethod(
        get_path, dirname="postprocessed", excl_name_component="sorting"
    )
    spikesorted = functools.partialmethod(
        get_path, dirname="spikesorted", excl_name_component="sorting"
    )

    @functools.cache
    def curated(self, probe: str) -> upath.UPath:
        """Path changed Jan 2024

        https://github.com/AllenNeuralDynamics/aind-ephys-spikesort-kilosort25-full/blob/main/RELEASE_NOTES.md#v30---jan-18-2024
        """
        try:
            dir_path = self.get_path("sorting_precurated")
        except FileNotFoundError:
            dir_path = self.get_path("curated")
        return self.get_path(dir_path.name, probe)

    @functools.cache
    def sorting_analyzer(self, probe: str) -> zarr.hierarchy.Group:
        """Open the probe's Zarr ``SortingAnalyzer`` store."""
        if not self._analyzer_metadata(probe):
            raise ValueError(f"{probe} uses the legacy SpikeInterface organization")
        path = self.postprocessed(probe=probe)
        store = path.fs.get_mapper(path.path)
        try:
            return zarr.open_consolidated(store, mode="r")
        except (KeyError, ValueError):
            return zarr.open(store, mode="r")

    def _analyzer_extension(self, probe: str, extension: str) -> zarr.hierarchy.Group:
        return self.sorting_analyzer(probe)[f"extensions/{extension}"]

    def _analyzer_metrics_df(self, probe: str, extension: str) -> pd.DataFrame:
        metrics = self._analyzer_extension(probe, extension)["metrics"]
        index = np.asarray(metrics["index"])
        data = {
            name: np.asarray(metrics[name])
            for name in metrics.array_keys()
            if name != "index"
        }
        return pd.DataFrame(data, index=index)

    @functools.cache
    def quality_metrics_dict(self, probe: str) -> dict:
        if self._analyzer_metadata(probe):
            return dict(
                self._analyzer_extension(probe, "quality_metrics").attrs["params"]
            )
        return self.read_json(
            self.get_correct_path(
                self.postprocessed(probe=probe), "quality_metrics", "params.json"
            )
        )

    @functools.cache
    def postprocessed_params_dict(self, probe: str) -> dict:
        if self._analyzer_metadata(probe):
            return dict(self.sorting_analyzer(probe).attrs.get("settings", {}))
        return self.read_json(
            self.get_correct_path(self.postprocessed(probe=probe), "params.json")
        )

    @functools.cache
    def quality_metrics_df(self, probe: str) -> pd.DataFrame:
        if self._analyzer_metadata(probe):
            return self._analyzer_metrics_df(probe, "quality_metrics")
        return self.read_csv(
            self.get_correct_path(
                self.postprocessed(probe=probe), "quality_metrics", "metrics.csv"
            )
        )

    @functools.cache
    def template_metrics_dict(self, probe: str) -> dict:
        if self._analyzer_metadata(probe):
            return dict(
                self._analyzer_extension(probe, "template_metrics").attrs["params"]
            )
        return self.read_json(
            self.get_correct_path(
                self.postprocessed(probe=probe), "template_metrics", "params.json"
            )
        )

    @functools.cache
    def template_metrics_df(self, probe: str) -> pd.DataFrame:
        if self._analyzer_metadata(probe):
            return self._analyzer_metrics_df(probe, "template_metrics")
        return self.read_csv(
            self.get_correct_path(
                self.postprocessed(probe=probe), "template_metrics", "metrics.csv"
            )
        )

    def templates_average(self, probe: str) -> npt.NDArray[np.floating]:
        logger.debug("Loading average templates for %s - typically ~200 MB", probe)
        if self._analyzer_metadata(probe):
            return np.asarray(self._analyzer_extension(probe, "templates")["average"])
        return np.load(
            io.BytesIO(
                self.get_correct_path(
                    self.postprocessed(probe=probe), "templates_average.npy"
                ).read_bytes()
            )
        )

    def templates_std(self, probe: str) -> npt.NDArray[np.floating]:
        logger.debug("Loading template standard deviations for %s", probe)
        if self._analyzer_metadata(probe):
            return np.asarray(self._analyzer_extension(probe, "templates")["std"])
        return np.load(
            io.BytesIO(
                self.get_correct_path(
                    self.postprocessed(probe=probe), "templates_std.npy"
                ).read_bytes()
            )
        )

    @functools.cache
    def sorting_cached(self, probe: str) -> dict[str, npt.NDArray]:
        if not self.is_pre_v0_99:
            raise AttributeError("sorting_cached.npz not used for SpikeInterface>=0.99")
        return np.load(
            io.BytesIO(
                self.get_correct_path(
                    self.curated(probe), "sorting_cached.npz"
                ).read_bytes()
            ),
            allow_pickle=True,
        )

    @functools.cache
    def provenance(self, probe: str) -> dict:
        return self.read_json(
            self.get_correct_path(self.curated(probe), "provenance.json")
        )

    @functools.cache
    def sparsity(self, probe: str) -> dict:
        if self._analyzer_metadata(probe):
            analyzer = self.sorting_analyzer(probe)
            mask = np.asarray(analyzer["sparsity_mask"])
            unit_ids = np.asarray(analyzer["sorting/unit_ids"])
            channel_ids = self.recording_attributes_json(probe)["channel_ids"]
            return {
                "unit_id_to_channel_ids": {
                    unit_id.item() if hasattr(unit_id, "item") else unit_id: [
                        channel_ids[index] for index in np.flatnonzero(mask[unit_index])
                    ]
                    for unit_index, unit_id in enumerate(unit_ids)
                },
                "channel_ids": list(channel_ids),
                "unit_ids": unit_ids.tolist(),
            }
        return self.read_json(
            self.get_correct_path(self.postprocessed(probe=probe), "sparsity.json")
        )

    @functools.cache
    def numpysorting_info(self, probe: str) -> dict:
        if self.is_pre_v0_99:
            raise AttributeError(
                "numpysorting_info.json not used for SpikeInterface<0.99"
            )
        return self.read_json(
            self.get_correct_path(self.curated(probe), "numpysorting_info.json")
        )

    @functools.cache
    def spikes_npy(self, probe: str) -> npt.NDArray[np.floating]:
        """format: array[(sample_index, unit_index, segment_index), ...]"""
        if self.is_pre_v0_99:
            raise AttributeError("spikes.npy not used for SpikeInterface<0.99")
        if self._analyzer_metadata(probe):
            analyzer = self.sorting_analyzer(probe)
            if analyzer["sorting"].attrs["num_segments"] > 1:
                raise AttributeError("num_segments > 1 not supported yet")
            sample_indexes = np.asarray(analyzer["sorting/spikes/sample_index"])
            unit_indexes = np.asarray(analyzer["sorting/spikes/unit_index"])
            return np.column_stack(
                (sample_indexes, unit_indexes, np.zeros_like(sample_indexes))
            )
        if self.numpysorting_info(probe)["num_segments"] > 1:
            raise AttributeError("num_segments > 1 not supported yet")
        return np.load(
            io.BytesIO(
                self.get_correct_path(self.curated(probe), "spikes.npy").read_bytes()
            )
        )

    @functools.cache
    def spike_indexes(self, probe: str) -> npt.NDArray[np.floating]:
        if self._analyzer_metadata(probe):
            original = np.asarray(
                self.sorting_analyzer(probe)["sorting/spikes/sample_index"]
            )
        elif self.is_pre_v0_99:
            original = self.sorting_cached(probe)["spike_indexes_seg0"]
        else:
            spikes = self.spikes_npy(probe)
            original = (
                cast(Any, spikes)["sample_index"]
                if spikes.dtype.names is not None
                else spikes[:, 0]
            )
        return original

    @functools.cache
    def unit_indexes(self, probe: str) -> npt.NDArray[np.int64]:
        if self._analyzer_metadata(probe):
            original = np.asarray(
                self.sorting_analyzer(probe)["sorting/spikes/unit_index"]
            )
        elif self.is_pre_v0_99:
            original = self.sorting_cached(probe)["spike_labels_seg0"]
        else:
            spikes = self.spikes_npy(probe)
            original = (
                cast(Any, spikes)["unit_index"]
                if spikes.dtype.names is not None
                else spikes[:, 1]
            )
        return original.astype(np.int64, copy=False)

    @functools.cache
    def cluster_indexes(self, probe: str) -> npt.NDArray[np.int64]:
        return self.original_cluster_id(probe)[self.unit_indexes(probe)]

    def device_indices_in_nwb_units(
        self, probe: str | npc_session.ProbeRecord
    ) -> npt.NDArray:
        probe = npc_session.ProbeRecord(probe)
        devices = np.array(
            [
                npc_session.ProbeRecord(device)
                for device in self.nwb_zarr["units/device_name"][:]
            ]
        )

        return np.argwhere(devices == probe).squeeze()

    def get_nwb_units_device_property(self, metric: str, probe: str) -> npt.NDArray:
        return self.nwb_zarr[f"units/{metric}"][
            self.device_indices_in_nwb_units(probe)
        ].squeeze()

    @functools.cache
    def original_cluster_id(self, probe: str) -> npt.NDArray[np.int64]:
        """Array of cluster IDs, one per unit in unique('unit_indexes')"""
        if self._analyzer_metadata(probe):
            return self._sorting_property(probe, "original_cluster_id").astype(
                np.int64, copy=False
            )
        if self.is_nextflow_pipeline:
            return self.get_nwb_units_device_property("ks_unit_id", probe).astype(
                np.int64
            )

        with contextlib.suppress(FileNotFoundError):
            return np.load(
                io.BytesIO(
                    self.get_correct_path(
                        self.curated(probe),
                        "properties",
                        "original_cluster_id.npy",
                    ).read_bytes()
                )
            ).astype(np.int64, copy=False)

        if self.is_pre_v0_99:
            # TODO! verify this is correct
            return self.sorting_cached(probe)["unit_ids"]

        raise ValueError(
            f"Unknown format of sorted output for SI {self.version=}, "
            f"{self.is_nextflow_pipeline=}"
        )

    @npc_io.cached_property
    def nwb_path(self) -> upath.UPath:
        if not self.is_nextflow_pipeline:
            raise ValueError("NWB not part of output from stand alone capsule")

        assert self.root is not None
        return next((self.root / "nwb").glob("*.nwb"))

    @npc_io.cached_property
    def nwb_zarr(self) -> zarr.hierarchy.Group:
        return zarr.open(
            self.nwb_path.fs.get_mapper(self.nwb_path.path),
            mode="r",
        )

    @npc_io.cached_property
    def nwb_file(self):
        """Returns pynwb.NWBFile, provided pynwb and hdmf-zarr are installed"""
        try:
            import hdmf_zarr
        except ImportError:
            raise ImportError(
                "hdmf-zarr is required to output NWBFile object from SpikeInterface pipeline output. Please install with `npc-ephys[nwb]`"
            ) from None
        return hdmf_zarr.NWBZarrIO(path=self.nwb_path.as_posix(), mode="r").read()

    @functools.cache
    def _sorting_property(self, probe: str, name: str) -> npt.NDArray:
        if self._analyzer_metadata(probe):
            return np.asarray(
                self.sorting_analyzer(probe)[f"sorting/properties/{name}"]
            )
        return np.load(
            io.BytesIO(
                self.get_correct_path(
                    self.curated(probe), "properties", f"{name}.npy"
                ).read_bytes()
            ),
            allow_pickle=True,
        )

    @functools.cache
    def default_qc(self, probe: str) -> npt.NDArray[np.bool_]:
        return self._sorting_property(probe, "default_qc").astype(np.bool_, copy=False)

    @functools.cache
    def decoder_probability(self, probe: str) -> npt.NDArray[np.floating]:
        return self._sorting_property(probe, "decoder_probability")

    @functools.cache
    def decoder_label(self, probe: str) -> npt.NDArray[np.str_]:
        return self._sorting_property(probe, "decoder_label")

    @functools.cache
    def spike_amplitudes(self, probe: str) -> tuple[npt.NDArray[np.floating], ...]:
        """array of amplitudes for each unit in order of unique('unit_indexes')"""
        if self._analyzer_metadata(probe):
            spike_amplitudes = np.asarray(
                self._analyzer_extension(probe, "spike_amplitudes")["amplitudes"]
            )
        else:
            spike_amplitudes = np.load(
                io.BytesIO(
                    self.get_correct_path(
                        self.postprocessed(probe=probe),
                        "spike_amplitudes",
                        "amplitude_segment_0.npy",
                    ).read_bytes()
                )
            )
        unit_indexes = self.unit_indexes(probe)
        spike_amplitudes_by_unit: list[npt.NDArray[np.floating]] = []
        for index in sorted(np.unique(unit_indexes)):
            spike_amplitudes_by_unit.append(spike_amplitudes[unit_indexes == index])
        return tuple(spike_amplitudes_by_unit)

    @functools.cache
    def unit_locations(self, probe: str) -> npt.NDArray[np.floating]:
        if self._analyzer_metadata(probe):
            return np.asarray(
                self._analyzer_extension(probe, "unit_locations")["unit_locations"]
            )
        return np.load(
            io.BytesIO(
                self.get_correct_path(
                    self.postprocessed(probe=probe),
                    "unit_locations",
                    "unit_locations.npy",
                ).read_bytes()
            )
        )

    @functools.cache
    def sorting_json(self, probe: str) -> dict:
        if self._analyzer_metadata(probe):
            return dict(self.sorting_analyzer(probe)["sorting"].attrs)
        return self.read_json(
            self.get_correct_path(self.postprocessed(probe=probe), "sorting.json")
        )

    @functools.cache
    def recording_attributes_json(self, probe: str) -> dict:
        if self._analyzer_metadata(probe):
            return dict(
                self.sorting_analyzer(probe)["recording_info"].attrs[
                    "recording_attributes"
                ]
            )
        return self.read_json(
            self.get_correct_path(
                self.postprocessed(probe=probe),
                "recording_info",
                "recording_attributes.json",
            )
        )

    @functools.cache
    def sparse_channel_indices(self, probe: str) -> tuple[int, ...]:
        """SpikeInterface output from sorting used to consistently store channels
        as 1-indexed integers: "AP1", ..., "AP384".

        As of late 2025 and `libraryName="Neuropix-PXI" libraryVersion="0.7.0"` (in settings.xml),
        Open Ephys started representing channels as 0-indexed in the GUI, and storing them this way
        in the structure.oebin files (channels in settings.xml have always been 0-indexed).
        Spikeinterface gets channel IDs from the structure.oebin files, so newer recordings
        may have 0-indexed channel IDs in the recording_attributes.json, but older ones will have 1-indexed channel IDs.

        This method returns the 0-indexed *integers* for each probe
        recorded, for use in indexing into the electrode table.
        """
        channel_indices = sorted(
            int("".join(i for i in str(id_) if i.isdigit()))
            for id_ in self.recording_attributes_json(probe)["channel_ids"]
        )
        if self._analyzer_metadata(probe):
            is_one_indexed = False
        elif channel_indices == list(range(len(channel_indices))):
            is_one_indexed = False
        elif channel_indices == list(range(1, len(channel_indices) + 1)):
            is_one_indexed = True
        else:
            is_one_indexed = self.settings_xml.neuropix_pxi_version < "0.7.0"
        if is_one_indexed:
            channel_indices = [i - 1 for i in channel_indices]
        return tuple(channel_indices)

    @functools.cache
    def electrode_locations_xy(self, probe: str) -> npt.NDArray[np.floating]:
        return np.array(
            self.sorting_json(probe)["annotations"]["__sorting_info__"]["recording"][
                "properties"
            ]["location"]
        )


# Backwards compatibility for callers written when only KS2.5 output was supported.
SpikeInterfaceKS25Data = SpikeInterfaceData


if __name__ == "__main__":
    from npc_ephys import testmod

    testmod()
