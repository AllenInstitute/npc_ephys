from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import re
from collections.abc import Iterable

import npc_io
import npc_session
import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm

import npc_ephys.openephys
import npc_ephys.spikeinterface

logger = logging.getLogger(__name__)


def bin_spike_times(
    spike_times: npt.NDArray[np.float64], bin_interval: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if not spike_times.any():
        raise ValueError("spike_times provided is empty")
    spike_times = np.concatenate(spike_times, axis=0)  # flatten array
    return np.histogram(
        spike_times,
        bins=np.arange(
            np.floor(np.min(spike_times)), np.ceil(np.max(spike_times)), bin_interval
        ),
    )


def get_aligned_spike_times(
    spike_times: npt.NDArray[np.floating],
    device_timing_on_sync: npc_ephys.openephys.EphysTimingInfo,
) -> npt.NDArray[np.float64]:
    return (
        spike_times / device_timing_on_sync.sampling_rate
    ) + device_timing_on_sync.start_time


@dataclasses.dataclass()
class AmplitudesWaveformsChannels:
    """Data class, each entry a sequence with len == N units"""

    amplitudes: tuple[np.floating, ...]
    templates_mean: tuple[npt.NDArray[np.floating], ...]
    templates_sd: tuple[npt.NDArray[np.floating], ...]
    peak_channels: tuple[np.intp, ...]
    channels: tuple[tuple[np.intp, ...], ...]

    def __post_init__(self):
        # check all attrs same length
        if not all(
            len(getattr(self, attr))
            == len(getattr(self, next(iter(self.__annotations__.keys()))))
            for attr in self.__annotations__
        ):
            raise ValueError("All attributes must have same length")


def get_amplitudes_waveforms_channels(
    spike_interface_data: npc_ephys.spikeinterface.SpikeInterfaceData,
    electrode_group_name: str,
) -> AmplitudesWaveformsChannels:
    unit_amplitudes: list[np.floating] = []
    templates_mean: list[npt.NDArray[np.floating]] = []
    templates_sd: list[npt.NDArray[np.floating]] = []
    peak_channels: list[np.intp] = []
    channels: list[tuple[np.intp, ...]] = []

    sparse_channel_indices = spike_interface_data.sparse_channel_indices(
        electrode_group_name
    )
    if (
        spike_interface_data.is_nextflow_pipeline
        and not spike_interface_data.is_analyzer
    ):
        # faster data access
        _templates_mean = spike_interface_data.get_nwb_units_device_property(
            "waveform_mean", electrode_group_name
        )[..., sparse_channel_indices]
        _templates_sd = spike_interface_data.get_nwb_units_device_property(
            "waveform_sd", electrode_group_name
        )[..., sparse_channel_indices]
    else:
        _templates_mean = spike_interface_data.templates_average(electrode_group_name)
        _templates_sd = spike_interface_data.templates_std(electrode_group_name)

    sparse_channel_indices = spike_interface_data.sparse_channel_indices(
        electrode_group_name
    )
    assert (
        len(sparse_channel_indices) == _templates_mean.shape[2]
    ), f"Expected {len(sparse_channel_indices)=} channels to match {_templates_mean.shape[2]=}"
    for unit_index in range(_templates_mean.shape[0]):
        _mean = _templates_mean[unit_index, :, :]
        pk_to_pk = np.max(_mean, axis=0) - np.min(
            _mean, axis=0
        )  # same method as Allen ecephys pipeline:
        # https://github.com/bjhardcastle/ecephys_spike_sorting/blob/7e567a6fc3fd2fc0eedef750b83b8b8a0d469544/ecephys_spike_sorting/modules/mean_waveforms/extract_waveforms.py#L87
        peak_channel = sparse_channel_indices[(m := np.argmax(pk_to_pk))]
        unit_amplitudes.append(pk_to_pk[m].item())
        _sd = _templates_sd[unit_index, :, :]

        idx_with_data = np.where(_mean.any(axis=0))[0]
        very_sparse_channel_indices = np.array(sparse_channel_indices)[idx_with_data]
        templates_mean.append(_mean[:, idx_with_data])
        templates_sd.append(_sd[:, idx_with_data])
        peak_channels.append(np.intp(peak_channel))
        logger.debug(f"very_sparse_channel_indices: {very_sparse_channel_indices}")
        channels.append(tuple(very_sparse_channel_indices))

    return AmplitudesWaveformsChannels(
        amplitudes=tuple(unit_amplitudes),
        templates_mean=tuple(templates_mean),
        templates_sd=tuple(templates_sd),
        peak_channels=tuple(peak_channels),
        channels=tuple(channels),
    )


def get_waveform_sd(
    templates_std: npt.NDArray[np.floating],
) -> list[npt.NDArray[np.floating]]:
    unit_templates_std: list[npt.NDArray[np.floating]] = []
    for unit_index in range(templates_std.shape[0]):
        template = templates_std[unit_index, :, :]
        unit_templates_std.append(template)

    return unit_templates_std


def get_units_x_spike_times(
    spike_times: npt.NDArray[np.float64],
    unit_indexes: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """
    Note: index in tuple != unit_index (may be gaps in unique(unit_indexes)))
    >>> spike_times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> unit_indexes = np.array([3, 0, 1, 1, 0])    # {0, 1, 3} - must handle gaps
    >>> get_units_x_spike_times(spike_times, unit_indexes)
    (array([0.2, 0.5]), array([0.3, 0.4]), array([0.1]))
    """
    if len(spike_times) != len(unit_indexes):
        raise ValueError(
            "spike_times and unit_indexes must be same length (values should correspond)"
        )
    units = sorted(np.unique(unit_indexes))  # may have gaps
    units_x_spike_times = tuple(spike_times[unit_indexes == unit] for unit in units)
    assert (
        spike_times[0]
        == units_x_spike_times[np.argwhere(units == unit_indexes[0]).item()][0]
    ), "Interpretation of unit_indexes/spike_times is faulty"
    return units_x_spike_times


def _device_helper(
    device_timing_on_sync: npc_ephys.openephys.EphysTimingInfo,
    spike_interface_data: npc_ephys.spikeinterface.SpikeInterfaceData,
    include_waveform_arrays: bool,
    sorting_name: str,
) -> pd.DataFrame:
    electrode_group_name = npc_session.ProbeRecord(
        device_timing_on_sync.device.name
    ).name
    spike_interface_data.electrode_locations_xy(sorting_name)

    df_device_metrics = spike_interface_data.quality_metrics_df(sorting_name).merge(
        spike_interface_data.template_metrics_df(sorting_name),
        left_index=True,
        right_index=True,
    )

    df_device_metrics["electrode_group_name"] = [str(electrode_group_name)] * len(
        df_device_metrics
    )
    if match := re.search(r"_group(\d+)$", sorting_name):
        df_device_metrics["shank"] = int(match.group(1))

    awc = get_amplitudes_waveforms_channels(
        spike_interface_data=spike_interface_data,
        electrode_group_name=sorting_name,
    )

    df_device_metrics["peak_channel"] = awc.peak_channels
    cluster_id = df_device_metrics.index.to_list()
    assert np.array_equal(
        cluster_id, spike_interface_data.original_cluster_id(sorting_name)
    ), "cluster-ids from npy file do not match index column in metrics.csv"
    df_device_metrics["cluster_id"] = df_device_metrics.index.to_list()
    df_device_metrics["default_qc"] = spike_interface_data.default_qc(sorting_name)
    df_device_metrics["decoder_label"] = spike_interface_data.decoder_label(
        sorting_name
    )
    df_device_metrics["decoder_probability"] = spike_interface_data.decoder_probability(
        sorting_name
    )
    df_device_metrics["amplitude"] = awc.amplitudes
    if include_waveform_arrays:
        df_device_metrics["waveform_mean"] = awc.templates_mean
        df_device_metrics["waveform_sd"] = awc.templates_sd
        df_device_metrics["channels"] = awc.channels

    spike_times = spike_interface_data.spike_indexes(sorting_name)
    if not device_timing_on_sync.device.is_sync_adjusted:
        spike_times_aligned = get_aligned_spike_times(
            spike_times,
            device_timing_on_sync,
        )
    else:
        spike_times_aligned = spike_times

    unit_indexes = spike_interface_data.unit_indexes(sorting_name)
    units_x_spike_times = get_units_x_spike_times(
        spike_times=spike_times_aligned,
        unit_indexes=unit_indexes,
    )
    assert len(units_x_spike_times) == len(
        df_device_metrics
    ), "Mismatch number of units in spike_times and metrics.csv"
    assert np.array_equal(
        df_device_metrics["num_spikes"].array,
        [len(unit) for unit in units_x_spike_times],
    ), "Mismatch between rows in spike_times and metrics.csv"
    df_device_metrics["spike_times"] = units_x_spike_times
    df_device_metrics["spike_amplitudes"] = spike_interface_data.spike_amplitudes(
        sorting_name
    )
    assert all(
        len(df_device_metrics["spike_amplitudes"].iloc[i]) == len(spike_times)
        for i, spike_times in enumerate(units_x_spike_times)
    ), "Mismatch between spike_times and spike_amplitudes"

    return df_device_metrics


def _is_ap_device_timing(
    timing: npc_ephys.openephys.EphysTimingInfo,
) -> bool:
    """Include legacy ``-AP`` streams and newer unsuffixed 30 kHz probe streams."""
    name = timing.device.name
    if name.endswith("-LFP"):
        return False
    try:
        npc_session.ProbeRecord(name)
    except ValueError:
        return False
    return name.endswith("-AP") or timing.sampling_rate >= 10_000


def make_units_table_from_spike_interface(
    session_or_spikeinterface_data_or_path: (
        str
        | npc_session.SessionRecord
        | npc_io.PathLike
        | npc_ephys.spikeinterface.SpikeInterfaceData
    ),
    device_timing_on_sync: Iterable[npc_ephys.openephys.EphysTimingInfo],
    include_waveform_arrays: bool = False,
) -> pd.DataFrame:
    """Make a units table from legacy or analyzer SpikeInterface output.

    >>> import npc_lims
    >>> device_timing_on_sync = npc_ephys.openephys.get_ephys_timing_on_sync(npc_lims.get_h5_sync_from_s3('662892_20230821'), npc_lims.get_recording_dirs_experiment_path_from_s3('662892_20230821'), only_devices_including='ProbeA')
    >>> units = make_units_table_from_spike_interface('662892_20230821', device_timing_on_sync)
    >>> len(units[units['electrode_group_name'] == 'probeA'])
    240
    """
    spike_interface_data = npc_ephys.spikeinterface.get_spikeinterface_data(
        session_or_spikeinterface_data_or_path
    )

    devices_timing = tuple(
        timing for timing in device_timing_on_sync if _is_ap_device_timing(timing)
    )

    task_to_future: dict[tuple[str, str], concurrent.futures.Future[pd.DataFrame]] = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for device_timing in devices_timing:
            sorting_names = spike_interface_data.sorting_names_for_device(
                device_timing.device.name
            )
            if not sorting_names:
                sorting_names = (
                    npc_session.ProbeRecord(device_timing.device.name).name,
                )
            for sorting_name in sorting_names:
                task = (device_timing.device.name, sorting_name)
                task_to_future[task] = executor.submit(
                    _device_helper,
                    device_timing,
                    spike_interface_data,
                    include_waveform_arrays,
                    sorting_name,
                )

        for future in tqdm.tqdm(
            iterable=concurrent.futures.as_completed(task_to_future.values()),
            desc="fetching units",
            unit="sorting",
            total=len(task_to_future),
            ncols=80,
            ascii=False,
        ):
            task = next(k for k, v in task_to_future.items() if v == future)
            device, sorting_name = task
            session = spike_interface_data.session or ""
            try:
                _ = future.result()
            except npc_ephys.spikeinterface.ProbeNotFoundError:
                logger.warning(
                    f"Path to {session}{' ' if session else ''}{device} sorted data not found: likely skipped by SpikeInterface"
                )
                del task_to_future[task]
            except AssertionError as e:
                logger.error(
                    f"{session}{' ' if session else ''}{device} ({sorting_name})"
                )
                raise e from None
            except Exception as e:
                logger.error(
                    f"{session}{' ' if session else ''}{device} ({sorting_name})"
                )
                raise RuntimeError(
                    f"Error fetching units for {session} - see original exception above/below"
                ) from e

    return pd.concat(task_to_future[task].result() for task in sorted(task_to_future))


# Backwards-compatible names from when only Kilosort 2.5 output was supported.
get_amplitudes_waveforms_channels_ks25 = get_amplitudes_waveforms_channels
get_waveform_sd_ks25 = get_waveform_sd
make_units_table_from_spike_interface_ks25 = make_units_table_from_spike_interface


def add_electrode_annotations_to_units(
    units: pd.DataFrame,
    annotated_electrodes: pd.DataFrame,
) -> pd.DataFrame:
    """Join units table with tissuecyte electrode locations table and drop redundant columns."""
    annotation_columns = [
        "group_name",
        "channel",
        "x",
        "y",
        "z",
        "structure",
        "location",
    ]
    if any(col not in annotated_electrodes.columns for col in annotation_columns):
        raise ValueError(
            f"annotated_electrodes must contain all columns {annotation_columns}"
        )
    units = units.merge(
        annotated_electrodes[annotation_columns],
        left_on=["electrode_group_name", "peak_channel"],
        right_on=["group_name", "channel"],
    )
    units.drop(columns=["group_name", "channel"], inplace=True)
    units.rename(
        columns={
            "x": "ccf_ap",
            "y": "ccf_dv",
            "z": "ccf_ml",
        },
        inplace=True,
    )
    return units


def add_global_unit_ids(
    units: pd.DataFrame,
    session: str | npc_session.SessionRecord,
    unit_id_column: str = "unit_id",
) -> pd.DataFrame:
    """Add session, probe letter, optional shank, and cluster ID."""

    def make_unit_id(row: pd.Series) -> str:
        probe = row.electrode_group_name.replace("probe", "")
        shank = (
            f"-{int(row['shank'])}"
            if "shank" in units.columns and pd.notna(row["shank"])
            else ""
        )
        return f"{session}_{probe}{shank}-{row['cluster_id']}"

    units[unit_id_column] = [make_unit_id(row) for _, row in units.iterrows()]
    return units


def good_units(units: pd.DataFrame, qc_column: str = "default_qc") -> pd.DataFrame:
    units = units[:]
    if units[qc_column].dtype != bool:
        raise NotImplementedError(
            f"currently qc_column {qc_column} must be boolean - either don't use this function or add a fix"
        )
    return units[units[qc_column]]


if __name__ == "__main__":
    from npc_ephys import testmod

    testmod()
