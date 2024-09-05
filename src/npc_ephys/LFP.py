from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import npc_lims
import npc_session
import numpy as np
import numpy.typing as npt
import upath
import zarr
import zarr.core

import npc_ephys.openephys

LFP_SUBSAMPLED_SAMPLING_RATE = 1250


@dataclasses.dataclass()
class LFP:
    """Data class with LFP traces n samples x m channels, LFP aligned timestamps to sync size n timestamps,
    LFP channels selected size m channels output from LFP subsampling capsule"""

    traces: zarr.core.Array  # samples x channels
    timestamps: npt.NDArray[np.float64]
    channel_ids: tuple[int, ...]  # 0 indexed, output from spike interface is 1 indexed
    probe: str
    sampling_rate: float

    def __post_init__(self):
        # check shapes of attributes
        if self.traces.shape[0] != self.timestamps.shape[0]:
            raise ValueError(
                "Mismatch in time dimension between traces and aligned timestamps"
            )

        if self.traces.shape[1] != len(self.channel_ids):
            raise ValueError(
                "Mismatch in channel dimension between traces and selected channels"
            )


def _get_LFP_subsampled_output(
    LFP_subsampled_files: upath.UPath,
) -> tuple[zarr.core.Array, ...]:
    LFP_time_samples_path = tuple(LFP_subsampled_files.glob("*_samples.zarr"))
    if not LFP_time_samples_path:
        raise FileNotFoundError("No LFP time samples found. Check codeocean")

    LFP_traces_path = tuple(LFP_subsampled_files.glob("*_subsampled.zarr"))
    if not LFP_traces_path:
        raise FileNotFoundError("No LFP traces found. Check codeocean")

    LFP_channels_path = tuple(LFP_subsampled_files.glob("*_channels.zarr"))
    if not LFP_channels_path:
        raise FileNotFoundError("No LFP channels found. Check codeocean")

    LFP_traces = zarr.open(LFP_traces_path[0], mode="r")["traces_seg0"]
    LFP_time_samples = zarr.open(LFP_time_samples_path[0], mode="r")
    LFP_channels = zarr.open(LFP_channels_path[0], mode="r")

    return LFP_traces, LFP_time_samples, LFP_channels


def _get_LFP_channel_ids(LFP_channels: zarr.core.Array | list[str]) -> tuple[int, ...]:
    """
    >>> ids = ['LFP1', 'LFP4', 'LFP380']
    >>> _get_LFP_channel_ids(ids)
    (0, 3, 379)
    """
    # spike interface channel ids are 1-indexed
    channel_ids = sorted(
        [
            int("".join(i for i in channel if i.isdigit())) - 1
            for channel in LFP_channels[:]
        ]
    )
    assert (
        m := min(channel_ids)
    ) >= 0, (
        f"Expected all channel_ids from SpikeInterface to be 1-indexed: min = {m + 1}"
    )

    return tuple(channel_ids)


def _get_LFP_probe_result(
    probe: str,
    device_timing: npc_ephys.openephys.EphysTimingInfo,
    LFP_subsampled_directories: tuple[upath.UPath, ...],
    temporal_factor: int = 2,
) -> LFP:
    probe_LFP_subsampled_directory = tuple(
        directory
        for directory in LFP_subsampled_directories
        if directory.is_dir() and probe in str(directory)
    )
    if not probe_LFP_subsampled_directory:
        raise FileNotFoundError(
            f"No LFP subsampled results for probe {probe}. Check codeocean"
        )

    probe_LFP_directory = probe_LFP_subsampled_directory[0]
    probe_LFP_traces, probe_LFP_samples, probe_LFP_channels = (
        _get_LFP_subsampled_output(probe_LFP_directory)
    )
    # time samples from output are evenly spaced at 1/1250, apply scale: (actual sampling rate / temporal factor) / 1250
    probe_LFP_aligned_timestamps = (
        probe_LFP_samples[:]
        / (
            (device_timing.sampling_rate / temporal_factor)
            / LFP_SUBSAMPLED_SAMPLING_RATE
        )
    ) + device_timing.start_time
    probe_LFP_channel_ids = _get_LFP_channel_ids(probe_LFP_channels)

    return LFP(
        traces=probe_LFP_traces,
        timestamps=probe_LFP_aligned_timestamps,
        channel_ids=probe_LFP_channel_ids,
        probe=probe,
        sampling_rate=device_timing.sampling_rate / temporal_factor,
    )


def get_LFP_subsampled_results(
    session: str | npc_session.SessionRecord,
    device_timing_on_sync: Iterable[npc_ephys.openephys.EphysTimingInfo],
) -> tuple[LFP, ...]:
    """
    Gets the LFP subsampled output for the session. Returns an object for each probe with subsampled traces, aligned timestamps, channel ids,
    probe name, and sampling rate
    >>> device_timing_on_sync = npc_ephys.openephys.get_ephys_timing_on_sync(npc_lims.get_h5_sync_from_s3('674562_2023-10-05'), npc_lims.get_recording_dirs_experiment_path_from_s3('674562_2023-10-05'), only_devices_including='ProbeA')
    >>> LFP_probeA = get_LFP_subsampled_results('674562_2023-10-05', device_timing_on_sync)
    >>> LFP_probeA[0].traces.shape
    (8813052, 96)
    >>> LFP_probeA[0].timestamps.shape
    (8813052,)
    >>> len(LFP_probeA[0].channel_ids)
    96
    >>> LFP_probeA[0].probe
    'ProbeA'
    >>> LFP_probeA[0].sampling_rate
    1250.0025845802068
    """
    LFP_subsampled_results = []
    session = npc_session.SessionRecord(session)
    session_LFP_subsampled_directories = npc_lims.get_LFP_subsampling_paths_from_s3(
        session
    )

    devices_LFP_timing = tuple(
        timing for timing in device_timing_on_sync if timing.device.name.endswith("LFP")
    )

    for device_timing in devices_LFP_timing:
        probe = f"Probe{npc_session.ProbeRecord(device_timing.device.name)}"
        probe_LFP_subsampled_result = _get_LFP_probe_result(
            probe, device_timing, session_LFP_subsampled_directories
        )
        LFP_subsampled_results.append(probe_LFP_subsampled_result)

    return tuple(LFP_subsampled_results)


if __name__ == "__main__":
    from npc_ephys import testmod

    testmod()
