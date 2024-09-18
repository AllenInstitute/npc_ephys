from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable

import npc_lims
import npc_session
import numpy as np
import numpy.typing as npt
import upath
import zarr
import zarr.core

import npc_ephys.openephys

logger = logging.getLogger(__name__)

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
    LFP_subsampled_files: tuple[upath.UPath, ...],
    temporal_factor: int = 2,
) -> LFP | None:
    probe_LFP_subsampled_file = tuple(
        file for file in LFP_subsampled_files if probe in str(file)
    )
    if not probe_LFP_subsampled_file:
        logger.warning(
            f"No LFP subsampled results for probe {probe}. Potentially skipped due to no surface channel index. Check codeocean"
        )

        return None

    probe_LFP_zarr_file = probe_LFP_subsampled_file[0]
    probe_LFP_zarr = zarr.open(probe_LFP_zarr_file, mode="r")
    probe_LFP_traces, probe_LFP_channels = (
        probe_LFP_zarr["traces_seg0"],
        probe_LFP_zarr["channel_ids"],
    )

    probe_LFP_aligned_timestamps = (
        np.arange(probe_LFP_traces.shape[0])
        / (device_timing.sampling_rate / temporal_factor)
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
    >>> device_timing_on_sync = npc_ephys.openephys.get_ephys_timing_on_sync(npc_lims.get_h5_sync_from_s3('674562_2023-10-03'), npc_lims.get_recording_dirs_experiment_path_from_s3('674562_2023-10-03'), only_devices_including='ProbeA')
    >>> LFP_probeA = get_LFP_subsampled_results('674562_2023-10-03', device_timing_on_sync)
    >>> LFP_probeA[0].traces.shape
    (8923628, 96)
    >>> LFP_probeA[0].timestamps.shape
    (8923628,)
    >>> len(LFP_probeA[0].channel_ids)
    96
    >>> LFP_probeA[0].probe
    'ProbeA'
    >>> LFP_probeA[0].sampling_rate
    1250.0031940878919
    """
    LFP_subsampled_results = []
    session = npc_session.SessionRecord(session)
    session_LFP_subsampled_files = npc_lims.get_LFP_subsampling_paths_from_s3(session)

    devices_LFP_timing = tuple(
        timing for timing in device_timing_on_sync if timing.device.name.endswith("LFP")
    )

    for device_timing in devices_LFP_timing:
        probe = f"Probe{npc_session.ProbeRecord(device_timing.device.name)}"

        probe_LFP_subsampled_result = _get_LFP_probe_result(
            probe, device_timing, session_LFP_subsampled_files
        )

        if probe_LFP_subsampled_result is None:
            continue

        LFP_subsampled_results.append(probe_LFP_subsampled_result)

    return tuple(LFP_subsampled_results)


if __name__ == "__main__":
    from npc_ephys import testmod

    testmod()
