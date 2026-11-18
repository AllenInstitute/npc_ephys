"""Tools for working with Open Ephys raw data files."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import re
from collections.abc import Generator, Iterable, Sequence
from typing import Any, Literal, Protocol

import npc_io
import npc_sync
import numpy as np
import numpy.typing as npt
import upath
import zarr

import npc_ephys.barcodes
import npc_ephys.sample_drops
import npc_ephys.settings_xml

logger = logging.getLogger(__name__)

DEFAULT_PROBES = "ABCDEF"
TIMESTAMPS_ADJUSTED_FLAG_FILENAME = "TIMESTAMPS_ADJUSTED.flag"
BARCODE_TTL_CHANNEL = 1


def get_sync_messages_data(
    sync_messages_path: npc_io.PathLike,
) -> dict[str, dict[Literal["start", "rate"], int]]:
    """
    Start Time for Neuropix-PXI (107) - ProbeA-AP @ 30000 Hz: 210069564
    Start Time for Neuropix-PXI (107) - ProbeA-LFP @ 2500 Hz: 17505797
    Start Time for NI-DAQmx (109) - PXI-6133 @ 30000 Hz: 210265001

    >>> path = 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1/sync_messages.txt'
    >>> dirname_to_sample = get_sync_messages_data(path)
    >>> dirname_to_sample['NI-DAQmx-105.PXI-6133']
    {'start': 257417001, 'rate': 30000}
    """

    def label(line) -> str:
        return "".join(
            line.split("Start Time for ")[-1]
            .split(" @")[0]
            .replace(") - ", ".")
            .replace(" (", "-")
        )

    def start(line) -> int:
        return int(line.strip(" ").split("Hz:")[-1])

    def rate(line) -> int:
        return int(line.split("@ ")[-1].split(" Hz")[0])

    return {
        label(line): {
            "start": start(line),
            "rate": rate(line),
        }
        for line in npc_io.from_pathlike(sync_messages_path)
        .read_text()
        .splitlines()[1:]
    }


@dataclasses.dataclass(frozen=True, eq=True, unsafe_hash=True)
class EphysDeviceInfo:
    name: str
    continuous: upath.UPath
    """Abs path to device's folder within raw data/continuous/"""
    events: upath.UPath
    """Abs path to device's folder within raw data/events/"""
    ttl: upath.UPath
    """Abs path to device's folder within events/"""
    compressed: upath.UPath | None
    """Abs path to device's zarr storage within ecephys_compressed/, or None if not found"""
    ttl_sample_numbers: npt.NDArray
    """Sample numbers on open ephys clock, after subtracting first sample reported in
    sync_messages.txt"""
    ttl_states: npt.NDArray
    """Signed TTL channel IDs from ttl/states.npy; the sign indicates edge polarity."""

    @npc_io.cached_property
    def num_samples(self) -> int:
        return get_ephys_data(self.continuous.parent.parent, device=self).shape[0]

    @npc_io.cached_property
    def is_sync_adjusted(self) -> bool:
        """Check if the device's timestamps have been adjusted to the sync
        clock. If one has been adjusted and the other hasn't, raise an AssertionError.
        """
        continuous = (self.continuous / TIMESTAMPS_ADJUSTED_FLAG_FILENAME).exists()
        events = (self.events / "TTL" / TIMESTAMPS_ADJUSTED_FLAG_FILENAME).exists()
        if continuous != events:
            not_adjusted = "continuous" if not continuous else "events"
            raise AssertionError(
                f"Continuous and events timestamps adjusted flags do not match in {self.continuous.parent} - one is adjusted, the other is not: {not_adjusted}"
            )
        return continuous


class EphysTimingInfo(Protocol):
    @property
    def device(self) -> EphysDeviceInfo: ...

    @property
    def sampling_rate(self) -> float: ...

    """Samples per second"""

    @property
    def start_time(self) -> float: ...

    """Sec, relative to start of experiment"""

    @property
    def stop_time(self) -> float: ...

    """Sec, relative to start of experiment"""


@dataclasses.dataclass(frozen=True, eq=True, unsafe_hash=True)
class EphysTimingInfoOnPXI:
    device: EphysDeviceInfo
    """Info with paths"""
    start_sample: int
    """Start sample reported in sync_messages.txt"""
    sampling_rate: float
    """Nominal sample rate reported in sync_messages.txt"""

    start_time: float = 0
    """No sync available, so start_time is meaningless (set to 0)"""

    @property
    def stop_time(self) -> float:
        """Simple length of recording using nominal sample rate"""
        return self.start_time + (self.device.num_samples / self.sampling_rate)


@dataclasses.dataclass(frozen=True, eq=True, unsafe_hash=True)
class EphysTimingInfoOnSync:
    """Timing of one Open Ephys device on the sync clock.

    Sync time of recorded sample index k (0 = first recorded sample):

        t(k) = start_time + (k + L(k)) / sampling_rate

    where L(k) is the number of samples lost (dropped by the acquisition system) at or before k,
    from `drops`. Without drops this is the usual affine map. Use `to_sync_time` rather than
    `start_time` and `sampling_rate` directly.

    Devices that fail QC have `status == 'fail'`, a `reason`, and nan `sampling_rate` and
    `start_time`.
    """

    device: EphysDeviceInfo
    """Info with paths"""
    sampling_rate: float
    """Sample rate assessed on the sync clock"""
    start_time: float
    """First sample time (sec) relative to the start of the sync clock"""
    drops: tuple[tuple[float, float], ...] = ()
    """(sample index of the first sample after a gap, number of samples lost), sorted"""
    invalid_intervals: tuple[tuple[float, float], ...] = ()
    """Sync-time intervals (sec) containing a loss of unknown position (timing inside them is
    uncertain by up to the loss)"""
    status: Literal["pass", "fail"] = "pass"
    reason: str = ""
    qc: tuple[tuple[str, Any], ...] = ()
    """QC metrics as (name, value) pairs - see `qc_metrics`"""

    @property
    def is_valid(self) -> bool:
        return self.status == "pass"

    @property
    def qc_metrics(self) -> dict[str, Any]:
        return dict(self.qc)

    @property
    def total_samples_lost(self) -> float:
        return float(sum(n for _, n in self.drops))

    def to_sync_time(self, sample_indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Sync time (sec) of recorded sample indices (relative to the first recorded sample;
        fractional indices allowed)."""
        k = np.asarray(sample_indices, dtype=float)
        if not self.drops:
            return self.start_time + k / self.sampling_rate
        pos, n = (np.array(v, dtype=float) for v in zip(*self.drops))
        return self.start_time + (
            k + npc_ephys.sample_drops.cumulative_loss(pos, n, k)
        ) / self.sampling_rate

    @property
    def stop_time(self) -> float:
        """Last sample time (sec) relative to the start of the sync clock"""
        return float(self.to_sync_time(self.device.num_samples))


def read_array_range_from_npy(
    path: npc_io.PathLike, _range: int | slice
) -> npt.NDArray:
    """Read specific range without downloading entire array. For 1-D array only, currently."""
    if not isinstance(_range, slice):
        _range = slice(_range, _range + 1)
    path = npc_io.from_pathlike(path)
    ver_major = int.from_bytes(path.fs.read_bytes(path, start=6, end=7), "little")
    header_len_stop = 10 if ver_major == 1 else 12
    header_len = int.from_bytes(
        path.fs.read_bytes(path, start=8, end=header_len_stop), "little"
    )
    array_start = header_len_stop + header_len
    header_bytes = path.fs.read_bytes(path, start=header_len_stop, end=array_start)
    header = eval(header_bytes.decode("utf-8").strip("\n").strip())
    assert len(header["shape"]) == 1, "Currently supporting 1-D array only"
    dtype = header["descr"]
    num_bytes_per_value = np.dtype(dtype).itemsize
    return np.frombuffer(
        path.fs.read_bytes(
            path,
            start=array_start + _range.start * num_bytes_per_value,
            end=array_start + _range.stop * num_bytes_per_value,
        ),
        dtype=dtype,
    )


def get_ephys_timing_on_pxi(
    recording_dirs: Iterable[npc_io.PathLike],
    only_devices_including: str | None = None,
) -> Generator[EphysTimingInfoOnPXI, None, None]:
    """
    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> next(get_ephys_timing_on_pxi(path)).sampling_rate
    30000.0
    """
    if not isinstance(recording_dirs, Iterable):
        recording_dirs = (recording_dirs,)

    for recording_dir in recording_dirs:
        recording_dir = npc_io.from_pathlike(recording_dir)
        device_to_sync_messages_data = get_sync_messages_data(
            recording_dir / "sync_messages.txt"
        )  # includes name of each input device used (probe, nidaq)
        for device in device_to_sync_messages_data:
            if (
                only_devices_including
                and only_devices_including.lower() not in device.lower()
            ):
                continue
            continuous = recording_dir / "continuous" / device
            if not continuous.exists():
                continue
            events = recording_dir / "events" / device
            ttl = next(events.glob("TTL*"))

            first_sample_from_continuous_sample_numbers = read_array_range_from_npy(
                continuous / "sample_numbers.npy", 0
            ).item()
            first_sample_from_sync_messages = device_to_sync_messages_data[device][
                "start"
            ]
            if (
                first_sample_from_continuous_sample_numbers
                != first_sample_from_sync_messages
            ):
                logger.debug(
                    f"{first_sample_from_sync_messages =} != {first_sample_from_continuous_sample_numbers =}. This may be due to Record Nodes being out-of-sync (green indicator in GUI). Using value from sample_numbers.npy"
                )
            first_sample_on_ephys_clock = first_sample_from_continuous_sample_numbers

            sampling_rate = device_to_sync_messages_data[device]["rate"]
            ttl_sample_numbers = (
                np.load(io.BytesIO((ttl / "sample_numbers.npy").read_bytes()))
                - first_sample_on_ephys_clock
            )
            ttl_states = np.load(io.BytesIO((ttl / "states.npy").read_bytes()))
            try:
                compressed = clipped_path_to_compressed(continuous)
            except ValueError:
                logger.info(f"No compressed data found for {continuous}")
                compressed = None
            yield EphysTimingInfoOnPXI(
                device=EphysDeviceInfo(
                    name=device,
                    continuous=continuous,
                    events=events,
                    ttl=ttl,
                    compressed=compressed,
                    ttl_sample_numbers=ttl_sample_numbers,
                    ttl_states=ttl_states,
                ),
                start_sample=first_sample_on_ephys_clock,
                sampling_rate=float(sampling_rate),
            )


def clipped_path_to_compressed(path: npc_io.PathLike) -> upath.UPath:
    """
    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeA-AP')
    >>> clipped_path_to_compressed(path).as_posix()
    's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_compressed/experiment1_Record Node 102#Neuropix-PXI-100.ProbeA-AP.zarr'
    """
    path = npc_io.from_pathlike(path)
    if "ecephys_clipped" not in path.as_posix():
        raise ValueError(f'Expected path to contain "ecephys_clipped", got {path}')
    experiment_re = re.search(r".*(experiment\d+)", path.as_posix())
    record_node_re = re.search(r".*(Record Node \d+)", path.as_posix())
    # /recording?/ isn't part of compressed path: assumes 1 recording per folder, or concats multiple recordings
    if not (experiment_re and record_node_re):
        raise ValueError(f"Could not parse experiment and record node from {path}")
    experiment, record_node = experiment_re.groups()[0], record_node_re.groups()[0]
    device_re = re.match(
        rf".*{record_node}/{experiment}/recording\d+/[^/]+/(.*)", path.as_posix()
    )
    if not device_re:
        raise ValueError(f"Could not parse device from {path}")
    compressed_name = f"{experiment}_{record_node}#{device_re.groups()[0]}.zarr"
    root_path = next(p for p in path.parents if p.name == "ecephys_clipped")
    # cannot construct S3Path de novo from a string including `#`, but we can return
    # the actual path that exists
    compressed_path = next(
        (
            path
            for path in root_path.with_name("ecephys_compressed").iterdir()
            if path.name == compressed_name
        ),
        None,
    )
    if compressed_path is None:
        raise FileNotFoundError(f"Could not find {compressed_name} in {root_path}")
    return compressed_path


def open_zarr(path: npc_io.PathLike, mode: Literal["r"] = "r") -> zarr.Group:
    """Open a zarr store at a local or cloud path.

    zarr doesn't accept `UPath` objects directly (zarr 3, and zarr 2 with
    universal-pathlib>=0.2, where `UPath` is no longer `os.PathLike`), so we pass an
    fsspec mapper, which works with zarr 2 and 3, and preserves storage options.
    """
    path = npc_io.from_pathlike(path)
    return zarr.open(path.fs.get_mapper(path.path), mode=mode)


def get_ephys_data(
    *recording_dirs: npc_io.PathLike,
    device: str | EphysDeviceInfo,
) -> npt.NDArray[np.int16]:
    if not isinstance(device, EphysDeviceInfo):
        device = next(
            get_ephys_timing_on_pxi(recording_dirs, only_devices_including=device)
        ).device
    if device.compressed:
        data = open_zarr(device.compressed)
        return data["traces_seg0"]
    else:
        device_metadata = next(
            (
                _
                for _ in get_merged_oebin_file(
                    next(npc_io.from_pathlike(p).glob("*.oebin"))
                    for p in recording_dirs
                )["continuous"]
                if device.name in _["folder_name"]
            ),
            None,
        )
        if device_metadata is None:
            raise ValueError(
                f"Could not find device metadata for {device.name}: looked for `structure.oebin` files in {recording_dirs}"
            )
        num_channels: int = device_metadata["num_channels"]

        dat: npt.NDArray[np.int16]
        if device.continuous.protocol in ("", "file"):
            # local file we can memory-map
            dat = np.memmap(
                device.continuous / "continuous.dat", dtype="int16", mode="r"
            )
        else:
            logger.warning(
                f"Reading entirety of uncompressed OpenEphys data from {device.continuous}. If you only need part of this data, consider using `read_array_range_from_npy` with the path instead."
            )
            dat = np.frombuffer(
                (device.continuous / "continuous.dat").read_bytes(), dtype=np.int16
            )
        return np.reshape(dat, (int(dat.size / num_channels), -1))


def get_pxi_nidaq_data(
    *recording_dirs: npc_io.PathLike,
    device_name: str | None = None,
) -> npt.NDArray[np.int16]:
    """
    -channel_idx: 0-indexed
    - if device_name not specified, first and only (assumed) NI-DAQ will be used

    ```
    speaker_channel, mic_channel = 1, 3
    ```

    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> get_pxi_nidaq_data(path).shape
    (142823472, 8)
    """
    if device_name:
        device = next(
            get_ephys_timing_on_pxi(recording_dirs, only_devices_including=device_name)
        ).device
    else:
        device = get_pxi_nidaq_info(recording_dirs).device
    return get_ephys_data(*recording_dirs, device=device)


def get_pxi_nidaq_info(
    recording_dir: Iterable[npc_io.PathLike],
) -> EphysTimingInfoOnPXI:
    """Convenience method to get timing info for one NI-DAQmx device.

    - if more than one device is found:
      - if the device names match, they're assumed to be duplicate recordings
      across record nodes and the first is returned (with a warning logged)
      - if the device names differ, a ValueError is raised

    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> get_pxi_nidaq_info(path).device.ttl.parent.name
    'NI-DAQmx-105.PXI-6133'
    """
    info = tuple(
        t
        for t in get_ephys_timing_on_pxi(
            recording_dir, only_devices_including="NI-DAQmx-"
        )
    )
    if not info:
        raise FileNotFoundError(
            f"No */continuous/NI-DAQmx-*/ dir found in {recording_dir = }"
        )
    if info and len(info) != 1:
        warning_msg = f"Expected a single NI-DAQmx folder to exist, but found: {[d.device.continuous for d in info]}."
        advice_msg = "Supply a specific `recording_dir` to narrow down the search."
        if len(set(i.device.name for i in info)) == 1:
            logger.warning(
                f"{warning_msg} Names are the same, so we'll assume they're duplicates and return the first one found. {advice_msg}"
            )
        else:
            raise ValueError(
                f"{warning_msg} Names differ, so we don't know which to use. {advice_msg}"
            )
    return info[0]


SYNC_INVALID_INTERVAL_PADDING_S = 2.0
"""Barcode edges this close to a sync invalid interval (replayed buffers) are not used."""


def _sync_barcodes(sync: npc_sync.SyncDataset) -> tuple[npt.NDArray, npt.NDArray]:
    on = sync.get_rising_edges("barcode_ephys", units="seconds")
    off = sync.get_falling_edges("barcode_ephys", units="seconds")
    invalid = tuple(getattr(sync, "invalid_intervals", ()) or ())
    if invalid:
        pad = SYNC_INVALID_INTERVAL_PADDING_S
        bad = [(a - pad, b + pad) for a, b in invalid]

        def inside(t):
            m = np.zeros(len(t), bool)
            for a, b in bad:
                m |= (t >= a) & (t < b)
            return m

        on, off = on[~inside(on)], off[~inside(off)]
    return npc_ephys.barcodes.extract_barcodes_from_times(
        on_times=on, off_times=off, total_time_on_line=sync.total_seconds
    )


def _device_barcodes(info: EphysTimingInfoOnPXI) -> tuple[npt.NDArray, npt.NDArray]:
    """Barcode start times in samples (relative to first recorded sample) and values."""
    fs = info.sampling_rate
    d = info.device
    t, ids = npc_ephys.barcodes.extract_barcodes_from_times(
        on_times=d.ttl_sample_numbers[d.ttl_states == BARCODE_TTL_CHANNEL] / fs,
        off_times=d.ttl_sample_numbers[d.ttl_states == -BARCODE_TTL_CHANNEL] / fs,
        total_time_on_line=(
            d.ttl_sample_numbers[-1] / fs if len(d.ttl_sample_numbers) > 0 else 0.0
        ),
    )
    return np.asarray(t, dtype=float) * fs, np.asarray(ids)


_DROP_MESSAGE_CACHE: dict[str, npt.NDArray[np.int64]] = {}


def _drop_messages(recording_dir: upath.UPath) -> npt.NDArray[np.int64]:
    key = recording_dir.as_posix()
    if key not in _DROP_MESSAGE_CACHE:
        _DROP_MESSAGE_CACHE[key] = npc_ephys.sample_drops.read_drop_messages(
            recording_dir
        )
    return _DROP_MESSAGE_CACHE[key]


def _ap_to_lfp_ratio(lfp_info: EphysTimingInfoOnPXI) -> float:
    rates = get_sync_messages_data(
        lfp_info.device.continuous.parent.parent / "sync_messages.txt"
    )
    ap = lfp_info.device.name.replace("-LFP", "-AP")
    ap_rate = rates[ap]["rate"] if ap in rates else 30_000
    return float(ap_rate) / lfp_info.sampling_rate


def get_logged_drops(
    info: EphysTimingInfoOnPXI,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str]:
    """Logged drops for a Neuropixels device, in the device's own samples relative to its first
    recorded sample: (positions, sizes, note). Empty for NI-DAQ devices and recordings without
    drop messages. Timestamp-reset messages (not data loss) are excluded.

    LFP positions/sizes are derived from the AP sample numbers in the log: the LFP sample number
    is the AP sample number / (AP rate / LFP rate) (12 for NP1.0).
    """
    empty = np.zeros(0), np.zeros(0)
    m = re.search(r"\.(Probe[A-Z])-(AP|LFP)$", info.device.name)
    if m is None:
        return *empty, ""
    recording_dir = info.device.continuous.parent.parent
    msgs = _drop_messages(recording_dir)
    if msgs.size == 0:
        return *empty, ""
    xml = npc_ephys.sample_drops._settings_xml_for_recording(recording_dir)
    slot_port = (
        npc_ephys.sample_drops.get_probe_slot_port(xml, m.group(1)) if xml else None
    )
    if slot_port is None:
        raise LookupError(
            f"drop messages found in {recording_dir} but slot/port of {m.group(1)} is unknown"
        )
    sel = (msgs[:, 0] == slot_port[0]) & (msgs[:, 1] == slot_port[1])
    N, J = msgs[sel, 2], msgs[sel, 3]
    reset = npc_ephys.sample_drops.is_timestamp_reset(N, J)
    N, J = N[~reset], J[~reset]
    sizes = npc_ephys.sample_drops.ticks_to_lost_samples(J)
    note = f"slot {slot_port[0]} port {slot_port[1]}: {len(N)} logged drops, {int(reset.sum())} timestamp resets ignored"
    if m.group(2) == "AP":
        return (N - info.start_sample).astype(float), sizes, note
    ratio = _ap_to_lfp_ratio(info)
    return N / ratio - info.start_sample, sizes / ratio, note


def get_ephys_timing_on_sync(
    sync: npc_sync.SyncPathOrDataset,
    recording_dirs: Iterable[npc_io.PathLike] | None = None,
    devices: Iterable[EphysTimingInfoOnPXI] | None = None,
    only_devices_including: str | None = None,
    max_residual_ms: dict[str, float] | None = None,
    step_threshold_ms: dict[str, float] | None = None,
) -> tuple[EphysTimingInfoOnSync, ...]:
    """
    One of `recording_dir` or `devices` must be supplied:
        - all devices in `recording_dir` will be returned
        - or just those specified in `devices`
        (use `get_ephys_timing_on_pxi()` to get a filtered iterable of devices
        in a recording_dir)

    Every device is returned, each with its own `status` ('pass'/'fail') and `reason`. Failed
    devices have nan `sampling_rate`/`start_time`: callers should exclude them (per device).

    Timing model (see `npc_ephys.sample_drops`): sync time = start + (k + L(k)) / rate, fitted to
    all barcodes whose value is unique on both lines, with samples lost by the acquisition system
    (logged in the MessageCenter, refined against barcodes) added back. On a probe's LFP stream
    the refined drop table from its AP stream is reused.

    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> sync = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5')
    >>> timing_info = get_ephys_timing_on_sync(sync, path)[0]
    >>> timing_info.device.name, timing_info.status, round(timing_info.sampling_rate), round(timing_info.start_time, 2)
    ('Neuropix-PXI-100.ProbeA-AP', 'pass', 30000, 20.08)
    """
    if not (recording_dirs or devices):
        raise ValueError("Must specify recording_dir or devices")
    thresholds = {
        **npc_ephys.sample_drops.DEFAULT_MAX_RESIDUAL_MS,
        **(max_residual_ms or {}),
    }
    step_thresholds = {
        **npc_ephys.sample_drops.DEFAULT_STEP_THRESHOLD_MS,
        **(step_threshold_ms or {}),
    }

    sync = npc_sync.get_sync_data(sync)
    sync_barcode_times, sync_barcode_ids = _sync_barcodes(sync)

    if devices and not isinstance(devices, Iterable):
        devices = (devices,)
    if recording_dirs and (
        isinstance(recording_dirs, str) or not isinstance(recording_dirs, Iterable)
    ):
        recording_dirs = (recording_dirs,)
    if recording_dirs and not devices:
        devices = get_ephys_timing_on_pxi(recording_dirs, only_devices_including)
    assert devices is not None
    infos = tuple(devices)

    stream_type = npc_ephys.sample_drops.stream_type
    results: dict[int, EphysTimingInfoOnSync] = {}
    ap_fits: dict[
        str, tuple[EphysTimingInfoOnPXI, npc_ephys.sample_drops.TimingFit]
    ] = {}
    # AP streams first, so each LFP stream can reuse its probe's refined AP drop table
    order = sorted(
        range(len(infos)),
        key=lambda i: {"AP": 0, "LFP": 1}.get(stream_type(infos[i].device.name), 2),
    )
    for i in order:
        info = infos[i]
        name = info.device.name
        kind = stream_type(name)
        try:
            pos, sizes, note = get_logged_drops(info)
        except LookupError as exc:
            results[i] = EphysTimingInfoOnSync(
                device=info.device,
                sampling_rate=np.nan,
                start_time=np.nan,
                status="fail",
                reason=str(exc),
            )
            continue
        refine = True
        inherited: tuple[tuple[float, float], ...] = ()
        ap_name = name.replace("-LFP", "-AP")
        if (
            kind == "LFP"
            and ap_name in ap_fits
            and ap_fits[ap_name][1].status != "pass"
            and not ap_fits[ap_name][1].reason.startswith("too few")
        ):
            # same probe, same clock, same lost samples: an unexplained AP failure applies
            # to the LFP stream too (AP barcodes are 12x more precise)
            results[i] = EphysTimingInfoOnSync(
                device=info.device,
                sampling_rate=np.nan,
                start_time=np.nan,
                status="fail",
                reason=f"{ap_name} failed timing QC: {ap_fits[ap_name][1].reason}",
            )
            continue
        if (
            kind == "LFP"
            and ap_name in ap_fits
            and ap_fits[ap_name][1].status == "pass"
        ):
            ap_info, ap_fit = ap_fits[ap_name]
            ratio = _ap_to_lfp_ratio(info)
            pos = (ap_fit.drop_positions + ap_info.start_sample) / ratio - info.start_sample
            sizes = ap_fit.drop_sizes / ratio
            refine = False
            inherited = ap_fit.invalid_intervals
            note += f"; drop table from {ap_name}"
        dev_samples, dev_ids = _device_barcodes(info)
        fit = npc_ephys.sample_drops.fit_device_timing(
            sync_barcode_times,
            sync_barcode_ids,
            dev_samples,
            dev_ids,
            info.sampling_rate,
            pos,
            sizes,
            refine=refine,
            max_residual_ms=thresholds[kind],
            step_threshold_ms=step_thresholds[kind],
        )
        if kind == "AP":
            ap_fits[name] = (info, fit)
        metrics = dict(fit.metrics, note=note)
        if fit.status != "pass":
            logger.warning(f"{name}: timing QC failed: {fit.reason}")
            results[i] = EphysTimingInfoOnSync(
                device=info.device,
                sampling_rate=np.nan,
                start_time=np.nan,
                status="fail",
                reason=fit.reason,
                qc=tuple(metrics.items()),
            )
            continue
        drops = tuple(
            (float(p), float(n))
            for p, n in zip(fit.drop_positions, fit.drop_sizes)
            if n != 0
        )
        invalid = tuple(sorted(set(fit.invalid_intervals) | set(inherited)))
        results[i] = EphysTimingInfoOnSync(
            device=info.device,
            sampling_rate=fit.sampling_rate,
            start_time=fit.start_time,
            drops=drops,
            invalid_intervals=invalid,
            status="pass",
            reason="",
            qc=tuple(metrics.items()),
        )
    return tuple(results[i] for i in range(len(infos)))


def is_new_ephys_folder(path: npc_io.PathLike) -> bool:
    """Look for any hallmarks of a v0.6.x Open Ephys recording in path or subfolders."""
    path = npc_io.from_pathlike(path)
    globs = (
        "Record Node*",
        "structure*.oebin",
    )
    components = tuple(_.replace("*", "") for _ in globs)

    if any(_.lower() in path.as_posix().lower() for _ in components):
        return True

    for glob in globs:
        if next(path.rglob(glob), None):
            return True
    return False


def is_complete_ephys_folder(path: npc_io.PathLike) -> bool:
    """Look for all hallmarks of a complete v0.6.x Open Ephys recording."""
    # TODO use structure.oebin to check for completeness
    path = npc_io.from_pathlike(path)
    if not is_new_ephys_folder(path):
        return False
    for glob in ("continuous.dat", "spike_times.npy", "spike_clusters.npy"):
        if not next(path.rglob(glob), None):
            logger.debug(f"Could not find {glob} in {path}")
            return False
    return True


def is_valid_ephys_folder(
    path: npc_io.PathLike,
    min_size_gb: int | float | None = None,
) -> bool:
    """Check a single dir of raw data for size and v0.6.x+ Open Ephys."""
    path = npc_io.from_pathlike(path)
    if not path.is_dir():
        return False
    if not is_new_ephys_folder(path):
        return False
    if (
        min_size_gb is not None and npc_io.get_size_gb(path) < min_size_gb * 1024**3
    ):  # GB
        return False
    return True


def get_ephys_root(path: npc_io.PathLike) -> upath.UPath:
    """Returns the parent of the first `Record Node *` found in the supplied
    path.

    >>> get_ephys_root(upath.UPath('A:/test/Record Node 0/Record Node test')).as_posix()
    'A:/test'
    """
    path = npc_io.from_pathlike(path)
    if "Record Node" not in path.as_posix():
        raise ValueError(
            f"Could not find 'Record Node' in {path} - is this a valid raw ephys path?"
        )
    return next(
        p.parent for p in path.parents if "Record Node".lower() in p.name.lower()
    )


def get_filtered_ephys_paths_relative_to_record_node_parents(
    toplevel_ephys_path: npc_io.PathLike,
) -> Generator[tuple[upath.UPath, upath.UPath], None, None]:
    """For restructuring the raw ephys data in a session folder, we want to
    discard superfluous recording folders and only keep the "good" data, but
    with the overall directory structure relative to `Record Node*` folders intact.

    Supply a top-level path that contains `Record Node *`
    subfolders somewhere in its tree.

    Returns a generator akin to `path.rglob('Record Node*')` except:
    - only paths associated with the "good" ephys data are returned (with some
    assumptions made about the ephys session)
    - a tuple of two paths is supplied:
        - `(abs_path, abs_path.relative_to(record_node.parent))`
        ie. path[1] should always start with `Record Node *`

    Expectation is:
    - `/npexp_path/ephys_*/Record Node */ recording1 / ... / continuous.dat`

    ie.
    - one recording per `Record Node *` folder
    - one subfolder in `npexp_path/` per `Record Node *` folder (following the
    pipeline `*_probeABC` / `*_probeDEF` convention for recordings split across two
    drives)


    Many folders have:
    - `/npexp_path/Record Node */ recording*/ ...`

    ie.
    - multiple recording folders per `Record Node *` folder
    - typically there's one proper `recording` folder: the rest are short,
    aborted recordings during setup

    We filter out the superfluous small recording folders here.

    Some folders (Templeton) have:
    - `/npexp_path/Record Node */ ...`

    ie.
    - contents of expected ephys subfolders directly deposited in npexp_path

    """
    toplevel_ephys_path = npc_io.from_pathlike(toplevel_ephys_path)
    record_nodes = toplevel_ephys_path.rglob("Record Node*")

    for record_node in record_nodes:
        superfluous_recording_dirs = tuple(
            _.parent for _ in get_superfluous_oebin_paths(record_node)
        )
        logger.debug(
            f"Found {len(superfluous_recording_dirs)} superfluous recording dirs to exclude: {superfluous_recording_dirs}"
        )

        for abs_path in record_node.rglob("*"):
            is_superfluous_path = any(
                _ in abs_path.parents for _ in superfluous_recording_dirs
            )

            if is_superfluous_path:
                continue

            yield abs_path, abs_path.relative_to(record_node.parent)


def get_raw_ephys_subfolders(
    path: npc_io.PathLike, min_size_gb: int | float | None = None
) -> tuple[upath.UPath, ...]:
    """
    Return raw ephys recording folders, defined as the root that Open Ephys
    records to, e.g. `A:/1233245678_366122_20220618_probeABC`.
    """
    path = npc_io.from_pathlike(path)

    subfolders = set()

    for f in upath.UPath(path).rglob("continuous.dat"):
        if any(
            k in f.as_posix().lower()
            for k in [
                "sorted",
                "extracted",
                "curated",
            ]
        ):
            # skip sorted/extracted folders
            continue

        subfolders.add(get_ephys_root(f))

    if min_size_gb is not None:
        subfolders = {
            _ for _ in subfolders if npc_io.get_size(_) < min_size_gb * 1024**3
        }

    return tuple(sorted(subfolders, key=lambda s: str(s)))


# - If we have probeABC and probeDEF raw data folders, each one has an oebin file:
#     we'll need to merge the oebin files and the data folders to create a single session
#     that can be processed in parallel
def get_single_oebin_path(path: npc_io.PathLike) -> upath.UPath:
    """Get the path to a single structure.oebin file in a folder of raw ephys data.

    - There's one structure.oebin per `recording*` folder
    - Raw data folders may contain multiple `recording*` folders
    - Datajoint expects only one structure.oebin file per Session for sorting
    - If we have multiple `recording*` folders, we assume that there's one
        good folder - the largest - plus some small dummy / accidental recordings
    """
    path = npc_io.from_pathlike(path)
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")

    oebin_paths = tuple(path.rglob("structure*.oebin"))

    if not oebin_paths:
        raise FileNotFoundError(f"No structure.oebin file found in {path}")

    if len(oebin_paths) == 1:
        return oebin_paths[0]

    oebin_parents = (_.parent for _ in oebin_paths)
    dir_sizes = tuple(npc_io.get_size(_) for _ in oebin_parents)
    return oebin_paths[dir_sizes.index(max(dir_sizes))]


def get_superfluous_oebin_paths(path: npc_io.PathLike) -> tuple[upath.UPath, ...]:
    """Get the paths to any oebin files in `recording*` folders that are not
    the largest in a folder of raw ephys data.

    Companion to `get_single_oebin_path`.
    """
    path = npc_io.from_pathlike(path)
    all_oebin_paths = tuple(path.rglob("structure*.oebin"))

    if len(all_oebin_paths) == 1:
        return ()

    return tuple(set(all_oebin_paths) - {(get_single_oebin_path(path))})


def assert_xml_files_match(*path: npc_io.PathLike) -> None:
    """Check that all xml files are identical, as they should be for
    recordings split across multiple locations e.g. A:/*_probeABC, B:/*_probeDEF
    or raise an error.

    Update: xml files on two nodes can be created at slightly different times, so their `date`
    fields may differ. Everything else should be identical.
    """
    paths = tuple(npc_io.from_pathlike(p) for p in path)
    if not all(s == ".xml" for s in [p.suffix for p in paths]):
        raise ValueError(f"Not all paths are XML files: {paths}")
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(
            f"Not all paths are files, or they do not exist: {paths}"
        )
    if not npc_io.checksums_match(*paths):
        data = [npc_ephys.settings_xml.get_settings_xml_data(p) for p in paths]
        if not all(d == data[0] for d in data):
            raise AssertionError(f"XML files do not match: {paths}")


def get_merged_oebin_file(
    paths: Iterable[npc_io.PathLike], exclude_probe_names: Sequence[str] | None = None
) -> dict[Literal["continuous", "events", "spikes"], list[dict[str, Any]]]:
    """Merge two or more structure.oebin files into one.

    For recordings split across multiple locations e.g. A:/*_probeABC,
    B:/*_probeDEF
    - if items in the oebin files have 'folder_name' values that match any
    entries in `exclude_probe_names`, they will be removed from the merged oebin
    """
    if not isinstance(paths, Iterable):
        paths = tuple(paths)
    oebin_paths = tuple(npc_io.from_pathlike(p) for p in paths)
    if len(oebin_paths) == 1:
        return get_oebin_data(oebin_paths[0])

    # ensure oebin files can be merged - if from the same exp they will have the same settings.xml file
    if any(p.suffix != ".oebin" for p in oebin_paths):
        raise ValueError(f"Not all paths are .oebin files: {oebin_paths}")
    assert_xml_files_match(
        *[p / "settings.xml" for p in [o.parent.parent.parent for o in oebin_paths]]
    )

    logger.debug(f"Creating merged oebin file from {oebin_paths}")
    merged_oebin: dict = {}
    for oebin_path in sorted(oebin_paths):
        oebin_data = get_oebin_data(oebin_path)

        for key in oebin_data:
            # skip if already in merged oebin
            if merged_oebin.get(key) == oebin_data[key]:
                continue

            # 'continuous', 'events', 'spikes' are lists, which we want to concatenate across files
            if isinstance(oebin_data[key], list):
                for item in oebin_data[key]:
                    # skip if already in merged oebin
                    if item in merged_oebin.get(key, []):
                        continue

                    # skip probes in excl list (ie. not inserted)
                    if exclude_probe_names and any(
                        p.lower() in item.get("folder_name", "").lower()
                        for p in exclude_probe_names
                    ):
                        continue

                    # insert in merged oebin
                    merged_oebin.setdefault(key, []).append(item)

    if not merged_oebin:
        raise ValueError("No data found in structure.oebin files")
    return merged_oebin


def get_oebin_data(
    path: npc_io.PathLike,
) -> dict[Literal["continuous", "events", "spikes"], list[dict[str, Any]]]:
    return json.loads(npc_io.from_pathlike(path).read_text())


def validate_recording_folder(
    recording_path: npc_io.PathLike,
    sync_path_or_dataset: npc_sync.SyncPathOrDataset | None = None,
) -> None:
    sync = (
        npc_sync.get_sync_data(sync_path_or_dataset) if sync_path_or_dataset else None
    )
    recording_path = npc_io.from_pathlike(recording_path)
    logging.debug(f"Validating ephys data in {recording_path}")

    for device in get_oebin_data(recording_path / "structure.oebin")["continuous"]:
        device_name = device["folder_name"].strip("/")
        try:
            info = next(
                get_ephys_timing_on_pxi(
                    recording_path, only_devices_including=device_name
                )
            )
        except StopIteration:
            raise AssertionError(
                f"Could not find {device_name} data in {recording_path}"
            )
        if sync:
            try:
                (timing,) = get_ephys_timing_on_sync(sync, devices=(info,))
                if not timing.is_valid:
                    logger.warning(
                        f"{info.device.name} failed timing QC: {timing.reason}"
                    )
            except Exception as exc:
                raise AssertionError(
                    f"Could not validate {info.device.name} with sync"
                ) from exc
        logging.debug(
            f"Validated {info.device.name} {'with' if sync else 'without'} sync"
        )


def validate_ephys(
    root_paths: npc_io.PathLike | Iterable[npc_io.PathLike],
    sync_path_or_dataset: npc_sync.SyncPathOrDataset | None = None,
    ignore_small_folders: bool = True,
) -> None:
    """
    >>> root = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped')
    >>> sync = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5')
    >>> validate_ephys(root, sync)
    """
    for root in npc_io.iterable_from_pathlikes(root_paths):
        oebin_paths = (
            (get_single_oebin_path(root),)
            if ignore_small_folders
            else root.rglob("structure.oebin")
        )
        for p in oebin_paths:
            validate_recording_folder(
                p.parent, sync_path_or_dataset=sync_path_or_dataset
            )
        logging.info(f"Validated ephys data in {root}")


@dataclasses.dataclass
class SyncAdjustedTimestamps:
    timing: EphysTimingInfoOnSync
    """Info with paths"""
    continuous: npt.ArrayLike
    """Timestamps in seconds, relative to the start of the sync clock"""
    events: npt.ArrayLike
    """Timestamps in seconds, relative to the start of the sync clock"""


def get_adjusted_timestamps(
    ephys_timing: EphysTimingInfoOnSync,
) -> SyncAdjustedTimestamps:
    """Adjust timestamps from ephys clock to sync clock using the provided
    `EphysTimingInfoOnSync` object.

    >>> sync = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5')
    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> ephys_timing = get_ephys_timing_on_sync(sync, path)[0]
    >>> t = get_adjusted_timestamps(ephys_timing)
    >>> t.continuous[[0, -1]].round(3)   # within 0.05 ms of the old first/last-barcode map
    array([  20.08 , 4776.217])
    >>> t.events[[0, -1]].round(3)
    array([  41.196, 4779.861])
    """
    if not getattr(ephys_timing, "is_valid", True):
        raise ValueError(
            f"{ephys_timing.device.name} failed timing QC: {ephys_timing.reason}"
        )
    default_continuous = np.arange(
        start=0,
        stop=ephys_timing.device.num_samples,
        dtype=np.float64,
    )
    adjusted_continuous = ephys_timing.to_sync_time(default_continuous)
    adjusted_events = ephys_timing.to_sync_time(ephys_timing.device.ttl_sample_numbers)
    return SyncAdjustedTimestamps(
        timing=ephys_timing,
        continuous=adjusted_continuous,
        events=adjusted_events,
    )


def overwrite_timestamps(
    ephys_timing: Iterable[EphysTimingInfoOnSync],
) -> None:
    """Overwrite timestamps in the continuous and events folders with
    sync-adjusted versions.

    - this allows spike-sorting output to be used directly
    """
    if not isinstance(ephys_timing, Iterable):
        ephys_timing = (ephys_timing,)

    def overwrite(path: upath.UPath, data: npt.ArrayLike) -> None:
        assert path.exists(), f"{path} does not exist - check the path construction"
        flag = path.parent / TIMESTAMPS_ADJUSTED_FLAG_FILENAME
        if flag.exists():
            raise FileExistsError(
                f"{flag} exists: timestamps have already been adjusted"
            )
        logger.debug(f"Overwriting {path.as_posix()}")
        if path.is_symlink():
            path.unlink()  # otherwise the symlink target will be overwritten
        np.save(path.as_posix(), data)
        logger.debug(f"Creating flag file {flag.as_posix()}")
        flag.write_text(
            "Original timestamps.npy file has been overwritten with sync-adjusted timestamps"
        )

    for info in ephys_timing:
        adjusted = get_adjusted_timestamps(info)
        overwrite(info.device.continuous / "timestamps.npy", adjusted.continuous)
        overwrite(info.device.events / "TTL" / "timestamps.npy", adjusted.events)


if __name__ == "__main__":
    from npc_ephys import testmod

    testmod()
