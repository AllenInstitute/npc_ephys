from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import upath

import npc_ephys


def test_import_package() -> None:
    pass


def _get_test_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: Iterable[tuple[float, int]],
) -> npc_ephys.EphysTimingInfoOnSync:
    sync = SimpleNamespace(
        total_seconds=52.0,
        get_rising_edges=lambda line, units: np.array([12.0, 43.0]),
        get_falling_edges=lambda line, units: np.array([12.01, 43.01]),
    )
    monkeypatch.setattr(npc_ephys.openephys.npc_sync, "get_sync_data", lambda _: sync)

    device = npc_ephys.EphysDeviceInfo(
        name="Neuropix-PXI-100.ProbeD-AP",
        continuous=upath.UPath(tmp_path / "continuous"),
        events=upath.UPath(tmp_path / "events"),
        ttl=upath.UPath(tmp_path / "events" / "TTL"),
        compressed=None,
        ttl_sample_numbers=np.array([round(time * 1000) for time, _ in events]),
        ttl_states=np.array([state for _, state in events]),
    )
    timing_on_pxi = npc_ephys.EphysTimingInfoOnPXI(
        device=device,
        start_sample=0,
        sampling_rate=1000.0,
    )

    return next(npc_ephys.get_ephys_timing_on_sync("sync", devices=(timing_on_pxi,)))


def test_get_ephys_timing_on_sync_ignores_non_barcode_ttl_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    barcode_events = (
        (3.0, 1),
        (3.01, -1),
        (34.0, 1),
        (34.01, -1),
    )
    noisy_events = tuple(
        event
        for time in np.arange(0.1, 47.0, 0.1)
        for event in ((time, 2), (time + 0.01, -2))
    )

    timing = _get_test_timing(
        monkeypatch, tmp_path, sorted((*barcode_events, *noisy_events))
    )

    assert timing.start_time == 9.0
    assert timing.sampling_rate == 1000.0


def test_get_ephys_timing_on_sync_preserves_single_ttl_channel_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    timing = _get_test_timing(
        monkeypatch,
        tmp_path,
        (
            (3.0, 1),
            (3.01, -1),
            (34.0, 1),
            (34.01, -1),
            (47.0, 1),
            (47.01, -1),
        ),
    )

    assert timing.start_time == 9.0
    assert timing.sampling_rate == 1000.0
