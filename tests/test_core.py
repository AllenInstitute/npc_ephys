from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import upath

import npc_ephys


def test_import_package() -> None:
    pass


BARCODE_TIMES = 12.0 + 30.0 * np.arange(12)  # sync times of barcode starts
DEVICE_START_ON_SYNC = 9.0


def _barcode_edges(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Rising/falling edge times (s) of 12 barcodes with distinct random bit patterns.
    All edges on whole milliseconds, so they are exact at a 1 kHz device rate."""
    rng = np.random.default_rng(seed)
    on, off = [], []
    for t0 in BARCODE_TIMES:
        t = t0
        on.append(t)
        off.append(t + 0.030)
        t += 0.030
        for _ in range(12):
            gap, width = rng.integers(5, 40) / 1000, rng.integers(5, 40) / 1000
            on.append(round(t + gap, 3))
            off.append(round(t + gap + width, 3))
            t += gap + width
    return np.array(on), np.array(off)


def _get_test_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_events: Iterable[tuple[float, int]] = (),
) -> npc_ephys.EphysTimingInfoOnSync:
    on, off = _barcode_edges()
    sync = SimpleNamespace(
        total_seconds=BARCODE_TIMES[-1] + 10.0,
        get_rising_edges=lambda line, units: on,
        get_falling_edges=lambda line, units: off,
    )
    monkeypatch.setattr(npc_ephys.openephys.npc_sync, "get_sync_data", lambda _: sync)

    events = sorted(
        [(t - DEVICE_START_ON_SYNC, 1) for t in on]
        + [(t - DEVICE_START_ON_SYNC, -1) for t in off]
        + list(extra_events)
    )
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
    results = npc_ephys.get_ephys_timing_on_sync("sync", devices=(timing_on_pxi,))
    assert isinstance(results, tuple) and len(results) == 1
    return results[0]


def test_get_ephys_timing_on_sync_ignores_non_barcode_ttl_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    noisy_events = tuple(
        event
        for time in np.arange(0.1, 47.0, 0.1)
        for event in ((round(time, 3), 2), (round(time + 0.01, 3), -2))
    )
    timing = _get_test_timing(monkeypatch, tmp_path, noisy_events)
    assert timing.status == "pass", timing.reason
    assert timing.start_time == pytest.approx(DEVICE_START_ON_SYNC, abs=1e-9)
    assert timing.sampling_rate == pytest.approx(1000.0, rel=1e-9)


def test_get_ephys_timing_on_sync_clean_device_is_affine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    timing = _get_test_timing(monkeypatch, tmp_path)
    assert timing.status == "pass", timing.reason
    assert timing.drops == () and timing.invalid_intervals == ()
    k = np.arange(0, 400_000, 997)
    np.testing.assert_allclose(
        timing.to_sync_time(k), timing.start_time + k / timing.sampling_rate
    )
    assert timing.start_time == pytest.approx(DEVICE_START_ON_SYNC, abs=1e-9)
