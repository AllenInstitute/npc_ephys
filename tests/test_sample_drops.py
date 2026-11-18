"""Synthetic tests for the dropped-sample timing model (npc_ephys.sample_drops)."""

from __future__ import annotations

import numpy as np
import pytest

import npc_ephys.sample_drops as sd

FS = 30_000.0
RATE = 30_000.37  # true device rate on sync clock
T0 = 20.0  # sync time of device sample 0


def make(
    duration_s=7000.0,
    drops_true=(),
    seed=0,
    jitter_samples=0.5,
    interval_s=30.0,
):
    """Synthetic barcodes. drops_true: (true sample index, n_lost) on the device's true timeline.
    Returns sync times, ids, device barcode samples (recorded index), drop table in recorded
    indices, and a function true-sync-time(recorded index)."""
    rng = np.random.default_rng(seed)
    s = np.arange(T0 + 5, T0 + duration_s, interval_s) + rng.uniform(0, 1, 1)
    ids = rng.choice(2**31, size=len(s), replace=False)
    k_true = (s - T0) * RATE
    dt = np.array(sorted(drops_true), dtype=float).reshape(-1, 2)
    cum_before = np.concatenate([[0], np.cumsum(dt[:, 1])])[:-1] if len(dt) else np.zeros(0)
    rec_pos = dt[:, 0] - cum_before if len(dt) else np.zeros(0)  # recorded index of first sample after gap
    sizes = dt[:, 1] if len(dt) else np.zeros(0)

    def lost_before_true(k):
        return np.array([dt[dt[:, 0] <= x, 1].sum() for x in np.atleast_1d(k)])

    k_rec = k_true - lost_before_true(k_true) + rng.normal(0, jitter_samples, len(s))

    def true_time(k_rec_query):
        k = np.atleast_1d(np.asarray(k_rec_query, float))
        L = sd.cumulative_loss(rec_pos, sizes, k)
        return T0 + (k + L) / RATE

    return s, ids, k_rec, rec_pos, sizes, true_time


def test_clean_device_is_affine():
    s, ids, k, _, _, true_time = make()
    fit = sd.fit_device_timing(s, ids, k, ids, FS)
    assert fit.status == "pass", fit.reason
    assert fit.drop_positions.size == 0 and fit.invalid_intervals == ()
    assert fit.sampling_rate == pytest.approx(RATE, rel=1e-8)
    assert fit.start_time == pytest.approx(T0, abs=1e-4)
    q = np.linspace(0, 7000 * FS, 1000)
    # identical to the plain affine map
    np.testing.assert_allclose(fit.to_sync_time(q), fit.start_time + q / fit.sampling_rate)
    assert np.abs(fit.to_sync_time(q) - true_time(q)).max() < 5e-5


def test_logged_drops_corrected_and_refined():
    rng = np.random.default_rng(1)
    true_pos = np.sort(rng.uniform(0, 7000 * RATE, 5000))
    n_true = rng.choice([1.0, 9.0, 10.0, 12.0], size=5000)
    s, ids, k, rec_pos, sizes, true_time = make(drops_true=list(zip(true_pos, n_true)))
    logged = sizes + rng.normal(0, 0.4, sizes.size)  # tick quantisation error
    # uncorrected: badly non-affine
    raw = sd.fit_device_timing(s, ids, k, ids, FS, detect_steps=False)
    assert raw.status == "fail"
    fit = sd.fit_device_timing(s, ids, k, ids, FS, rec_pos, logged)
    assert fit.status == "pass", fit.reason
    assert fit.invalid_intervals == ()
    q = np.linspace(k[0], k[-1], 5000)
    err_ms = np.abs(fit.to_sync_time(q) - true_time(q)) * 1e3
    assert err_ms.max() < 0.15
    # refinement beats logged sizes alone
    unref = sd.fit_device_timing(s, ids, k, ids, FS, rec_pos, logged, refine=False, max_residual_ms=10)
    err_unref = np.abs(unref.to_sync_time(q) - true_time(q)).max() * 1e3
    assert err_ms.max() < err_unref


def test_unlogged_drop_becomes_invalid_interval():
    true_pos = 3000.0 * RATE
    s, ids, k, rec_pos, sizes, true_time = make(drops_true=[(true_pos, 0.337 * RATE)])
    fit = sd.fit_device_timing(s, ids, k, ids, FS)  # nothing logged
    assert fit.status == "pass", fit.reason
    assert len(fit.invalid_intervals) == 1
    a, b = fit.invalid_intervals[0]
    t_drop = T0 + true_pos / RATE
    assert a <= t_drop <= b and b - a < 31
    assert fit.metrics["unlogged_loss_ms"] == pytest.approx(337, abs=0.5)
    q = np.linspace(k[0], k[-1], 5000)
    t_q = true_time(q)
    outside = (t_q < a) | (t_q > b)
    assert np.abs(fit.to_sync_time(q)[outside] - t_q[outside]).max() * 1e3 < 0.1


def test_downward_step_fails():
    s, ids, k, _, _, _ = make()
    k = k.copy()
    k[s > T0 + 3000] += 0.05 * RATE  # samples appear *later*: impossible for a loss
    fit = sd.fit_device_timing(s, ids, k, ids, FS)
    assert fit.status == "fail"
    assert "downward step" in fit.reason


def test_mispaired_endpoint_rejected():
    s, ids, k, _, _, true_time = make()
    # duplicate value on sync: first device barcode's value also appears 900 s later
    ids_dup = ids.copy()
    ids_dup[30] = ids_dup[0]
    fit = sd.fit_device_timing(s, ids_dup, k, ids, FS)
    assert fit.status == "pass" and fit.metrics["n_paired"] == len(s) - 2
    # unique on both lines but pairs two different events (decode error): isolated outlier
    s2 = s.copy()
    s2[-1] = s[-1] - 4000
    fit2 = sd.fit_device_timing(s2, ids, k, ids, FS)
    assert fit2.status == "pass", fit2.reason
    assert fit2.metrics["n_rejected_pairs"] == 1
    q = np.linspace(k[0], k[-2], 100)
    assert np.abs(fit2.to_sync_time(q) - true_time(q)).max() < 1e-4


def test_too_few_barcodes_fails():
    s, ids, k, _, _, _ = make(duration_s=200)
    fit = sd.fit_device_timing(s, ids, k, ids, FS)
    assert fit.status == "fail" and "too few" in fit.reason


def test_message_parsing_and_resets():
    msgs = [
        b"NPX TIMESTAMP JUMP: 35, expected 3 or 4...Possible data loss on slot 3, probe 1 at sample number 719424",
        "NPX TIMESTAMP JUMP: 35, expected 3 or 4...Possible data loss on slot 3, probe 1 at sample number 719424",
        "NPX TIMESTAMP JUMP: 35933973, expected 3 or 4...Possible data loss on slot 3, probe 1 at sample number 10778641",
        "something else",
    ]
    d = sd.parse_drop_messages(msgs)
    assert d.tolist() == [[3, 1, 719424, 35], [3, 1, 10778641, 35933973]]
    assert sd.is_timestamp_reset(d[:, 2], d[:, 3]).tolist() == [False, True]
    assert sd.ticks_to_lost_samples([34]) == pytest.approx([9.2])
