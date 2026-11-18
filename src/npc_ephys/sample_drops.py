"""Dropped-sample model for Open Ephys Neuropixels recordings aligned to sync barcodes.

When the acquisition PC cannot keep up, the Neuropix-PXI plugin drops blocks of samples from a
probe's stream, and Open Ephys numbers the surviving samples contiguously. Every sample after a
drop is then placed too early by the cumulative amount lost, while the probe clock itself stays
affine. The plugin logs each drop to the MessageCenter:

    NPX TIMESTAMP JUMP: 35, expected 3 or 4...Possible data loss on slot 3, probe 1 at sample number 719424

The jump is counted in headstage timestamp ticks (100 kHz, 10/3 ticks per 30 kHz AP sample) and the
sample number is the (contiguous) AP sample number of the first sample after the gap.

Model: sync time of recorded sample index k (relative to the first recorded sample)

    t(k) = start_time + (k + L(k)) / sampling_rate,     L(k) = sum of n_j over drops with pos_j <= k

Drop positions come from the log (exact); drop sizes are estimated from the logged ticks
(0.3 * J - 1 samples) and then refined so that the corrected barcodes lie on one affine line.
Losses that were not logged appear as persistent upward steps in the barcode residual; if larger
than `step_threshold_ms` the interval between the bracketing barcodes is marked invalid (the
position of the loss inside it is unknown). A downward step cannot be produced by lost samples
and fails the device.

All functions here are pure numpy apart from `read_drop_messages` and `get_probe_slot_port`.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import re
from collections.abc import Iterable, Sequence
from typing import Literal

import npc_io
import numpy as np
import numpy.typing as npt
import upath

logger = logging.getLogger(__name__)

TICKS_PER_AP_SAMPLE = 10 / 3
"""Neuropixels headstage timestamp ticks (100 kHz) per 30 kHz AP sample."""

DROP_MESSAGE_PATTERN = re.compile(
    r"TIMESTAMP JUMP: (\d+), expected.*?slot (\d+), probe (\d+) at sample number (\d+)"
)

MISPAIR_TOLERANCE_S = 1.0
"""A paired barcode whose offset differs from its neighbours' running median by more than this
is treated as mispaired (barcodes are ~30 s apart, so a real mispair is >= ~30 s off)."""

MIN_PAIRED_BARCODES = 10
MAX_REJECTED_FRACTION = 0.05
DEFAULT_MAX_RESIDUAL_MS: dict[str, float] = {"AP": 1.0, "LFP": 1.0, "NI-DAQ": 1.0}
"""Gate T on the max leave-one-out residual at barcodes (absolute timing error at any barcode,
not cumulative), per stream type. 1 ms agreed 2026-11-18: above every clean and drop-corrected
device in the survey (max 0.43 ms AP, 0.61 ms LFP, 0.33 ms NI-DAQ). Deliberately decoupled from
DEFAULT_STEP_THRESHOLD_MS, which stays tighter so that unlogged losses > 0.5 ms still become
invalid intervals."""
DEFAULT_STEP_THRESHOLD_MS: dict[str, float] = {"AP": 0.5, "LFP": 0.8, "NI-DAQ": 0.5}
"""Unexplained steps between consecutive barcodes larger than this become unlogged drops
(upward) or fail the device (downward)."""


def stream_type(device_name: str) -> Literal["AP", "LFP", "NI-DAQ"]:
    if device_name.endswith("-LFP"):
        return "LFP"
    if device_name.endswith("-AP"):
        return "AP"
    return "NI-DAQ"


# ---------------------------------------------------------------------------------------------
# reading the log
# ---------------------------------------------------------------------------------------------


def parse_drop_messages(
    messages: Iterable[str | bytes],
) -> npt.NDArray[np.int64]:
    """Parse MessageCenter text into a deduplicated (n, 4) int64 array of (slot, port, N, J).

    Each message is saved more than once (twice per record node, and once per record node), so
    rows are deduplicated on all four values.

    >>> msgs = ['NPX TIMESTAMP JUMP: 35, expected 3 or 4...Possible data loss on slot 3, probe 1 at sample number 719424'] * 2
    >>> parse_drop_messages(msgs).tolist()
    [[3, 1, 719424, 35]]
    """
    rows = set()
    for m in messages:
        if isinstance(m, bytes):
            m = m.decode(errors="ignore")
        match = DROP_MESSAGE_PATTERN.search(str(m))
        if match:
            J, S, P, N = (int(g) for g in match.groups())
            rows.add((S, P, N, J))
    return np.array(sorted(rows), dtype=np.int64).reshape(-1, 4)


def is_timestamp_reset(sample_number: npt.ArrayLike, ticks: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Jumps that are not data loss: the 'jump' is (about) the whole headstage timestamp, ie the
    previous timestamp was zero/invalid. These have J >= half the ticks elapsed since acquisition
    start (J ~ 10/3 * N in practice). A real loss that large would remove more than had been
    recorded up to that point.

    >>> is_timestamp_reset([10778641, 96006300], [35933973, 443181]).tolist()
    [True, False]
    """
    N = np.asarray(sample_number, dtype=float)
    J = np.asarray(ticks, dtype=float)
    return J >= 0.5 * TICKS_PER_AP_SAMPLE * N


def ticks_to_lost_samples(ticks: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Logged jump (headstage ticks between the samples either side of a gap) -> estimated number
    of AP samples lost (fractional; refined against barcodes later).

    >>> ticks_to_lost_samples([4, 34]).round(2).tolist()
    [0.2, 9.2]
    """
    return np.asarray(ticks, dtype=float) / TICKS_PER_AP_SAMPLE - 1.0


def _settings_xml_for_recording(recording_dir: upath.UPath) -> upath.UPath | None:
    record_node = recording_dir.parent.parent
    experiment_idx = int(re.sub(r"\D", "", recording_dir.parent.name) or 1)
    name = "settings.xml" if experiment_idx == 1 else f"settings_{experiment_idx}.xml"
    path = record_node / name
    return path if path.exists() else None


def get_probe_slot_port(
    settings_xml: npc_io.PathLike, probe_name: str
) -> tuple[int, int] | None:
    """(slot, port) of the probe whose `custom_probe_name` is `probe_name` (eg 'ProbeD')."""
    text = npc_io.from_pathlike(settings_xml).read_text(errors="ignore")
    for m in re.finditer(r"<NP_PROBE (.*?)>", text, re.S):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        if attrs.get("custom_probe_name") == probe_name and "slot" in attrs and "port" in attrs:
            return int(attrs["slot"]), int(attrs["port"])
    return None


def read_drop_messages(recording_dir: npc_io.PathLike) -> npt.NDArray[np.int64]:
    """All drop messages saved for one recording, from every record node's MessageCenter copy.

    Returns a deduplicated (n, 4) array of (slot, port, N, J) - not yet filtered for timestamp
    resets.
    """
    recording_dir = npc_io.from_pathlike(recording_dir)
    rel = f"{recording_dir.parent.name}/{recording_dir.name}"
    root = recording_dir.parent.parent.parent
    texts: list = []
    candidates = [recording_dir] + [
        p / rel for p in root.glob("Record Node *") if (p / rel) != recording_dir
    ]
    for rec in candidates:
        path = rec / "events" / "MessageCenter" / "text.npy"
        if not path.exists():
            continue
        texts.extend(np.load(io.BytesIO(path.read_bytes()), allow_pickle=True).tolist())
    return parse_drop_messages(texts)


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------


def cumulative_loss(
    drop_positions: npt.ArrayLike, drop_sizes: npt.ArrayLike, sample_indices: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """L(k): total samples lost at or before each sample index k (drops sorted or not)."""
    pos = np.asarray(drop_positions, dtype=float)
    sizes = np.asarray(drop_sizes, dtype=float)
    k = np.asarray(sample_indices, dtype=float)
    if pos.size == 0:
        return np.zeros(k.shape)
    o = np.argsort(pos, kind="stable")
    c = np.concatenate([[0.0], np.cumsum(sizes[o])])
    return c[np.searchsorted(pos[o], k, side="right")]


@dataclasses.dataclass(frozen=True)
class TimingFit:
    """Result of fitting the dropped-sample model to one device's barcodes."""

    status: Literal["pass", "fail"]
    reason: str
    sampling_rate: float
    """Samples per second on the sync clock (nan if no fit)"""
    start_time: float
    """Sync time (s) of sample index 0 (nan if no fit)"""
    drop_positions: npt.NDArray[np.float64]
    """Sample index of the first sample after each gap (device samples)"""
    drop_sizes: npt.NDArray[np.float64]
    """Samples lost in each gap (refined; fractional)"""
    invalid_intervals: tuple[tuple[float, float], ...]
    """Sync-time intervals containing a loss of unknown position"""
    metrics: dict

    def to_sync_time(self, sample_indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        k = np.asarray(sample_indices, dtype=float)
        return self.start_time + (k + cumulative_loss(self.drop_positions, self.drop_sizes, k)) / self.sampling_rate


def pair_barcodes(
    sync_times: npt.ArrayLike,
    sync_ids: npt.ArrayLike,
    device_times: npt.ArrayLike,
    device_ids: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pair barcodes whose value occurs exactly once on each line; sorted by sync time.

    Values repeated on either line are never used (repeats happen; see
    outputs/barcode-duplicate-mispairing.md).
    """
    st, sid = np.asarray(sync_times, float), np.asarray(sync_ids)
    dt, did = np.asarray(device_times, float), np.asarray(device_ids)
    sv, sc = np.unique(sid, return_counts=True)
    dv, dc = np.unique(did, return_counts=True)
    common = np.intersect1d(sv[sc == 1], dv[dc == 1])
    s = np.array([st[sid == v][0] for v in common], dtype=float)
    d = np.array([dt[did == v][0] for v in common], dtype=float)
    o = np.argsort(s)
    return s[o], d[o]


def _running_median(x: npt.NDArray, half_width: int = 2) -> npt.NDArray:
    n = len(x)
    return np.array([np.median(x[max(0, i - half_width): min(n, i + half_width + 1)]) for i in range(n)])


def reject_mispairs(
    s: npt.NDArray, x: npt.NDArray, tolerance_s: float = MISPAIR_TOLERANCE_S
) -> npt.NDArray[np.bool_]:
    """Keep-mask for paired barcodes: drop pairs whose offset (s - x) is isolated from its
    neighbours (running median of 5) by more than `tolerance_s`. Persistent steps (drops) are not
    rejected: a median of 5 follows a step."""
    keep = np.ones(len(s), bool)
    for _ in range(3):
        idx = np.flatnonzero(keep)
        o = s[idx] - x[idx]
        bad = np.abs(o - _running_median(o)) > tolerance_s
        if not bad.any():
            break
        keep[idx[bad]] = False
    # monotonicity: device time must increase with sync time
    idx = np.flatnonzero(keep)
    nonmono = np.flatnonzero(np.diff(x[idx]) <= 0)
    keep[idx[nonmono + 1]] = False
    return keep


def _affine(x: npt.NDArray, y: npt.NDArray) -> tuple[float, float, npt.NDArray]:
    A = np.c_[np.ones_like(x), x]
    (c0, c1), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c0), float(c1), y - (c0 + c1 * x)


def _refine(
    p: npt.NDArray, adj: npt.NDArray, pos: npt.NDArray, sizes: npt.NDArray
) -> npt.NDArray:
    """Scale the drop sizes inside each barcode interval (p[k], p[k+1]] so that its total changes by
    adj[k] samples. Intervals without drops are left unchanged."""
    new = sizes.astype(float).copy()
    if pos.size == 0:
        return new
    k_of_drop = np.searchsorted(p, pos, side="left") - 1  # interval index: p[k] < pos <= p[k+1]
    for k in np.unique(k_of_drop):
        if k < 0 or k >= len(p) - 1:
            continue
        sel = k_of_drop == k
        tot = new[sel].sum()
        if tot > 0:
            new[sel] *= max(tot + adj[k], 0.0) / tot
    return new


def _fit_once(
    s: npt.NDArray, p: npt.NDArray, fs: float, pos: npt.NDArray, sizes: npt.NDArray, refine: bool
) -> tuple[float, float, npt.NDArray, npt.NDArray]:
    """Affine fit on corrected positions, optional refinement, refit.
    Returns (c0, c1, refined sizes, residual s - model at p)."""
    L = cumulative_loss(pos, sizes, p)
    c0, c1, r = _affine((p + L) / fs, s)
    if refine and pos.size:
        adj = np.diff(r) / c1 * fs  # samples
        sizes = _refine(p, adj, pos, sizes)
        L = cumulative_loss(pos, sizes, p)
        c0, c1, r = _affine((p + L) / fs, s)
    return c0, c1, sizes, r


def fit_device_timing(
    sync_times: npt.ArrayLike,
    sync_ids: npt.ArrayLike,
    device_barcode_samples: npt.ArrayLike,
    device_ids: npt.ArrayLike,
    nominal_sampling_rate: float,
    drop_positions: npt.ArrayLike = (),
    drop_sizes: npt.ArrayLike = (),
    *,
    refine: bool = True,
    max_residual_ms: float = 0.5,
    step_threshold_ms: float = 0.5,
    min_barcodes: int = MIN_PAIRED_BARCODES,
    max_rejected_fraction: float = MAX_REJECTED_FRACTION,
    detect_steps: bool = True,
) -> TimingFit:
    """Fit sync time = start + (k + L(k)) / rate to one device's barcodes.

    - `device_barcode_samples`: barcode start times on the device in samples relative to the
      first recorded sample (float ok)
    - `drop_positions`, `drop_sizes`: logged drops in the same sample units (already
      converted, eg for LFP); empty for none. With `refine=True` sizes are scaled per barcode
      interval to match the barcodes; pass refine=False to keep them (eg sizes already refined on
      the probe's AP stream).

    QC (status 'fail' with a reason): too few paired barcodes; too many inconsistent pairs;
    a downward step > step_threshold_ms; leave-one-out residual > max_residual_ms.
    """
    fs = float(nominal_sampling_rate)
    pos = np.asarray(drop_positions, dtype=float).ravel()
    sizes = np.asarray(drop_sizes, dtype=float).ravel()
    o = np.argsort(pos, kind="stable")
    pos, sizes = pos[o], sizes[o]
    metrics: dict = dict(n_logged_drops=int(pos.size), logged_loss_ms=float(sizes.sum() / fs * 1e3))

    def fail(reason: str, **kw) -> TimingFit:
        metrics.update(kw)
        return TimingFit("fail", reason, np.nan, np.nan, pos, sizes, (), metrics)

    s, d = pair_barcodes(sync_times, sync_ids, device_barcode_samples, device_ids)
    metrics["n_paired"] = int(len(s))
    if len(s) < min_barcodes:
        return fail(f"too few paired barcodes ({len(s)} < {min_barcodes})")

    L0 = cumulative_loss(pos, sizes, d)
    keep = reject_mispairs(s, (d + L0) / fs)
    n_rej = int((~keep).sum())
    metrics["n_rejected_pairs"] = n_rej
    if n_rej > max(2, max_rejected_fraction * len(s)):
        return fail(f"inconsistent barcode pairing ({n_rej} of {len(s)} pairs rejected)")
    s, p = s[keep], d[keep]
    if len(s) < min_barcodes:
        return fail(f"too few consistent barcodes ({len(s)} < {min_barcodes})")

    # unlogged steps: iterate on the (logged-corrected, unrefined) residual
    synth: dict[int, float] = {}  # barcode interval index -> unlogged loss (samples)
    step_thr = step_threshold_ms / 1e3 * fs  # samples

    def synth_arrays() -> tuple[npt.NDArray, npt.NDArray]:
        ks = sorted(synth)
        return (
            np.array([0.5 * (p[k] + p[k + 1]) for k in ks], dtype=float),
            np.array([synth[k] for k in ks], dtype=float),
        )

    if detect_steps:
        for _ in range(100):
            sp, ss = synth_arrays()
            P = np.concatenate([pos, sp])
            S = np.concatenate([sizes, ss])
            c0, c1, r = _affine((p + cumulative_loss(P, S, p)) / fs, s)
            inc = np.diff(r) / c1 * fs
            inc = inc - np.median(inc)
            k_lo, k_hi = int(np.argmin(inc)), int(np.argmax(inc))
            if inc[k_lo] < -step_thr and -inc[k_lo] >= inc[k_hi]:
                return fail(
                    f"downward step of {-inc[k_lo] / fs * 1e3:.3g} ms between barcodes at "
                    f"{s[k_lo]:.1f} s and {s[k_lo + 1]:.1f} s (sync): not explained by lost samples",
                    max_step_down_ms=float(-inc[k_lo] / fs * 1e3),
                )
            if inc[k_hi] <= step_thr:
                break
            synth[k_hi] = synth.get(k_hi, 0.0) + float(inc[k_hi])
        else:
            return fail("step detection did not converge")
    invalid = [(float(s[k]), float(s[k + 1])) for k in sorted(synth)]
    sp, ss = synth_arrays()
    metrics["n_unlogged_steps"] = len(invalid)
    metrics["unlogged_loss_ms"] = float(ss.sum() / fs * 1e3)

    P = np.concatenate([pos, sp])
    S = np.concatenate([sizes, ss])
    o = np.argsort(P, kind="stable")
    P, S = P[o], S[o]
    c0, c1, S_ref, r = _fit_once(s, p, fs, P, S, refine)
    metrics["max_residual_ms"] = float(np.abs(r).max() * 1e3)

    # leave-one-out residual at interior barcodes (honest between-barcode error)
    loo = np.zeros(len(s) - 2)
    for j, i in enumerate(range(1, len(s) - 1)):
        m = np.ones(len(s), bool)
        m[i] = False
        a0, a1, Si, _ = _fit_once(s[m], p[m], fs, P, S, refine)
        loo[j] = s[i] - (a0 + a1 * (p[i] + cumulative_loss(P, Si, p[i])) / fs)
    # an interval marked invalid has a loss of unknown position: exclude barcodes adjacent to it
    in_invalid = np.zeros(len(s), bool)
    for a, b in invalid:
        in_invalid |= (s >= a) & (s <= b)
    loo_valid = np.abs(loo[~in_invalid[1:-1]]) * 1e3
    metrics["loo_max_ms"] = float(loo_valid.max()) if loo_valid.size else float("nan")
    metrics["loo_p99_ms"] = float(np.quantile(loo_valid, 0.99)) if loo_valid.size else float("nan")
    metrics["total_loss_ms"] = float(S_ref.sum() / fs * 1e3)
    rate = fs / c1
    start = c0
    status: Literal["pass", "fail"] = "pass"
    reason = ""
    if metrics["loo_max_ms"] > max_residual_ms:
        status, reason = "fail", (
            f"leave-one-out residual {metrics['loo_max_ms']:.3g} ms > {max_residual_ms} ms"
        )
    return TimingFit(status, reason, float(rate), float(start), P, S_ref, tuple(invalid), metrics)
