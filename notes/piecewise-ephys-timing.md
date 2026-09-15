# Proposal: piecewise ephys timing on the sync clock

## Motivation

`EphysTimingInfoOnSync` currently reduces the relationship between a probe's sample
clock and the sync clock to one `start_time` and one `sampling_rate`. That is correct
for recordings whose barcode pairs follow a single affine transform, but it cannot
represent a clock discontinuity or nonlinear drift. The current first/last-barcode
fit can therefore agree at both endpoints while being substantially wrong in the
middle of a recording.

This was exposed while recovering Probe D from Task 2 pilot recordings. Filtering
Open Ephys TTL events to barcode channel 1 restores valid barcode decoding, but an
all-barcode residual check shows that several recovered timebases are not globally
linear. Approximate maximum residuals from an endpoint fit are:

| Session | Probe D maximum residual |
| --- | ---: |
| `DRpilot_840160_20260428` | 580 ms; a roughly 1 s sync-wide discontinuity |
| `DRpilot_840160_20260429` | 29 ms |
| `DRpilot_841308_20260505` | 19 ms |
| `DRpilot_841308_20260506` | 25 ms |
| `DRpilot_844567_20260408` | 180 ms |
| `DRpilot_850217_20260402` | 335 ms |
| `DRpilot_850217_20260403` | 400 ms |
| `DRpilot_850217_20260406` | 13 ms |

The AP and LFP residual traces for Probe D are nearly identical, which points to a
shared probe timebase problem rather than failed barcode decoding. In the April 28
session the discontinuity is also visible on healthy probes, indicating a sync-wide
event.

## Goals

- Use every unambiguous shared barcode, rather than only the first and last pair.
- Quantify fit quality before declaring a timing result valid.
- Represent smooth nonlinear drift and discrete clock discontinuities without
  silently flattening them into one sampling rate.
- Preserve existing behavior and data structures for recordings verified to have a
  linear timebase.
- Give downstream code one conversion method for probe sample indices to sync time.

Stimulus display timing, recording-launch selection, and recovery of probes with no
decodable barcode channel are separate concerns.

## Proposed model

Introduce a timebase abstraction owned by `EphysTimingInfoOnSync`:

```python
@dataclasses.dataclass(frozen=True)
class TimingAnchor:
    sample_index: int
    sync_time: float


@dataclasses.dataclass(frozen=True)
class TimingSegment:
    start_sample: int
    stop_sample: int
    anchors: tuple[TimingAnchor, ...]
    max_abs_residual: float
    rms_residual: float


class EphysTimebase(Protocol):
    def to_sync_time(self, sample_indices: npt.ArrayLike) -> npt.NDArray: ...
    def to_sample_index(self, sync_times: npt.ArrayLike) -> npt.NDArray: ...
```

Provide two implementations:

1. `AffineTimebase` stores the current `start_time` and adjusted `sampling_rate`.
2. `PiecewiseTimebase` stores ordered segments and the matched barcode anchors used
   to construct them. Within a continuous segment, conversion uses monotonic linear
   interpolation between anchors, with an explicitly recorded extrapolation policy
   outside the first and last anchors.

A detected discontinuity separates segments. The interval between the last anchor
before the jump and the first anchor after it is intrinsically ambiguous when
barcodes are the only evidence. The model should retain that uncertainty interval
and either raise or emit a quality flag for samples in it; it should not smear the
jump across the whole barcode interval without disclosure.

`EphysTimingInfoOnSync` gains a `timebase` field and a `to_sync_time()` method. Its
existing `start_time` and `sampling_rate` remain unchanged for `AffineTimebase` so
old experiments and callers retain their current results. For a piecewise mapping,
scalar timing properties must not be presented as exact: during migration they can
return a documented summary with a warning, followed by a deprecation cycle that
requires callers to use `to_sync_time()`.

## Construction and validation

1. Decode barcode events from the designated TTL channel.
2. Match all unique barcode IDs on the probe and sync clocks. Reject or separately
   resolve duplicates; do not let first/last matching hide ambiguity.
3. Fit a robust affine model to all matched anchors and retain per-anchor residuals.
4. Classify the mapping as affine only when residuals and local slopes satisfy
   empirically chosen tolerances.
5. When the affine model fails, detect change points in residual offset and local
   slope. Split discrete jumps into independent segments; use anchor interpolation
   for continuous within-segment drift.
6. Cross-check AP and LFP from the same probe. Their mappings should agree within a
   sample-rate-scaled tolerance, and disagreement should be surfaced as invalid
   timing rather than averaged away.
7. Store diagnostics with the result: number of decoded and matched barcodes,
   residual summary, segment boundaries, extrapolated ranges, ambiguous ranges, and
   the classification reason.

Thresholds should be selected from a population of known-good historical sessions,
not hard-coded from the Task 2 failures alone.

## Integration sequence

1. Add all-match barcode matching and residual diagnostics without changing the
   returned affine result. Warn on suspect fits and build a regression corpus from
   known-good and affected sessions.
2. Add `AffineTimebase`, `PiecewiseTimebase`, and conversion tests covering constant
   drift, nonlinear drift, a discrete jump, extrapolation, and inversion.
3. Add `timebase` to `EphysTimingInfoOnSync`; construct `AffineTimebase` for all
   current linear sessions and verify byte-for-byte-equivalent timestamp arrays.
4. Route `get_adjusted_timestamps` through `to_sync_time`, then audit units, LFP,
   timestamp-adjustment, and NWB export paths for direct use of scalar
   `start_time`/`sampling_rate`.
5. Enable `PiecewiseTimebase` only after thresholds and quality behavior are
   validated against the regression corpus. Persist provenance and quality metadata
   in derived outputs.

## Acceptance criteria

- Existing linear-session tests produce unchanged start times, rates, and adjusted
  timestamps.
- Every timing result includes all-match residual diagnostics.
- A synthetic clock jump is represented by separate segments and is not silently
  distributed as drift.
- Smooth nonlinear drift converts barcode anchors to their sync times within the
  selected tolerance.
- Ambiguous and extrapolated regions are queryable by downstream consumers.
- AP/LFP consistency failures and insufficient/ambiguous barcode matches have
  explicit error or quality states.
