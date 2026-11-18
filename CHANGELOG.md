# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## Unreleased

<small>[Compare with latest](https://github.com/AllenInstitute/npc_ephys/compare/v0.1.1...HEAD)</small>

### Added

- Dropped-sample timing model (`npc_ephys.sample_drops`): Open Ephys MessageCenter drop messages are added back to the sample index, sizes refined against barcodes, unlogged steps become invalid intervals; per-device QC (status/reason, leave-one-out residual gate). Evidence: `outputs/ephys-dropped-samples-report.md` (2026-11-18, local branch `sample-drops`, not released).

### Changed

- `get_ephys_timing_on_sync` returns a tuple of per-device results (with `status`, `reason`, `drops`, `invalid_intervals`, `to_sync_time`); barcodes are paired only on values unique on both lines; failed devices have nan rate/start.

### Fixed

- Fix varname in docs ([29f44cd](https://github.com/AllenInstitute/npc_ephys/commit/29f44cd29f9391f1aef3a2f02f6d7673a8265f86) by bjhardcastle).
- Fix input parsing ([7d6782c](https://github.com/AllenInstitute/npc_ephys/commit/7d6782c83c9944d48e7ad3cf8a0e89c48b7e8738) by bjhardcastle).

### Changed

- Change version-dependent error type in SpikeInterface data ([f187924](https://github.com/AllenInstitute/npc_ephys/commit/f18792422896bdb20ea842bf58aebeca8137206e) by bjhardcastle).

<!-- insertion marker -->
## [v0.1.1](https://github.com/AllenInstitute/npc_ephys/releases/tag/v0.1.1) - 2024-02-01

<small>[Compare with first commit](https://github.com/AllenInstitute/npc_ephys/compare/edcb643c274b5fdfdb76785021c92fbf9c613868...v0.1.1)</small>

