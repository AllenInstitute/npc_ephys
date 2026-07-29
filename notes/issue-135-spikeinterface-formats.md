# Issue 135: SpikeInterface formats

## Problem

Unit loading assumed the legacy folder layout and Kilosort 2.5 naming. New SpikeInterface output uses `SortingAnalyzer` Zarr stores.

## Findings

- SpikeInterface version, not Kilosort version, determines the layout.
- Analyzer metadata identifies the format and version.
- New output may use long stream names, `#` in paths, and several `_groupN` stores per probe.
- New AP stream names may omit the legacy `-AP` suffix.

## Changes

- Added automatic legacy/analyzer detection behind one `SpikeInterfaceData` API.
- Added analyzer readers for metrics, templates, spikes, properties, locations, and amplitudes.
- Added generic `make_units_table_from_spike_interface()`; kept the KS2.5 name as an alias.
- Added exact stream matching and grouped-store fan-out so all units are loaded.
- Preserved configured S3 access when opening Zarr stores.
- Added regression tests for both layouts, exact names, and grouped outputs.

These changes keep existing callers working while removing layout and sorter-version assumptions.
