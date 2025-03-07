from __future__ import annotations

import collections
import datetime
import logging
from collections.abc import Iterable

import npc_io
import npc_session
import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

SERIAL_NUM_TO_PROBE_LETTER = (
    {
        "SN32148": "A",
        "SN32142": "B",
        "SN32144": "C",
        "SN32149": "D",
        "SN24272": "D",
        "SN32135": "E",
        "SN24273": "F",
    }  # NP.0
    | {
        "SN40911": "A",
        "SN40900": "B",
        "SN40912": "C",
        "SN40913": "D",
        "SN40914": "E",
        "SN40910": "F",
    }  # NP.1
    | {
        "SN45356": "A",
        "SN45484": "B",
        "SN45485": "C",
        "SN45359": "D",
        "SN45482": "E",
        "SN45361": "F",
    }  # NP.2
    | {
        "SN40906": "A",
        "SN40908": "B",
        "SN40907": "C",
        "SN41084": "D",
        "SN40903": "E",
        "SN40902": "F",
    }  # NP.3
)
SHORT_TRAVEL_SERIAL_NUMBERS = {
    "SN32148",
    "SN32142",
    "SN32144",
    "SN32149",
    "SN32135",
    "SN24273",
}
SHORT_TRAVEL_RANGE = 6_000
LONG_TRAVEL_RANGE = 15_000
NEWSCALE_LOG_COLUMNS = (
    "last_movement_dt",
    "device_name",
    "x",
    "y",
    "z",
    "x_virtual",
    "y_virtual",
    "z_virtual",
)


def get_newscale_data(path: npc_io.PathLike) -> pl.DataFrame:
    """May be empty if log.csv is empty.

    >>> df = get_newscale_data('s3://aind-ephys-data/ecephys_686740_2023-10-23_14-11-05/behavior/log.csv')
    """
    try:
        return pl.read_csv(
            source=npc_io.from_pathlike(path).as_posix(),
            new_columns=NEWSCALE_LOG_COLUMNS,
            try_parse_dates=True,
            ignore_errors=True,
            # some log files have leading null values on first row, which cause date-parsing errors:
            # alternative is to use `infer_schema_length=int(1e9)` to read more rows,
            # but it's slower than `ignore_errors` and sometimes still doesn't parse
            # dates correctly.
            # since we only have one column that needs parsing, this seems safe to use
        )
    except pl.exceptions.NoDataError:
        return pl.DataFrame()


def get_newscale_coordinates(
    newscale_log_path: npc_io.PathLike,
    recording_start_time: (
        str | datetime.datetime | npc_session.DatetimeRecord | None
    ) = None,
) -> pd.DataFrame:
    """Returns the coordinates of each probe at the given time, by scanning for the most-recent prior movement on each motor.

    - looks up the timestamp of movement preceding `recording_start_time`
    - if not provided, attempt to parse experiment (sync) start time from `newscale_log_path`:
      assumes manipulators were not moved after the start time

    >>> df = get_newscale_coordinates('s3://aind-ephys-data/ecephys_686740_2023-10-23_14-11-05/behavior/log.csv', '2023-10-23 14-11-05')
    >>> list(df['x'])
    [6278.0, 6943.5, 7451.0, 4709.0, 4657.0, 5570.0]
    >>> list(df['z'])
    [11080.0, 8573.0, 6500.0, 8107.0, 8038.0, 9125.0]
    """
    newscale_log_path = npc_io.from_pathlike(newscale_log_path)
    if recording_start_time is None:
        try:
            start = npc_session.DatetimeRecord(newscale_log_path.as_posix())
        except ValueError as exc:
            raise ValueError(
                f"`recording_start_time` must be provided to indicate start of ephys recording: no time could be parsed from {newscale_log_path.as_posix()}"
            ) from exc
    else:
        start = npc_session.DatetimeRecord(recording_start_time)

    movement = pl.col(NEWSCALE_LOG_COLUMNS[0])
    serial_number = pl.col(NEWSCALE_LOG_COLUMNS[1])
    df = get_newscale_data(newscale_log_path)

    # if experiment date isn't in df, the log file didn't cover this experiment -
    # we can't continue
    if df.is_empty() or start.dt.date() not in df["last_movement_dt"].dt.date():
        raise IndexError(
            f"no movement data found for experiment date {start.dt.date()} in {newscale_log_path.as_posix()}"
        )

    recent_df = df.filter(
        pl.col("last_movement_dt").dt.date()
        > (start.dt.date() - datetime.timedelta(hours=24))
    )
    recent_z_values = recent_df["z"].str.strip_chars().cast(pl.Float32).to_numpy()
    z_inverted: bool = is_z_inverted(recent_z_values)

    df = (
        df.filter(movement < start.dt)
        .select(NEWSCALE_LOG_COLUMNS[:-3])
        .group_by(serial_number)
        .agg(pl.all().sort_by(movement).last())  # get last-moved for each manipulator
        .top_k(6, by=movement)
    )

    # serial numbers have an extra leading space
    manipulators = df.get_column(NEWSCALE_LOG_COLUMNS[1]).str.strip_chars()
    df = df.with_columns(manipulators)
    # convert str floats to floats
    for column in NEWSCALE_LOG_COLUMNS[2:8]:
        if column not in df.columns:
            continue
        df = df.with_columns(df.get_column(column).str.strip_chars().cast(pl.Float64))
    probes = manipulators.replace(
        {k: f"probe{v}" for k, v in SERIAL_NUM_TO_PROBE_LETTER.items()}
    ).alias("electrode_group_name")

    # correct z values
    z = df["z"]
    for idx, device in enumerate(df["device_name"]):
        if z_inverted:
            z[idx] = get_z_travel(device) - z[idx]
    df = df.with_columns(z)

    # add time of last movement relative to start of recording
    df = df.with_columns(
        (pl.col("last_movement_dt") - start.dt)
        .dt.total_seconds()
        .alias("last_movement_time")
    )

    df = (
        df.insert_column(index=0, column=probes)
        .sort(pl.col("electrode_group_name"))
        .to_pandas()
    )
    # nwb doesn't support `Timestamp`
    df.last_movement_dt = df.last_movement_dt.astype("str")  # type: ignore[attr-defined]
    return df


def get_z_travel(serial_number: str) -> int:
    """
    >>> get_z_travel('SN32144')
    6000
    >>> get_z_travel('SN40911')
    15000
    """
    if serial_number not in SERIAL_NUM_TO_PROBE_LETTER:
        raise ValueError(
            f"{serial_number=} is not a known serial number: need to update {__file__}"
        )
    if serial_number in SHORT_TRAVEL_SERIAL_NUMBERS:
        return SHORT_TRAVEL_RANGE
    return LONG_TRAVEL_RANGE


def is_z_inverted(z_values: Iterable[float]) -> bool:
    """
    The limits of the z-axis are [0-6000] for NP.0 and [0-15000] for NP.1-3. The
    NewScale software sometimes (but not consistently) inverts the z-axis, so
    retracted probes have a z-coordinate of 6000 or 15000 not 0. This function checks
    the values in the z-column and tries to determine if the z-axis is inverted.

    Assumptions:
    - the manipulators spend more time completely retracted than completely extended

    >>> is_z_inverted([0, 3000, 3000, 0])
    False
    >>> is_z_inverted([15000, 3000, 3000, 15000])
    True
    """
    c = collections.Counter(np.round(list(z_values), -2))
    is_long_travel = bool(c[LONG_TRAVEL_RANGE])
    travel_range = LONG_TRAVEL_RANGE if is_long_travel else SHORT_TRAVEL_RANGE
    return c[0] < c[travel_range]


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
