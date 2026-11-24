"""
>>> path = upath.UPath('s3://aind-ephys-data/ecephys_668759_2023-07-11_13-07-32/ecephys_clipped/Record Node 102/settings.xml')
>>> et = ET.parse(io.BytesIO(path.read_bytes()))
>>> _hostname(et)
'W10DT05516'
>>> _date_time(et)
(datetime.date(2023, 7, 11), datetime.time(13, 7, 53))
>>> _open_ephys_version(et)
'0.6.4'
>>> _settings_xml_md5(path)
'5c1b33293cb7c5f72df56fbdd4b72fc3'
>>> _probe_serial_numbers(et)
(18194810652, 18005123131, 18005102491, 19192719021, 18005118602, 19192719061)
>>> _probe_letters(et)
('A', 'B', 'C', 'D', 'E', 'F')
>>> _probe_types(et)
('Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0')
>>> isinstance(get_settings_xml_data(path), SettingsXmlInfo)
True
>>> _probe_serial_number_to_channel_pos_xy(et)[18194810652]['CH0']
(27, 0)
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import hashlib
import io
import re
import xml.etree.ElementTree as ET
from typing import Literal

import npc_io
import numpy as np
import upath


@dataclasses.dataclass
class SettingsXmlInfo:
    """Info from a settings.xml file from an Open Ephys recording.

    assumptions:
        - 2 slots
        - 3 ports in use per slot
        - probes ABCDEF connected in sequence
    """

    path: upath.UPath
    probe_serial_numbers: tuple[int, ...]
    probe_types: tuple[str, ...]
    probe_letters: tuple[Literal["A", "B", "C", "D", "E", "F"], ...]
    hostname: str
    date: datetime.date
    start_time: datetime.time
    open_ephys_version: str
    channel_pos_xy: tuple[dict[str, tuple[int, int]], ...]
    is_tip_channel_bank: tuple[bool, ...]
    """All channels are in the bank closest to the tip"""
    is_tip_referenced: tuple[bool, ...]
    neuropix_pxi_version: str

    def __eq__(self, other) -> bool:
        if not isinstance(other, SettingsXmlInfo):
            return NotImplemented
        # files from multiple nodes can be created at slightly different times, so their `date`
        # fields may differ. Everything else should be identical.
        return all(
            getattr(self, field.name) == getattr(other, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("path", "start_time")
        )


def get_settings_xml_etree(path: npc_io.PathLike) -> ET.ElementTree:
    """Info from a settings.xml file from an Open Ephys recording."""
    return ET.parse(io.BytesIO(npc_io.from_pathlike(path).read_bytes()))


def get_settings_xml_data(path: npc_io.PathLike | SettingsXmlInfo) -> SettingsXmlInfo:
    if isinstance(path, SettingsXmlInfo):
        return path
    et = get_settings_xml_etree(path)
    return SettingsXmlInfo(
        path=npc_io.from_pathlike(path),
        probe_serial_numbers=_probe_serial_numbers(et),
        probe_types=_probe_types(et),
        probe_letters=_probe_letters(et),
        hostname=_hostname(et),
        date=_date_time(et)[0],
        start_time=_date_time(et)[1],
        open_ephys_version=_open_ephys_version(et),
        channel_pos_xy=tuple(_probe_serial_number_to_channel_pos_xy(et).values()),
        is_tip_channel_bank=_is_tip_channel_bank(et),
        is_tip_referenced=_tip_referenced(et),
        neuropix_pxi_version=_get_tag_attrib(
            et, "PROCESSOR", "libraryVersion", name="Neuropix-PXI"
        )
        or "",
    )


def _get_tag_text(et: ET.ElementTree, tag: str) -> str | None:
    result = [
        element.text for element in et.getroot().iter() if element.tag == tag.upper()
    ]
    if not (result and any(result)):
        result = [element.attrib.get(tag.lower()) for element in et.getroot().iter()]
    return str(result[0]) if (result and any(result)) else None


def _get_tag_attrib(
    et: ET.ElementTree, tag: str, attrib: str, **condition_map
) -> str | None:
    result = [
        element.attrib.get(attrib)
        for element in et.getroot().iter()
        if element.tag == tag.upper()
        and all(element.attrib.get(k) == v for k, v in condition_map.items())
    ]
    return str(result[0]) if (result and any(result)) else None


def _hostname(et: ET.ElementTree) -> str:
    result = (
        # older, pre-0.6.x:
        _get_tag_text(et, "machine")
        # newer, 0.6.x:
        or _get_tag_attrib(et, "MACHINE", "name")
    )
    if not result:
        raise LookupError(f"No hostname: {result!r}")
    return result


@functools.cache
def _date_time(et: ET.ElementTree) -> tuple[datetime.date, datetime.time]:
    """Date and recording start time."""
    result = _get_tag_text(et, "date")
    if not result:
        raise LookupError(f"No datetime found: {result!r}")
    dt = datetime.datetime.strptime(result, "%d %b %Y %H:%M:%S")
    return dt.date(), dt.time()


@functools.cache
def _probe_attrib_dicts(et: ET.ElementTree) -> tuple[dict[str, str], ...]:
    return tuple(
        probe_dict.attrib
        for probe_dict in et.getroot().iter()
        if "probe_serial_number" in probe_dict.attrib
    )


def _probe_attrib(et: ET.ElementTree, attrib: str) -> tuple[str, ...]:
    return tuple(probe[attrib] for probe in _probe_attrib_dicts(et))


def _probe_serial_numbers(et: ET.ElementTree) -> tuple[int, ...]:
    return tuple(int(_) for _ in _probe_attrib(et, "probe_serial_number"))


def _probe_types(et: ET.ElementTree) -> tuple[str, ...]:
    try:
        return _probe_attrib(et, "probe_name")
    except KeyError:
        return tuple("unknown" for _ in _probe_attrib_dicts(et))


def _tip_referenced(et: ET.ElementTree) -> tuple[bool, ...]:
    """Whether each probe is tip-referenced.

    Returns:
        tuple of bool, one per probe
    """
    ref_attribs = _probe_attrib(et, "referenceChannel")
    return tuple(ref.lower() == "tip" for ref in ref_attribs)


def _probe_serial_number_to_channel_pos_xy(
    et: ET.ElementTree,
) -> dict[int, dict[str, tuple[int, int]]]:
    x_pos_iter = tuple(a for a in et.getroot().iter() if a.tag == "ELECTRODE_XPOS")
    y_pos_iter = tuple(a for a in et.getroot().iter() if a.tag == "ELECTRODE_YPOS")

    return dict(
        zip(
            _probe_serial_numbers(et),
            (
                {k: (int(x.attrib[k]), int(y.attrib[k])) for k in x.attrib.keys()}
                for x, y in zip(x_pos_iter, y_pos_iter)
            ),
        )
    )


def _probe_serial_number_to_bank_idx(
    et: ET.ElementTree,
) -> dict[int, dict[str, int]]:
    """`CHANNELS` in settings xml, corresponding to the "bank" index for each
    channel.
    - 0 is bank closest to tip
    """
    channels_iter = tuple(a for a in et.getroot().iter() if a.tag == "CHANNELS")

    return dict(
        zip(
            _probe_serial_numbers(et),
            ({k: int(c.attrib[k]) for k in c.attrib.keys()} for c in channels_iter),
        )
    )


def _probe_idx(et: ET.ElementTree) -> tuple[int, ...]:
    """Try to reconstruct index from probe slot and port.

    assumptions:
        - 2 slots
        - 3 ports in use per slot (but from 11/1 onwards, probes may be in ports 2,3,4 rather than 1,2,3)
        - probes ABCDEF connected in sequence
    """
    slots, ports = _probe_attrib(et, "slot"), _probe_attrib(et, "port")
    is_full_complement = len(set(zip(slots, ports))) == 6
    if is_full_complement:
        return tuple(range(6))
    else:
        idx: list[int] = []
        for port_idx, slot in enumerate(sorted(set(slots))):
            current_ports = sorted(
                [int(p) for s, p in zip(slots, ports) if int(s) == int(slot)]
            )
            if current_ports == [2, 3]:
                raise ValueError(
                    f"Ambiguous behavior for {slot=}: cannot determine whether first probe was disconnected from port 1 or last probe from port 4 {current_ports}"
                )
            max_port = 4 if 4 in current_ports else 3
            port_shift: Literal[1] | Literal[0] = 1 if max_port == 4 else 0
            current_idx = np.array(current_ports) - 1 - port_shift
            idx.extend((current_idx + port_idx * 3).tolist())
        return tuple(idx)


def _custom_probe_letters(
    et: ET.ElementTree,
) -> tuple[Literal["A", "B", "C", "D", "E", "F"], ...] | None:
    """Probe letters from `custom_probe_name` attributes (e.g. 'ProbeB'), or None if
    any probe is missing a name, any name is not of the form 'Probe[A-F]', or
    letters are duplicated.

    >>> _custom_probe_letters(_test_et(('ProbeA', 'ProbeB'), ((2, 1), (2, 3))))
    ('A', 'B')
    >>> _custom_probe_letters(_test_et(('18005108082', 'ProbeB'), ((2, 1), (2, 3)))) is None
    True
    >>> _custom_probe_letters(_test_et(('ProbeB', 'ProbeB'), ((2, 1), (2, 3)))) is None
    True
    """
    names = tuple(probe.get("custom_probe_name") for probe in _probe_attrib_dicts(et))
    if not names or not all(
        name is not None and re.fullmatch("Probe[A-F]", name) for name in names
    ):
        return None
    letters = tuple(name[-1] for name in names)  # type: ignore [index]
    if len(set(letters)) != len(letters):
        return None
    return letters  # type: ignore [return-value]


def _test_et(
    custom_names: tuple[str, ...], slot_ports: tuple[tuple[int, int], ...]
) -> ET.ElementTree:
    """Minimal settings.xml etree for tests."""
    probes = "".join(
        f'<NP_PROBE slot="{s}" port="{p}" dock="1" probe_serial_number="{i}"'
        f' probe_name="Neuropixels 1.0" custom_probe_name="{name}"/>'
        for i, (name, (s, p)) in enumerate(zip(custom_names, slot_ports))
    )
    return ET.ElementTree(
        ET.fromstring(f"<SETTINGS><EDITOR>{probes}</EDITOR></SETTINGS>")
    )


def _probe_letters(
    et: ET.ElementTree,
) -> tuple[Literal["A", "B", "C", "D", "E", "F"], ...]:
    """Probe letters inferred from slot/port, unless every probe has a valid
    `custom_probe_name` ('Probe[A-F]') and they disagree with the inference, in
    which case the custom names are used.

    Probe B in slot 2, port 3 with port 2 empty (slot/port inference gives 'C'):
    >>> slot_ports = ((2, 1), (2, 3), (3, 1), (3, 2), (3, 3))
    >>> _probe_letters(_test_et(('ProbeA', 'ProbeB', 'ProbeD', 'ProbeE', 'ProbeF'), slot_ports))
    ('A', 'B', 'D', 'E', 'F')

    Custom names missing or malformed: fall back to slot/port inference:
    >>> _probe_letters(_test_et(('ProbeA', '18005108082', 'ProbeD', 'ProbeE', 'ProbeF'), slot_ports))
    ('A', 'C', 'D', 'E', 'F')
    """
    custom_letters = _custom_probe_letters(et)
    try:
        inferred_letters = _inferred_probe_letters(et)
    except ValueError:
        if custom_letters is not None:
            return custom_letters
        raise
    if custom_letters is not None and custom_letters != inferred_letters:
        return custom_letters
    return inferred_letters


def _inferred_probe_letters(
    et: ET.ElementTree,
) -> tuple[Literal["A", "B", "C", "D", "E", "F"], ...]:
    probe_idx = _probe_idx(et)
    if all(np.diff(probe_idx) == 1) and len(probe_idx) == 6:
        return tuple("ABCDEF")  # type: ignore [arg-type]
    return tuple("ABCDEF"[idx] for idx in probe_idx)  # type: ignore [misc]


def _open_ephys_version(et: ET.ElementTree) -> str:
    result = _get_tag_text(et, "version")
    if not result:
        raise LookupError(f"No version found: {result!r}")
    return result


def _is_tip_channel_bank(
    et: ET.ElementTree,
) -> tuple[bool, ...]:
    """
    >>> x = get_settings_xml_etree('s3://aind-ephys-data/ecephys_660023_2023-08-08_15-11-14/ecephys_clipped/Record Node 102/settings.xml')
    >>> _is_tip_channel_bank(x)
    (True, True, True, False, True, True)
    """
    return tuple(
        all(channel == 0 for channel in all_channels_for_probe.values())
        for all_channels_for_probe in _probe_serial_number_to_bank_idx(et).values()
    )


def _settings_xml_md5(path: str | upath.UPath) -> str:
    return hashlib.md5(upath.UPath(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    from npc_ephys import testmod

    testmod()
