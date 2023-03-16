# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Utilities for extracellular experiments """
import json
import logging
import re

logging.getLogger(__name__).addHandler(logging.NullHandler())


def entry_time(entry):
    """Return the timestamp of an entry as a floating point number"""
    from arf import timestamp_to_float

    return timestamp_to_float(entry.attrs["timestamp"])


def entry_datetime(entry):
    """Return the timestamp of an entry as a floating point number"""
    from arf import timestamp_to_datetime

    return timestamp_to_datetime(entry.attrs["timestamp"])


def iter_entries(data_file):
    """Iterate through the entries in an arf file in order of time"""
    return enumerate(sorted(data_file.values(), key=entry_time))


def find_stim_dset(entry):
    """Returns the first dataset that matches 'Network_Events.*_TEXT'"""
    rex = re.compile(r"Network_Events-.*?TEXT")
    for name in entry:
        if rex.match(name) is not None:
            logging.debug("  - stim log dataset: %s", name)
            return entry[name]


def entry_metadata(entry):
    """Extracts metadata from an entry in an oeaudio-present experiment ARF file.

    Metadata are passed to open-ephys through the network events socket as a
    json-encoded dictionary. There should be one and only one metadata message
    per entry, so only the first is returned.

    """
    re_metadata = re.compile(r"metadata: (\{.*\})")
    stims = find_stim_dset(entry)
    for row in stims:
        message = row["message"].decode("utf-8")
        m = re_metadata.match(message)
        try:
            metadata = json.loads(m.group(1))
        except (AttributeError, json.JSONDecodeError):
            pass
        else:
            metadata.update(name=entry.name, sampling_rate=stims.attrs["sampling_rate"])
            return metadata
