# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Utilities for extracellular experiments """
import re
import json
import logging
from functools import lru_cache


log = logging.getLogger("dlab")


def iter_entries(data_file):
    """Iterate through the entries in an arf file in order of time"""
    from arf import timestamp_to_float

    return enumerate(
        sorted(
            data_file.values(),
            key=lambda entry: timestamp_to_float(entry.attrs["timestamp"]),
        )
    )


def find_stim_dset(entry):
    """Returns the first dataset that matches 'Network_Events.*_TEXT'"""
    rex = re.compile(r"Network_Events-.*?TEXT")
    for name in entry:
        if rex.match(name) is not None:
            log.debug("  - stim log dataset: %s", name)
            return entry[name]


@lru_cache(maxsize=None)
def stim_duration(stim_name):
    """
    Returns the duration of a stimulus (in s). This can only really be done by
    downloading the stimulus from the registry, because the start/stop times are
    not reliable. We try to speed this up by memoizing the function and caching
    the downloaded files.

    """
    import wave
    from nbank import default_registry
    from dlab.core import fetch_resource

    neurobank_registry = default_registry()
    target = fetch_resource(neurobank_registry, stim_name)
    with open(target, "rb") as fp:
        reader = wave.open(fp)
        return 1.0 * reader.getnframes() / reader.getframerate()


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
