# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Process pprox (point process) data

See https://meliza.org/spec:2/pprox/ for specification

This package deliberately does not include any functions for serialization, as
pprox objects are just python dictionaries.

Copyright (C) Dan Meliza, 2006-2020 (dan@meliza.org)

"""

_schema = "https://meliza.org/spec:2/pprox.json#"


def empty():
    """Returns a new, empty pprox object"""
    return from_trials([])


def from_trials(trials, **metadata):
    """Wrap a sequence of trials in a pprox object, optionally specifying top-level metadata"""
    d = { "$schema": _schema, "pprox": tuple(trials) }
    d.update(**metadata)
    return d


def wrap_uuid(b):
    """ Wrap a UUID (string or bytes) in a URN string """
    import uuid
    try:
        b = b.decode("ascii")
    except AttributeError:
        pass
    return uuid.UUID(b).urn


def groupby(obj, *keys):
    """Iterate through pprocs based on keys

    For example, if "stim" and "trial" are metadata on the trials, collate(obj, "stim") will yield
    (stim0, [events_trial0, events_trial1]), (stim1, [events_trial0, events_trial1]), ...

    """
    import itertools
    import operator
    keyfun = operator.itemgetter(keys)
    evsorted = sorted(obj['pprox'], keyfun)
    return itertools.groupby(evsorted, keyfun)


def validate(obj):
    """Validates object against pprox schema"""
    import jsonschema
    import requests
    pass


def trial_iterator(pprox):
    """ Iterates through trials in pprox, yielding (index, trial) """
    for i, trial in enumerate(pprox["pprox"]):
        # annotate with sampling rate
        if "sampling_rate" not in trial["recording"]:
            entry = trial["recording"]["entry"]
            sampling_rate = pprox["entry_metadata"][entry]["sampling_rate"]
            trial["recording"]["sampling_rate"] = sampling_rate
        yield i, trial


def aggregate_events(pprox, use_recording=False):
    """Aggregate all the events in a pprox into a single array.

    This function is primarily used for testing, as this should be the reverse
    operation to whatever function is used to assign events to trials (assuming
    no gaps in the recording). If there are gaps, then set `use_recording` to
    True.

    """
    import numpy as np
    all_events = []
    for trial in pprox["pprox"]:
        sampling_rate = trial["recording"]["sampling_rate"]
        events = np.asarray(trial["events"])
        if use_recording:
            events = (events * sampling_rate).astype("i8") + trial["recording"]["start"]
        else:
            events = ((events + trial["offset"]) * sampling_rate).astype("i8")
        all_events.append(events)
    return np.concatenate(all_events)
