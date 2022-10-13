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
    d = {"$schema": _schema, "pprox": tuple(trials)}
    d.update(**metadata)
    return d


def wrap_uuid(b):
    """Wrap a UUID (string or bytes) in a URN string"""
    import uuid

    try:
        b = b.decode("ascii")
    except AttributeError:
        pass
    return uuid.UUID(b).urn


def groupby(obj, keyfun):
    """Iterate through pprocs based on keys

    For example, if "stim" and "trial" are metadata on the trials, groupby(obj, lambda x: x["stim"]) will yield
    (stim0, [events_trial0, events_trial1]), (stim1, [events_trial0, events_trial1]), ...

    """
    import itertools

    evsorted = sorted(obj["pprox"], key=keyfun)
    return itertools.groupby(evsorted, key=keyfun)


def validate(obj):
    """Validates object against pprox schema"""
    import jsonschema
    import requests

    pass


def trial_iterator(pprox):
    """Iterates through trials in pprox, yielding (index, trial)"""
    for i, trial in enumerate(pprox["pprox"]):
        # annotate with sampling rate
        if "sampling_rate" not in trial["recording"]:
            entry = trial["recording"]["entry"]
            sampling_rate = pprox["entry_metadata"][entry]["sampling_rate"]
            trial["recording"]["sampling_rate"] = sampling_rate
        yield i, trial


def aggregate_events(pprox):
    """Aggregate all the events in a pprox into a single array, using the offset field to adjust times"""
    import numpy as np

    all_events = [
        np.asarray(trial["events"]) + trial["offset"] for trial in pprox["pprox"]
    ]
    return np.concatenate(all_events)


def split_trial(trial, split_fun):
    """Split a trial into multiple intervals.

    split_fun: a function that takes the name of the stimulus and returns an
    list of dictionaries, one per split. Each dict needs to have at least three
    fields: `stim_begin`, the time when the split begins (relative to the start
    of the stimulus; `stim_end`, the time when the split ends; and `name`, the
    name that should be given to the split. Any additional fields will be stored
    as metadata on the new trials. Alternatively, `split_fun` can return a
    pandas dataframe. Make sure the metadata fields are valid Python identifiers.

    Returns a pandas DataFrame with one row per split.

    """
    import pandas as pd

    stimulus = trial["stimulus"]
    stim_on = stimulus["interval"][0]
    splits = split_fun(stimulus["name"])
    # calculate average gap between splits and use this to determine when
    # the last split should end
    gaps = splits.stim_begin.shift(-1) - splits.stim_end
    gaps.iloc[-1] = gaps.mean()
    splits["interval_end"] = splits.stim_end + gaps
    last_split_end = splits.interval_end.iloc[-1]
    spikes = pd.Series(trial["events"]) - stim_on
    spikes = spikes[spikes < last_split_end]
    # this expression uses searchsorted to assign each spike to a split,
    # then groups the spikes by split and merges this with the table of splits
    df = splits.join(
        spikes.groupby(splits.stim_begin.searchsorted(spikes, side="left") - 1)
        .apply(lambda x: x.to_numpy())
        .rename("events")
    )
    df["offset"] = trial["offset"] + df.stim_begin + stim_on
    df["events"] -= df.stim_begin
    df["stim_end"] -= df.stim_begin
    df["interval_end"] -= df.stim_begin
    df["source_trial"] = trial["index"]
    return df.drop(columns=["stim_begin"]).rename_axis(index="interval").reset_index()
