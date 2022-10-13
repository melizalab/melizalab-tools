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


def split_trials(pprox, split_fun):
    """Split the trials in `pprox` based on the stimulus.

    split_fun: a function that takes the name of the stimulus and returns an
    list of dictionaries, one per split. Each dict needs to have at least three
    fields: `stim_begin`, the time when the split begins (relative to the start
    of the stimulus; `stim_end`, the time when the split ends; and `name`, the
    name that should be given to the split. Any additional fields will be stored
    as metadata on the new trials. Alternatively, `split_fun` can return a
    pandas dataframe. Make sure the metadata fields are valid Python identifiers.

    Returns a copy of the input pprox with the "pprox" field replaced by the split trials.

    """
    import pandas as pd

    required_fields = ["stim_begin", "stim_end", "interval_end"]
    out = pprox.copy()
    trials = out.pop("pprox")
    out["pprox"] = []
    for trial in trials:
        stimulus = trial["stimulus"]
        stim_on = stimulus["interval"][0]
        splits = split_fun(stimulus["name"])
        if not isinstance(splits, pd.DataFrame):
            splits = pd.DataFrame(splits)
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
        split_data = splits[required_fields].join(
            spikes.groupby(splits.stim_begin.searchsorted(spikes, side="left") - 1)
            .apply(lambda x: x.to_numpy())
            .rename("events")
        )
        # kludgy way to select out metadata
        split_meta = splits.drop(columns=required_fields)
        # finally, iterate through the splits table and generate pproc objects
        # with adjusted spike times, offsets, etc
        for sdata, smeta in zip(
            split_data.itertuples(), split_meta.itertuples(index=False)
        ):
            # check for empty trial
            if isinstance(sdata.events, float):
                evts = []
            else:
                evts = sdata.events - sdata.stim_begin
            pproc = {
                "events": evts,
                "offset": trial["offset"] + sdata.stim_begin + stim_on,
                "index": sdata.Index,
                "interval": [0.0, sdata.interval_end - sdata.stim_begin],
                "stimulus": {"interval": [0.0, sdata.stim_end - sdata.stim_begin]},
                "source_trial": trial["index"],
            }
            pproc["stimulus"].update(smeta._asdict())
            out["pprox"].append(pproc)
    return out
