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
