# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Functions for using mountainlab data """
import os
import shutil
import logging

from dlab import core, __version__

log = logging.getLogger('dlab.mountain')

def load_spikes(path):
    """Loads spike time data from an mda file and wrangles it into pandas """


def assign_events(pprox, events):
    """Assign events to trials within a pprox based on recording time.

    trials: an iterable list of pproc objects, sorted in order of time. Each
    object must have a "recording" field that contains "start" and "stop"
    subfields. The values of these fields must indicate the start and stop time
    of the trial.

    """
    from copy import deepcopy
    from collections import defaultdict
    def clone():
        return deepcopy(trials)

    clusters = defaultdict(clone)
    trial_iter = enumerate(trials)
    index, trial = next(trial_iter)
    for channel, time, clust in events:
        if time < trial["recording"]["start"]:
            log.debug("spike at %d is before the start of trial %d", time, index)
            continue
        while time > trial["recording"]["stop"]:
            index, trial = next(trial_iter)
        t_seconds = (time - trial["recording"]["start"]) / trial["recording"]["sampling_rate"]
        clusters[int(clust)][index]["events"].append(t_seconds)
    return clusters


def assign_events_sorted(pprox, events):
    """Assign events to trials within a pprox based on recording time.

    trials: an iterable list of pproc objects, sorted in order of time. Each
    object must have a "recording" field that contains "start" and "stop"
    subfields. The values of these fields must indicate the start and stop time
    of the trial.

    """
    import pandas as pd
    from copy import deepcopy
    E = pd.DataFrame(events, columns=("channel", "time", "cluster")).set_index("cluster")
    clusters = E.index.unique().sort_values()

    for clust in clusters:
        data = deepcopy(pprox)
        trial_iter = enumerate(data["pprox"])
        index, trial = next(trial_iter)
        for cluster, row in E.loc[clust].iterrows():
            if row.time < trial["recording"]["start"]:
                log.warning("spike at %d is before the start of trial %d", row.time, index)
            while row.time > trial["recording"]["stop"]:
                index, trial = next(trial_iter)
            t_seconds = (
                (row.time - trial["recording"]["start"]) / trial["recording"]["sampling_rate"]
            )
            trial["events"].append(t_seconds)
        data.update(cluster=clust,
                    channel=row.channel)
        yield data

if __name__=="__main__":
    import json
    from arfx import mdaio
    pprox = json.load(open("data/P33_080320_p1r1/trials.pprox"))
    events = mdaio.mdafile("data/P33_080320_p1r1/firings.mda").read().astype("i8")
