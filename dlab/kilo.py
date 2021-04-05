# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Functions for using kilsort/phy data """
import os
import shutil
import logging
import numpy as np
import pandas as pd

from dlab import core, __version__

log = logging.getLogger("dlab.kilo")


def assign_events(pprox, events):
    """Assign events to trials within a pprox based on recording time.

    trials: an iterable list of pproc objects, sorted in order of time. Each
    object must have a "recording" field that contains "start" and "stop"
    subfields. The values of these fields must indicate the start and stop time
    of the trial.

    """
    from dlab.pprox import trial_iterator
    from copy import deepcopy

    clusters = {}
    trial_iter = trial_iterator(pprox)
    index, trial = next(trial_iter)
    for channel, time, clust in events:
        if time < trial["recording"]["start"]:
            log.warning(
                "warning: spike at %d is before the start of trial %d", time, index
            )
            continue
        while time > trial["recording"]["stop"]:
            try:
                index, trial = next(trial_iter)
            except StopIteration:
                log.warning(
                    "warning: spike at %d is after the end of the last trial (%d samples)",
                    time,
                    trial["recording"]["stop"],
                )
                break
        t_seconds = (time - trial["recording"]["start"]) / trial["recording"][
            "sampling_rate"
        ]
        if clust not in clusters:
            cluster = deepcopy(pprox)
            cluster.update(cluster=clust, channel=channel)
            clusters[clust] = cluster
        clusters[clust]["pprox"][index]["events"].append(t_seconds)
    return clusters


def group_spikes_script(argv=None):
    import nbank
    import argparse
    import json
    from dlab.util import setup_log, json_serializable

    __version__ = "0.1.0"

    p = argparse.ArgumentParser(
        description="group kilosorted spikes into pprox files based on cluster and trial"
    )
    p.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    p.add_argument(
        "--output",
        "-o",
        help="directory to output pprox files (default current directory)",
    )
    p.add_argument(
        "--mua",
        action="store_true",
        help="save multiunit clusters along with single units",
    )
    p.add_argument(
        "--name",
        "-n",
        help="base name of the unit (default is based on 'recording' field of trials pprox) ",
    )
    p.add_argument(
        "trials", help="pprox file with the trial structure of the experiment"
    )
    p.add_argument(
        "sortdir",
        help="kilosort output directory. Needs to contain 'spike_times.npy', 'spike_clusters.npy',"
        " and 'cluster_info.tsv'"
    )
    args = p.parse_args(argv)
    setup_log(log, args.debug)

    log.info("- loading data:")
    log.info("  - experiment file: %s", args.trials)
    with open(args.trials, "rt") as fp:
        pprox = json.load(fp)
    log.info("  - kilosort output directory: %s", args.sortdir)
    timefile = os.path.join(args.sortdir, "spike_times.npy")
    clustfile = os.path.join(args.sortdir, "spike_clusters.npy")
    infofile = os.path.join(args.sortdir, "cluster_info.tsv")
    groupfile = os.path.join(args.sortdir, "cluster_group.tsv")
    log.info("    - spike times: %s", timefile)
    log.info("    - spike clusters: %s", clustfile)
    events = np.column_stack((np.load(timefile), np.load(clustfile)))
    log.info("    - cluster info: %s", infofile)
    info = pd.read_csv(infofile, sep="\t", index_col=0)

    if args.name is None:
        base, rec_id = nbank.parse_resource_id(pprox["recording"])
        args.name = rec_id

    log.info("- grouping spikes by cluster and trial...")
    clusters = assign_events(pprox, events)
    for clust_id, cluster in clusters.items():
        outfile = os.path.join(
            args.output or "", "{}_c{}.pprox".format(args.name, clust_id)
        )
        log.info("  - cluster %d -> %s", clust_id, outfile)
        try:
            pb = cluster["processed_by"]
        except KeyError:
            pb = []
            cluster["processed_by"] = pb
        pb.append("{} {}".format(p.prog, __version__))
        with open(outfile, "wt") as ofp:
            json.dump(cluster, ofp, default=json_serializable)
