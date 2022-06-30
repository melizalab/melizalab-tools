# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Functions for using kilsort/phy data """
import os
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("dlab.kilo")


def assign_events_flat(event_times, event_clusts, sampling_rate):
    """Assign event_times to clusters, generating a large toelis object"""
    from collections import defaultdict

    tl = defaultdict(list)
    for time, clust in zip(event_times, event_clusts):
        tl[clust].append(time / sampling_rate * 1000.0)
    max_clust = max(tl.keys())
    return [np.asarray(tl[i]) for i in range(max_clust + 1)]


def assign_events(pprox, event_times, event_clusts, warn_unassigned=False):
    """Assign event_times to trials within a pprox based on recording time.

    pprox: an iterable list of pproc objects, sorted in order of time. Each
    object must have a "recording" field that contains "start" and "stop"
    subfields. The values of these fields must indicate the start and stop time
    of the trial.

    """
    from copy import deepcopy

    # clear out the events; the default for oeaudio-trials is to store the click times
    for trial in pprox["pprox"]:
        trial["events"] = []
    # trial iterator pulls sampling rate from entry metadata
    def trial_iterator():
        for i, trial in enumerate(pprox["pprox"]):
            entry = trial["recording"]["entry"]
            fs = pprox["entry_metadata"][entry]["sampling_rate"]
            yield (i, trial, fs)

    clusters = {}
    trial_iter = trial_iterator()
    index, trial, sampling_rate = next(trial_iter)
    for time, clust in zip(event_times, event_clusts):
        if time < trial["recording"]["start"]:
            if warn_unassigned:
                log.warning(
                    "warning: spike at %d is before the start of trial %d", time, index
                )
            continue
        while time > trial["recording"]["stop"]:
            try:
                index, trial, sampling_rate = next(trial_iter)
            except StopIteration:
                if warn_unassigned:
                    log.warning(
                        "warning: spike at %d is after the end of the last trial (%d samples)",
                        time,
                        trial["recording"]["stop"],
                    )
                break
        t_seconds = (time - trial["recording"]["start"]) / sampling_rate
        if clust not in clusters:
            cluster = deepcopy(pprox)
            cluster.update(cluster=clust)
            clusters[clust] = cluster
        clusters[clust]["pprox"][index]["events"].append(t_seconds)
    return clusters


def group_spikes_script(argv=None):
    import nbank
    import argparse
    import json
    from dlab.util import setup_log, json_serializable

    __version__ = "2021.04.08"

    p = argparse.ArgumentParser(
        description="group kilosorted spikes into pprox files based on cluster and trial"
    )
    p.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    p.add_argument(
        "--toelis",
        action="store_true",
        help="output toelis instead of pprox. one file will be generated for the entire recording",
    )
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
        " and 'cluster_info.tsv'",
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
    event_times = np.load(timefile).squeeze()
    event_clusts = np.load(clustfile)
    log.info("    - cluster info: %s", infofile)
    info = pd.read_csv(infofile, sep="\t", index_col=0)

    if args.name is None:
        base, rec_id = nbank.parse_resource_id(pprox["recording"])
        args.name = rec_id

    if args.toelis:
        import toelis

        log.info("- grouping %d spikes by cluster...", event_times.size)
        sampling_rate = pprox["entry_metadata"][0]["sampling_rate"]
        clusters = assign_events_flat(event_times, event_clusts, sampling_rate)
        outfile = os.path.join(args.output or "", "all_units.toe_lis")
        with open(outfile, "wt") as ofp:
            toelis.write(ofp, clusters)
        log.info("- saved %d spikes to '%s'", toelis.count(clusters), outfile)
    else:
        log.info("- grouping %d spikes by cluster and trial...", event_times.size)
        clusters = assign_events(
            pprox, event_times, event_clusts, warn_unassigned=args.debug
        )
        total_spikes = 0
        total_clusters = 0
        for clust_id, cluster in clusters.items():
            clust_info = info.loc[clust_id]
            clust_type = clust_info["group"]
            n_spikes = sum(len(t["events"]) for t in cluster["pprox"])
            if clust_type == "noise" or (clust_type == "mua" and not args.mua):
                log.info(
                    "  - cluster %d (%d spikes, %s) -> skipped",
                    clust_id,
                    n_spikes,
                    clust_type,
                )
                continue
            total_spikes += n_spikes
            total_clusters += 1
            # annotate with cluster info
            cluster.update(
                kilosort_amplitude=clust_info["Amplitude"],
                kilosort_contam_pct=clust_info["ContamPct"],
                kilosort_source_channel=clust_info["ch"],
                kilosort_probe_depth=clust_info["depth"],
                kilosort_n_spikes=clust_info["n_spikes"],
            )
            outfile = os.path.join(
                args.output or "", "{}_c{}.pprox".format(args.name, clust_id)
            )
            log.info(
                "  - cluster %d (%d spikes, %s) -> %s",
                clust_id,
                n_spikes,
                clust_type,
                outfile,
            )
            try:
                pb = cluster["processed_by"]
            except KeyError:
                pb = []
                cluster["processed_by"] = pb
            pb.append("{} {}".format(p.prog, __version__))
            with open(outfile, "wt") as ofp:
                json.dump(cluster, ofp, default=json_serializable)
        log.info(
            "- a total of %d spikes were assigned to %d clusters",
            total_spikes,
            total_clusters,
        )
