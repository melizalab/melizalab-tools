# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Functions for using kilsort/phy data """
import os
import logging
import numpy as np
import pandas as pd
import quickspikes as qs
from tqdm import tqdm

log = logging.getLogger("dlab.kilo")


def assign_events_flat(events, sampling_rate):
    """Assign event_times to clusters, generating a large toelis object"""
    from collections import defaultdict

    tl = defaultdict(list)
    nevents, _ = events.shape
    log.info("- grouping %d spikes by cluster...", nevents)
    for event in tqdm(events.itertuples(index=False), total=nevents):
        tl[event.clust].append(event.time / sampling_rate * 1000.0)
    max_clust = max(tl.keys())
    return [np.asarray(tl[i]) for i in range(max_clust + 1)]


def assign_events(pprox, events, warn_unassigned=False, only_clusters=None):
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
    nevents, _ = events.shape
    log.info("- grouping %d spikes by cluster and trial...", nevents)
    for event in tqdm(events.itertuples(index=False), total=nevents):
        if only_clusters is not None and event.clust not in only_clusters:
            continue
        if event.time < trial["recording"]["start"]:
            if warn_unassigned:
                log.warning(
                    "warning: spike at %d is before the start of trial %d", time, index
                )
            continue
        while event.time > trial["recording"]["stop"]:
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
        t_seconds = (event.time - trial["recording"]["start"]) / sampling_rate
        if event.clust not in clusters:
            cluster = deepcopy(pprox)
            cluster.update(cluster=event.clust)
            clusters[event.clust] = cluster
        clusters[event.clust]["pprox"][index]["events"].append(t_seconds)
    return clusters


# def extract_waveforms(


def group_spikes_script(argv=None):
    import nbank
    import argparse
    import json
    from dlab.util import setup_log, json_serializable

    __version__ = "2022.07.01"

    p = argparse.ArgumentParser(
        description="group kilosorted spikes into pprox files based on cluster and trial"
    )
    p.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    p.add_argument(
        "--dry-run",
        help="do everything except write the output files",
        action="store_true",
    )
    p.add_argument(
        "--toelis",
        action="store_true",
        help="output toelis instead of pprox. one file will be generated for the entire recording",
    )
    p.add_argument(
        "--cluster",
        "-c",
        help="only save data for the specified clusters (as comma-separated list)",
        type=lambda s: [int(item) for item in s.split(",")],
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
        "--save-waveforms",
        "-W",
        action="store_true",
        help="save representative waveforms from each unit's main channel in npy format",
    )
    p.add_argument(
        "--waveform-num-spikes",
        type=int,
        default=200,
        help="maximum number of spikes to use for average waveform",
    )
    p.add_argument(
        "--waveform-seed",
        type=int,
        default=12345,
        help="random seed for selecting spikes",
    )
    p.add_argument(
        "--waveform-pre-peak",
        type=float,
        default=2.0,
        help="samples before the spike to keep (default %(default).1f ms)",
    )
    p.add_argument(
        "--waveform-post-peak",
        type=float,
        default=5.0,
        help="samples after the spike to keep (default %(default).1f ms)",
    )
    p.add_argument(
        "--waveform-upsample",
        type=int,
        default=3,
        help="factor to upsample spikes before aligning them (default %(default)0.1f)",
    )
    p.add_argument(
        "trials", help="pprox file with the trial structure of the experiment"
    )
    p.add_argument(
        "sortdir",
        help="kilosort output directory. Needs to contain 'spike_times.npy', 'spike_clusters.npy',"
        " 'cluster_info.tsv', and 'temp_wh.dat'",
    )
    args = p.parse_args(argv)
    setup_log(log, args.debug)

    log.info("- loading data:")
    log.info("  - experiment file: %s", args.trials)
    with open(args.trials, "rt") as fp:
        pprox = json.load(fp)
    sampling_rate = pprox["entry_metadata"][0]["sampling_rate"]
    log.info("  - kilosort output directory: %s", args.sortdir)
    timefile = os.path.join(args.sortdir, "spike_times.npy")
    clustfile = os.path.join(args.sortdir, "spike_clusters.npy")
    infofile = os.path.join(args.sortdir, "cluster_info.tsv")
    log.info("    - spike times: %s", timefile)
    log.info("    - spike clusters: %s", clustfile)
    events = pd.DataFrame(
        {"time": np.load(timefile).squeeze(), "clust": np.load(clustfile)}
    )
    log.info("    - cluster info: %s", infofile)
    info = pd.read_csv(infofile, sep="\t", index_col=0)
    recfile = os.path.join(args.sortdir, "temp_wh.dat")
    filtfile = os.path.join(args.sortdir, "whitening_mat.npy")
    nchannels, _ = np.load(filtfile).shape
    # TODO get dtype from params.py
    recording = np.memmap(recfile, mode="c", dtype="int16")
    recording = np.reshape(recording, (recording.size // nchannels, nchannels))
    nsamples, nchannels = recording.shape
    log.info("    - filtered recording: %s", recfile)
    log.info("      - %d samples, %d channels", nsamples, nchannels)

    if args.name is None:
        base, rec_id = nbank.parse_resource_id(pprox["recording"])
        args.name = rec_id

    if args.toelis:
        import toelis

        clusters = assign_events_flat(events, sampling_rate)
        outfile = os.path.join(args.output or "", "all_units.toe_lis")
        if not args.dry_run:
            with open(outfile, "wt") as ofp:
                toelis.write(ofp, clusters)
            log.info("- saved %d spikes to '%s'", toelis.count(clusters), outfile)
    else:
        clusters = assign_events(
            pprox,
            events,
            warn_unassigned=args.debug,
            only_clusters=args.cluster,
        )
        total_spikes = 0
        total_clusters = 0
        events = events.set_index("clust")
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
            n_before = int(args.waveform_pre_peak * sampling_rate / 1000)
            n_after = int(args.waveform_post_peak * sampling_rate / 1000)
            clust = events.loc[clust_id]
            clust = clust[(clust.time > n_before) & (clust.time < (nsamples - n_after))]
            nspikes, _ = clust.shape
            selected = clust.sample(
                min(args.waveform_num_spikes, nspikes), random_state=args.waveform_seed
            )
            times = sorted(selected.time)
            waveforms = qs.peaks(
                recording[:, clust_info["ch"]], times, n_before, n_after
            )
            if args.waveform_upsample > 1:
                # quick and dirty check for inverted signal
                mean_spike = waveforms.mean(0)
                flip = np.sign(mean_spike.min() + mean_spike.max())
                _, waveforms = qs.realign_spikes(
                    times,
                    waveforms * flip,
                    args.waveform_upsample,
                    expected_peak=n_before * args.waveform_upsample,
                )
                waveforms *= flip
            cluster["waveform"] = {
                "mean": waveforms.mean(0),
                "sampling_rate": sampling_rate * args.waveform_upsample,
            }
            if args.save_waveforms:
                np.save(os.path.splitext(outfile)[0] + "_spikes.npy", waveforms)

            try:
                pb = cluster["processed_by"]
            except KeyError:
                pb = []
                cluster["processed_by"] = pb
            pb.append("{} {}".format(p.prog, __version__))
            if not args.dry_run:
                with open(outfile, "wt") as ofp:
                    json.dump(cluster, ofp, default=json_serializable)
        log.info(
            "- a total of %d spikes were assigned to %d clusters",
            total_spikes,
            total_clusters,
        )
