# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Functions for using kilsort/phy data """
import logging
import json
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
import quickspikes as qs
import h5py as h5
from dlab import pprox

log = logging.getLogger("dlab.kilo")

Trial = namedtuple(
    "Trial",
    [
        "recording_entry",
        "recording_start",
        "recording_end",
        "stimulus_name",
        "stimulus_start",
        "stimulus_end",
    ],
)

Stimulus = namedtuple("Stimulus", ["name", "start", "end"], defaults=[None])


def read_kilo_params(fname):
    """Read the kilosort params.py file"""
    from itertools import chain
    from configparser import ConfigParser

    parser = ConfigParser()
    with open(fname, "rt") as lines:
        lines = chain(("[top]",), lines)
        parser.read_file(lines)
    sect = parser["top"]
    return dict(
        dtype=sect["dtype"].strip("'"),
        nchannels=int(sect["n_channels_dat"]),
        sampling_rate=float(sect["sample_rate"]),
    )


def oeaudio_stims(dset):
    """Parse the messages in the 'stim' dataset to get a table of stimuli with
    start samples. Note that these need to be corrected for offset of the
    recording and network lag.

    """
    import re

    re_start = re.compile(r"start (.*)")

    for row in dset:
        time = row["start"]
        message = row["message"].decode("utf-8")
        m = re_start.match(message)
        if m is not None:
            stim_name = Path(m.group(1)).stem
            yield Stimulus(stim_name, time)


def oeaudio_to_trials(data_file, sync_dset, sync_thresh=1.0, prepad=1.0):

    """Extracts trial information from an oeaudio-present experiment ARF file

    When using oeaudio-present, a single recording is made in response to all
    the stimuli. The stimulus presentation script sends network events to
    open-ephys to mark the start and stop of each stimulus. There is typically a
    significant lag between the 'start' event and the onset of the stimulus, due
    to buffering of the audio playback. However, the oeaudio-present script will
    play a synchronization click on a second channel by default. As long as the
    user remembers to record this channel, it can be used to correct the onset
    and offset values.

    The continuous recording is broken up into trials based on the stimulus
    presentation, such that each trial encompasses one and only one stimulus.
    The `prepad` parameter specifies, in seconds, when trials begin relative to
    stimulus onset. The default is 1.0 s.

    """
    from itertools import zip_longest
    from dlab.extracellular import (
        find_stim_dset,
        entry_time,
        iter_entries,
        stim_duration,
    )
    from arf import timestamp_to_datetime

    expt_start = None
    det = qs.detector(sync_thresh, 10)
    for entry_num, entry in iter_entries(data_file):
        log.info(" - entry: '%s'", entry.name)
        entry_start = entry_time(entry)
        log.info("  - start time: %s", timestamp_to_datetime(entry.attrs["timestamp"]))
        if expt_start is None:
            expt_start = entry_start

        log.info("  - parsing stimulus log")
        stims = list(oeaudio_stims(find_stim_dset(entry)))
        log.info("    - detected %d stimuli", len(stims))
        log.info("  - sync track: '%s'", sync_dset)
        sync = entry[sync_dset]
        sync_data = sync[:].astype("d")
        det.scale_thresh(sync_data.mean(), sync_data.std())
        stim_onsets = np.asarray(det(sync_data))
        log.info("    - detected %d clicks", stim_onsets.size)
        dset_offset = sync.attrs["offset"]
        dset_end = sync.size
        sampling_rate = sync.attrs["sampling_rate"]
        stim_sample_offset = int(dset_offset * sampling_rate)
        log.info("  - recording clock offset: %d", stim_sample_offset)

        if len(stims) != stim_onsets.size:
            log.warning(
                "  - WARNING: number of stimuli does not match number of clicks. This recording may need to be discarded"
            )

        padding_samples = int(prepad * sampling_rate)
        for stim, onset, offset in zip_longest(
            stims, stim_onsets, stim_onsets[1:], fillvalue=dset_end + padding_samples
        ):
            stim_dur = int(stim_duration(stim.name) * sampling_rate)
            if stim_dur > offset - onset:
                log.warning(
                    "  - WARNING: stimulus %s is longer than the duration of the trial",
                    stim,
                )
            yield Trial(
                entry_num,
                onset - padding_samples,
                offset - padding_samples,
                stim.name,
                onset,
                onset + stim_dur,
            )


def assign_events_flat(events, sampling_rate):
    """Assign event_times to clusters, generating a large toelis object"""
    nevents, _ = events.shape
    log.info("- grouping %d spikes by cluster...", nevents)
    return events.groupby("clust").apply(
        lambda df: df.time.sort_values().to_numpy() / sampling_rate * 1000.0
    )


def trials_to_pprox(trials, sampling_rate):
    """Convert pandas trials to pproc"""
    for trial in trials.itertuples():
        # TODO need to handle empty trials
        pproc = {
            "events": (trial.events.astype("d") - trial.stimulus_start) / sampling_rate,
            "offset": trial.stimulus_start / sampling_rate,
            "index": trial.Index,
            "interval": (
                (trial.recording_start - trial.stimulus_start) / sampling_rate,
                (trial.recording_end - trial.stimulus_start) / sampling_rate,
            ),
            "stimulus": {
                "name": trial.stimulus_name,
                "interval": (
                    0.0,
                    (trial.stimulus_end - trial.stimulus_start) / sampling_rate,
                ),
            },
            "recording": {
                "entry": trial.recording_entry,
                "start": trial.recording_start,
                "end": trial.recording_end,
            },
        }
        yield pproc


def group_spikes_script(argv=None):
    import nbank
    import argparse
    from dlab.util import setup_log, json_serializable
    from dlab.core import __version__, get_or_verify_datafile
    from dlab.extracellular import entry_metadata, iter_entries

    version = "2022.10.11"

    p = argparse.ArgumentParser(
        description="group kilosorted spikes into pprox files based on cluster and trial"
    )
    p.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {version} (melizalab-tools {__version__})",
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    p.add_argument(
        "--dry-run",
        help="do everything except write the output files",
        action="store_true",
    )
    p.add_argument(
        "--sync",
        default="sync",
        help="name of channel with synchronization signal (default '%(default)s')",
    )
    p.add_argument(
        "--sync-thresh",
        default=30.0,
        type=float,
        help="threshold (z-score) for detecting sync clicks (default %(default)0.1f)",
    )
    p.add_argument(
        "--prepad",
        type=float,
        default=1.0,
        help="sets trial start time relative to stimulus onset (default %(default)0.1f s)",
    )
    p.add_argument(
        "--toelis",
        action="store_true",
        help="output toelis instead of pprox. one file will be generated for "
        "the entire recording (including multiunits)",
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
        type=Path,
        default=".",
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
    p.add_argument("recording", help="ARF recording (local file, or neurobank id/URL)")
    p.add_argument(
        "sortdir",
        type=Path,
        help="kilosort output directory. Needs to contain 'spike_times.npy', 'spike_clusters.npy',"
        " 'cluster_info.tsv', and 'temp_wh.dat'",
    )
    args = p.parse_args(argv)
    setup_log(log, args.debug)

    datafile, resource_info = get_or_verify_datafile(args.recording, args.debug)
    recording_name = resource_info["name"]
    resource_url = nbank.full_url(recording_name)
    log.info("- kilosort output directory: %s", args.sortdir)
    timefile = args.sortdir / "spike_times.npy"
    clustfile = args.sortdir / "spike_clusters.npy"
    infofile = args.sortdir / "cluster_info.tsv"
    log.info("  - spike times: %s", timefile)
    log.info("  - spike clusters: %s", clustfile)
    events = pd.DataFrame(
        {"time": np.load(timefile).squeeze(), "clust": np.load(clustfile)},
    ).set_index("clust")
    log.info("  - cluster info: %s", infofile)
    info = pd.read_csv(infofile, sep="\t", index_col=0)
    recfile = args.sortdir / "temp_wh.dat"
    params = read_kilo_params(args.sortdir / "params.py")
    recording = np.memmap(recfile, mode="c", dtype=params["dtype"])
    recording = np.reshape(
        recording, (recording.size // params["nchannels"], params["nchannels"])
    )
    nsamples, nchannels = recording.shape
    log.info("  - filtered recording: %s", recfile)
    log.info("    - %d samples, %d channels", nsamples, nchannels)
    if args.cluster is not None:
        log.info("- only analyzing clusters: %s", args.cluster)
        events = events.loc[args.cluster]

    if args.toelis:
        import toelis

        clusters = assign_events_flat(events, params["sampling_rate"])
        outfile = (args.output / recording_name).with_suffix(".toe_lis")
        if not args.dry_run:
            with open(outfile, "wt") as ofp:
                toelis.write(ofp, clusters)
            log.info("- saved %d spikes to '%s'", toelis.count(clusters), outfile)
        return

    log.info("- splitting recording into trials:")
    with h5.File(datafile, "r") as afp:
        trials = pd.DataFrame(
            oeaudio_to_trials(afp, args.sync, args.sync_thresh, args.prepad)
        )
        entry_attrs = tuple(entry_metadata(e) for _, e in iter_entries(afp))

    # this pandas magic sorts the events by cluster and trial
    log.info("- sorting events into trials:")
    events["trial"] = trials.recording_start.searchsorted(events.time, side="left") - 1
    clusters = (
        events.groupby("clust")
        .apply(lambda df: df.groupby("trial").apply(lambda x: x.time.to_numpy()))
        .unstack("clust")
        .unstack("clust")  # I have no idea why this is necessary
    )

    total_spikes = 0
    total_clusters = 0
    for clust_id, cluster in clusters.items():
        clust_info = info.loc[clust_id]
        clust_type = clust_info["group"]
        clust_trials = trials.join(cluster.rename("events"))
        n_spikes = clust_trials.events.apply(len).agg("sum")
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
        outfile = args.output / f"{recording_name}_c{clust_id}.pprox"
        log.info(
            "  ✓ cluster %d (%d spikes, %s) -> %s",
            clust_id,
            n_spikes,
            clust_type,
            outfile,
        )

        clust_trials = pprox.from_trials(
            trials_to_pprox(clust_trials, params["sampling_rate"]),
            recording=resource_url,
            processed_by=[f"{p.prog} {version}"],
            kilosort_amplitude=clust_info["Amplitude"],
            kilosort_contam_pct=clust_info["ContamPct"],
            kilosort_source_channel=clust_info["ch"],
            kilosort_probe_depth=clust_info["depth"],
            kilosort_n_spikes=clust_info["n_spikes"],
            entry_metadata=entry_attrs,
            **resource_info["metadata"],
        )
        log.info(
            "    - computing average spike waveform from channel %d", clust_info["ch"]
        )
        n_before = int(args.waveform_pre_peak * params["sampling_rate"] / 1000)
        n_after = int(args.waveform_post_peak * params["sampling_rate"] / 1000)
        spikes = events.loc[clust_id]
        spikes = spikes[(spikes.time > n_before) & (spikes.time < (nsamples - n_after))]
        selected = spikes.sample(
            min(args.waveform_num_spikes, n_spikes), random_state=args.waveform_seed
        )
        times = sorted(selected.time)
        waveforms = qs.peaks(recording[:, clust_info["ch"]], times, n_before, n_after)
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
        clust_trials["waveform"] = {
            "mean": waveforms.mean(0),
            "sampling_rate": params["sampling_rate"] * args.waveform_upsample,
        }
        if args.save_waveforms:
            np.save(args.output / (outfile.stem + "_spikes.npy"), waveforms)

        if not args.dry_run:
            with open(outfile, "wt") as ofp:
                json.dump(clust_trials, ofp, default=json_serializable)
    log.info(
        "- a total of %d spikes were assigned to %d clusters",
        total_spikes,
        total_clusters,
    )
