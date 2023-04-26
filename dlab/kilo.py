# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Functions for using kilosort/phy data """
import json
import logging
import re
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator

import anyio
import ewave
import h5py as h5
import numpy as np
import pandas as pd
import quickspikes as qs
import toelis
from httpx import AsyncClient

from dlab import nbank, pprox
from dlab.spikes import SpikeWaveforms, save_waveforms

logging.getLogger(__name__).addHandler(logging.NullHandler())

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


def read_kilo_params(fname: Path) -> Dict:
    """Read the kilosort params.py file"""
    from configparser import ConfigParser
    from itertools import chain

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


def oeaudio_stims(dset: h5.Dataset) -> Iterator[Stimulus]:
    """Parse the messages in the 'stim' dataset to get a table of stimuli with
    start samples. Note that these need to be corrected for offset of the
    recording and network lag.

    """
    re_start = re.compile(r"start (.*)")
    for row in dset:
        time = row["start"]
        message = row["message"].decode("utf-8")
        m = re_start.match(message)
        if m is not None:
            stim_name = Path(m.group(1)).stem
            yield Stimulus(stim_name, time)


@lru_cache(maxsize=None)
async def stim_duration(session: AsyncClient, stim_name: str) -> float:
    """
    Returns the duration of a stimulus (in s). This can only really be done by
    downloading the stimulus from the registry, because the start/stop times are
    not reliable. We try to speed this up by memoizing the function and caching
    the downloaded files.

    """
    target = await nbank.find_resource(session, nbank.default_registry, stim_name)
    with ewave.wavfile(str(target)) as fp:
        return 1.0 * fp.nframes / fp.sampling_rate


async def oeaudio_to_trials(
    session: AsyncClient,
    data_file: h5.File,
    sync_dset: str,
    sync_thresh: float = 1.0,
    prepad: float = 1.0,
) -> Iterator[Trial]:
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
        entry_datetime,
        entry_time,
        find_stim_dset,
        iter_entries,
    )

    expt_start = None
    det = qs.detector(sync_thresh, 10)
    trials = []
    for entry_num, entry in iter_entries(data_file):
        logging.info(" - entry: '%s'", entry.name)
        entry_start = entry_time(entry)
        logging.info("  - start time: %s", entry_datetime(entry))
        if expt_start is None:
            expt_start = entry_start

        logging.info("  - parsing stimulus log")
        stims = list(oeaudio_stims(find_stim_dset(entry)))
        logging.info("    - detected %d stimuli", len(stims))
        logging.info("  - sync track: '%s'", sync_dset)
        sync = entry[sync_dset]
        sync_data = sync[:].astype("d")
        det.scale_thresh(sync_data.mean(), sync_data.std())
        stim_onsets = np.asarray(det(sync_data))
        logging.info("    - detected %d clicks", stim_onsets.size)
        dset_offset = sync.attrs["offset"]
        dset_end = sync.size
        sampling_rate = sync.attrs["sampling_rate"]
        stim_sample_offset = int(dset_offset * sampling_rate)
        logging.info("  - recording clock offset: %d", stim_sample_offset)

        if len(stims) != stim_onsets.size:
            logging.warning(
                "  - WARNING: number of stimuli does not match number of clicks. This recording may need to be discarded"
            )

        padding_samples = int(prepad * sampling_rate)
        for stim, onset, offset in zip_longest(
            stims, stim_onsets, stim_onsets[1:], fillvalue=dset_end + padding_samples
        ):
            stim_seconds = await stim_duration(session, stim.name)
            stim_samples = int(stim_seconds * sampling_rate)
            if stim_samples > offset - onset:
                logging.warning(
                    "  - WARNING: stimulus %s is longer than the duration of the trial",
                    stim,
                )
            trials.append(
                Trial(
                    entry_num,
                    onset - padding_samples,
                    offset - padding_samples,
                    stim.name,
                    onset,
                    onset + stim_samples,
                )
            )
    return trials


def assign_events_flat(events: pd.DataFrame, sampling_rate: float):
    """Assign event_times to clusters, generating a large toelis object"""
    nevents, _ = events.shape
    nclusters = events.index.unique().size
    logging.info("- grouping %d spikes into %d clusters...", nevents, nclusters)
    return events.groupby("clust").apply(
        lambda df: df.time.sort_values().to_numpy() / sampling_rate * 1000.0
    )


def trials_to_pprox(trials: pd.DataFrame, sampling_rate: float):
    """Convert pandas trials to pproc"""
    for trial in trials.itertuples():
        if isinstance(trial.events, float):
            events = []
        else:
            events = (trial.events.astype("d") - trial.stimulus_start) / sampling_rate
        pproc = {
            "events": events,
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


async def group_spikes_script(argv=None):
    import argparse
    import os

    from dlab.core import __version__
    from dlab.extracellular import entry_metadata, iter_entries
    from dlab.util import json_serializable

    version = "2023.04.26"

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
        "--artifact-reject-thresh",
        type=float,
        default=6.0,
        help="threshold for rejecting artifact spikes (max absolute amplitude"
        " more than x times max absolute amplitude of the mean spike)",
    )
    p.add_argument(
        "--save-waveforms",
        "-W",
        action="store_true",
        help="save representative waveforms from each unit's main channel in an hdf5 file",
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
    p.add_argument("recording", type=Path, help="path of ARF recording file")
    p.add_argument(
        "sortdir",
        type=Path,
        help="kilosort output directory. Needs to contain 'spike_times.npy', 'spike_clusters.npy',"
        " 'cluster_info.tsv', and 'temp_wh.dat'",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if args.debug else logging.INFO
    )
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    logging.info("- %s version %s", p.prog, version)

    async with AsyncClient(timeout=None) as session:
        resource_info = await nbank.fetch_metadata(
            session, nbank.default_registry, args.recording.stem
        )
        recording_name = resource_info["name"]
        resource_url = nbank.registry.full_url(nbank.default_registry, recording_name)
        logging.info("- kilosort output directory: %s", args.sortdir)
        timefile = args.sortdir / "spike_times.npy"
        clustfile = args.sortdir / "spike_clusters.npy"
        infofile = args.sortdir / "cluster_info.tsv"
        logging.info("  - spike times: %s", timefile)
        logging.info("  - spike clusters: %s", clustfile)
        events = pd.DataFrame(
            {"time": np.load(timefile).squeeze(), "clust": np.load(clustfile)},
        ).set_index("clust")
        logging.info("  - cluster info: %s", infofile)
        info = pd.read_csv(infofile, sep="\t", index_col=0)
        recfile = args.sortdir / "temp_wh.dat"
        params = read_kilo_params(args.sortdir / "params.py")
        recording = np.memmap(recfile, mode="c", dtype=params["dtype"])
        recording = np.reshape(
            recording, (recording.size // params["nchannels"], params["nchannels"])
        )
        nsamples, nchannels = recording.shape
        logging.info("  - filtered recording: %s", recfile)
        logging.info("    - %d samples, %d channels", nsamples, nchannels)
        if args.cluster is not None:
            logging.info("- only analyzing clusters: %s", args.cluster)
            events = events.loc[args.cluster]

        if args.toelis:
            clusters = assign_events_flat(events, params["sampling_rate"])
            outfile = (args.output / recording_name).with_suffix(".toe_lis")
            if not args.dry_run:
                with open(outfile, "wt") as ofp:
                    toelis.write(ofp, clusters)
                logging.info(
                    "- saved %d spikes to '%s'", toelis.count(clusters), outfile
                )
            return

        if args.recording.is_file():
            datafile = args.recording
        else:
            datafile = await nbank.find_resource(
                session, nbank.default_registry, str(args.recording)
            )
        logging.info("- splitting '%s' into trials:", datafile)
        with h5.File(datafile, "r") as afp:
            trials = pd.DataFrame(
                await oeaudio_to_trials(
                    session, afp, args.sync, args.sync_thresh, args.prepad
                )
            )
            entry_attrs = tuple(entry_metadata(e) for _, e in iter_entries(afp))

        # this pandas magic sorts the events by cluster and trial
        logging.info("- sorting events into trials:")
        events["trial"] = (
            trials.recording_start.searchsorted(events.time, side="left") - 1
        )

        total_spikes = 0
        total_clusters = 0
        for clust_id, cluster in events.groupby("clust"):
            clust_info = info.loc[clust_id]
            clust_type = clust_info["group"]
            n_spikes = len(cluster)
            if clust_type == "noise" or (clust_type == "mua" and not args.mua):
                logging.info(
                    "  - cluster %d (%d spikes, %s) -> skipped",
                    clust_id,
                    n_spikes,
                    clust_type,
                )
                continue
            logging.info(
                "  ✓ cluster %d (%d spikes, %s)",
                clust_id,
                n_spikes,
                clust_type,
            )
            # remove artifact spikes
            n_before = int(args.waveform_pre_peak * params["sampling_rate"] / 1000)
            n_after = int(args.waveform_post_peak * params["sampling_rate"] / 1000)
            spikes = cluster[
                (cluster.time > n_before) & (cluster.time < (nsamples - n_after))
            ]
            waveforms = qs.peaks(
                recording[:, clust_info["ch"]],
                spikes.time,
                n_before=n_before,
                n_after=n_after,
            )
            mean_spike = waveforms.mean(0)
            included = np.abs(waveforms).max(-1) < (
                np.abs(mean_spike).max(-1) * args.artifact_reject_thresh
            )
            n_included = included.sum()
            if n_included < n_spikes:
                cluster = cluster[included]
                waveforms = waveforms[included]
                logging.info(
                    "    - %d artifact spike(s) excluded", n_spikes - n_included
                )
            # aggregate spikes by trial and left join to trial information table
            # - empty trials will be nan
            clust_trials = trials.join(
                cluster.groupby("trial")
                .apply(lambda x: x.time.to_numpy())
                .rename("events")
            )
            total_spikes += n_spikes
            total_clusters += 1
            outfile = args.output / f"{recording_name}_c{clust_id}.pprox"
            logging.info(
                "    - %d spikes -> %s",
                n_included,
                outfile,
            )
            clust_trials = pprox.from_trials(
                trials_to_pprox(clust_trials, params["sampling_rate"]),
                schema=pprox._stimtrial_schema,
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
            if not args.dry_run:
                with open(outfile, "wt") as ofp:
                    json.dump(clust_trials, ofp, default=json_serializable)
                if args.save_waveforms:
                    outfile = args.output / (outfile.stem + "_spikes.h5")
                    logging.info(
                        "    - waveforms on channel %d -> %s",
                        clust_info["ch"],
                        outfile,
                    )
                    save_waveforms(
                        outfile,
                        SpikeWaveforms(
                            waveforms,
                            cluster.time.to_numpy(),
                            params["sampling_rate"],
                            n_before,
                        ),
                        recording=resource_url,
                        processed_by=f"{p.prog} {version}",
                        kilosort_amplitude=clust_info["Amplitude"],
                        kilosort_contam_pct=clust_info["ContamPct"],
                        kilosort_source_channel=clust_info["ch"],
                        kilosort_probe_depth=clust_info["depth"],
                        kilosort_n_spikes=clust_info["n_spikes"],
                    )

        logging.info(
            "- a total of %d spikes were assigned to %d clusters",
            total_spikes,
            total_clusters,
        )


def run_group_spikes():
    anyio.run(group_spikes_script)
