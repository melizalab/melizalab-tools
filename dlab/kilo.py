# -*- mode: python -*-
"""Functions for using kilosort/phy data"""

import json
import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import arf
import ewave
import h5py as h5
import numpy as np
import pandas as pd
import quickspikes as qs
import toelis

from dlab import neurobank as nbank
from dlab import pprox
from dlab.spikes import SpikeWaveforms, save_waveforms

log = logging.getLogger(__name__)


class Trial(NamedTuple):
    """Represents the structure of a trial. All time units are in samples."""

    recording_entry: int
    recording_start: int
    recording_end: int
    stimulus_name: str
    stimulus_start: int
    stimulus_end: int


class Stimulus(NamedTuple):
    name: str
    start: int
    end: int | None = None


def read_kilo_params(fname: Path) -> dict:
    """Read the kilosort params.py file"""
    from configparser import ConfigParser
    from itertools import chain

    parser = ConfigParser()
    with open(fname) as lines:
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
    start samples. Note that these will need to be corrected for offset of the
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


def entry_time(entry):
    """Return the timestamp of an entry as a floating point number"""
    from arf import timestamp_to_float

    return timestamp_to_float(entry.attrs["timestamp"])


def iter_entries(data_file):
    """Iterate through the entries in an arf file in order of time"""
    return enumerate(sorted(data_file.values(), key=entry_time))


def find_stim_dset(entry):
    """Returns the first dataset that matches 'Network_Events.*_TEXT'"""
    rex = re.compile(r"Network_Events-.*?TEXT")
    for name in entry:
        if rex.match(name) is not None:
            log.debug("  - stim log dataset: %s", name)
            return entry[name]


def entry_metadata(entry):
    """Extracts metadata from an entry in an oeaudio-present experiment ARF file.

    Metadata are passed to open-ephys through the network events socket as a
    json-encoded dictionary. There should be one and only one metadata message
    per entry, so only the first is returned.

    """
    re_metadata = re.compile(r"metadata: (\{.*\})")
    stims = find_stim_dset(entry)
    for row in stims:
        message = row["message"].decode("utf-8")
        m = re_metadata.match(message)
        try:
            metadata = json.loads(m.group(1))
        except (AttributeError, json.JSONDecodeError):
            pass
        else:
            metadata.update(name=entry.name, sampling_rate=stims.attrs["sampling_rate"])
            return metadata


class StimulusFinder:
    """Looks up stimuli using neurobank and/or files in a local directory"""

    def __init__(self, nbank_registry_url: str, alt_base: Path | None = None):
        self.registry_url = nbank_registry_url
        self.alt_base = alt_base

    def get_durations(self, names: Iterable[str]) -> dict[str, float]:
        """Looks up durations (in s) for a sequence of stimuli. Searches
        neurobank first and then tries local directory.

        """
        output = {}
        for name, res in nbank.find_resources(*names, registry_url=self.registry_url):
            if isinstance(res, FileNotFoundError):
                if self.alt_base is None:
                    raise res
                path = (self.alt_base / name).with_suffix(".wav")
                if not path.exists():
                    raise res
            else:
                path = res
            log.debug("  - found '%s' at %s", name, path)
            with ewave.wavfile(path) as fp:
                output[name] = 1.0 * fp.nframes / fp.sampling_rate
        return output


def oeaudio_to_trials(
    data_file: h5.File,
    stim_finder: StimulusFinder,
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

    expt_start = None
    det = qs.detector(sync_thresh, 10)
    trials = []
    for entry_num, entry in iter_entries(data_file):
        log.info(" - entry: '%s'", entry.name)
        entry_start = arf.timestamp_to_float(entry.attrs["timestamp"])
        log.info(
            "  - start time: %s", arf.timestamp_to_datetime(entry.attrs["timestamp"])
        )
        if expt_start is None:
            expt_start = entry_start

        log.info("  - parsing stimulus log")
        entry_stimuli = list(oeaudio_stims(find_stim_dset(entry)))
        try:
            stim_durations = stim_finder.get_durations(
                stim.name for stim in entry_stimuli
            )
        except FileNotFoundError as err:
            raise RuntimeError(
                "unable to find a stimulus to look up duration. Was it deposited in neurobank?"
            ) from err
        log.info("    - detected %d stimuli", len(entry_stimuli))
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

        entry_stimuli = match_clicks(entry_stimuli, stim_onsets)

        padding_samples = int(prepad * sampling_rate)
        for stim, onset, offset in zip_longest(
            entry_stimuli,
            stim_onsets,
            stim_onsets[1:],
            fillvalue=dset_end + padding_samples,
        ):
            stim_seconds = stim_durations[stim.name]
            stim_samples = int(stim_seconds * sampling_rate)
            if stim_samples > offset - onset:
                log.warning(
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


def match_clicks(entry_stimuli, stim_onsets):
    """Match clicks in the sync track to the stimulus onset log.

    If the number of clicks matches the number of stimuli, nothing happens. If
    there are more clicks than stimuli, this is an error. If there are more
    stimuli than clicks, attempts to match each click with the next onset,
    discarding any stimuli that don't have a matching click.

    """
    if len(entry_stimuli) == stim_onsets.size:
        logging.debug(" - Number of stimuli matches number of clicks")
        return entry_stimuli
    elif len(entry_stimuli) < stim_onsets.size:
        logging.error(
            "  - Number of stimuli (%d) is fewer than the number of clicks (%d)",
            len(entry_stimuli),
            stim_onsets.size,
        )
        raise ValueError(
            "  - Error: unable to match clicks in the sync track with the stimulus list. Either discard recording or change sync threshold."
        )
    logging.info(
        "  - Number of stimuli (%d) is greater than number of clicks (%d). Trying to repair.",
        len(entry_stimuli),
        stim_onsets.size,
    )
    # this algorithm assumes that the click comes before the message, which is
    # pretty reasonable given that network delays will be longer than analog
    # signal propagation.
    matched_stims = []
    click_times = set(stim_onsets)
    for i, stim in enumerate(entry_stimuli):
        idx = np.searchsorted(stim_onsets, stim.start)
        closest = stim_onsets[idx - 1]
        logging.info(
            "   - stim %d (start=%d) matched to click at %d", i, stim.start, closest
        )
        if closest in click_times:
            matched_stims.append(stim)
            click_times.remove(closest)
        else:
            logging.info("     - this click was already used, dropping the trial")
    return matched_stims


def assign_events_flat(events: pd.DataFrame, sampling_rate: float):
    """Assign event_times to clusters, generating a large toelis object"""
    nevents, _ = events.shape
    nclusters = events.index.unique().size
    log.info("- grouping %d spikes into %d clusters...", nevents, nclusters)
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


def group_spikes_script(argv=None):
    import argparse
    import os

    from dlab import __version__
    from dlab.util import json_serializable, setup_log

    version = "2025.07.28"

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
    nbank.add_registry_argument(p)
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
        help="threshold for rejecting artifact spikes (default %(default).1f; max absolute amplitude"
        " more than x times max absolute amplitude of the mean spike)",
    )
    p.add_argument(
        "--no-waveforms",
        "-W",
        action="store_true",
        help="if set, do not save representative waveforms from each unit's main channel in an hdf5 file",
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
        "--local-stim-dir",
        type=Path,
        help="DEBUG/TESTING ONLY. Search this directory for stimulus files.",
    )
    p.add_argument("recording", type=Path, help="path of ARF recording file")
    p.add_argument(
        "sortdir",
        type=Path,
        help="kilosort output directory. Needs to contain 'spike_times.npy', 'spike_clusters.npy',"
        " 'cluster_info.tsv', and 'temp_wh.dat'",
    )
    args = p.parse_args(argv)
    setup_log(args.debug)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    log.info("- %s version %s", p.prog, version)

    log.info("- using neurobank registry at %s", args.registry_url)
    log.info("- using stimulus times from %s", args.recording)
    try:
        resource_info = nbank.describe(args.registry_url, args.recording.stem)
        recording_name = resource_info["name"]
        resource_url = nbank.registry.full_url(args.registry_url, recording_name)
        log.info("  - registered at %s", resource_url)
    except TypeError:
        if args.debug:
            resource_info = {"metadata": {}}
            recording_name = args.recording.stem
            resource_url = "(debug)"
            log.warning(
                "  - warning: recording has not been deposited, proceeding anyway in debug mode"
            )
        else:
            log.error("  - error: recording must be deposited in neurobank")
            p.exit(-1)

    if args.local_stim_dir is not None:
        log.info("  - using %s as fallback for looking up stimuli", args.local_stim_dir)
    stim_finder = StimulusFinder(args.registry_url, args.local_stim_dir)

    log.info("- kilosort output directory: %s", args.sortdir)
    timefile = args.sortdir / "spike_times.npy"
    clustfile = args.sortdir / "spike_clusters.npy"
    infofile = args.sortdir / "cluster_info.tsv"
    log.info("  - spike times: %s", timefile)
    log.info("  - spike clusters: %s", clustfile)
    events = pd.DataFrame(
        {"time": np.load(timefile).squeeze(), "clust": np.load(clustfile)},
    )
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
        events = events[events.cluster == args.cluster]

    # find duplicates - this is rare but needs to be caught
    duplicates = events.duplicated()
    if duplicates.any():
        dupl_clusts = events[duplicates].clust
        log.warning(
            "  - warning: removing spikes with duplicate times from clusters %s",
            ",".join(str(c) for c in dupl_clusts),
        )
        events = events[~duplicates]

    events.set_index("clust", inplace=True)
    if args.toelis:
        clusters = assign_events_flat(events, params["sampling_rate"])
        outfile = (args.output / recording_name).with_suffix(".toe_lis")
        if not args.dry_run:
            with open(outfile, "w") as ofp:
                toelis.write(ofp, clusters)
            log.info("- saved %d spikes to '%s'", toelis.count(clusters), outfile)
        return

    if args.recording.is_file():
        datafile = args.recording
    else:
        datafile = nbank.find_resource(
            str(args.recording), registry_url=nbank.default_registry
        )
    log.info("- splitting '%s' into trials:", datafile)
    with h5.File(datafile, "r") as afp:
        trials = pd.DataFrame(
            oeaudio_to_trials(
                afp, stim_finder, args.sync, args.sync_thresh, args.prepad
            )
        )
        entry_attrs = tuple(entry_metadata(e) for _, e in iter_entries(afp))

    # this pandas magic sorts the events by cluster and trial
    log.info("- sorting events into trials:")
    events["trial"] = trials.recording_start.searchsorted(events.time, side="left") - 1

    total_spikes = 0
    total_clusters = 0
    good_clust_types = ("good",)
    if args.mua:
        good_clust_types += ("mua",)
    for clust_id, cluster in events.groupby("clust"):
        clust_info = info.loc[clust_id]
        clust_type = clust_info["group"]
        n_spikes = len(cluster)
        if clust_type not in good_clust_types:
            log.info(
                "  - cluster %d (%d spikes, %s) -> skipped",
                clust_id,
                n_spikes,
                clust_type,
            )
            continue
        log.info(
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
        if n_included == 0:
            log.warning("   - all spikes marked as artifacts (sorting error?)")
            continue
        elif n_included < n_spikes:
            cluster = cluster[included]
            waveforms = waveforms[included]
            log.info("    - %d artifact spike(s) excluded", n_spikes - n_included)
        # aggregate spikes by trial and left join to trial information table
        # - empty trials will be nan
        clust_trials = trials.join(
            cluster.groupby("trial")
            .apply(lambda x: x.time.to_numpy(), include_groups=False)
            .rename("events")
        )
        total_spikes += n_spikes
        total_clusters += 1
        outfile = args.output / f"{recording_name}_c{clust_id}.pprox"
        log.info(
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
            with open(outfile, "w") as ofp:
                json.dump(clust_trials, ofp, default=json_serializable)
            if not args.no_waveforms:
                outfile = args.output / (outfile.stem + "_spikes.h5")
                log.info(
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

    log.info(
        "- a total of %d spikes were assigned to %d clusters",
        total_spikes,
        total_clusters,
    )


if __name__ == "__main__":
    group_spikes_script()
