# -*- mode: python -*-
"""Functions for using kilosort/phy data"""

import datetime
import io
import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

import arf
import ewave
import h5py as h5
import numpy as np
import pandas as pd
import toelis

from dlab import neurobank as nbank
from dlab import pprox
from dlab.spikes import SpikeWaveforms, save_waveforms

log = logging.getLogger(__name__)


class Trial(NamedTuple):
    """Represents the structure of a trial. All time units are in samples.

    aux is None if no auxiliary channels were requested; otherwise it holds a
    (name, start, end) tuple for each auxiliary pulse that starts in the trial.
    """

    recording_entry: int
    recording_start: int
    recording_end: int
    stimulus_name: str
    stimulus_start: int
    stimulus_end: int
    aux: tuple | None = None


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


def oeaudio_log_stims(
    oeaudio_log: io.TextIOBase, sampling_rate: int
) -> Iterator[Stimulus]:
    """Parse an open-ephys-audio log to get a table of stimuli with start
    samples. This function can be used when the 'stim' dataset is missing from
    the recording (e.g., during the time period when we were falsely assuming
    that the new version of the NetworkEvents plugin was storing these
    messages)"""
    re_start = re.compile(r'"start (.*)"')
    start_acq_time = None
    for i, line in enumerate(oeaudio_log):
        stripped = line.strip()
        if stripped.startswith("#") or len(stripped) == 0:
            continue
        timestamp, message = stripped.split(",", maxsplit=1)
        try:
            ts = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError as err:
            log.warning("      - line %d: error parsing timestampe: %s", i, err)
            continue
        if message == '"StartAcquisition"':
            start_acq_time = ts
            log.debug("      - acquisition started at %s", ts)
            continue
        m = re_start.match(message)
        if m is not None:
            if start_acq_time is None:
                raise ValueError(
                    f"line {i}: stimulus started before StartAcquisition in the log"
                )
            offset = (ts - start_acq_time).total_seconds() * sampling_rate
            stim_name = Path(m.group(1)).stem
            yield Stimulus(stim_name, int(offset))


def entry_time(entry):
    """Return the timestamp of an entry as a floating point number"""
    from arf import timestamp_to_float

    return timestamp_to_float(entry.attrs["timestamp"])


def iter_entries(data_file):
    """Iterate through the entries in an arf file in order of time"""
    return enumerate(sorted(data_file.values(), key=entry_time))


def find_stim_dset(entry):
    """Returns the dataset with the network messages from the stimulus
    presentation script, or None if there isn't one.

    arfx-oephys names this dataset 'MessageCenter' for open-ephys GUI >= 0.6
    and after the Network Events plugin's text channel (e.g.
    'Network_Events-104.0_TEXT_group_1') for earlier versions. Empty datasets
    are skipped: some recordings have an empty message dataset because logging
    to it was not enabled.

    """
    rex = re.compile(r"MessageCenter|Network_Events-.*_TEXT_")
    for name in entry:
        if rex.match(name) is not None and entry[name].size > 0:
            log.debug("  - stim log dataset: %s", name)
            return entry[name]


def entry_metadata(entry):
    """Extracts metadata from an entry in an oeaudio-present experiment ARF file.

    Metadata are passed to open-ephys through the network events socket as a
    json-encoded dictionary. There should be at most one metadata message
    per entry, so only the first is returned. If there is none (jpresent does
    not send one), only the entry name and sampling rate are returned.

    """
    re_metadata = re.compile(r"metadata: (\{.*\})")
    stim_dset = find_stim_dset(entry)
    if stim_dset is None:
        log.warning(
            "  - no stimulus log dataset in %s; not saving entry metadata", entry.name
        )
        # use sampling rate in the first sampled dataset
        for dset_name in entry:
            try:
                sampling_rate = entry[dset_name].attrs["sampling_rate"]
                return {"sampling_rate": sampling_rate}
            except KeyError:
                pass
        log.warning("  - unable to infer sampling rate for the entry")
        return {"sampling_rate": "unknown"}
    for row in stim_dset:
        message = row["message"].decode("utf-8")
        m = re_metadata.match(message)
        try:
            metadata = json.loads(m.group(1))
        except (AttributeError, json.JSONDecodeError):
            pass
        else:
            metadata.update(
                name=entry.name, sampling_rate=stim_dset.attrs["sampling_rate"]
            )
            return metadata
    # jpresent does not send a metadata message
    log.debug("  - no metadata message in %s", stim_dset.name)
    return {"name": entry.name, "sampling_rate": stim_dset.attrs["sampling_rate"]}


def detect_pulses(
    data: np.ndarray, thresh: float = 0.5, min_snr: float = 20.0
) -> np.ndarray:
    """Returns the onset and offset of each pulse in a signal, as an (n, 2) array.

    The threshold is set `thresh` of the way from the baseline (5th percentile)
    to the peak (maximum) of the signal. This works for brief clicks and for
    pulses that stay high for the duration of the stimulus, however long they
    are, as long as the signal is high less than 95% of the time. The onset is
    the first sample at or above the threshold, and the offset the first sample
    after it below the threshold (or the length of the data, if the pulse is
    still high at the end). A pulse already high at the start of the data is
    not included.

    The baseline and noise are estimated from about a million evenly spaced
    samples. Sync events should be unambiguous, so if the peak is less than
    `min_snr` times the baseline noise (a robust SD of the samples below the
    threshold) above the baseline, the signal is treated as having no events
    and an empty array is returned.

    """
    if not 0 < thresh < 1:
        raise ValueError(f"threshold must be between 0 and 1 (got {thresh})")
    # baseline and noise are estimated from a subsample, which bounds memory use
    # for long recordings; the peak and the crossings use every sample
    sample = data[:: max(1, data.size // 1_000_000)]
    baseline = np.percentile(sample, 5)
    peak = data.max()
    level = baseline + thresh * (peak - baseline)
    below = sample[sample < level].astype("d")
    noise = 1.4826 * np.median(np.abs(below - np.median(below)))
    if peak - baseline < min_snr * noise:
        log.debug("    - peak is only %.1f x the noise", (peak - baseline) / noise)
        return np.empty((0, 2), dtype=int)
    above = data >= level
    rises = np.flatnonzero(~above[:-1] & above[1:]) + 1
    falls = np.flatnonzero(above[:-1] & ~above[1:]) + 1
    offsets = np.append(falls, data.size)[np.searchsorted(falls, rises)]
    return np.column_stack([rises, offsets])


def detect_sync_onsets(
    data: np.ndarray, thresh: float = 0.5, min_snr: float = 20.0
) -> np.ndarray:
    """Returns the sample indices where a sync signal rises through a threshold.

    These are the onsets of the pulses found by detect_pulses (see there for
    how the threshold is set).

    """
    return detect_pulses(data, thresh, min_snr)[:, 0]


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
    sync_thresh: float = 0.5,
    prepad: float = 1.0,
    *,
    oeaudio_log: Path | None,
    aux: Mapping[str, str] | None = None,
) -> list[Trial]:
    """Extracts trial information from an oeaudio-present experiment ARF file

    When using oeaudio-present, a single recording is made in response to all
    the stimuli. The stimulus presentation script sends network events to
    open-ephys to mark the start and stop of each stimulus. There is typically a
    significant lag between the 'start' event and the onset of the stimulus, due
    to buffering of the audio playback. However, the presentation script will
    play a synchronization signal on a second channel: a brief click at each
    stimulus onset (old style), or a pulse that stays high for the duration of
    the stimulus (new style). As long as the user remembers to record this
    channel, it can be used to correct the onset values. `sync_thresh` sets the
    detection threshold as a fraction of the way from the sync channel's
    baseline to its peak (see detect_sync_onsets).

    The continuous recording is broken up into trials based on the stimulus
    presentation, such that each trial encompasses one and only one stimulus.
    The `prepad` parameter specifies, in seconds, when trials begin relative to
    stimulus onset. The default is 1.0 s.

    If oeaudio_log is set, it's used instead of the network event datasets.

    aux maps names to the datasets of auxiliary channels (e.g. {"led":
    "ADC4"}) carrying pulses from other devices, such as an optogenetic light
    source or a sensor's TTL output. Pulses are detected as for the sync
    channel, and each is assigned, unclipped, to the trial in which it starts
    (see Trial.aux).

    """
    from itertools import zip_longest

    expt_start = None
    trials = []

    for entry_num, entry in iter_entries(data_file):
        log.info(" - entry: '%s'", entry.name)
        entry_start = arf.timestamp_to_float(entry.attrs["timestamp"])
        log.info(
            "  - start time: %s", arf.timestamp_to_datetime(entry.attrs["timestamp"])
        )
        if expt_start is None:
            expt_start = entry_start

        log.info("  - sync track: '%s'", sync_dset)
        try:
            sync = entry[sync_dset]
        except KeyError as err:
            available_tracks = ", ".join(entry.keys())
            raise RuntimeError(
                f"unable to find sync track. Use --sync to configure. Options are: {available_tracks}"
            ) from err

        stim_onsets = detect_sync_onsets(sync[:], sync_thresh)
        log.info("    - detected %d sync events", stim_onsets.size)
        if stim_onsets.size == 0:
            raise RuntimeError(
                f"no sync events detected in '{sync_dset}'. Check --sync and --sync-thresh."
            )
        dset_offset = sync.attrs["offset"]
        dset_end = sync.size
        sampling_rate = sync.attrs["sampling_rate"]
        stim_sample_offset = round(dset_offset * sampling_rate)
        log.info("  - recording clock offset: %d", stim_sample_offset)

        stim_dset = None
        if oeaudio_log is not None:
            log.info("  - parsing stimulus log from %s", oeaudio_log)
            with open(oeaudio_log) as fp:
                entry_stimuli = list(oeaudio_log_stims(fp, sampling_rate))
        else:
            stim_dset = find_stim_dset(entry)
            if stim_dset is None:
                raise RuntimeError(
                    "unable to find stimulus list in ARF file. You may need to provide the oeaudio logfile"
                )
            log.info("  - parsing stimulus log from %s", stim_dset)
            entry_stimuli = list(oeaudio_stims(stim_dset))
        try:
            stim_durations = stim_finder.get_durations(
                stim.name for stim in entry_stimuli
            )
        except FileNotFoundError as err:
            raise RuntimeError(
                "unable to find a stimulus to look up duration. Was it deposited in neurobank?"
            ) from err
        log.info("    - detected %d stimuli", len(entry_stimuli))

        # message times are open-ephys sample numbers, which count from the
        # start of acquisition; convert to samples from the start of the sync
        # track. Log times are assumed to have the same origin (StartAcquisition).
        entry_stimuli = [
            stim._replace(start=stim.start - stim_sample_offset)
            for stim in entry_stimuli
        ]
        entry_stimuli = match_clicks(entry_stimuli, stim_onsets)
        starts = np.array([stim.start for stim in entry_stimuli])
        lags = (stim_onsets - starts) / sampling_rate
        log.info("    - sync events follow start messages by %.3f s", np.median(lags))
        for i in sync_lag_outliers(starts, stim_onsets, sampling_rate):
            log.warning(
                "  - WARNING: sync event for stimulus %d (%s) is %.3f s after its "
                "start message (median %.3f s). Check the sync track.",
                i,
                entry_stimuli[i].name,
                lags[i],
                np.median(lags),
            )

        padding_samples = int(prepad * sampling_rate)
        entry_trials = []
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
            entry_trials.append(
                Trial(
                    entry_num,
                    onset - padding_samples,
                    offset - padding_samples,
                    stim.name,
                    onset,
                    onset + stim_samples,
                )
            )
        if aux:
            entry_trials = assign_aux_pulses(
                entry_trials, aux_pulses(entry, aux, sync_thresh)
            )
            if stim_dset is not None:
                conditions = [
                    c._replace(start=c.start - stim_sample_offset)
                    for c in oeaudio_conditions(stim_dset)
                ]
                if conditions:
                    check_aux_conditions(
                        entry_trials, entry_stimuli, conditions, sampling_rate
                    )
        trials.extend(entry_trials)
    return trials


def oeaudio_conditions(dset: h5.Dataset) -> list[Stimulus]:
    """Parse the 'condition_start <stimulus>' messages in the stimulus message
    dataset. jpresent sends one with each stimulus presented under an
    experimental condition (e.g. optogenetic stimulation). Times are sample
    numbers, as for oeaudio_stims."""
    re_condition = re.compile(r"condition_start (.*)")
    out = []
    for row in dset:
        m = re_condition.match(row["message"].decode("utf-8"))
        if m is not None:
            out.append(Stimulus(Path(m.group(1)).stem, int(row["start"])))
    return out


def check_aux_conditions(
    trials: list[Trial],
    stimuli: list[Stimulus],
    conditions: list[Stimulus],
    sampling_rate: float,
    tolerance: float = 0.5,
) -> set[int]:
    """Checks that the trials with condition messages are the trials with
    auxiliary pulses, and logs a warning for each that isn't. Returns the
    indices of the trials with a condition message.

    trials and stimuli are the trials of one entry and their (matched)
    stimuli, and conditions the condition messages, with start times in the
    same units (samples from the start of the sync track). Each condition
    message is assigned to the trial whose start message is nearest, if that
    is within tolerance (in s) and names the same stimulus.

    """
    starts = np.array([stim.start for stim in stimuli])
    with_condition = set()
    for cond in conditions:
        i = int(np.argmin(np.abs(starts - cond.start)))
        if (
            abs(starts[i] - cond.start) > tolerance * sampling_rate
            or stimuli[i].name != cond.name
        ):
            log.warning(
                "  - WARNING: condition message for %s (sample %d) does not match "
                "any trial (was the trial dropped?)",
                cond.name,
                cond.start,
            )
        else:
            with_condition.add(i)
    with_pulses = {i for i, trial in enumerate(trials) if trial.aux}
    log.info(
        "    - %d trials with condition messages, %d with aux pulses",
        len(with_condition),
        len(with_pulses),
    )
    for i in sorted(with_condition - with_pulses):
        log.warning(
            "  - WARNING: trial %d (%s) has a condition message but no aux pulses",
            i,
            trials[i].stimulus_name,
        )
    for i in sorted(with_pulses - with_condition):
        log.warning(
            "  - WARNING: trial %d (%s) has aux pulses but no condition message",
            i,
            trials[i].stimulus_name,
        )
    return with_condition


def aux_pulses(entry, aux: Mapping[str, str], thresh: float) -> list[tuple]:
    """Detects the pulses on each auxiliary channel in an entry. Returns a list
    of (name, start, end) tuples, sorted by start."""
    pulses = []
    for name, dset_name in aux.items():
        try:
            dset = entry[dset_name]
        except KeyError as err:
            raise RuntimeError(
                f"unable to find auxiliary channel '{dset_name}' for '{name}'. "
                f"Options are: {', '.join(entry.keys())}"
            ) from err
        detected = detect_pulses(dset[:], thresh)
        log.info(
            "  - aux '%s' (%s): detected %d pulses", name, dset_name, len(detected)
        )
        pulses.extend((name, int(on), int(off)) for on, off in detected)
    return sorted(pulses, key=lambda pulse: pulse[1])


def assign_aux_pulses(trials: list[Trial], pulses: list[tuple]) -> list[Trial]:
    """Returns trials (from one entry, in order) with each pulse assigned to the
    trial in which it starts. Pulses that start before the first trial are
    dropped with a warning."""
    starts = np.array([trial.recording_start for trial in trials])
    assigned = [[] for _ in trials]
    for pulse in pulses:
        i = np.searchsorted(starts, pulse[1], side="right") - 1
        if i < 0:
            log.warning(
                "  - aux pulse %s starts before the first trial; dropped", pulse
            )
        else:
            assigned[i].append(pulse)
    return [
        trial._replace(aux=tuple(pulses))
        for trial, pulses in zip(trials, assigned, strict=True)
    ]


def match_clicks(
    entry_stimuli: list[Stimulus], stim_onsets: np.ndarray
) -> list[Stimulus]:
    """Match sync events to stimuli, returning the stimulus for each sync event.

    Stimulus start times and sync onsets must be in the same units (samples
    from the start of the sync track). The presentation script sends each
    stimulus's start message before the sound, and its sync event, comes out
    of the audio buffer, so each sync event is matched to the last stimulus
    that started at or before it. Stimuli with no sync event (e.g. a sync
    event that was not detected) are dropped with a warning. A sync event
    before any stimulus, or two sync events after the same stimulus, can't be
    resolved and raise ValueError.

    """
    starts = np.array([stim.start for stim in entry_stimuli])
    if np.any(np.diff(starts) < 0):
        raise ValueError("stimulus start times are not in order")
    idx = np.searchsorted(starts, stim_onsets, side="right") - 1
    if np.any(idx < 0):
        raise ValueError(
            f"sync event at sample {stim_onsets[idx < 0][0]} comes before any "
            "stimulus. Check --sync-thresh, or discard the recording."
        )
    repeated = np.flatnonzero(np.diff(idx) == 0)
    if repeated.size > 0:
        i = idx[repeated[0]]
        raise ValueError(
            f"more than one sync event after stimulus {i} ({entry_stimuli[i].name}). "
            "Check --sync-thresh, or discard the recording."
        )
    for i in sorted(set(range(len(entry_stimuli))) - set(idx.tolist())):
        log.warning(
            "  - no sync event for stimulus %d (%s); dropping the trial",
            i,
            entry_stimuli[i].name,
        )
    return [entry_stimuli[i] for i in idx]


def sync_lag_outliers(
    starts: np.ndarray, onsets: np.ndarray, sampling_rate: float, tolerance: float = 0.1
) -> np.ndarray:
    """Returns the indices of trials whose sync lag is out of line with the rest.

    starts and onsets are the start-message times of matched stimuli and their
    sync onsets, in samples. Each sync event follows its message by a lag that
    depends on the presentation setup (0.25-1 s in the example recordings) but
    varies little within a recording (by less than 65 ms). A lag that differs
    from the median by more than tolerance (in s) suggests a misdetected or
    mismatched sync event, or stimulus times that don't belong to the recording.
    The median is taken as the right lag, so this relies on most trials being
    right; if half or more are wrong, it flags the wrong trials.

    """
    lags = (np.asarray(onsets) - np.asarray(starts)) / sampling_rate
    if lags.size == 0:
        return np.array([], dtype=int)
    return np.flatnonzero(np.abs(lags - np.median(lags)) > tolerance)


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
        aux = getattr(trial, "aux", None)
        if isinstance(aux, tuple):
            pproc["aux"] = [
                {
                    "name": name,
                    "interval": (
                        (start - trial.stimulus_start) / sampling_rate,
                        (end - trial.stimulus_start) / sampling_rate,
                    ),
                }
                for name, start, end in aux
            ]
        yield pproc


def group_spikes_script(argv=None):
    import argparse
    import os

    from dlab import __version__
    from dlab.util import ParseKeyVal, json_serializable, setup_log

    version = "2026.07.15"

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
        default="ADC3",
        help="name of channel with synchronization signal (default '%(default)s')",
    )
    p.add_argument(
        "--sync-thresh",
        default=0.5,
        type=float,
        help="threshold for detecting sync events, as a fraction of the way from the "
        "sync channel's baseline to its peak (default %(default)0.2f)",
    )
    p.add_argument(
        "--aux",
        action=ParseKeyVal,
        metavar="NAME=CHANNEL",
        help="record pulses on an auxiliary channel (e.g. an optogenetic light "
        "source or a sensor's TTL output) in each trial as NAME. May be repeated.",
    )
    p.add_argument(
        "--oeaudio-log",
        type=Path,
        help="use an open-ephys-audio logfile to determine list of stimuli instead of using network messages",
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
    # read-only: a copy-on-write map of a whole sort can exceed available memory
    recording = np.memmap(recfile, mode="r", dtype=params["dtype"])
    recording = np.reshape(
        recording, (recording.size // params["nchannels"], params["nchannels"])
    )
    nsamples, nchannels = recording.shape
    log.info("  - filtered recording: %s", recfile)
    log.info("    - %d samples, %d channels", nsamples, nchannels)
    if args.cluster is not None:
        log.info("- only analyzing clusters: %s", args.cluster)
        events = events[events.clust.isin(args.cluster)]

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
                afp,
                stim_finder,
                args.sync,
                args.sync_thresh,
                args.prepad,
                oeaudio_log=args.oeaudio_log,
                aux=args.aux,
            )
        )
        entry_attrs = tuple(entry_metadata(e) for _, e in iter_entries(afp))

    # this pandas magic sorts the events by cluster and trial
    log.info("- sorting events into trials:")
    events["trial"] = trials.recording_start.searchsorted(events.time, side="left") - 1

    # describes the auxiliary channels; only written if there are any
    aux_tracks = (
        {"aux_tracks": {name: {"channel": dset} for name, dset in args.aux.items()}}
        if args.aux
        else {}
    )
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
        n_clean = len(spikes)
        # the same windows qs.peaks would extract; indexing the read-only
        # memmap directly reads only the samples around each spike
        windows = spikes.time.to_numpy()[:, None] + np.arange(-n_before, n_after)
        waveforms = recording[windows, clust_info["ch"]]
        mean_spike = waveforms.mean(0)
        included = np.abs(waveforms).max(-1) < (
            np.abs(mean_spike).max(-1) * args.artifact_reject_thresh
        )
        n_included = included.sum()

        if (n_clean < n_spikes) & (n_clean > n_spikes - 2):
            log.info(
                "    - %d spike(s) with insufficient samples excluded.",
                n_spikes - n_clean,
            )
        if (n_included < n_clean) & (n_included > n_clean / 2):
            spikes = spikes[included]
            waveforms = waveforms[included]
            log.info("    - %d artifact spike(s) excluded", n_spikes - n_included)

        if (n_included < n_clean / 2) | (n_clean < n_spikes - 2):
            log.warning(
                "    - too many spikes in cluster %d excluded as artifact or tail spikes. Recheck sorting data.",
                clust_id,
            )
            input("group-kilo-spikes will skip this unit. Press any key to continue.")
            continue

        # aggregate spikes by trial and left join to trial information table
        # - empty trials will be nan
        clust_trials = trials.join(
            spikes.groupby("trial")
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
            **aux_tracks,
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
                        spikes.time.to_numpy(),
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
