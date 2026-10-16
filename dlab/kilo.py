# -*- mode: python -*-
"""Functions for using kilosort/phy data

The main entry point is group-kilo-spikes (group_spikes_script), which sorts the
spikes in a kilosort/phy output directory into one pprox file (in the stimtrial
format, https://meliza.org/spec:2/stimtrial/) per cluster, split into trials by
stimulus. Stimulus onsets come from a sync channel recorded with the neural
data (clicks or pulses; see detect_pulses), matched to the start messages sent
by the presentation script (oeaudio-present or jpresent; see match_sync_events).
Pulses on other channels (e.g. optogenetic stimulation) can be recorded with
each trial (see arf_to_trials).
"""

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

from dlab import kilo_window, pprox
from dlab import neurobank as nbank
from dlab.spikes import SpikeWaveforms, save_waveforms

log = logging.getLogger(__name__)
# version of group-kilo-spikes, recorded in its output files
SCRIPT_VERSION = "2026.10.14"


class Trial(NamedTuple):
    """Represents the structure of a trial. All time units are in samples.

    aux is None if no auxiliary channels were requested; otherwise it holds a
    (name, start, end) tuple for each auxiliary pulse that starts in the trial.
    audio_sync is None unless the onset was found on an audio channel; then it
    describes the match (see audio_sync_trials).
    """

    recording_entry: int
    recording_start: int
    recording_end: int
    stimulus_name: str
    stimulus_start: int
    stimulus_end: int
    aux: tuple | None = None
    audio_sync: dict | None = None


class Stimulus(NamedTuple):
    """A stimulus start message: the stimulus name and its time, in samples."""

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


def fields_to_attrs(fields: Mapping) -> dict:
    """pprox fields as hdf5 attributes: a dict value is written as one
    attribute per item, named '<field>_<key>'"""
    out = {}
    for name, value in fields.items():
        if isinstance(value, Mapping):
            out.update({f"{name}_{k}": v for k, v in value.items()})
        else:
            out[name] = value
    return out


def kilosort_log_to_sorter(path: Path) -> str | None:
    """The sorter and its version from kilosort4.log, which kilosort 4 writes
    in its output directory (e.g. 'kilosort 4.1.7'), or 'kilosort 4' if the log
    doesn't give the version. None if there is no log."""
    if not path.exists():
        return None
    with open(path) as fp:
        for line in fp:
            m = re.search(r"Kilosort version (\S+)", line)
            if m is not None:
                return f"kilosort {m.group(1)}"
    return "kilosort 4"


# '<verb> <name>' with an optional '<stream>_' prefix on the verb (see
# messages_to_events)
_re_message = re.compile(r"(?:([A-Za-z0-9]+)_)?(start|stop) (.*)")


def messages_to_events(
    dset: h5.Dataset, duplicate_window: int = 1500
) -> dict[str, list[Stimulus]]:
    """Parse the start and stop messages in the stimulus message dataset (see
    find_message_dset) into events, by stream.

    The presentation script sends '<verb> <name>' for the stimulus, where verb
    is 'start' or 'stop'. jpresent (through jrelay) sends events on other MIDI
    channels with the channel as a prefix on the verb: 'trial_' for a window
    around the stimulus, 'condition_' for trials carrying a manipulation, and
    'channelN_' for any other channel N. These are returned by stream name
    ('stimulus', 'trial', 'condition', 'channelN'), each a list of
    Stimulus(name, start, end) in the order the start messages were sent; end
    is the time of the matching stop message, or None. Names are reduced to
    their stem (directory and extension removed). Times are open-ephys sample
    numbers, which count from the start of acquisition.

    A start message for the event that is still open on its stream, within
    duplicate_window samples (50 ms at 30 kHz) of its start, is a duplicate and
    is dropped, keeping the earlier time: jpresent sent every start message
    twice in some recordings (E1), up to 32 ms apart and not always in order.

    """
    events: dict[str, list[Stimulus]] = {}
    for row in dset:
        m = _re_message.fullmatch(row["message"].decode("utf-8"))
        if m is None:
            continue
        prefix, verb, name = m.groups()
        stream = events.setdefault(prefix or "stimulus", [])
        name, time = Path(name).stem, int(row["start"])
        if verb == "start":
            last = stream[-1] if stream else None
            if (
                last is not None
                and last.name == name
                and last.end is None
                and abs(time - last.start) <= duplicate_window
            ):
                stream[-1] = last._replace(start=min(last.start, time))
                continue
            stream.append(Stimulus(name, time))
            continue
        # a stop closes the most recent open event with the same name
        for i in range(len(stream) - 1, -1, -1):
            if stream[i].name == name and stream[i].end is None:
                stream[i] = stream[i]._replace(end=time)
                break
    return events


def yaml_to_stimulus_names(path: Path) -> dict[str, str]:
    """Reads a YAML file that maps the names of stimuli as presented (the stems
    in the messages or the log) to the names they were deposited under in
    neurobank. Names must be strings; quote ones that YAML would read as
    numbers."""
    import yaml

    with open(path) as fp:
        names = yaml.safe_load(fp)
    if not isinstance(names, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in names.items()
    ):
        raise ValueError(
            f"{path} should map stimulus names to neurobank names (quote any "
            "names YAML would read as numbers)"
        )
    return names


def rename_stimuli(
    stimuli: Iterable[Stimulus], names: Mapping[str, str]
) -> list[Stimulus]:
    """Replaces the names of stimuli that are keys of names with their values"""
    return [stim._replace(name=names.get(stim.name, stim.name)) for stim in stimuli]


def messages_to_stimuli(dset: h5.Dataset) -> list[Stimulus]:
    """The stimulus events in the message dataset (see messages_to_events).
    arf_to_trials converts their times to samples from the start of the
    recording and matches them to the sync events, which follow the messages by
    a few hundred ms."""
    return messages_to_events(dset).get("stimulus", [])


def oeaudio_log_to_stimuli(
    oeaudio_log: io.TextIOBase, sampling_rate: int
) -> Iterator[Stimulus]:
    """Parse the 'start <stimulus>' lines of an open-ephys-audio log. Times are
    in samples from StartAcquisition, the same origin as open-ephys sample
    numbers (see messages_to_stimuli).

    This is for recordings without the messages in the ARF file (some
    recordings with open-ephys GUI >= 0.6 were made without logging them).
    Ideally the log is from the same session as the recording. A log from
    another session can be used if the stimuli were presented in the same order
    (see match_log_stimuli).
    """
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


def find_message_dset(entry):
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


def entry_to_metadata(entry):
    """Extracts metadata from an entry in an ARF file recorded with a stimulus
    presentation script (oeaudio-present or jpresent).

    Metadata are passed to open-ephys through the network events socket as a
    json-encoded dictionary. There should be at most one metadata message
    per entry, so only the first is returned. If there is none (jpresent does
    not send one), only the entry name and sampling rate are returned.

    """
    re_metadata = re.compile(r"metadata: (\{.*\})")
    stim_dset = find_message_dset(entry)
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
    data: np.ndarray,
    thresh: float = 0.5,
    min_snr: float = 20.0,
    *,
    level: float | None = None,
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

    If `level` is given, it is used as the threshold instead (in the units of
    data), and the noise check is skipped.

    """
    if level is not None:
        return _pulses_above(data, level)
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
    return _pulses_above(data, level)


def _pulses_above(data: np.ndarray, level: float) -> np.ndarray:
    """Onsets and offsets of the stretches of data at or above level"""
    above = data >= level
    rises = np.flatnonzero(~above[:-1] & above[1:]) + 1
    falls = np.flatnonzero(above[:-1] & ~above[1:]) + 1
    offsets = np.append(falls, data.size)[np.searchsorted(falls, rises)]
    return np.column_stack([rises, offsets])


def detect_sync_onsets(
    data: np.ndarray,
    thresh: float = 0.5,
    min_snr: float = 20.0,
    *,
    level: float | None = None,
) -> np.ndarray:
    """Returns the sample indices where a sync signal rises through a threshold.

    These are the onsets of the pulses found by detect_pulses (see there for
    how the threshold is set).

    """
    return detect_pulses(data, thresh, min_snr, level=level)[:, 0]


class StimulusFinder:
    """Looks up stimuli using neurobank and/or files in a local directory"""

    def __init__(self, nbank_registry_url: str, alt_base: Path | None = None):
        self.registry_url = nbank_registry_url
        self.alt_base = alt_base

    def find(self, names: Iterable[str]) -> Iterator[tuple[str, Path]]:
        """Yields the name and path of each stimulus file. Searches neurobank
        first and then the local directory; raises FileNotFoundError if a
        stimulus can't be found."""
        for name, res in nbank.find_resources(*names, registry_url=self.registry_url):
            if isinstance(res, FileNotFoundError):
                if self.alt_base is None:
                    raise res
                path = (self.alt_base / name).with_suffix(".wav")
                if not path.exists():
                    raise FileNotFoundError(f"{res}; {path} does not exist") from res
            else:
                path = res
            log.debug("  - found '%s' at %s", name, path)
            yield name, path

    def get_durations(self, names: Iterable[str]) -> dict[str, float]:
        """Looks up durations (in s) for a sequence of stimuli (see find)"""
        output = {}
        for name, path in self.find(names):
            with ewave.wavfile(path) as fp:
                output[name] = 1.0 * fp.nframes / fp.sampling_rate
        return output

    def get_waveforms(
        self, names: Iterable[str], sampling_rate: float
    ) -> dict[str, np.ndarray]:
        """Reads a sequence of stimuli (see find), resampled to sampling_rate.
        Multichannel files are mixed down to one channel."""
        from fractions import Fraction

        from scipy.signal import resample_poly

        output = {}
        for name, path in self.find(names):
            with ewave.wavfile(path) as fp:
                data = fp.read().astype("d")
                rate = fp.sampling_rate
            if data.ndim > 1:
                data = data.mean(axis=1)
            ratio = Fraction(sampling_rate / rate).limit_denominator(1000)
            if ratio != 1:
                data = resample_poly(data, ratio.numerator, ratio.denominator)
            output[name] = data
        return output


# the lag assumed between the start messages and their sync events when the
# messages are on another clock (see arf_to_trials): typical of our rigs
# (0.07-1 s), and short enough that the aux pulses that follow other messages
# (0.27 s on one rig, where sync events follow by 0.07 s) still come
# between their message and the next
NOMINAL_SYNC_LAG = 0.25


def arf_to_trials(
    data_file: h5.File,
    stim_finder: StimulusFinder,
    sync_dset: str,
    sync_thresh: float | None = None,
    prepad: float = 1.0,
    *,
    oeaudio_log: Path | None,
    aux: Mapping[str, str] | None = None,
    audio_sync: str | None = None,
    stimulus_names: Mapping[str, str] | None = None,
) -> list[Trial]:
    """Extracts trial information from an ARF file recorded with a stimulus
    presentation script (oeaudio-present or jpresent)

    When using oeaudio-present or jpresent, a single recording is made in
    response to all the stimuli. The stimulus presentation script sends network events to
    open-ephys to mark the start and stop of each stimulus. There is typically a
    significant lag between the 'start' event and the onset of the stimulus, due
    to buffering of the audio playback. However, the presentation script will
    play a synchronization signal on a second channel: a brief click at each
    stimulus onset (old style), or a pulse that stays high for the duration of
    the stimulus (new style). As long as the user remembers to record this
    channel, it can be used to correct the onset values.

    By default, sync events are detected with a threshold halfway between the
    sync channel's baseline and its peak, and a channel whose peak doesn't
    stand well clear of its noise has no events (see detect_pulses).
    `sync_thresh` overrides this with an absolute threshold, in the channel's
    units. Either way, it is an error if there are more sync events than
    stimuli, or more than 1% of the stimuli (at least one) have none, since
    that indicates the wrong channel, a recording without a sync signal, or a
    bad threshold.

    The continuous recording is broken up into trials based on the stimulus
    presentation, such that each trial encompasses one and only one stimulus.
    The `prepad` parameter specifies, in seconds, when trials begin relative to
    stimulus onset. The default is 1.0 s. A trial whose stimulus doesn't fit
    the sync track, usually a presentation stopped early, is dropped (see
    check_stimulus_lengths); its interval is in no trial.

    If oeaudio_log is set, it's used instead of the network event datasets.

    stimulus_names maps the names of stimuli in the messages (or log) to the
    names they were deposited under in neurobank, which are used to look them
    up and to label the trials (see rename_stimuli). Stimuli not in it keep
    their names.

    If audio_sync names a channel with a copy of the audio (instead of a sync
    track, which sync_dset and sync_thresh then don't apply to), each stimulus
    is found on it by cross-correlation, near where its message says it was
    presented (see audio_sync_trials and dlab.audio_sync).

    aux maps names to the datasets of auxiliary channels (e.g. {"led":
    "ADC4"}) carrying pulses from other devices, such as an optogenetic light
    source or a sensor's TTL output. Pulses are detected as for the sync
    channel, and each is assigned, unclipped, to the trial in which it starts
    (see Trial.aux). A channel may be given as 'CHANNEL:STREAM' (e.g.
    "ADC4:condition") to check its pulses against the messages on that stream
    of the message dataset (see messages_to_events and check_aux_pulses).

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

        track, flag = (
            (audio_sync, "--audio-sync") if audio_sync else (sync_dset, "--sync")
        )
        log.info("  - %s track: '%s'", "audio" if audio_sync else "sync", track)
        try:
            sync = entry[track]
        except KeyError as err:
            available_tracks = ", ".join(entry.keys())
            raise RuntimeError(
                f"unable to find the {'audio' if audio_sync else 'sync'} track. Use "
                f"{flag} to configure. Options are: {available_tracks}"
            ) from err

        if audio_sync is None:
            sync_pulses = detect_pulses(sync[:], level=sync_thresh)
            stim_onsets = sync_pulses[:, 0]
            log.info("    - detected %d sync events", stim_onsets.size)
            if stim_onsets.size == 0:
                raise RuntimeError(
                    f"no sync events detected in '{sync_dset}': it may not be the "
                    "sync track, or the sync signal may not have been recorded. "
                    "Check --sync, or set --sync-thresh."
                )
        dset_offset = sync.attrs["offset"]
        dset_end = sync.size
        sampling_rate = sync.attrs["sampling_rate"]
        stim_sample_offset = round(dset_offset * sampling_rate)
        log.info("  - recording clock offset: %d", stim_sample_offset)

        stim_dset = None
        events = None  # messages on all streams (not available from an oeaudio log)
        if oeaudio_log is not None:
            log.info("  - reading stimuli from oeaudio log %s", oeaudio_log)
            with open(oeaudio_log) as fp:
                entry_stimuli = list(oeaudio_log_to_stimuli(fp, sampling_rate))
        else:
            stim_dset = find_message_dset(entry)
            if stim_dset is None:
                raise RuntimeError(
                    "unable to find the stimulus messages in the ARF file. If they "
                    "weren't recorded, use --oeaudio-log to read them from the log."
                )
            log.info("  - reading stimuli from message dataset %s", stim_dset.name)
            events = messages_to_events(stim_dset)
            log.info(
                "    - messages: %s",
                ", ".join(f"{len(v)} {k}" for k, v in events.items()),
            )
            entry_stimuli = events.get("stimulus", [])
        if stimulus_names:
            renamed = sum(stim.name in stimulus_names for stim in entry_stimuli)
            entry_stimuli = rename_stimuli(entry_stimuli, stimulus_names)
            log.info("    - %d stimuli renamed to their neurobank names", renamed)
        try:
            stim_durations = stim_finder.get_durations(
                stim.name for stim in entry_stimuli
            )
        except FileNotFoundError as err:
            raise RuntimeError(
                f"unable to look up a stimulus's duration: {err}. Stimuli must be"
                " deposited in neurobank (or given with --local-stim-dir)"
            ) from err
        log.info("    - detected %d stimuli", len(entry_stimuli))

        # message times are open-ephys sample numbers, which count from the
        # start of acquisition; convert to samples from the start of the sync
        # track. Log times are assumed to have the same origin (StartAcquisition).
        entry_stimuli = [
            stim._replace(start=stim.start - stim_sample_offset)
            for stim in entry_stimuli
        ]
        quality = None
        if audio_sync is not None:
            entry_stimuli, stim_onsets, quality = audio_sync_trials(
                sync, entry_stimuli, stim_finder, sampling_rate
            )
            # one-sample events, so the stimulus length check treats them as clicks
            sync_pulses = np.column_stack([stim_onsets, stim_onsets + 1])
            times_fit = True
        elif oeaudio_log is not None:
            check_sync_count(stim_onsets.size, len(entry_stimuli), sync_dset)
            entry_stimuli, times_fit = match_log_stimuli(
                entry_stimuli, stim_onsets, sampling_rate
            )
        else:
            check_sync_count(stim_onsets.size, len(entry_stimuli), sync_dset)
            entry_stimuli, times_fit = (
                match_sync_events(entry_stimuli, stim_onsets),
                True,
            )
        # lags (sync event after start message) are meaningless for a log from
        # another session. If the messages are on another clock from the sync
        # track, they're shifted so the start messages come NOMINAL_SYNC_LAG
        # before their sync events, for the aux checks
        message_shift = 0
        if times_fit:
            starts = np.array([stim.start for stim in entry_stimuli])
            lags = (stim_onsets - starts) / sampling_rate
            log.info(
                "    - sync events follow start messages by %.3f s", np.median(lags)
            )
            if not 0 <= np.median(lags) <= 2.0:
                message_shift = round(
                    np.median(stim_onsets - starts) - NOMINAL_SYNC_LAG * sampling_rate
                )
                log.warning(
                    "  - WARNING: sync events follow their start messages by %.3f s; "
                    "the message times don't line up with the sync track. The "
                    "stimuli are paired with the sync events in order, and the "
                    "messages shifted to %.2f s before them for the aux checks.",
                    np.median(lags),
                    NOMINAL_SYNC_LAG,
                )
            lo, hi = sync_lag_range(lags)
            for i in sync_lag_outliers(starts, stim_onsets, sampling_rate):
                log.warning(
                    "  - WARNING: sync event for stimulus %d (%s) is %.3f s after its "
                    "start message (others %.3f-%.3f s); message timing only, "
                    "the stimulus lengths check the pairing.",
                    i,
                    entry_stimuli[i].name,
                    lags[i],
                    lo,
                    hi,
                )

        # trials whose stimulus doesn't fit the sync track (stopped early) are
        # dropped
        mismatched = set(
            check_stimulus_lengths(
                entry_stimuli, stim_durations, sync_pulses, sync.size, sampling_rate
            ).tolist()
        )

        padding_samples = int(prepad * sampling_rate)
        entry_trials = []
        for i, (stim, onset, offset) in enumerate(
            zip_longest(
                entry_stimuli,
                stim_onsets,
                stim_onsets[1:],
                fillvalue=dset_end + padding_samples,
            )
        ):
            if i in mismatched:
                continue
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
                    audio_sync=None if quality is None else quality[i],
                )
            )
        if aux:
            pulses = aux_pulses(entry, aux)
            entry_trials = assign_aux_pulses(entry_trials, pulses)
            for name, spec in aux.items():
                _, _, stream = spec.partition(":")
                if not stream:
                    continue
                if events is None or stream not in events:
                    log.warning(
                        "  - WARNING: no '%s' messages to check aux '%s' against",
                        stream,
                        name,
                    )
                    continue
                shift = message_shift - stim_sample_offset
                messages = [
                    ev._replace(
                        start=ev.start + shift,
                        end=None if ev.end is None else ev.end + shift,
                    )
                    for ev in events[stream]
                ]
                check_aux_pulses(
                    name,
                    [pulse for pulse in pulses if pulse[0] == name],
                    stream,
                    messages,
                    sampling_rate,
                )
        trials.extend(entry_trials)
    return trials


def audio_sync_trials(
    channel: h5.Dataset,
    stimuli: list[Stimulus],
    stim_finder: StimulusFinder,
    sampling_rate: float,
    max_missing: float = 0.01,
) -> tuple[list[Stimulus], np.ndarray, list[dict]]:
    """Finds the stimuli (with start times in samples on the channel) on a
    channel with a copy of the audio (see dlab.audio_sync.audio_sync_onsets).
    Returns the stimuli that were found, their onsets, and a description of
    each match for the pprox: how the onset was found ('audio', or 'estimated'
    for silent stimuli, placed at the message plus the typical lag, with a
    warning), the lag from the message (s), the correlation and its contrast,
    the stimuli of the same length it couldn't be told from, and the best of
    them (rival). Stimuli that weren't found are dropped with a warning;
    raises RuntimeError if more than max_missing of them (at least one)
    weren't, as for a sync track (see check_sync_count)."""
    from dlab import audio_sync

    waveforms = stim_finder.get_waveforms({s.name for s in stimuli}, sampling_rate)
    matches = audio_sync.audio_sync_onsets(
        lambda a, b: channel[a:b].astype("d"), stimuli, waveforms, sampling_rate
    )
    found, onsets, quality = [], [], []
    for i, (stim, m) in enumerate(zip(stimuli, matches, strict=True)):
        if m.method == "unmatched":
            log.warning(
                "  - stimulus %d (%s) not found on the audio channel (%s); "
                "dropping the trial",
                i,
                stim.name,
                m.reason,
            )
            continue
        if m.method == "estimated":
            log.warning(
                "  - WARNING: stimulus %d (%s) is silent; its onset is estimated from "
                "its message and the typical lag (%.3f s)",
                i,
                stim.name,
                m.lag,
            )
        record = {"method": m.method, "lag": round(m.lag, 6)}
        if m.method == "audio":
            record.update(r=round(m.r, 4), contrast=round(m.contrast, 4))
            if m.variants:
                record["variants"] = list(m.variants)
            if m.rival is not None:
                record["rival"] = {"name": m.rival, "r": round(m.rival_r, 4)}
        found.append(stim)
        onsets.append(m.onset)
        quality.append(record)
    allowed = max(1, int(len(stimuli) * max_missing))
    if len(stimuli) - len(found) > allowed:
        raise RuntimeError(
            f"{len(stimuli) - len(found)} of {len(stimuli)} stimuli weren't found on "
            f"the audio channel (at most {allowed} may be missing): it may not be the "
            "copy of the audio. Check --audio-sync."
        )
    return found, np.array(onsets, dtype=int), quality


def check_stimulus_lengths(
    stimuli: list[Stimulus],
    durations: Mapping[str, float],
    pulses: np.ndarray,
    nsamples: int,
    sampling_rate: float,
    max_missing: float = 0.01,
) -> np.ndarray:
    """Checks that the stimuli paired with the sync events (one per event) fit
    the sync track: the gaps between them, and the pulse widths if the sync
    track has pulses (see stimulus_length_mismatches). Returns the indices of
    the trials that don't, with a warning for each: usually presentations
    stopped early (e.g. by a jack xrun), which arf_to_trials drops. Raises
    RuntimeError if more than max_missing of the trials (at least one) don't
    fit, since the stimuli are then probably paired with the wrong sync events,
    and the trials would be mislabeled."""
    onsets = pulses[:, 0]
    lengths = np.array([durations[stim.name] for stim in stimuli])
    widths = None
    if is_pulse_track(pulses, sampling_rate):
        # a pulse still high at the end of the track has no known width
        widths = np.where(pulses[:, 1] < nsamples, pulses[:, 1] - pulses[:, 0], np.nan)
    bad = stimulus_length_mismatches(onsets, lengths, sampling_rate, widths)
    for i in bad:
        log.warning(
            "  - WARNING: stimulus %d (%s, %.3f s) doesn't fit the sync track "
            "(%s; next sync event after %.3f s); dropping the trial",
            i,
            stimuli[i].name,
            lengths[i],
            "clicks" if widths is None else f"pulse {widths[i] / sampling_rate:.3f} s",
            (onsets[i + 1] - onsets[i]) / sampling_rate
            if i + 1 < onsets.size
            else np.nan,
        )
    allowed = max(1, int(len(stimuli) * max_missing))
    if bad.size > allowed:
        raise RuntimeError(
            f"{bad.size} of {len(stimuli)} stimuli don't fit the sync track: the "
            "sync events are probably paired with the wrong stimuli (a missed or "
            "spurious sync event), or the sync track or stimulus durations are "
            "wrong. Check --sync, or set --sync-thresh."
        )
    return bad


def match_aux_pulses(
    onsets: np.ndarray,
    messages: list[Stimulus],
    sampling_rate: float,
    tolerance: float = 0.1,
) -> dict:
    """Matches the onsets of the pulses on an auxiliary channel (samples) to the
    messages on the stream that drives it (Stimulus(name, start, end), in
    samples from the start of the sync track).

    Like the sync pulses, each pulse follows its message by a lag (from audio
    buffering). The first pulse after each start message, before the next one,
    is that message's; its lag should agree with the others (see
    sync_lag_outliers). More pulses may follow (e.g. a train) until the stop
    message, plus the lag, plus tolerance (s); without a stop message, until
    the next message.

    Returns a dict with the indices of the messages without a pulse
    ('missing'), of the pulses outside any message's window ('unexpected'), and
    of the messages whose pulse is out of line ('late'); the median lag
    ('lag', samples); and the number of messages with a pulse ('matched').
    """
    onsets = np.sort(np.asarray(onsets, dtype=int))
    starts = np.array([msg.start for msg in messages], dtype=int)
    nexts = np.append(starts[1:], np.iinfo(np.int64).max)
    first = np.searchsorted(onsets, starts, side="left")
    has_pulse = (first < onsets.size) & (np.append(onsets, nexts[-1])[first] < nexts)
    missing = np.flatnonzero(~has_pulse).tolist()
    lags = onsets[first[has_pulse]] - starts[has_pulse]
    late = (
        np.flatnonzero(has_pulse)[
            sync_lag_outliers(np.zeros(lags.size), lags, sampling_rate, tolerance)
        ].tolist()
        if lags.size
        else []
    )
    lag = int(np.median(lags)) if lags.size else 0
    unexpected = []
    for k, onset in enumerate(onsets):
        i = np.searchsorted(starts, onset, side="right") - 1
        if i < 0:
            unexpected.append(k)
            continue
        end = messages[i].end
        limit = nexts[i] if end is None else end + lag + tolerance * sampling_rate
        if onset > limit:
            unexpected.append(k)
    return {
        "missing": missing,
        "unexpected": unexpected,
        "late": late,
        "lag": lag,
        "matched": int(has_pulse.sum()),
    }


def check_aux_pulses(
    name: str,
    pulses: list[tuple],
    stream: str,
    messages: list[Stimulus],
    sampling_rate: float,
    tolerance: float = 0.1,
) -> dict[str, list[int]]:
    """Checks the pulses on an auxiliary channel against the messages on the
    stream that drives it (see match_aux_pulses), logging a warning for each
    problem. pulses are (name, start, end) tuples. Returns the indices of the
    messages without a pulse ('missing'), the pulses outside any message's
    window ('unexpected'), and the messages whose pulse is out of line
    ('late')."""
    onsets = np.array(sorted(pulse[1] for pulse in pulses), dtype=int)
    found = match_aux_pulses(onsets, messages, sampling_rate, tolerance)
    log.info(
        "  - aux '%s': %d '%s' messages, %d with pulses (lag %.3f s), %d pulses",
        name,
        len(messages),
        stream,
        found["matched"],
        found["lag"] / sampling_rate,
        onsets.size,
    )
    for i in found["missing"]:
        log.warning(
            "  - WARNING: no '%s' pulse for %s message %d (%s)",
            name,
            stream,
            i,
            messages[i].name,
        )
    for k in found["unexpected"]:
        log.warning(
            "  - WARNING: '%s' pulse at sample %d is not in the window of any %s message",
            name,
            onsets[k],
            stream,
        )
    for i in found["late"]:
        log.warning(
            "  - WARNING: '%s' pulse for %s message %d (%s) is out of line with the others",
            name,
            stream,
            i,
            messages[i].name,
        )
    return {k: found[k] for k in ("missing", "unexpected", "late")}


def aux_pulses(entry, aux: Mapping[str, str]) -> list[tuple]:
    """Detects the pulses on each auxiliary channel in an entry, with the
    automatic threshold (see detect_pulses). aux maps names to 'CHANNEL' or
    'CHANNEL:STREAM' (see arf_to_trials). Returns a list of (name, start, end)
    tuples, sorted by start."""
    pulses = []
    for name, spec in aux.items():
        dset_name = spec.partition(":")[0]
        try:
            dset = entry[dset_name]
        except KeyError as err:
            raise RuntimeError(
                f"unable to find auxiliary channel '{dset_name}' for '{name}'. "
                f"Options are: {', '.join(entry.keys())}"
            ) from err
        detected = detect_pulses(dset[:])
        log.info(
            "  - aux '%s' (%s): detected %d pulses", name, dset_name, len(detected)
        )
        pulses.extend((name, int(on), int(off)) for on, off in detected)
    return sorted(pulses, key=lambda pulse: pulse[1])


def assign_aux_pulses(trials: list[Trial], pulses: list[tuple]) -> list[Trial]:
    """Returns trials (from one entry, in order) with each pulse assigned to the
    trial in which it starts. Pulses that start before the first trial, or
    after the end of the trial before them (in a dropped trial), are dropped
    with a warning."""
    starts = np.array([trial.recording_start for trial in trials])
    assigned = [[] for _ in trials]
    for pulse in pulses:
        i = np.searchsorted(starts, pulse[1], side="right") - 1
        if i < 0:
            log.warning(
                "  - aux pulse %s starts before the first trial; dropped", pulse
            )
        elif pulse[1] >= trials[i].recording_end:
            log.warning("  - aux pulse %s is in a dropped trial; dropped", pulse)
        else:
            assigned[i].append(pulse)
    return [
        trial._replace(aux=tuple(pulses))
        for trial, pulses in zip(trials, assigned, strict=True)
    ]


def check_sync_count(
    nsync: int, nstimuli: int, sync_dset: str, max_missing: float = 0.01
) -> None:
    """Raises RuntimeError if the number of sync events doesn't fit the number
    of stimuli: if there are more sync events, or if more than max_missing of
    the stimuli (at least one) would have none."""
    hint = (
        f"'{sync_dset}' may not be the sync track, or the threshold may be wrong. "
        "Check --sync, or set --sync-thresh."
    )
    if nsync > nstimuli:
        raise RuntimeError(
            f"{nsync} sync events in '{sync_dset}' but only {nstimuli} stimuli: {hint}"
        )
    allowed = max(1, int(nstimuli * max_missing))
    if nstimuli - nsync > allowed:
        raise RuntimeError(
            f"only {nsync} sync events in '{sync_dset}' for {nstimuli} stimuli "
            f"(at most {allowed} may be missing): {hint}"
        )


def match_sync_events(
    entry_stimuli: list[Stimulus], stim_onsets: np.ndarray
) -> list[Stimulus]:
    """Match sync events to stimuli, returning the stimulus for each sync event.

    The start messages give the order of the stimuli, and the sync events
    their onsets, so the events are paired with the stimuli in order. If there
    are fewer sync events than stimuli (e.g. a sync event that was not
    detected), the stimuli without one are dropped with a warning. Which ones
    they are is found from the message times: of the pairings in order that
    skip that many stimuli, the one whose lags (sync event after message) are
    closest overall to the recording's typical lag. The messages need not be in
    time order, and a message may come after its sync event. More sync events
    than stimuli is a ValueError. The pairing is checked against the stimulus
    lengths afterwards (see check_stimulus_lengths).

    Stimulus start times and sync onsets must be in the same units (samples
    from the start of the sync track).

    """
    onsets = np.asarray(stim_onsets)
    n, m = onsets.size, len(entry_stimuli) - onsets.size
    if m < 0:
        raise ValueError(
            f"{n} sync events but only {len(entry_stimuli)} stimuli. Check "
            "--sync-thresh, or discard the recording."
        )
    if m == 0 or n == 0:
        idx = np.arange(n)
    else:
        idx = _skip_unmatched(np.array([s.start for s in entry_stimuli]), onsets, m)
    for i in sorted(set(range(len(entry_stimuli))) - set(idx.tolist())):
        log.warning(
            "  - no sync event for stimulus %d (%s); dropping the trial",
            i,
            entry_stimuli[i].name,
        )
    return [entry_stimuli[i] for i in idx]


def _skip_unmatched(starts: np.ndarray, onsets: np.ndarray, m: int) -> np.ndarray:
    """The indices of the stimuli paired with the sync events, in order,
    skipping m stimuli: the pairing whose lags are closest in total to the
    typical lag (the median lag from each sync event back to the last message
    before it, or if no message comes before any sync event, of the pairing in
    order without skips). Dynamic programming over the number of stimuli
    skipped before each event."""
    ordered = np.sort(starts)
    before = np.searchsorted(ordered, onsets, side="right") - 1
    if np.any(before >= 0):
        typical = np.median(onsets[before >= 0] - ordered[before[before >= 0]])
    else:
        typical = np.median(onsets - starts[: onsets.size])
    n = onsets.size
    # cost[i, k]: event i paired with stimulus i + k
    k = np.arange(m + 1)
    cost = np.abs(onsets[:, None] - starts[np.arange(n)[:, None] + k] - typical)
    total = np.zeros((n + 1, m + 1))
    for i in range(n):
        total[i + 1] = np.minimum.accumulate(total[i]) + cost[i]
    skips = np.empty(n, dtype=int)
    limit = m
    for i in range(n, 0, -1):
        skips[i - 1] = int(np.argmin(total[i][: limit + 1]))
        limit = skips[i - 1]
    return np.arange(n) + skips


def match_log_stimuli(
    stimuli: list[Stimulus], onsets: np.ndarray, sampling_rate: float
) -> tuple[list[Stimulus], bool]:
    """Match sync events to the stimuli in an oeaudio log, which may be from a
    different session than the recording (session logs aren't kept long).

    Returns the stimulus for each sync event, and whether the log's times fit
    the recording (every sync event follows its stimulus by a consistent lag;
    see sync_lag_outliers). If there is a sync event for every stimulus, they
    are paired in order; if the times don't fit, the log is probably from
    another session, and a warning is logged, since the labels are only right if
    the stimuli were presented in the same order. If sync events are missing,
    repairing the gap needs the times (see match_sync_events), so a log whose times
    don't fit is an error.

    """
    if len(stimuli) == onsets.size:
        starts = np.array([stim.start for stim in stimuli])
        fits = (onsets >= starts).all() and sync_lag_outliers(
            starts, onsets, sampling_rate
        ).size == 0
        if not fits:
            log.warning(
                "  - WARNING: the times in the oeaudio log don't fit this recording, "
                "so it is probably from another session. Trials are labeled by "
                "presentation order, which is only right if the stimuli were "
                "presented in the same order."
            )
        return list(stimuli), fits
    try:
        matched = match_sync_events(stimuli, onsets)
        starts = np.array([stim.start for stim in matched])
        fits = sync_lag_outliers(starts, onsets, sampling_rate).size == 0
    except ValueError:
        fits = False
    if not fits:
        raise RuntimeError(
            f"{len(stimuli) - onsets.size} sync event(s) are missing, and the times "
            "in the oeaudio log don't fit this recording (is it from another "
            "session?), so the trials can't be labeled: by order, every trial after "
            "a missing sync event would get the wrong stimulus. Use the log from the "
            "same session."
        )
    return matched, True


def sync_lag_range(
    lags: np.ndarray, max_spread: float = 0.3, min_count: int = 20
) -> tuple[float, float]:
    """The range of sync lags (s) taken as normal for a recording.

    Each sync event follows its message by a lag that depends on the
    presentation setup: 0.25-1 s, varying by about 30 ms within most
    recordings, but spread evenly over about 0.2 s in some (a larger audio
    buffer). The normal range is the 5th to 95th percentile of the lags, which
    covers either. If that is wider than max_spread (s), the lags don't have a
    consistent range (e.g. onsets drifting away from the messages, or many
    wrong onsets), and the median is used, as the right lag if most trials are
    right. The median is also used for fewer than min_count lags, whose
    percentiles would include the outliers.

    """
    if len(lags) < min_count:
        lo = hi = np.median(lags)
        return float(lo), float(hi)
    lo, hi = np.percentile(lags, [5, 95])
    if hi - lo > max_spread:
        lo = hi = np.median(lags)
    return float(lo), float(hi)


def sync_lag_outliers(
    starts: np.ndarray, onsets: np.ndarray, sampling_rate: float, tolerance: float = 0.1
) -> np.ndarray:
    """Returns the indices of trials whose sync lag is out of line with the rest.

    starts and onsets are the start-message times of matched stimuli and their
    sync onsets, in samples. A lag more than tolerance (in s) outside the
    recording's normal range (see sync_lag_range) suggests a misdetected or
    mismatched sync event, or stimulus times that don't belong to the recording.
    This relies on most trials being right; if half or more are wrong, it flags
    the wrong trials.

    """
    lags = (np.asarray(onsets) - np.asarray(starts)) / sampling_rate
    if lags.size == 0:
        return np.array([], dtype=int)
    lo, hi = sync_lag_range(lags)
    return np.flatnonzero((lags < lo - tolerance) | (lags > hi + tolerance))


def is_pulse_track(pulses: np.ndarray, sampling_rate: float) -> bool:
    """True if the sync events (onset, offset pairs from detect_pulses) are
    sustained pulses that last as long as the stimulus, not ~2 ms clicks."""
    widths = (pulses[:, 1] - pulses[:, 0]) / sampling_rate
    return widths.size > 0 and float(np.median(widths)) > 0.05


def stimulus_length_mismatches(
    onsets: np.ndarray,
    durations: np.ndarray,
    sampling_rate: float,
    widths: np.ndarray | None = None,
    tolerance: float = 0.01,
    gap_tolerance: float = 0.05,
) -> np.ndarray:
    """Returns the indices of trials whose stimulus length doesn't fit the sync
    track, which suggests the sync events and stimuli are paired wrongly (e.g.
    one sync event missed, and the rest paired in order) or an onset that isn't
    a sync event.

    onsets are the trials' sync onsets (samples, increasing) and durations the
    lengths of their stimuli (s). The presenters wait a nearly constant time
    after each stimulus before the next (exactly, for jpresent; within ~0.35 s
    for oeaudio-present), so a trial fails if the gap from the end of its
    stimulus to the next onset is shorter than the other gaps (below the lower
    Tukey fence of the gaps, less gap_tolerance s; with fewer than 4 gaps,
    only if the next onset comes before the stimulus ends). Longer gaps are
    allowed: a trial may have been dropped or the presentation paused. If widths (the
    widths of the trials' sync pulses, in samples, NaN where unknown) are given,
    a trial also fails if its pulse is more than tolerance s longer or shorter
    than its stimulus.

    """
    onsets = np.asarray(onsets)
    durations = np.asarray(durations, dtype=float)
    bad = np.zeros(onsets.size, dtype=bool)
    if widths is not None:
        widths = np.asarray(widths, dtype=float) / sampling_rate
        bad |= np.abs(widths - durations) > tolerance  # False where NaN
    if onsets.size > 1:
        gaps = np.diff(onsets) / sampling_rate - durations[:-1]
        bad[:-1] |= gaps < 0
        if gaps.size >= 4:
            q1, q3 = np.percentile(gaps, [25, 75])
            bad[:-1] |= gaps < q1 - 1.5 * (q3 - q1) - gap_tolerance
    return np.flatnonzero(bad)


def trials_in_window(
    trials: list[Trial], start: int, stop: int
) -> tuple[list[Trial], int, int]:
    """The trials of a sort of the window [start, stop) of the recording (see
    dlab.kilo_window): those whose prepad and stimulus are within the window,
    with their ends clipped to its stop, and without aux pulses that start
    after it. Also returns the numbers of trials dropped before and after the
    window."""
    kept = [
        trial._replace(
            recording_end=min(trial.recording_end, stop),
            aux=trial.aux and tuple(pulse for pulse in trial.aux if pulse[1] < stop),
        )
        for trial in trials
        if trial.recording_start >= start and trial.stimulus_end <= stop
    ]
    before = sum(trial.recording_start < start for trial in trials)
    return kept, before, len(trials) - len(kept) - before


def assign_spikes(times: np.ndarray, trial_starts: np.ndarray) -> np.ndarray:
    """The index of the trial each spike falls in, given the trials' start
    samples (increasing). A spike exactly at a trial's start goes to the
    previous trial, and spikes after the last trial's start go to the last
    trial. Spikes before the first trial get -1."""
    return np.searchsorted(trial_starts, times, side="left") - 1


def waveforms_to_events(
    times: np.ndarray, pprox_trials: list[dict], sampling_rate: float
) -> tuple[list[np.ndarray], int]:
    """Rebuilds the events of a unit's pprox trials from the spike times in its
    waveform file (samples, see dlab.spikes.load_waveforms), using the trial
    table (offset and recording.start) of a pprox from the same run. Returns
    the events for each trial (in s from the stimulus onset, as in the pprox)
    and the number of spikes before the first trial, which are in no trial."""
    times = np.sort(np.asarray(times))
    starts = np.array([trial["recording"]["start"] for trial in pprox_trials])
    onsets = np.array(
        [round(trial["offset"] * sampling_rate) for trial in pprox_trials]
    )
    trial = assign_spikes(times, starts)
    bounds = np.searchsorted(trial, np.arange(len(pprox_trials) + 1), side="left")
    events = [
        (times[bounds[i] : bounds[i + 1]] - onsets[i]) / sampling_rate
        for i in range(len(pprox_trials))
    ]
    return events, int((trial < 0).sum())


def events_to_toelis(events: pd.DataFrame, sampling_rate: float):
    """Assign event_times to clusters, generating a large toelis object"""
    nevents, _ = events.shape
    nclusters = events.index.unique().size
    log.info("- grouping %d spikes into %d clusters...", nevents, nclusters)
    return events.groupby("clust").apply(
        lambda df: df.time.sort_values().to_numpy() / sampling_rate * 1000.0
    )


def trials_to_pprox(trials: pd.DataFrame, sampling_rate: float):
    """Yields a stimtrial point process (dict) for each row of trials.

    trials is a DataFrame of Trial rows with an `events` column (spike times in
    samples, or NaN for none). `events` in the output are in seconds from the
    stimulus onset; `offset` is the onset, in seconds from the start of the
    recording; `interval` is the trial (and `stimulus.interval` the stimulus)
    relative to the onset; `recording` gives the entry and the trial's sample
    range; `aux` lists any auxiliary pulses, if they were requested; and
    `audio_sync` describes how the onset was found on an audio channel, if one
    was used (see audio_sync_trials).
    """
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
        audio = getattr(trial, "audio_sync", None)
        if isinstance(audio, dict):
            pproc["audio_sync"] = audio
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
    """group-kilo-spikes: write a pprox file (and a waveform file) for each
    good cluster in a kilosort/phy output directory. See --help."""
    import argparse
    import os

    from dlab import __version__
    from dlab.util import ParseKeyVal, add_log_arguments, json_serializable, setup_log

    version = SCRIPT_VERSION

    p = argparse.ArgumentParser(
        prog="group-kilo-spikes",
        description="group kilosorted spikes into pprox files based on cluster and trial",
    )
    p.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {version} (melizalab-tools {__version__})",
    )
    add_log_arguments(p)
    nbank.add_registry_argument(p)
    p.add_argument(
        "--dry-run",
        help="do everything except write the output files",
        action="store_true",
    )
    p.add_argument(
        "--sync",
        default="ADC3",
        help="name of the channel with the sync signal, clicks or pulses at stimulus "
        "onset (default '%(default)s')",
    )
    p.add_argument(
        "--sync-thresh",
        type=float,
        help="threshold for detecting sync events, in the units of the sync channel "
        "(default: halfway between the channel's baseline and its peak, with checks "
        "that the channel has clear events). Only needed to override the default.",
    )
    p.add_argument(
        "--audio-sync",
        metavar="CHANNEL",
        help="find the stimulus onsets on this channel, a copy of the audio sent to "
        "the speaker, by cross-correlation with the stimulus files, instead of using "
        "a sync track (for recordings without one). A hardwired copy is best: its "
        "onsets match a sync track's to the sample. A microphone also works, YMMV: "
        "its onsets are later by the acoustic path (~1.3-2.6 ms in our rigs) and "
        "vary by a few ms.",
    )
    p.add_argument(
        "--aux",
        action=ParseKeyVal,
        metavar="NAME=CHANNEL[:STREAM]",
        help="record pulses on an auxiliary channel (e.g. an optogenetic light "
        "source or a sensor's TTL output) in each trial as NAME. If STREAM is given "
        "(condition, trial, or channelN), check the pulses against the messages on "
        "that jrelay stream. May be repeated.",
    )
    p.add_argument(
        "--oeaudio-log",
        type=Path,
        help="get the list of stimuli from this open-ephys-audio log file, for "
        "recordings without the messages in the ARF file. Ideally the log from the "
        "same session; a log from another session with the stimuli in the same "
        "order can be used (with a warning) if no sync events are missing.",
    )
    p.add_argument(
        "--prepad",
        type=float,
        default=1.0,
        help="sets trial start time relative to stimulus onset (default %(default)0.1f s)",
    )
    p.add_argument(
        "--sort-start",
        type=float,
        metavar="SECONDS",
        help="if only part of the recording was sorted (e.g. with kilosort's "
        "tmin), the time in the recording (s) where the sorted part starts. "
        "Whether temp_wh.dat and the spike times count from there or from the "
        "recording's start is found from the data; the output is timed from the "
        "recording's start, with only the trials within the sorted part.",
    )
    p.add_argument(
        "--sort-stop",
        type=float,
        metavar="SECONDS",
        help="if only part of the recording was sorted (e.g. with kilosort's "
        "tmax), the time in the recording (s) where the sorted part stops. The "
        "sorted part is checked against the recording's neural channels.",
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
        help="time before the spike to keep in the waveforms (default %(default).1f ms)",
    )
    p.add_argument(
        "--waveform-post-peak",
        type=float,
        default=5.0,
        help="time after the spike to keep in the waveforms (default %(default).1f ms)",
    )
    p.add_argument(
        "--local-stim-dir",
        type=Path,
        help="DEBUG/TESTING ONLY. Search this directory for stimulus files.",
    )
    p.add_argument(
        "--sorter",
        metavar="NAME_VERSION",
        help="the spike sorter and its version (e.g. 'kilosort 2.5'), recorded "
        "in the outputs. Read from kilosort4.log for kilosort 4; give it for "
        "other sorters, which don't record it.",
    )
    p.add_argument(
        "--alt-stim-names",
        type=Path,
        metavar="YAML",
        help="YAML file mapping the names of stimuli in the messages (or log) to "
        "the names they were deposited under in neurobank, for stimuli presented "
        "under other names. Trials are labeled with the neurobank names.",
    )
    p.add_argument("recording", type=Path, help="path of ARF recording file")
    p.add_argument(
        "sortdir",
        type=Path,
        help="kilosort output directory. Needs to contain 'spike_times.npy', 'spike_clusters.npy',"
        " 'cluster_info.tsv', and 'temp_wh.dat'",
    )
    args = p.parse_args(argv)
    if args.audio_sync and args.sync_thresh is not None:
        p.error("--sync-thresh doesn't apply to --audio-sync")
    windowed = args.sort_start is not None or args.sort_stop is not None
    if args.toelis and windowed:
        p.error("--sort-start and --sort-stop don't apply to --toelis")
    setup_log(args.debug, args.debug_http)
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
    stimulus_names = {}
    if args.alt_stim_names is not None:
        stimulus_names = yaml_to_stimulus_names(args.alt_stim_names)
        log.info(
            "  - using %s for the neurobank names of %d stimuli",
            args.alt_stim_names,
            len(stimulus_names),
        )
    stim_finder = StimulusFinder(args.registry_url, args.local_stim_dir)

    log.info("- kilosort output directory: %s", args.sortdir)
    sorted_by = kilosort_log_to_sorter(args.sortdir / "kilosort4.log")
    if sorted_by is not None and args.sorter is not None and sorted_by != args.sorter:
        raise RuntimeError(
            f"--sorter is '{args.sorter}', but kilosort4.log says '{sorted_by}'"
        )
    sorted_by = sorted_by or args.sorter
    if sorted_by is None:
        log.warning("  - WARNING: sorter version unknown; give it with --sorter")
    else:
        log.info("  - sorted by %s", sorted_by)
    timefile = args.sortdir / "spike_times.npy"
    clustfile = args.sortdir / "spike_clusters.npy"
    infofile = args.sortdir / "cluster_info.tsv"
    log.info("  - spike times: %s", timefile)
    log.info("  - spike clusters: %s", clustfile)
    # kilosort 2.5 saves uint64, which numpy promotes to float when combined
    # with signed integers, and which wraps instead of going negative
    events = pd.DataFrame(
        {
            "time": np.load(timefile).squeeze().astype("int64"),
            "clust": np.load(clustfile),
        },
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

    # the spikes of each cluster, and how many are dropped at each step
    n_sorted = events.clust.value_counts()
    # find duplicates - this is rare but needs to be caught
    duplicates = events.duplicated()
    n_duplicate = events[duplicates].clust.value_counts()
    if duplicates.any():
        dupl_clusts = events[duplicates].clust
        log.warning(
            "  - warning: removing spikes with duplicate times from clusters %s",
            ",".join(str(c) for c in dupl_clusts),
        )
        events = events[~duplicates]

    events.set_index("clust", inplace=True)
    if args.toelis:
        clusters = events_to_toelis(events, params["sampling_rate"])
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
            str(args.recording), registry_url=args.registry_url
        )

    sampling_rate = params["sampling_rate"]
    good_clust_types = ("good",)
    if args.mua:
        good_clust_types += ("mua",)
    n_before = int(args.waveform_pre_peak * sampling_rate / 1000)
    n_after = int(args.waveform_post_peak * sampling_rate / 1000)

    log.info("- splitting '%s' into trials:", datafile)
    with h5.File(datafile, "r") as afp:
        trials = arf_to_trials(
            afp,
            stim_finder,
            args.sync,
            args.sync_thresh,
            args.prepad,
            oeaudio_log=args.oeaudio_log,
            aux=args.aux,
            audio_sync=args.audio_sync,
            stimulus_names=stimulus_names,
        )
        entry_attrs = tuple(entry_to_metadata(e) for _, e in iter_entries(afp))
        entries = [entry for _, entry in iter_entries(afp)]
        # the part of the recording that was sorted; only checked for
        # single-entry recordings (temp_wh.dat for several is not settled)
        window = None
        sorted_window = {}
        if len(entries) > 1 and windowed:
            raise RuntimeError(
                "--sort-start and --sort-stop are only supported for recordings "
                "with one entry"
            )
        if len(entries) == 1:
            entry = entries[0]
            track = entry[args.audio_sync or args.sync]
            if track.attrs["sampling_rate"] != sampling_rate:
                raise RuntimeError(
                    f"the recording's sampling rate ({track.attrs['sampling_rate']}) "
                    f"isn't the sort's ({sampling_rate})"
                )
            window = kilo_window.sort_window(
                args.sort_start, args.sort_stop, track.size, nsamples, sampling_rate
            )
        if windowed:
            log.info(
                "- sorted window: samples %d-%d (%.3f-%.3f s) of the recording",
                window.start,
                window.stop,
                window.start / sampling_rate,
                window.stop / sampling_rate,
            )
            mapfile = args.sortdir / "channel_map.npy"
            channel_map = (
                np.load(mapfile).ravel() if mapfile.exists() else np.arange(nchannels)
            )
            # the largest units to be written, on different channels
            units = []
            for clust_id in events.index.value_counts().index:
                if clust_id not in info.index:
                    continue
                if info.loc[clust_id, "group"] not in good_clust_types:
                    continue
                column = int(info.loc[clust_id, "ch"])
                if column in (u[1] for u in units):
                    continue
                times = events.loc[[clust_id], "time"].to_numpy(copy=True)
                units.append((int(clust_id), column, times))
                if len(units) == kilo_window.CHECK_UNITS:
                    break
            if not units:
                raise RuntimeError("no units to check the sorted window against")
            log.info("  - finding where the sort and its spike times start:")
            window, window_check = kilo_window.check_sort_window(
                recording,
                entry,
                window,
                units,
                channel_map,
                n_before,
                n_after,
                sampling_rate,
            )
            trials, trials_before, trials_after = trials_in_window(
                trials, window.start, window.stop
            )
            log.info(
                "  - %d trials in the window (%d before and %d after it dropped)",
                len(trials),
                trials_before,
                trials_after,
            )
            if not trials:
                raise RuntimeError("no trials in the sorted window")
            sorted_window = {
                "sorted_window": {
                    "entry": 0,
                    "start": window.start,
                    "stop": window.stop,
                    "start_s": window.start / sampling_rate,
                    "stop_s": window.stop / sampling_rate,
                    "first_sample": round(track.attrs["offset"] * sampling_rate),
                    "sorted_samples": window.sorted_samples,
                    "sorted_data_start": window.data_origin,
                    "spike_times_start": window.spike_origin,
                    "trials_before": trials_before,
                    "trials_after": trials_after,
                    **window_check,
                }
            }
        trials = pd.DataFrame(trials)

    # spike times may count from the window's start, and temp_wh.dat may start
    # there too (see dlab.kilo_window); time the spikes from the recording's
    # start, and index temp_wh.dat from data_origin
    spike_origin = 0 if window is None else window.spike_origin
    data_origin = 0 if window is None else window.data_origin
    events["time"] += spike_origin

    # this pandas magic sorts the events by cluster and trial
    log.info("- sorting events into trials:")
    events["trial"] = assign_spikes(events.time, trials.recording_start)
    # spikes before the first trial are in no trial; drop them from the waveform
    # files too, so these and the pprox files have the same spikes
    before = events.trial < 0
    n_before_first = events[before].index.value_counts()
    if before.any():
        log.info("  - dropping %d spikes before the first trial", before.sum())
        events = events[~before]
    # and spikes after the end of the trial they fall in: in the interval of a
    # dropped trial, or after the last trial of a sorted window
    ends = trials.recording_end.to_numpy()[events.trial.to_numpy()]
    after = events.time.to_numpy() > ends
    n_after_end = events[after].index.value_counts()
    if after.any():
        log.info(
            "  - dropping %d spikes after the end of their trial (in dropped trials)",
            after.sum(),
        )
        events = events[~after]

    # the options that determine the trials, so they can be reconstructed;
    # written to the pprox and waveform files (flat, for hdf5 attributes)
    if args.audio_sync:
        trial_options = {"audio_sync_track": args.audio_sync, "prepad": args.prepad}
    else:
        trial_options = {"sync_track": args.sync, "prepad": args.prepad}
    if args.sync_thresh is not None:
        trial_options["sync_thresh"] = args.sync_thresh
    if args.oeaudio_log is not None:
        trial_options["oeaudio_log"] = args.oeaudio_log.name
    if args.local_stim_dir is not None:
        trial_options["local_stim_dir"] = args.local_stim_dir.name
    # the sort, and the options that decide which spikes are kept
    sort_fields = {
        "sort": {
            "directory": args.sortdir.resolve().name,
            "samples": nsamples,
            "channels": nchannels,
            "dtype": params["dtype"],
            "sampling_rate": params["sampling_rate"],
        },
        **({"sorted_by": sorted_by} if sorted_by is not None else {}),
        "artifact_reject_thresh": args.artifact_reject_thresh,
        "waveform_pre_peak": args.waveform_pre_peak,
        "waveform_post_peak": args.waveform_post_peak,
    }
    # the stimuli presented under other names; only written if there are any
    alt_stim_names = {"stimulus_names": stimulus_names} if stimulus_names else {}
    # describes the auxiliary channels; only written if there are any
    aux_tracks = {}
    if args.aux:
        aux_tracks = {"aux_tracks": {}}
        for name, spec in args.aux.items():
            channel, _, stream = spec.partition(":")
            track = {"channel": channel}
            if stream:
                track["stream"] = stream
            aux_tracks["aux_tracks"][name] = track
    # how the onsets were found on the audio channel; only in the pprox
    audio_parameters = {}
    if args.audio_sync:
        from dlab import audio_sync

        audio_parameters = {"audio_sync_parameters": audio_sync.parameters()}
    total_spikes = 0
    total_clusters = 0
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
        sorted_times = cluster.time - data_origin  # in temp_wh.dat
        spikes = cluster[
            (sorted_times > n_before) & (sorted_times < (nsamples - n_after))
        ]
        n_clean = len(spikes)
        # the same windows qs.peaks would extract; indexing the read-only
        # memmap directly reads only the samples around each spike
        windows = (spikes.time.to_numpy() - data_origin)[:, None] + np.arange(
            -n_before, n_after
        )
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

        spike_counts = {
            "sorted": int(n_sorted.get(clust_id, 0)),
            "duplicate": int(n_duplicate.get(clust_id, 0)),
            "before_first_trial": int(n_before_first.get(clust_id, 0)),
            "after_trial_end": int(n_after_end.get(clust_id, 0)),
            "near_ends": n_spikes - n_clean,
            "artifact": n_clean - len(spikes),
            "kept": len(spikes),
        }
        unit_fields = {
            "kilosort_cluster": int(clust_id),
            "kilosort_group": clust_type,
            "spike_counts": spike_counts,
        }

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
            **unit_fields,
            **sort_fields,
            **trial_options,
            **sorted_window,
            **alt_stim_names,
            **aux_tracks,
            **audio_parameters,
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
                    **fields_to_attrs(
                        {
                            **unit_fields,
                            **sort_fields,
                            **trial_options,
                            **sorted_window,
                            **alt_stim_names,
                        }
                    ),
                )

    log.info(
        "- a total of %d spikes were assigned to %d clusters",
        total_spikes,
        total_clusters,
    )


if __name__ == "__main__":
    group_spikes_script()
