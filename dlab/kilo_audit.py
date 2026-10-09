# -*- mode: python -*-
"""Audit the group-kilo-spikes output for one recording, and find the units
of recordings to audit.

audit-kilo-spikes checks the pprox files (one per unit) sorted from a
recording against their waveform files, against the stimulus messages in the
recording's ARF file, and against each other. It reports likely errors but
changes nothing: deposited resources have permanent identifiers, so whether a
problem justifies reprocessing is a decision for a person.

Findings are graded by their effect on analyses of the data:

- info: no effect (e.g. spikes before the first trial in the waveform file,
  which versions before 2026.10.07 kept).
- warn: specific trials are unreliable and can be excluded (they are listed),
  or the files are inconsistent in a way that doesn't change the spike times.
- fail: the unit as a whole is unreliable (e.g. trials labeled with the wrong
  stimulus, most onsets out of line with the messages, or events that don't
  match the waveform file).

The report is JSON, with a status (the worst finding) for the recording and
each unit. The exit status is 0 if the audit ran, whatever it found, so batch
runs (e.g. with GNU parallel) only see failures to run.

find-kilo-units finds the units of recordings in the registry (given by name
fragment, as a list, or --all) and writes a control file for batch runs, one line per recording: the recording id, a tab,
and its units, comma-separated. For example:

    find-kilo-units --name P397 --reports reports -o audit.tsv
    parallel --colsep '\t' -a audit.tsv \
        'audit-kilo-spikes {1} --units {2} -o reports/{1}.json'

The recordings can also be listed in a file, or piped from a custom search:

    nbank search -d <arf dtype> -k <key>=<value> | find-kilo-units - -o audit.tsv

collect-kilo-audit summarizes the reports:

    collect-kilo-audit --control audit.tsv --tsv findings.tsv reports

Units are grouped by name, as group-kilo-spikes names them (<recording>_c<N>,
with waveform files <recording>_c<N>_spikes), so the audit checks that each
pprox names the same recording.

"""

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import h5py as h5
import httpx
import numpy as np
from nbank import core as nbank_core

from dlab import __version__, kilo, pprox
from dlab import neurobank as nbank
from dlab.util import add_log_arguments, setup_log

log = logging.getLogger("dlab")

SEVERITIES = ("ok", "info", "warn", "fail")
# the first version with the current sync detection; lag outliers in output
# from earlier versions are probably the known detection errors
SYNC_FIX_VERSION = "2026.10.07"
# lag outliers in more than this fraction of trials make a unit unreliable
MAX_OUTLIER_FRACTION = 0.5
# sync events follow their messages by 0.25-1 s; a longer lag isn't plausible
MAX_SYNC_LAG = 2.0
# lags usually vary by ~30 ms within a recording; a wider range (5th-95th
# percentile, s) means the messages were delayed by varying amounts
MAX_LAG_SPREAD = 0.1


def finding(check: str, severity: str, message: str, trials=None) -> dict:
    out = {"check": check, "severity": severity, "message": message}
    if trials is not None:
        out["trials"] = [int(t) for t in trials]
    return out


def worst(findings: Iterable[dict]) -> str:
    return max((f["severity"] for f in findings), key=SEVERITIES.index, default="ok")


@dataclass
class Unit:
    name: str
    pprox_path: Path
    pprox: dict
    waveforms_path: Path | None = None
    findings: list[dict] = field(default_factory=list)

    @property
    def trials(self) -> list[dict]:
        return self.pprox["pprox"]

    @property
    def processed_by(self) -> list[str]:
        value = self.pprox.get("processed_by", [])
        return [value] if isinstance(value, str) else list(value)

    def report(self) -> dict:
        return {
            "name": self.name,
            "pprox": str(self.pprox_path),
            "waveforms": None
            if self.waveforms_path is None
            else str(self.waveforms_path),
            "processed_by": self.processed_by,
            "status": worst(self.findings),
            "findings": self.findings,
        }


def old_sync_version(processed_by: list[str]) -> bool:
    """True if the unit was processed by a version of group-kilo-spikes (or
    group-klopto-spikes) from before the sync detection fixes."""
    for entry in processed_by:
        prog, _, version = entry.partition(" ")
        if prog in ("group-kilo-spikes", "group-klopto-spikes"):
            return version < SYNC_FIX_VERSION
    return False


# --- checks on a pprox alone


def _numbers(value, n: int | None = None) -> bool:
    """True if value is a list of n (or any number of) numbers"""
    return (
        isinstance(value, list | tuple)
        and (n is None or len(value) == n)
        and all(isinstance(x, int | float) and not isinstance(x, bool) for x in value)
    )


def _usable_trial(t) -> bool:
    """True if a trial has the fields stimtrial requires, with numeric times,
    so that the other checks can use it"""
    return (
        isinstance(t, dict)
        and _numbers(t.get("events"))
        and _numbers([t.get("offset")], 1)
        and _numbers(t.get("interval"), 2)
        and isinstance(t.get("stimulus"), dict)
        and "name" in t["stimulus"]
        and _numbers(t["stimulus"].get("interval"), 2)
    )


def check_pprox(unit: Unit, sampling_rate: float) -> list[dict]:
    """Checks the structure of a unit's pprox: the fields stimtrial requires,
    trial order, events within their trials, and consistency between the
    trial times and the recording sample ranges."""
    out = []
    trials = unit.trials
    missing = [i for i, t in enumerate(trials) if not _usable_trial(t)]
    if missing:
        out.append(
            finding(
                "pprox-fields",
                "fail",
                "trials without the events, offset, interval or stimulus that "
                "stimtrial requires (or with non-numeric times)",
                missing,
            )
        )
        return out
    offsets = np.array([t["offset"] for t in trials])
    if np.any(np.diff(offsets) < 0):
        out.append(finding("trial-order", "warn", "trials are not in time order"))
    indexes = [t.get("index") for t in trials]
    if len(set(indexes)) < len(indexes):
        out.append(finding("trial-index", "warn", "trial indexes are not unique"))
    outside = [
        i
        for i, t in enumerate(trials)
        if np.any(np.asarray(t["events"]) < t["interval"][0] - 1e-6)
        or np.any(np.asarray(t["events"]) > t["interval"][1] + 1e-6)
    ]
    if outside:
        out.append(
            finding(
                "events-in-interval",
                "warn",
                "trials with events outside the trial interval",
                outside,
            )
        )
    n_events = sum(len(t["events"]) for t in trials)
    n_sorted = unit.pprox.get("kilosort_n_spikes")
    if n_sorted is not None and n_events > n_sorted:
        out.append(
            finding(
                "spike-count",
                "warn",
                f"{n_events} events, but kilosort_n_spikes is {n_sorted}",
            )
        )
    if not all("recording" in t for t in trials):
        out.append(
            finding(
                "recording-field",
                "info",
                "trials without a recording field; not checked against the "
                "waveform file",
            )
        )
        return out
    starts = np.array([t["recording"]["start"] for t in trials])
    ends = np.array([t["recording"]["end"] for t in trials])
    if np.any(ends[:-1] > starts[1:]):
        out.append(
            finding(
                "trial-overlap",
                "warn",
                "trials that overlap the next trial",
                np.flatnonzero(ends[:-1] > starts[1:]),
            )
        )
    expected = np.array(
        [
            [round((t["offset"] + t["interval"][k]) * sampling_rate) for k in (0, 1)]
            for t in trials
        ]
    )
    bad = np.flatnonzero(np.abs(expected - np.c_[starts, ends]).max(1) > 1)
    if bad.size:
        out.append(
            finding(
                "recording-range",
                "warn",
                "trials whose recording sample range doesn't match their offset "
                "and interval",
                bad,
            )
        )
    return out


def _json_path(path) -> str:
    """A jsonschema error path as e.g. pprox[3].interval"""
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else f".{part}" if out else part
    return out or "(top level)"


def check_schema(unit: Unit, max_errors: int = 3) -> list[dict]:
    """Validates a unit's pprox against the schema named in its $schema, using
    the bundled copies of the published pprox and stimtrial schemas."""
    schema = unit.pprox.get("$schema")
    if schema is None:
        return [finding("schema", "info", "the pprox has no $schema; not validated")]
    try:
        errors = pprox.validation_errors(unit.pprox)
    except ValueError:
        return [finding("schema", "info", f"unknown $schema {schema}; not validated")]
    if not errors:
        return []
    described = []
    for err in errors[:max_errors]:
        message = err.message if len(err.message) <= 100 else err.message[:97] + "..."
        described.append(f"{_json_path(err.absolute_path)}: {message}")
    more = f" and {len(errors) - max_errors} more" if len(errors) > max_errors else ""
    trials = sorted(
        {
            e.absolute_path[1]
            for e in errors
            if len(e.absolute_path) > 1 and e.absolute_path[0] == "pprox"
        }
    )
    return [
        finding(
            "schema",
            "warn",
            f"{len(errors)} violation(s) of {schema}: " + "; ".join(described) + more,
            trials or None,
        )
    ]


# --- pprox vs waveform file


def check_waveforms(unit: Unit) -> list[dict]:
    """Checks that the events in a unit's pprox can be rebuilt from the spike
    times in its waveform file."""
    if unit.waveforms_path is None:
        return [finding("waveforms", "info", "no waveform file")]
    if not all("recording" in t for t in unit.trials):
        return []
    with h5.File(unit.waveforms_path, "r") as fp:
        times = fp["times"][:]
        sampling_rate = fp["times"].attrs["sampling_rate"]
        recording = fp.attrs.get("recording")
    out = []
    if recording is not None and recording != unit.pprox.get("recording"):
        out.append(
            finding(
                "waveforms-recording",
                "fail",
                f"the waveform file is from {recording}, the pprox from "
                f"{unit.pprox.get('recording')}",
            )
        )
    events, n_before = kilo.waveforms_to_events(times, unit.trials, sampling_rate)
    tolerance = 1.5 / sampling_rate
    bad = [
        i
        for i, (rebuilt, trial) in enumerate(zip(events, unit.trials, strict=True))
        if rebuilt.size != len(trial["events"])
        or not np.allclose(rebuilt, trial["events"], rtol=0, atol=tolerance)
    ]
    if bad:
        out.append(
            finding(
                "waveforms-events",
                "fail",
                "trials whose events don't match the spike times in the waveform file",
                bad,
            )
        )
    if n_before:
        out.append(
            finding(
                "waveforms-before-first-trial",
                "info",
                f"{n_before} spike(s) before the first trial in the waveform file "
                "(kept by versions before 2026.10.07)",
            )
        )
    return out


# --- pprox vs ARF messages


def entry_clock(entry, channel: str | None = None) -> tuple[float, int]:
    """The sampling rate of an entry's channels and the open-ephys sample number
    of their first sample, from the given channel (e.g. the sync track) or
    else the first channel with these attributes."""
    messages = kilo.find_message_dset(entry)
    names = [channel] if channel and channel in entry else []
    names += [name for name in entry if name != channel]
    for name in names:
        dset = entry[name]
        if dset == messages or not {"offset", "sampling_rate"} <= dset.attrs.keys():
            continue
        rate = float(dset.attrs["sampling_rate"])
        return rate, round(dset.attrs["offset"] * rate)
    raise RuntimeError(f"no channel in {entry.name} has the recording clock")


def entry_stimuli(entry, first_sample: int, sampling_rate: float, oeaudio_log):
    """The stimuli presented during an entry, with start times in samples from
    the start of the recording, from the message dataset or else the oeaudio
    log. Returns None if neither is available."""
    dset = kilo.find_message_dset(entry)
    if dset is not None:
        stimuli = kilo.messages_to_stimuli(dset)
    elif oeaudio_log is not None:
        with open(oeaudio_log) as fp:
            stimuli = list(kilo.oeaudio_log_to_stimuli(fp, sampling_rate))
    else:
        return None
    return [stim._replace(start=stim.start - first_sample) for stim in stimuli]


def clock_shift(
    names: list[str], onsets: np.ndarray, stimuli, sampling_rate: float
) -> tuple[int, float] | None:
    """Looks for a constant shift between a unit's trials and the stimulus
    messages: an offset k such that trial i follows message i+k throughout,
    with its label and with consistent lags (see kilo.sync_lag_range), as when
    a recording was sorted from some time after its start and the trials (and
    spikes) were timed from the start of the sort (P388_3_1, P390_3_1).
    Returns k and the median time from message to onset (s, negative if the
    onsets come first), or None if there is no such offset."""
    codes = {}
    trial_codes = np.array([codes.setdefault(n, len(codes)) for n in names])
    message_codes = np.array([codes.setdefault(s.name, len(codes)) for s in stimuli])
    n, m = trial_codes.size, message_codes.size
    if n < 20 or m < n:
        return None
    # every trial must have a message, so k is in [0, m - n]; score the first
    # trials, then check all of them
    head = trial_codes[: min(n, 200)]
    scores = [
        np.count_nonzero(message_codes[k : k + head.size] == head)
        for k in range(m - n + 1)
    ]
    k = int(np.argmax(scores))
    if np.mean(message_codes[k : k + n] == trial_codes) < 0.95:
        return None
    starts = np.array([s.start for s in stimuli[k : k + n]])
    lags = (onsets - starts) / sampling_rate
    lo, hi = kilo.sync_lag_range(lags)
    # no consistent range (e.g. drifting onsets), or lags of the usual size
    if lo == hi or (lo >= 0 and hi <= MAX_SYNC_LAG):
        return None
    return k, float(np.median(lags))


def check_messages(unit: Unit, stimuli, sampling_rate: float) -> list[dict]:
    """Checks a unit's trials against the stimulus start messages: each trial's
    onset (its sync event) should follow a start message for its stimulus by a
    lag consistent with the other trials. If the trials don't fit the messages
    as they are, but do with a constant shift (see clock_shift), the shift is
    reported and the trials are checked against the messages it pairs them
    with."""
    if stimuli is None:
        return [
            finding(
                "messages",
                "info",
                "no stimulus messages in the ARF file (use --oeaudio-log); "
                "trials not checked against them",
            )
        ]
    trials = unit.trials
    names = [t["stimulus"]["name"] for t in trials]
    starts = np.array([s.start for s in stimuli])
    onsets = np.array([round(t["offset"] * sampling_rate) for t in trials])
    idx = np.searchsorted(starts, onsets, side="right") - 1
    out = []

    def mislabeled(idx):
        return [
            i
            for i, (name, k) in enumerate(zip(names, idx, strict=True))
            if k >= 0 and name != stimuli[k].name
        ]

    early = np.flatnonzero(idx < 0)
    wrong = mislabeled(idx)
    if early.size or len(wrong) > MAX_OUTLIER_FRACTION * len(trials):
        shift = clock_shift(names, onsets, stimuli, sampling_rate)
        if shift is not None:
            k, lag = shift
            out.append(
                finding(
                    "clock-shift",
                    "warn",
                    f"trial i follows message i+{k}, with onsets {abs(lag):.3f} s "
                    f"{'before' if lag < 0 else 'after'} their messages: the trials "
                    "are timed from another origin (e.g. the start of a sort of "
                    "part of the recording); checked against those messages",
                )
            )
            idx = np.arange(len(trials)) + k
            early = np.array([], dtype=int)
            wrong = mislabeled(idx)
    if early.size:
        severity = "fail" if early.size > MAX_OUTLIER_FRACTION * len(trials) else "warn"
        out.append(
            finding(
                "messages-before",
                severity,
                "trials that start before any stimulus message",
                early,
            )
        )
    if wrong:
        out.append(
            finding(
                "stimulus-labels",
                "fail",
                "trials labeled with a different stimulus from the message "
                "before their onset",
                wrong,
            )
        )
    later = np.flatnonzero(idx >= 0)
    shared = later[1:][np.diff(idx[later]) == 0]
    if shared.size:
        out.append(
            finding(
                "messages-shared",
                "fail",
                "trials that follow the same stimulus message as the previous trial",
                shared,
            )
        )
    # the lags describe the message timing: the neural data and the sync track
    # share a clock, so an onset that is a stimulus's sync event is right
    # whatever its lag (check_stimulus_durations checks the onsets)
    outliers = later[
        kilo.sync_lag_outliers(starts[idx[later]], onsets[later], sampling_rate)
    ]
    lags = (onsets[later] - starts[idx[later]]) / sampling_rate
    if lags.size:
        p5, p95 = np.percentile(lags, [5, 95])
        timing = (
            f"onsets follow their messages by {np.median(lags):.3f} s "
            f"(5th-95th percentile {p5:.3f}-{p95:.3f} s)"
        )
    if outliers.size or (lags.size and p95 - p5 > MAX_LAG_SPREAD):
        notes = []
        if p95 - p5 > MAX_LAG_SPREAD:
            notes.append("spread more than the usual ~30 ms")
        if outliers.size:
            notes.append("the listed trials are more than 0.1 s out of line")
        message = (
            f"{timing}; {'; '.join(notes)}: the messages were delayed by varying "
            "amounts (message timing only; the onsets come from the sync track, "
            "and stimulus-durations checks them)"
        )
        out.append(
            finding("sync-lag", "info", message, outliers if outliers.size else None)
        )
    n_dropped = len(stimuli) - np.unique(idx[later]).size
    allowed = max(1, int(0.01 * len(stimuli)))
    if n_dropped > allowed:
        out.append(
            finding(
                "trials-dropped",
                "warn",
                f"{n_dropped} of {len(stimuli)} stimulus messages have no trial",
            )
        )
    elif n_dropped:
        out.append(
            finding(
                "trials-dropped",
                "info",
                f"{n_dropped} of {len(stimuli)} stimulus messages have no trial",
            )
        )
    return out


def pulse_sync_track(unit: Unit, entry, sampling_rate: float, cache: dict):
    """The name of the entry's sync track and its pulses, if it has sustained
    pulses, or None. The track is the unit's sync_track, or for older pprox
    files that don't record it, the pulse channel with pulses rising at (within
    2 ms of) the most of the unit's onsets, if that is at least half of them.
    cache holds the pulses detected on each channel, shared with check_aux."""
    onsets = np.array([round(t["offset"] * sampling_rate) for t in unit.trials])
    named = unit.pprox.get("sync_track")
    if named is not None and named in entry:
        names = [named]
    else:
        messages = kilo.find_message_dset(entry)
        names = [
            name
            for name, dset in entry.items()
            if not name.startswith("CH")
            and dset != messages
            and dset.ndim == 1
            and dset.dtype.kind in "iuf"
            and "sampling_rate" in dset.attrs
        ]
    window = round(0.002 * sampling_rate)
    best, best_match = None, 0.5
    for name in names:
        key = (entry.name, name)
        if key not in cache:
            cache[key] = kilo.detect_pulses(entry[name][:])
        pulses = cache[key]
        if not kilo.is_pulse_track(pulses, sampling_rate) or onsets.size == 0:
            continue
        if name == named:
            return name, pulses
        distance = np.abs(pulses[_nearest_rise(pulses, onsets), 0] - onsets)
        match = np.mean(distance <= window)
        if match >= best_match:
            best, best_match = (name, pulses), match
    return best


def _nearest_rise(pulses: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """The index of the pulse whose rise is nearest each onset"""
    rises = pulses[:, 0]
    after = np.clip(np.searchsorted(rises, onsets), 0, rises.size - 1)
    before = np.maximum(after - 1, 0)
    return np.where(
        np.abs(rises[after] - onsets) <= np.abs(rises[before] - onsets), after, before
    )


def check_stimulus_durations(unit: Unit, sampling_rate: float, sync=None) -> list[dict]:
    """Checks that the trials' stimulus lengths fit their onsets: the gap from
    the end of each stimulus to the next onset (from the pprox alone), and if
    sync gives the entry's pulse sync track (name, pulses), that each onset is
    the rise of a pulse as long as its stimulus (see
    kilo.stimulus_length_mismatches). Since the stimuli are presented in order,
    lengths that fit mean each sync event has the right stimulus."""
    trials = unit.trials
    onsets = np.array([round(t["offset"] * sampling_rate) for t in trials])
    lengths = np.array(
        [t["stimulus"]["interval"][1] - t["stimulus"]["interval"][0] for t in trials]
    )
    widths = None
    no_pulse = np.array([], dtype=int)
    if sync is not None:
        pulses = sync[1]
        k = _nearest_rise(pulses, onsets)
        at_rise = np.abs(pulses[k, 0] - onsets) <= round(0.002 * sampling_rate)
        widths = np.where(at_rise, pulses[k, 1] - pulses[k, 0], np.nan)
        no_pulse = np.flatnonzero(~at_rise)
    bad = np.union1d(
        kilo.stimulus_length_mismatches(onsets, lengths, sampling_rate, widths),
        no_pulse,
    )
    if bad.size == 0:
        return []
    message = (
        "trials whose stimulus length doesn't fit the sync track (the gap before "
        "the next onset"
        + ("" if sync is None else f", or the width of the pulse on {sync[0]}")
        + "): the sync events may be paired with the wrong stimuli"
    )
    if no_pulse.size:
        message += f"; {no_pulse.size} onsets are not the rise of a pulse"
        if old_sync_version(unit.processed_by):
            message += (
                f" (in versions before {SYNC_FIX_VERSION}, probably pulse onsets "
                "reported at the end of the pulse)"
            )
    allowed = max(1, int(0.01 * len(trials)))
    severity = "fail" if bad.size > allowed else "warn"
    return [finding("stimulus-durations", severity, message, bad)]


def check_recording_name(unit: Unit, recording: str) -> list[dict]:
    """Checks that a unit's pprox names the recording it is audited against
    (by neurobank id, the last part of its URL)."""
    url = unit.pprox.get("recording")
    if url is None:
        return [finding("recording-name", "info", "the pprox names no recording")]
    name = url.rstrip("/").rsplit("/", 1)[-1]
    if name != recording:
        return [
            finding(
                "recording-name",
                "warn",
                f"the pprox names recording {name}, not {recording}",
            )
        ]
    return []


# --- aux pulses


def unit_aux_pulses(unit: Unit, sampling_rate: float) -> dict[str, list[tuple]]:
    """The aux pulses in a unit's pprox, by name, as (trial, onset, offset), with
    onset and offset in samples from the start of the recording"""
    out: dict[str, list[tuple]] = {}
    for i, trial in enumerate(unit.trials):
        for pulse in trial.get("aux", []):
            try:
                start, end = (
                    round((trial["offset"] + x) * sampling_rate)
                    for x in pulse["interval"]
                )
                out.setdefault(pulse["name"], []).append((i, start, end))
            except (KeyError, TypeError, ValueError):
                continue  # reported by check_aux_fields
    return out


def check_aux_fields(unit: Unit) -> list[dict]:
    """Checks the aux pulses in a unit's pprox: aux_tracks describes them, every
    trial has an aux list, each pulse has a known name and an interval, and
    each starts in its trial (pulses are assigned to the trial in which they
    start)."""
    tracks = unit.pprox.get("aux_tracks")
    with_aux = [i for i, t in enumerate(unit.trials) if t.get("aux")]
    if tracks is None:
        if with_aux:
            return [
                finding(
                    "aux-tracks",
                    "warn",
                    "trials have aux pulses, but there is no aux_tracks field "
                    "describing their channels",
                    with_aux,
                )
            ]
        return []
    out = []
    no_list = [i for i, t in enumerate(unit.trials) if "aux" not in t]
    if no_list:
        out.append(
            finding(
                "aux-fields",
                "warn",
                "trials without an aux list (a trial without pulses should have an "
                "empty one)",
                no_list,
            )
        )
    malformed, unknown, outside = [], set(), []
    for i, trial in enumerate(unit.trials):
        lo, hi = trial["interval"]
        for pulse in trial.get("aux", []):
            try:
                name = pulse["name"]
                start, end = pulse["interval"]
            except (KeyError, TypeError, ValueError):
                malformed.append(i)
                continue
            if name not in tracks:
                unknown.add(name)
                malformed.append(i)
            elif end < start:
                malformed.append(i)
            elif not lo - 1e-6 <= start < hi + 1e-6:
                outside.append(i)
    if malformed:
        names = f" (names not in aux_tracks: {', '.join(sorted(map(str, unknown)))})"
        out.append(
            finding(
                "aux-fields",
                "warn",
                "trials with malformed aux pulses" + (names if unknown else ""),
                sorted(set(malformed)),
            )
        )
    if outside:
        out.append(
            finding(
                "aux-fields",
                "warn",
                "trials with aux pulses that don't start in the trial",
                sorted(set(outside)),
            )
        )
    return out


def check_aux_channel(
    unit: Unit, name: str, channel: str, detected: np.ndarray, sampling_rate: float
) -> list[dict]:
    """Checks a unit's aux pulses against the pulses detected on their channel
    (onsets and offsets, to within a sample). Pulses before the first trial
    are not expected in the pprox."""
    first = unit.trials[0]["recording"]["start"] if unit.trials else 0
    starts = np.array([t["recording"]["start"] for t in unit.trials])
    on_channel = [(int(on), int(off)) for on, off in detected if on >= first]
    in_pprox = unit_aux_pulses(unit, sampling_rate).get(name, [])
    channel_onsets = np.array([on for on, _ in on_channel])
    not_on_channel, durations, matched = [], [], set()
    for trial, on, off in in_pprox:
        k = np.flatnonzero(np.abs(channel_onsets - on) <= 1) if on_channel else []
        if len(k) == 0:
            not_on_channel.append(trial)
            continue
        matched.add(int(k[0]))
        if abs(on_channel[k[0]][1] - off) > 1:
            durations.append(trial)
    not_in_pprox = sorted(
        {
            int(np.searchsorted(starts, on, side="right") - 1)
            for k, (on, _) in enumerate(on_channel)
            if k not in matched
        }
    )
    out = []
    for trials, what in (
        (not_on_channel, f"aux '{name}' pulses in the pprox that aren't on {channel}"),
        (not_in_pprox, f"pulses on {channel} missing from the pprox's aux '{name}'"),
        (durations, f"aux '{name}' pulses whose end doesn't match {channel}"),
    ):
        if trials:
            out.append(
                finding(
                    "aux-pulses", "warn", f"trials with {what}", sorted(set(trials))
                )
            )
    return out


def check_aux_stream(
    unit: Unit,
    name: str,
    stream: str,
    detected: np.ndarray,
    messages: list | None,
    sampling_rate: float,
) -> list[dict]:
    """Checks the pulses on an aux channel against the messages on the stream
    that drives it (see kilo.match_aux_pulses), listing the trials affected.
    The pulse track is the ground truth: a message without a pulse (e.g. the
    LED didn't fire) is recorded correctly in aux, but a pulse without a
    message may be spurious."""
    if not messages:
        return [
            finding(
                "aux-stream",
                "info",
                f"no '{stream}' messages to check aux '{name}' against",
            )
        ]
    onsets = (
        np.sort(np.asarray(detected[:, 0], dtype=int))
        if len(detected)
        else np.array([], dtype=int)
    )
    found = kilo.match_aux_pulses(onsets, messages, sampling_rate)
    starts = np.array([t["recording"]["start"] for t in unit.trials])

    def trials_of(samples):
        idx = np.searchsorted(starts, samples, side="right") - 1
        return sorted({int(i) for i in idx if i >= 0})

    msg_starts = np.array([m.start for m in messages])
    out = []
    if found["unexpected"]:
        out.append(
            finding(
                "aux-stream",
                "warn",
                f"trials with aux '{name}' pulses outside every '{stream}' message's "
                "window (spurious pulses?)",
                trials_of(onsets[found["unexpected"]]),
            )
        )
    if found["missing"]:
        out.append(
            finding(
                "aux-stream",
                "info",
                f"{len(found['missing'])} of {len(messages)} '{stream}' messages have "
                f"no '{name}' pulse (recorded as such in aux)",
                trials_of(msg_starts[found["missing"]] + found["lag"]),
            )
        )
    if found["late"]:
        out.append(
            finding(
                "aux-stream",
                "info",
                f"trials with aux '{name}' pulses whose lag after their '{stream}' "
                f"message differs from the median ({found['lag'] / sampling_rate:.3f} s) "
                "by more than 0.1 s",
                trials_of(msg_starts[found["late"]] + found["lag"]),
            )
        )
    return out


def entry_events(entry, first_sample: int) -> dict[str, list] | None:
    """The events on every message stream of an entry (see
    kilo.messages_to_events), in samples from the start of the recording, or
    None if the entry has no message dataset"""
    dset = kilo.find_message_dset(entry)
    if dset is None:
        return None
    return {
        stream: [
            ev._replace(
                start=ev.start - first_sample,
                end=None if ev.end is None else ev.end - first_sample,
            )
            for ev in events
        ]
        for stream, events in kilo.messages_to_events(dset).items()
    }


def check_aux(unit: Unit, entry, first_sample: int, sampling_rate: float, cache: dict):
    """Checks a unit's aux pulses: their fields, against their channels in the
    ARF file, and against the message streams named in aux_tracks. cache holds
    the pulses detected on each channel and the messages of each entry, which
    are shared by the units of a recording."""
    out = check_aux_fields(unit)
    tracks = unit.pprox.get("aux_tracks") or {}
    # pulses are compared by sample, which needs the trials' sample ranges
    if not isinstance(tracks, dict) or not all("recording" in t for t in unit.trials):
        return out
    for name, track in tracks.items():
        channel = track.get("channel") if isinstance(track, dict) else None
        if channel is None:
            out.append(
                finding(
                    "aux-pulses", "warn", f"aux_tracks gives no channel for '{name}'"
                )
            )
            continue
        key = (entry.name, channel)
        if key not in cache:
            cache[key] = (
                kilo.detect_pulses(entry[channel][:]) if channel in entry else None
            )
        detected = cache[key]
        if detected is None:
            out.append(
                finding(
                    "aux-pulses",
                    "warn",
                    f"channel {channel} for aux '{name}' is not in the ARF file",
                )
            )
            continue
        out.extend(check_aux_channel(unit, name, channel, detected, sampling_rate))
        stream = track.get("stream")
        if stream:
            key = (entry.name, "events", first_sample)
            if key not in cache:
                cache[key] = entry_events(entry, first_sample)
            messages = (cache[key] or {}).get(stream)
            out.extend(
                check_aux_stream(unit, name, stream, detected, messages, sampling_rate)
            )
    return out


# --- metadata

# the metadata fields that describe a recording, as named in the registry and
# the ARF entry attributes (arfx-oephys 2.8.0 and later)
METADATA_FIELDS = ("bird", "pen", "site", "hemisphere", "protocol", "experimenter")
# oeaudio-present's metadata message names some of them differently
MESSAGE_FIELDS = {"experiment": "protocol"}
# a bird's short name before the date in an entry name, e.g. "E79" in
# "..._E79_2026-06-23_12-33-22_chorus_Record Node..."
_re_entry_animal = re.compile(r"(?:^|[_/])([A-Za-z]+\d+)_\d{4}-\d{2}-\d{2}_")


def _same(a, b) -> bool:
    # the ARF stores all values as strings
    return str(a).strip() == str(b).strip()


def entry_metadata_sources(entry, label: str) -> dict[str, dict]:
    """The metadata recorded in an ARF entry, by source: its attributes, and
    the metadata message from oeaudio-present (if any)"""
    attrs = {
        k: _plain_attr(entry.attrs[k]) for k in METADATA_FIELDS if k in entry.attrs
    }
    message = {
        MESSAGE_FIELDS.get(k, k): v
        for k, v in kilo.entry_to_metadata(entry).items()
        if MESSAGE_FIELDS.get(k, k) in METADATA_FIELDS
    }
    return {f"{label} attributes": attrs, f"{label} metadata message": message}


def _plain_attr(value):
    return value.decode() if isinstance(value, bytes) else value


def disagreements(sources: dict[str, dict]) -> list[str]:
    """For each field with different values in different sources, a
    description of the values and their sources"""
    out = []
    fields = sorted({k for values in sources.values() for k in values})
    for field_ in fields:
        found = [
            (src, values[field_]) for src, values in sources.items() if field_ in values
        ]
        if any(not _same(found[0][1], v) for _, v in found[1:]):
            described = ", ".join(f"{v} ({src})" for src, v in found)
            out.append(f"{field_}: {described}")
    return out


def check_recording_metadata(
    entries, recording: str, record: dict | None
) -> list[dict]:
    """Checks the metadata in the ARF file against itself and the recording's
    registry record (if given), and the bird named in the ARF file against the
    recording's id."""
    out = []
    sources = {}
    if record is not None:
        sources["registry"] = {
            k: v for k, v in record.get("metadata", {}).items() if k in METADATA_FIELDS
        }
    for i, entry in enumerate(entries):
        label = "ARF" if len(entries) == 1 else f"entry {i}"
        sources.update(entry_metadata_sources(entry, label))
    mismatched = disagreements(sources)
    if mismatched:
        out.append(
            finding(
                "metadata-arf",
                "warn",
                "the ARF file and the registry disagree: " + "; ".join(mismatched)
                if record is not None
                else "the ARF file disagrees with itself: " + "; ".join(mismatched),
            )
        )
    bird = recording.split("_", 1)[0]
    for entry in entries:
        names = []
        animal = kilo.entry_to_metadata(entry).get("animal")
        if animal is not None:
            names.append((str(animal), "metadata message"))
        m = _re_entry_animal.search(entry.name)
        if m is not None:
            names.append((m.group(1), "entry name"))
        for name, source in names:
            if name != bird:
                out.append(
                    finding(
                        "metadata-name",
                        "warn",
                        f"the ARF {source} is for bird {name}, but the recording "
                        f"is {recording}",
                    )
                )
    return out


def check_unit_metadata(
    unit: Unit, record: dict | None, unit_records: list[dict]
) -> list[dict]:
    """Checks a unit's pprox against the recording's current registry record,
    and against the registry records of the unit's own resources (pprox and
    waveform file)."""
    out = []
    if record is not None:
        registry = record.get("metadata", {})
        keys = set(registry) | {k for k in METADATA_FIELDS if k in unit.pprox}
        differ = [
            f"{k}: {unit.pprox[k]} (pprox), {registry[k]} (registry)"
            for k in sorted(keys)
            if k in registry
            and k in unit.pprox
            and not _same(unit.pprox[k], registry[k])
        ]
        if differ:
            out.append(
                finding(
                    "metadata-registry",
                    "warn",
                    "the pprox and the recording's registry record disagree: "
                    + "; ".join(differ),
                )
            )
        one_side = [
            f"{k} ({'registry' if k in registry else 'pprox'} only)"
            for k in sorted(keys)
            if (k in registry) != (k in unit.pprox)
        ]
        if one_side:
            out.append(
                finding(
                    "metadata-registry",
                    "info",
                    "fields in only one of the pprox and the recording's registry "
                    "record: " + ", ".join(one_side),
                )
            )
    for rec in unit_records:
        metadata = rec.get("metadata", {})
        differ = [
            f"{k}: {unit.pprox[k]} (pprox), {metadata[k]} ({rec['name']})"
            for k in sorted(metadata)
            if k in unit.pprox and not _same(unit.pprox[k], metadata[k])
        ]
        if differ:
            out.append(
                finding(
                    "metadata-unit",
                    "warn",
                    "the pprox and the registry record of the unit's resource "
                    "disagree: " + "; ".join(differ),
                )
            )
    return out


# --- across units


def trials_key(trials: list[dict]) -> tuple:
    """A hashable summary of a trial table (onset, stimulus and interval of
    each trial), for comparing the trials of different units"""
    return tuple(
        (
            round(t.get("offset", np.nan), 6),
            t.get("stimulus", {}).get("name"),
            *np.round(t.get("interval", (np.nan, np.nan)), 6),
        )
        for t in trials
    )


def trial_table(unit: Unit) -> tuple:
    return trials_key(unit.trials)


def check_units(units: list[Unit]) -> list[dict]:
    """Checks that the units of a recording have the same trials and were
    processed by the same version. Returns findings for the recording."""
    out = []
    tables: dict[tuple, list[str]] = {}
    for unit in units:
        tables.setdefault(trial_table(unit), []).append(unit.name)
    if len(tables) > 1:
        groups = sorted(tables.values(), key=len, reverse=True)
        out.append(
            finding(
                "trial-tables",
                "warn",
                "units have different trials: "
                + "; ".join(", ".join(names) for names in groups),
            )
        )
    # the version that made the trials (later entries, e.g. from
    # regenerate-pprox, don't change them)
    versions = {(unit.processed_by or ["unknown"])[0] for unit in units}
    if len(versions) > 1:
        out.append(
            finding(
                "versions",
                "info",
                "units processed by different versions: " + "; ".join(sorted(versions)),
            )
        )
    return out


# --- driver


def audit_recording(
    arf_path: Path,
    units: list[Unit],
    oeaudio_log: Path | None = None,
    recording: str | None = None,
    registry_url: str | None = None,
) -> dict:
    """Audits the units sorted from a recording. recording is its neurobank id
    (by default, the name of the ARF file without its extension). Metadata are
    checked against the registry if registry_url is given. Returns the
    report."""
    recording = recording or Path(arf_path).stem
    findings = check_units(units)
    records = {}
    if registry_url:
        ids = [
            recording,
            *(u.name for u in units),
            *(f"{u.name}_spikes" for u in units),
        ]
        records = {r["name"]: r for r in nbank_core.describe_many(registry_url, *ids)}
        if recording not in records:
            findings.append(
                finding(
                    "registry",
                    "warn",
                    f"the recording {recording} is not in the registry",
                )
            )
    record = records.get(recording)
    with h5.File(arf_path, "r") as afp:
        entries = [entry for _, entry in kilo.iter_entries(afp)]
        findings.extend(check_recording_metadata(entries, recording, record))
        stimuli_cache = {}
        aux_cache = {}
        for unit in units:
            if registry_url:
                unit_records = [
                    records[name]
                    for name in (unit.name, f"{unit.name}_spikes")
                    if name in records
                ]
                unit.findings.extend(check_unit_metadata(unit, record, unit_records))
            entry_ids = {
                t.get("recording", {}).get("entry", 0) for t in unit.trials
            } or {0}
            if max(entry_ids) >= len(entries):
                raise RuntimeError(
                    f"{unit.name} has trials from entry {max(entry_ids)}, but "
                    f"{arf_path} has {len(entries)} entries"
                )
            entry = entries[min(entry_ids)]
            sampling_rate, first_sample = entry_clock(
                entry, unit.pprox.get("sync_track")
            )
            unit.findings.extend(check_recording_name(unit, recording))
            # the schema check is safe on any input, so it runs first
            unit.findings.extend(check_schema(unit))
            unit.findings.extend(check_pprox(unit, sampling_rate))
            if worst(unit.findings) == "fail":
                continue
            unit.findings.extend(check_waveforms(unit))
            if len(entry_ids) > 1:
                unit.findings.append(
                    finding(
                        "messages",
                        "info",
                        "trials from more than one entry; not checked against "
                        "the messages",
                    )
                )
                continue
            key = (entry.name, first_sample)
            if key not in stimuli_cache:
                stimuli_cache[key] = entry_stimuli(
                    entry, first_sample, sampling_rate, oeaudio_log
                )
            unit.findings.extend(
                check_messages(unit, stimuli_cache[key], sampling_rate)
            )
            unit.findings.extend(
                check_stimulus_durations(
                    unit,
                    sampling_rate,
                    pulse_sync_track(unit, entry, sampling_rate, aux_cache),
                )
            )
            if "aux_tracks" in unit.pprox or any("aux" in t for t in unit.trials):
                unit.findings.extend(
                    check_aux(unit, entry, first_sample, sampling_rate, aux_cache)
                )
    status = worst([*findings, *(f for u in units for f in u.findings)])
    return {
        "recording": recording,
        "arf": str(arf_path),
        "audited_by": f"audit-kilo-spikes {__version__}",
        "registry": registry_url,
        "status": status,
        "findings": findings,
        "units": [unit.report() for unit in units],
    }


def find_local(name: str, registry_url: str | None) -> Path:
    """The path of a neurobank resource in an archive on this host (or the local
    cache). Never downloads: the ARF files are large, and the audit is meant to
    run on the archive host. Raises FileNotFoundError."""
    try:
        return nbank.find_resource(name, registry_url=registry_url, no_download=True)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"{name}: not in a neurobank archive on this host ({err})"
        ) from err


def locate(name: str, registry_url: str) -> Path:
    """A local path, or the path of a neurobank resource on this host"""
    path = Path(name)
    if path.exists():
        return path
    return find_local(name, registry_url)


def load_units(names: list[str], registry_url: str | None) -> list[Unit]:
    """Loads the units named by pprox paths, directories of pprox files, or
    neurobank ids. The waveform file of a local pprox is <name>_spikes.h5 next
    to it; that of a neurobank resource is the resource <id>_spikes."""
    units = []
    for name in names:
        path = Path(name)
        if path.is_dir():
            found = [
                (p, p.with_name(p.stem + "_spikes.h5"))
                for p in sorted(path.glob("*.pprox"))
            ]
        elif path.exists():
            found = [(path, path.with_name(path.stem + "_spikes.h5"))]
        else:
            pprox_path = find_local(name, registry_url)
            try:
                waveforms = find_local(f"{name}_spikes", registry_url)
            except FileNotFoundError:
                waveforms = None
            found = [(pprox_path, waveforms)]
        for pprox_path, waveforms in found:
            if waveforms is not None and not waveforms.exists():
                waveforms = None
            with open(pprox_path) as fp:
                pprox = json.load(fp)
            units.append(
                Unit(
                    Path(pprox_path).stem if path.exists() else name,
                    pprox_path,
                    pprox,
                    waveforms,
                )
            )
    return units


def script(argv=None):
    p = argparse.ArgumentParser(
        prog="audit-kilo-spikes",
        description="Check the group-kilo-spikes output for one recording against "
        "its waveform files, the stimulus messages in the ARF file, and each other. "
        "Writes a JSON report; changes nothing.",
    )
    p.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    add_log_arguments(p)
    nbank.add_registry_argument(p)
    p.add_argument(
        "--oeaudio-log",
        type=Path,
        help="the open-ephys-audio log, for recordings without the stimulus "
        "messages in the ARF file (from the same session: the trials are checked "
        "against its times)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        help="write the report here (default: standard output)",
    )
    p.add_argument(
        "--units",
        required=True,
        action="extend",
        type=lambda s: [x for x in s.split(",") if x],
        help="the units to audit: pprox files, directories of pprox files, or "
        "neurobank ids (comma-separated or repeated)",
    )
    p.add_argument("recording", help="the ARF file: a path or neurobank id")
    args = p.parse_args(argv)

    setup_log(args.debug, args.debug_http)
    try:
        arf_path = locate(args.recording, args.registry_url)
        units = load_units(args.units, args.registry_url)
        if not units:
            raise RuntimeError("no units to audit")
        log.info("- auditing %d units from %s", len(units), arf_path)
        recording = Path(args.recording).stem
        report = audit_recording(
            arf_path, units, args.oeaudio_log, recording, args.registry_url
        )
    except (OSError, RuntimeError, ValueError, KeyError) as err:
        log.error("audit-kilo-spikes: %s", err)
        sys.exit(1)

    # info findings are only logged with --debug; all are in the report
    def level(f):
        return logging.DEBUG if f["severity"] == "info" else logging.INFO

    for f in report["findings"]:
        log.log(level(f), "  - %s: %s", f["severity"], f["message"])
    for unit in report["units"]:
        log.info("  - %s: %s", unit["name"], unit["status"])
        for f in unit["findings"]:
            log.log(level(f), "    - %s: %s%s", f["severity"], f["message"], _trials(f))
    log.info("- status: %s", report["status"])
    text = json.dumps(report, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n")


# --- selection

PPROX_DTYPE = "spikes-pprox"
WAVEFORMS_DTYPE = "spikes-hdf5"
_re_unit = re.compile(r"(?P<recording>.+)_c\d+")


def group_units(
    pprox_names: Iterable[str], waveform_names: Iterable[str]
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """Groups unit resources by recording, using group-kilo-spikes's names
    (<recording>_c<N>.pprox and <recording>_c<N>_spikes.h5). Returns a dict
    mapping each recording to its 'units' (pprox ids), 'orphans' (waveform ids
    without a pprox), and 'no_waveforms' (pprox ids without a waveform file),
    and a list of the names that don't fit the pattern."""
    pprox = set(pprox_names)
    waveforms = {name.removesuffix("_spikes") for name in waveform_names}
    unmatched = sorted(
        {n for n in pprox if not _re_unit.fullmatch(n)}
        | {f"{n}_spikes" for n in waveforms if not _re_unit.fullmatch(n)}
    )
    groups: dict[str, dict[str, list[str]]] = {}
    for name in sorted(pprox | waveforms):
        m = _re_unit.fullmatch(name)
        if m is None:
            continue
        group = groups.setdefault(
            m["recording"], {"units": [], "orphans": [], "no_waveforms": []}
        )
        if name not in pprox:
            group["orphans"].append(f"{name}_spikes")
            continue
        group["units"].append(name)
        if name not in waveforms:
            group["no_waveforms"].append(name)
    return groups, unmatched


def fetch_locations(registry_url: str, names: Iterable[str]) -> dict[str, list]:
    """The locations of resources, by name, from the registry's bulk locations
    endpoint (the resource records only give them as strings)"""
    from httpx import Client
    from nbank import registry, util

    names = list(names)
    if not names:
        return {}
    url, query = registry.get_locations_bulk(registry_url, names)
    with Client(auth=nbank.default_auth) as client:
        return {
            r["name"]: r.get("locations", [])
            for r in util.query_registry_bulk(client, url, query)
        }


def unavailable_reason(locations: list) -> str:
    """Why a resource isn't in a neurobank archive on this host, from its
    locations (see fetch_locations)"""
    from nbank.registry import local_schemes

    schemes = {loc.get("scheme") for loc in locations if isinstance(loc, dict)}
    if not schemes:
        return "no locations in the registry"
    if any(_unreadable(loc) for loc in locations):
        return "in an archive here that this user can't read (check its permissions)"
    if schemes & set(local_schemes()):
        return "in an archive that isn't on this host"
    return "only on " + ", ".join(sorted(map(str, schemes)))


def _unreadable(location) -> bool:
    """True if location is in a neurobank archive on this host whose
    directories this user can't read"""
    from nbank.registry import local_schemes
    from nbank.util import parse_location

    if not isinstance(location, dict) or location.get("scheme") not in local_schemes():
        return False
    try:
        parse_location(location)
    except PermissionError:
        return True
    except (KeyError, ValueError):
        pass
    return False


def local_copy(locations: list) -> Path | None:
    """The path of a resource in a neurobank archive on this host, given its
    locations (see fetch_locations), or None. Copies elsewhere (other hosts,
    http, tape) are not used: the audit scripts never download. A copy in an
    archive here that this user can't read doesn't count."""
    from nbank.registry import local_schemes
    from nbank.util import parse_location

    for location in locations:
        if not isinstance(location, dict):
            continue
        if location.get("scheme") not in local_schemes():
            continue
        try:
            resource = parse_location(location)
        except (KeyError, ValueError, PermissionError):
            # an unreadable archive is reported by unavailable_reason
            continue
        if resource is not None:
            return resource.path
    return None


def already_audited(report: Path, units: list[str]) -> bool:
    """True if report is an audit of exactly these units"""
    try:
        done = {u["name"] for u in json.loads(report.read_text())["units"]}
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return done == set(units)


def write_control(path: Path | None, lines: dict[str, list[str]]) -> None:
    text = "".join(f"{rec}\t{','.join(lines[rec])}\n" for rec in sorted(lines))
    if path is None:
        sys.stdout.write(text)
    else:
        path.write_text(text)


def read_recordings(fp) -> list[str]:
    """Recording ids from a file, one per line (the first word of each line,
    as `nbank search` prints them); blank lines and comments are skipped."""
    out = []
    for line in fp:
        words = line.split("#", 1)[0].split()
        if words:
            out.append(words[0])
    return list(dict.fromkeys(out))


def find_units_script(argv=None):
    p = argparse.ArgumentParser(
        prog="find-kilo-units",
        description="Find the group-kilo-spikes units of recordings in the "
        "registry and write a control file for audit-kilo-spikes, one line per "
        "recording: the recording id, a tab, and its units, comma-separated.",
    )
    p.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    add_log_arguments(p)
    nbank.add_registry_argument(p)
    which = p.add_mutually_exclusive_group()
    which.add_argument(
        "--name",
        help="only resources whose names contain this (e.g. a bird or recording)",
    )
    which.add_argument(
        "recordings",
        nargs="?",
        type=argparse.FileType("r"),
        help="only the recordings listed in this file, one id per line ('-' for "
        "standard input, e.g. piped from nbank search)",
    )
    which.add_argument(
        "--all",
        action="store_true",
        help="every unit in the registry (slow: fetches every pprox and waveform "
        "record)",
    )
    p.add_argument(
        "--reports",
        type=Path,
        help="skip recordings with a report (<recording>.json) in this directory "
        "that covers the same units",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        help="write the control file here (default: standard output)",
    )
    p.add_argument(
        "--orphans",
        type=Path,
        help="write a control file of waveform files without a pprox here, in the "
        "same format (for regenerating the pprox files)",
    )
    p.add_argument(
        "--unavailable",
        type=Path,
        help="write a control file of the recordings skipped because their ARF "
        "files aren't in a neurobank archive on this host (e.g. they are in cold "
        "storage) here, in the same format",
    )
    p.add_argument("--pprox-dtype", default=PPROX_DTYPE, help="default: %(default)s")
    p.add_argument(
        "--waveforms-dtype", default=WAVEFORMS_DTYPE, help="default: %(default)s"
    )
    args = p.parse_args(argv)
    if args.name is None and args.recordings is None and not args.all:
        p.error("give recordings (a file, or '-'), --name, or --all")

    setup_log(args.debug, args.debug_http)
    if args.recordings is not None:
        recordings = read_recordings(args.recordings)
        log.info("- finding units for %d recordings", len(recordings))
        # one search per recording; a name search matches fragments, so the
        # results include other recordings (e.g. P397_1_10 for P397_1_1)
        queries = [{"name": rec} for rec in recordings]
    else:
        recordings = None
        queries = [{"name": args.name} if args.name else {}]

    def names(dtype: str) -> list[str]:
        found = {
            r["name"]
            for query in queries
            for r in nbank_core.search(args.registry_url, dtype=dtype, **query)
        }
        log.info("- %d %s resources", len(found), dtype)
        return sorted(found)

    try:
        groups, unmatched = group_units(
            names(args.pprox_dtype), names(args.waveforms_dtype)
        )
        no_units = []
        if recordings is not None:
            groups = {rec: groups[rec] for rec in recordings if rec in groups}
            unmatched = []
            # e.g. deposited before sorting showed there were no good units
            no_units = [rec for rec in recordings if rec not in groups]
        records = {
            r["name"]: r for r in nbank_core.describe_many(args.registry_url, *groups)
        }
        locations = fetch_locations(args.registry_url, [*records, *no_units])
    except (OSError, httpx.HTTPError) as err:
        log.error("find-kilo-units: %s", err)
        sys.exit(1)

    # recordings without units are harmless; only those still in the archive here
    # (taking up space) are worth mentioning
    no_units_here = [r for r in no_units if local_copy(locations.get(r, []))]
    if no_units_here:
        log.info(
            "- %d of the recordings have no units, but their ARF files are in the "
            "archive here%s",
            len(no_units_here),
            "" if args.debug else " (listed with --debug)",
        )
    for rec in no_units:
        where = (
            "ARF file in the archive here"
            if rec in no_units_here
            else unavailable_reason(locations.get(rec, []))
        )
        log.debug("  - %s: no units (%s)", rec, where)
    if unmatched:
        log.info(
            "- %d resources skipped: names don't match <recording>_c<N>%s",
            len(unmatched),
            "" if args.debug else " (listed with --debug)",
        )
    for name in unmatched:
        log.debug("  - %s: name doesn't match <recording>_c<N>", name)
    unregistered = sorted(set(groups) - set(records))
    if unregistered:
        log.info(
            "- %d recordings skipped: not in the registry%s",
            len(unregistered),
            "" if args.debug else " (listed with --debug)",
        )
    for rec in unregistered:
        log.debug("  - %s: recording not in the registry", rec)
        del groups[rec]
    to_audit, orphans, unavailable, skipped = {}, {}, {}, 0
    reasons: dict[str, int] = {}
    for rec, group in groups.items():
        for name in group["no_waveforms"]:
            log.debug("  - %s: no waveform file", name)
        # orphans don't need the ARF file (their trials come from other units)
        if group["orphans"]:
            orphans[rec] = group["orphans"]
        if not group["units"]:
            continue
        if local_copy(locations.get(rec, [])) is None:
            reason = unavailable_reason(locations.get(rec, []))
            log.debug("  - %s: ARF file not on this host (%s)", rec, reason)
            reasons[reason] = reasons.get(reason, 0) + 1
            unavailable[rec] = group["units"]
            continue
        if args.reports and already_audited(
            args.reports / f"{rec}.json", group["units"]
        ):
            skipped += 1
            continue
        to_audit[rec] = group["units"]
    log.info(
        "- %d recordings to audit (%d units); %d already audited",
        len(to_audit),
        sum(map(len, to_audit.values())),
        skipped,
    )
    if unavailable:
        log.info(
            "- %d recordings (%d units) skipped: their ARF files aren't in a "
            "neurobank archive on this host%s",
            len(unavailable),
            sum(map(len, unavailable.values())),
            "" if args.unavailable else " (list them with --unavailable)",
        )
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            log.info("  - %d %s", n, reason)
        if not to_audit and not skipped:
            log.warning(
                "- none of the recordings is in a neurobank archive on this host; "
                "run find-kilo-units (and the audits) on the archive host"
            )
    log.info(
        "- %d waveform files without a pprox, in %d recordings",
        sum(map(len, orphans.values())),
        len(orphans),
    )
    write_control(args.output, to_audit)
    if args.orphans is not None:
        write_control(args.orphans, orphans)
    if args.unavailable is not None:
        write_control(args.unavailable, unavailable)


# --- collection


def load_reports(paths: Iterable[Path]) -> tuple[list[dict], list[str]]:
    """Loads audit reports from files and directories of them (*.json).
    Returns the reports, sorted by recording, and a description of each file
    that couldn't be read."""
    reports, errors = [], []
    for path in paths:
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for file in files:
            try:
                report = json.loads(file.read_text())
                if not {"recording", "status", "units"} <= report.keys():
                    raise ValueError("not an audit-kilo-spikes report")
            except (OSError, ValueError, AttributeError) as err:
                errors.append(f"{file}: {err}")
                continue
            reports.append(report)
    return sorted(reports, key=lambda r: r["recording"]), errors


def unit_version(unit: dict) -> str:
    return unit["processed_by"][0] if unit.get("processed_by") else "unknown"


def report_findings(reports: list[dict]) -> Iterable[dict]:
    """Yields each finding in the reports as a flat row (unit is None for
    findings about the recording as a whole)."""
    for report in reports:
        for f in report["findings"]:
            yield {"recording": report["recording"], "unit": None, "version": None, **f}
        for unit in report["units"]:
            for f in unit["findings"]:
                yield {
                    "recording": report["recording"],
                    "unit": unit["name"],
                    "version": unit_version(unit),
                    **f,
                }


def summarize(reports: list[dict], level: str = "warn") -> str:
    """A text summary of audit reports: recordings and units by status,
    findings by check, units by version, and the recordings with findings at or
    above level."""
    lines = []
    units = [u for r in reports for u in r["units"]]
    lines.append(f"{len(reports)} recordings, {len(units)} units")
    lines.append("")
    lines.append(f"{'status':<10}{'recordings':>12}{'units':>8}")
    for sev in SEVERITIES:
        n_rec = sum(r["status"] == sev for r in reports)
        n_unit = sum(u["status"] == sev for u in units)
        lines.append(f"{sev:<10}{n_rec:>12}{n_unit:>8}")

    rows = list(report_findings(reports))
    checks: dict[tuple[str, str], tuple[set, set]] = {}
    for row in rows:
        recs, unit_names = checks.setdefault(
            (row["check"], row["severity"]), (set(), set())
        )
        recs.add(row["recording"])
        if row["unit"] is not None:
            unit_names.add((row["recording"], row["unit"]))
    if checks:
        lines.append("")
        lines.append(f"{'check':<30}{'severity':<10}{'recordings':>12}{'units':>8}")
        for (check, sev), (recs, unit_names) in sorted(
            checks.items(), key=lambda kv: (-SEVERITIES.index(kv[0][1]), kv[0][0])
        ):
            lines.append(f"{check:<30}{sev:<10}{len(recs):>12}{len(unit_names):>8}")

    versions: dict[str, dict[str, int]] = {}
    for unit in units:
        counts = versions.setdefault(unit_version(unit), dict.fromkeys(SEVERITIES, 0))
        counts[unit["status"]] += 1
    if versions:
        lines.append("")
        lines.append(f"{'version':<40}" + "".join(f"{s:>7}" for s in SEVERITIES))
        for version, counts in sorted(versions.items()):
            lines.append(
                f"{version:<40}" + "".join(f"{counts[s]:>7}" for s in SEVERITIES)
            )

    # worst first, then by recording
    flagged = sorted(
        (
            r
            for r in reports
            if SEVERITIES.index(r["status"]) >= SEVERITIES.index(level)
        ),
        key=lambda r: (-SEVERITIES.index(r["status"]), r["recording"]),
    )
    if flagged:
        lines.append("")
        lines.append(
            f"recordings with {' or '.join(SEVERITIES[SEVERITIES.index(level) :])}:"
        )
        for report in flagged:
            # each check at or above level, with the number of units it was
            # found in (none for a finding about the recording as a whole)
            found: dict[str, int] = {}
            for row in report_findings([report]):
                if SEVERITIES.index(row["severity"]) >= SEVERITIES.index(level):
                    key = f"{row['check']} ({row['severity']})"
                    found[key] = found.get(key, 0) + (row["unit"] is not None)
            described = ", ".join(
                f"{check} x{n}" if n else check for check, n in sorted(found.items())
            )
            lines.append(f"  {report['recording']:<20}{report['status']:<6}{described}")
    return "\n".join(lines) + "\n"


def write_findings(path: Path, reports: list[dict]) -> None:
    """Writes every finding as a tab-separated row: recording, unit (empty for
    the recording as a whole), version, check, severity, trials
    (comma-separated), message."""
    columns = ("recording", "unit", "version", "check", "severity", "trials", "message")
    with open(path, "w") as fp:
        fp.write("\t".join(columns) + "\n")
        for row in report_findings(reports):
            row["trials"] = ",".join(map(str, row.get("trials", [])))
            fp.write(
                "\t".join("" if row.get(c) is None else str(row[c]) for c in columns)
                + "\n"
            )


def collect_script(argv=None):
    p = argparse.ArgumentParser(
        prog="collect-kilo-audit",
        description="Summarize the reports written by audit-kilo-spikes.",
    )
    p.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    add_log_arguments(p)
    p.add_argument(
        "--level",
        choices=SEVERITIES[1:],
        default="warn",
        help="list the recordings with findings at or above this level "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--tsv",
        type=Path,
        help="write every finding to this file, one tab-separated row each",
    )
    p.add_argument(
        "--control",
        type=Path,
        help="the control file the audits were run from; recordings in it without "
        "a report (the audit couldn't run) are listed",
    )
    p.add_argument(
        "reports", type=Path, nargs="+", help="report files or directories of them"
    )
    args = p.parse_args(argv)
    setup_log(args.debug, args.debug_http)

    reports, errors = load_reports(args.reports)
    for error in errors:
        log.warning("- unable to read %s", error)
    if args.control is not None:
        with open(args.control) as fp:
            expected = read_recordings(fp)
        done = {r["recording"] for r in reports}
        missing = [rec for rec in expected if rec not in done]
        if missing:
            log.warning(
                "- %d of %d recordings in %s have no report: %s",
                len(missing),
                len(expected),
                args.control,
                ", ".join(missing),
            )
    sys.stdout.write(summarize(reports, args.level))
    if args.tsv is not None:
        write_findings(args.tsv, reports)


def _trials(f: dict, n: int = 10) -> str:
    if "trials" not in f:
        return ""
    trials = f["trials"]
    more = f" and {len(trials) - n} more" if len(trials) > n else ""
    return f" (trials {', '.join(map(str, trials[:n]))}{more})"


if __name__ == "__main__":
    script()
