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

find-kilo-units finds the units of recordings in the registry and writes a
control file for batch runs, one line per recording: the recording id, a tab,
and its units, comma-separated. For example:

    find-kilo-units --name P397 --reports reports -o audit.tsv
    parallel --colsep '\t' -a audit.tsv \
        'audit-kilo-spikes {1} --units {2} -o reports/{1}.json'

The recordings can also be listed in a file, or piped from a custom search:

    nbank search -d <arf dtype> -k <key>=<value> | find-kilo-units - -o audit.tsv

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
import numpy as np
from nbank import core as nbank_core

from dlab import __version__, kilo
from dlab import neurobank as nbank
from dlab.util import setup_log

log = logging.getLogger("dlab")

SEVERITIES = ("ok", "info", "warn", "fail")
# the first version with the current sync detection; lag outliers in output
# from earlier versions are probably the known detection errors
SYNC_FIX_VERSION = "2026.10.07"
# lag outliers in more than this fraction of trials make a unit unreliable
MAX_OUTLIER_FRACTION = 0.5


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


def check_pprox(unit: Unit, sampling_rate: float) -> list[dict]:
    """Checks the structure of a unit's pprox: the fields stimtrial requires,
    trial order, events within their trials, and consistency between the
    trial times and the recording sample ranges."""
    out = []
    trials = unit.trials
    missing = [
        i
        for i, t in enumerate(trials)
        if not {"events", "offset", "interval", "stimulus"} <= t.keys()
        or not {"name", "interval"} <= t["stimulus"].keys()
    ]
    if missing:
        out.append(
            finding(
                "pprox-fields",
                "fail",
                "trials without the events, offset, interval or stimulus that "
                "stimtrial requires",
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


def check_messages(unit: Unit, stimuli, sampling_rate: float) -> list[dict]:
    """Checks a unit's trials against the stimulus start messages: each trial's
    onset (its sync event) should follow a start message for its stimulus by a
    lag consistent with the other trials."""
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
    starts = np.array([s.start for s in stimuli])
    onsets = np.array([round(t["offset"] * sampling_rate) for t in trials])
    idx = np.searchsorted(starts, onsets, side="right") - 1
    out = []
    early = np.flatnonzero(idx < 0)
    if early.size:
        out.append(
            finding(
                "messages-before",
                "fail",
                "trials that start before any stimulus message",
                early,
            )
        )
        return out
    mislabeled = [
        i
        for i, (t, k) in enumerate(zip(trials, idx, strict=True))
        if t["stimulus"]["name"] != stimuli[k].name
    ]
    if mislabeled:
        out.append(
            finding(
                "stimulus-labels",
                "fail",
                "trials labeled with a different stimulus from the message "
                "before their onset",
                mislabeled,
            )
        )
    shared = np.flatnonzero(np.diff(idx) == 0) + 1
    if shared.size:
        out.append(
            finding(
                "messages-shared",
                "fail",
                "trials that follow the same stimulus message as the previous trial",
                shared,
            )
        )
    outliers = kilo.sync_lag_outliers(starts[idx], onsets, sampling_rate)
    if outliers.size:
        lags = (onsets - starts[idx]) / sampling_rate
        message = (
            f"trials whose onset lag differs from the median ({np.median(lags):.3f} s) "
            "by more than 0.1 s"
        )
        if old_sync_version(unit.processed_by):
            message += (
                f"; expected in versions before {SYNC_FIX_VERSION} if the sync "
                "track was pulses"
            )
        severity = (
            "fail" if outliers.size > MAX_OUTLIER_FRACTION * len(trials) else "warn"
        )
        out.append(finding("sync-lag", severity, message, outliers))
    n_dropped = len(stimuli) - np.unique(idx).size
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


# --- across units


def trial_table(unit: Unit) -> tuple:
    return tuple(
        (
            round(t.get("offset", np.nan), 6),
            t.get("stimulus", {}).get("name"),
            *np.round(t.get("interval", (np.nan, np.nan)), 6),
        )
        for t in unit.trials
    )


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
    versions = {tuple(unit.processed_by) for unit in units}
    if len(versions) > 1:
        out.append(
            finding(
                "versions",
                "info",
                "units processed by different versions: "
                + "; ".join(" + ".join(v) for v in sorted(versions)),
            )
        )
    return out


# --- driver


def audit_recording(
    arf_path: Path,
    units: list[Unit],
    oeaudio_log: Path | None = None,
    recording: str | None = None,
) -> dict:
    """Audits the units sorted from a recording. recording is its neurobank id
    (by default, the name of the ARF file without its extension). Returns the
    report."""
    recording = recording or Path(arf_path).stem
    findings = check_units(units)
    with h5.File(arf_path, "r") as afp:
        entries = [entry for _, entry in kilo.iter_entries(afp)]
        stimuli_cache = {}
        for unit in units:
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
    status = worst([*findings, *(f for u in units for f in u.findings)])
    return {
        "recording": recording,
        "arf": str(arf_path),
        "audited_by": f"audit-kilo-spikes {__version__}",
        "status": status,
        "findings": findings,
        "units": [unit.report() for unit in units],
    }


def locate(name: str, registry_url: str) -> Path:
    """A local path, or the path of a neurobank resource"""
    path = Path(name)
    if path.exists():
        return path
    return nbank.find_resource(name, registry_url=registry_url)


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
            pprox_path = nbank.find_resource(name, registry_url=registry_url)
            try:
                waveforms = nbank.find_resource(
                    f"{name}_spikes", registry_url=registry_url
                )
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
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
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

    setup_log(args.debug)
    try:
        arf_path = locate(args.recording, args.registry_url)
        units = load_units(args.units, args.registry_url)
        if not units:
            raise RuntimeError("no units to audit")
        log.info("- auditing %d units from %s", len(units), arf_path)
        recording = Path(args.recording).stem
        report = audit_recording(arf_path, units, args.oeaudio_log, recording)
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
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
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
    p.add_argument("--pprox-dtype", default=PPROX_DTYPE, help="default: %(default)s")
    p.add_argument(
        "--waveforms-dtype", default=WAVEFORMS_DTYPE, help="default: %(default)s"
    )
    args = p.parse_args(argv)

    setup_log(args.debug)
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
        if recordings is not None:
            groups = {rec: groups[rec] for rec in recordings if rec in groups}
            unmatched = []
            for rec in recordings:
                if rec not in groups:
                    log.info("  - %s: no units found", rec)
        registered = {
            r["name"] for r in nbank_core.describe_many(args.registry_url, *groups)
        }
    except OSError as err:
        log.error("find-kilo-units: %s", err)
        sys.exit(1)

    for name in unmatched:
        log.info("  - %s: name doesn't match <recording>_c<N>; skipped", name)
    for rec in sorted(set(groups) - registered):
        log.info("  - %s: recording not in the registry; skipped", rec)
        del groups[rec]
    to_audit, orphans, skipped = {}, {}, 0
    for rec, group in groups.items():
        for name in group["no_waveforms"]:
            log.debug("  - %s: no waveform file", name)
        if group["orphans"]:
            orphans[rec] = group["orphans"]
        if not group["units"]:
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
    log.info(
        "- %d waveform files without a pprox, in %d recordings",
        sum(map(len, orphans.values())),
        len(orphans),
    )
    write_control(args.output, to_audit)
    if args.orphans is not None:
        write_control(args.orphans, orphans)


def _trials(f: dict, n: int = 10) -> str:
    if "trials" not in f:
        return ""
    trials = f["trials"]
    more = f" and {len(trials) - n} more" if len(trials) > n else ""
    return f" (trials {', '.join(map(str, trials[:n]))}{more})"


if __name__ == "__main__":
    script()
