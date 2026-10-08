# -*- mode: python -*-
"""Regenerate missing pprox files from their waveform files.

group-kilo-spikes writes two files per unit: a pprox with the spike times
split into trials, and a waveform file (_spikes.h5) with every spike's time
and waveform. A unit's pprox can be rebuilt from its waveform file and a
trial table (see kilo.waveforms_to_events). regenerate-pprox does this for
units whose pprox went missing or was never deposited (find-kilo-units
--orphans lists them).

The trial table comes from one of these, in order of preference:

- --trials: a pprox from the same recording and run, given explicitly.
- Another unit's pprox from the same recording (local, next to the waveform
  files, or in the registry). If they don't all have the same trials, only
  those processed by the same version as the waveform file are used. The
  table is only used if it reproduces that unit's own events from its
  waveform file. This reproduces the original pprox exactly.
- --from-arf: the trials made from the ARF file by the current version of
  group-kilo-spikes. This needs the sync track and prepad, from the options or
  the waveform file. The onsets may differ by a few samples from those of the
  original version (more for pulse sync before 2026.10.07), and aux pulses are
  not included.

The regenerated files are written to a directory for depositing by hand. Each
records the waveform file it was derived from (derived_from), the source of its
trials (trials_from), and both the original version and this script in
processed_by.

"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py as h5
import httpx
import numpy as np
import pandas as pd

from dlab import __version__, kilo, pprox
from dlab import neurobank as nbank
from dlab.kilo_audit import PPROX_DTYPE, _re_unit, nbank_core, trials_key
from dlab.util import json_serializable, setup_log

log = logging.getLogger("dlab")

PROG = "regenerate-pprox"
# top-level fields that describe the unit, not the recording or its trials
UNIT_FIELDS = ("processed_by", "derived_from", "trials_from")


@dataclass
class Waveforms:
    """The spike times and attributes of a unit's waveform file"""

    unit: str  # the unit's name (the pprox id it should have)
    source: str  # the resource it came from (neurobank URL or path)
    times: np.ndarray
    sampling_rate: float
    attrs: dict

    @classmethod
    def load(cls, path: Path, source: str) -> "Waveforms":
        with h5.File(path, "r") as fp:
            times = fp["times"][:]
            rate = float(fp["times"].attrs["sampling_rate"])
            attrs = {k: _plain(v) for k, v in fp.attrs.items()}
        unit = Path(path).stem.removesuffix("_spikes")
        return cls(unit, source, times, rate, attrs)

    @property
    def processed_by(self) -> str | None:
        return self.attrs.get("processed_by")


def _plain(value):
    """An hdf5 attribute as a plain python value"""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class TrialSource:
    """A trial table, and the pprox (or ARF file) it came from"""

    name: str
    doc: dict  # the source pprox (top-level fields and trials)

    @property
    def trials(self) -> list[dict]:
        return self.doc["pprox"]

    @property
    def processed_by(self) -> str | None:
        value = self.doc.get("processed_by") or [None]
        return value if isinstance(value, str) else value[0]


def rebuild(waveforms: Waveforms, trials: list[dict], quiet=False) -> list[dict]:
    """The trials with their events rebuilt from the waveform file"""
    events, n_before = kilo.waveforms_to_events(
        waveforms.times, trials, waveforms.sampling_rate
    )
    if n_before and not quiet:
        log.info("    - %d spikes before the first trial (not in the pprox)", n_before)
    return [{**t, "events": ev} for t, ev in zip(trials, events, strict=True)]


def reproduces(source: TrialSource, waveforms: Waveforms) -> bool:
    """True if rebuilding the source unit's events from its own waveform file
    gives the events in its pprox"""
    tolerance = 1.5 / waveforms.sampling_rate
    rebuilt = rebuild(waveforms, source.trials, quiet=True)
    return all(
        len(t["events"]) == len(r["events"])
        and np.allclose(r["events"], t["events"], rtol=0, atol=tolerance)
        for t, r in zip(source.trials, rebuilt, strict=True)
    )


# --- finding the trials


def local_siblings(recording: str, directories: set[Path]) -> list[tuple[Path, Path]]:
    """The pprox files of a recording's units in the given directories, with
    the paths their waveform files would have"""
    found = []
    for directory in sorted(directories):
        for path in sorted(directory.glob(f"{recording}_c*.pprox")):
            m = _re_unit.fullmatch(path.stem)
            if m and m["recording"] == recording:
                found.append((path, path.with_name(path.stem + "_spikes.h5")))
    return found


def registry_siblings(
    recording: str, registry_url: str
) -> list[tuple[Path, Path | None]]:
    """The pprox resources of a recording's units in the registry, with their
    waveform files (if registered)"""
    names = sorted(
        r["name"]
        for r in nbank_core.search(registry_url, dtype=PPROX_DTYPE, name=recording)
        if (m := _re_unit.fullmatch(r["name"])) and m["recording"] == recording
    )
    found = []
    for name in names:
        path = nbank.find_resource(name, registry_url=registry_url)
        try:
            waveforms = nbank.find_resource(f"{name}_spikes", registry_url=registry_url)
        except FileNotFoundError:
            waveforms = None
        found.append((path, waveforms))
    return found


def choose_trials(
    waveforms: Waveforms, siblings: list[tuple[Path, Path | None]]
) -> TrialSource:
    """Chooses the trial table for a unit from the pprox files of other units
    from the same recording. Siblings processed by another version than the
    waveform file are used only if every sibling has the same trials. The
    chosen table must reproduce its own unit's events, if that unit has a
    waveform file. Raises RuntimeError if there is no suitable table."""
    candidates = []
    for path, waveforms_path in siblings:
        if Path(path).stem == waveforms.unit:
            continue
        with open(path) as fp:
            source = TrialSource(Path(path).stem, json.load(fp))
        candidates.append((source, waveforms_path))
    if not candidates:
        raise RuntimeError(
            "no other pprox from this recording to take the trials from "
            "(use --trials or --from-arf)"
        )
    tables = {trials_key(s.trials) for s, _ in candidates}
    if len(tables) > 1:
        candidates = [
            (s, w) for s, w in candidates if s.processed_by == waveforms.processed_by
        ]
        tables = {trials_key(s.trials) for s, _ in candidates}
        if len(tables) != 1:
            raise RuntimeError(
                "the other units of this recording have different trials, and "
                f"{'several' if tables else 'none'} of them were processed by "
                f"{waveforms.processed_by}; choose one with --trials"
            )
    for source, waveforms_path in candidates:
        if waveforms_path is None or not Path(waveforms_path).exists():
            continue
        if reproduces(source, Waveforms.load(waveforms_path, str(waveforms_path))):
            log.info(
                "    - trials from %s (checked against its waveform file)", source.name
            )
            return source
        raise RuntimeError(
            f"the trials of {source.name} don't reproduce its own events from its "
            "waveform file; choose another with --trials"
        )
    source = candidates[0][0]
    log.warning(
        "    - trials from %s (no waveform file to check them against)", source.name
    )
    return source


def arf_trials(args, waveforms: list[Waveforms]) -> TrialSource:
    """The trial table made from the ARF file by the current version"""

    def option(name, flag, value):
        if value is not None:
            return value
        values = {w.attrs.get(name) for w in waveforms}
        if len(values) != 1 or None in values:
            raise RuntimeError(
                f"{flag} is needed (not recorded in all the waveform files)"
            )
        return values.pop()

    sync = option("sync_track", "--sync", args.sync)
    prepad = option("prepad", "--prepad", args.prepad)
    sync_thresh = args.sync_thresh
    if sync_thresh is None:
        sync_thresh = {w.attrs.get("sync_thresh") for w in waveforms}.pop()
    arf_path = Path(args.recording)
    if not arf_path.exists():
        arf_path = nbank.find_resource(args.recording, registry_url=args.registry_url)
    finder = kilo.StimulusFinder(args.registry_url, args.local_stim_dir)
    log.info(
        "- splitting '%s' into trials (sync %s, prepad %.2f s):", arf_path, sync, prepad
    )
    with h5.File(arf_path, "r") as afp:
        trials = kilo.arf_to_trials(
            afp, finder, sync, sync_thresh, prepad, oeaudio_log=args.oeaudio_log
        )
        entry_attrs = tuple(
            kilo.entry_to_metadata(e) for _, e in kilo.iter_entries(afp)
        )
    rate = waveforms[0].sampling_rate
    table = list(kilo.trials_to_pprox(pd.DataFrame(trials).assign(events=np.nan), rate))
    recording = Path(args.recording).stem
    try:
        metadata = nbank.describe(args.registry_url, recording)["metadata"]
        url = nbank.registry.full_url(args.registry_url, recording)
    except (TypeError, KeyError, httpx.HTTPError) as err:
        # describe() returns None for an unregistered recording
        log.warning("  - recording metadata not found in the registry (%s)", err)
        metadata, url = {}, waveforms[0].attrs.get("recording")
    trial_options = {"sync_track": sync, "prepad": prepad}
    if sync_thresh is not None:
        trial_options["sync_thresh"] = sync_thresh
    if args.oeaudio_log is not None:
        trial_options["oeaudio_log"] = args.oeaudio_log.name
    doc = pprox.from_trials(
        table,
        schema=pprox._stimtrial_schema,
        recording=url,
        entry_metadata=entry_attrs,
        **trial_options,
        **metadata,
    )
    return TrialSource(f"{arf_path.name} ({PROG} {__version__})", doc)


# --- output


def regenerate(waveforms: Waveforms, source: TrialSource) -> dict:
    """The pprox for a unit: the source's top-level fields and trials, with the
    events from the waveform file and the unit's kilosort fields"""
    doc = {
        k: v
        for k, v in source.doc.items()
        if k != "pprox" and k not in UNIT_FIELDS and not k.startswith("kilosort_")
    }
    doc["pprox"] = rebuild(waveforms, source.trials)
    doc.update({k: v for k, v in waveforms.attrs.items() if k.startswith("kilosort_")})
    doc["processed_by"] = [
        p for p in (waveforms.processed_by, f"{PROG} {__version__}") if p
    ]
    doc["derived_from"] = waveforms.source
    doc["trials_from"] = source.name
    recording = waveforms.attrs.get("recording")
    if recording is not None and recording != doc.get("recording"):
        log.warning(
            "    - the waveform file names recording %s; keeping %s from the trials",
            recording,
            doc.get("recording"),
        )
    return doc


def load_waveforms(names: list[str], registry_url: str | None) -> list[Waveforms]:
    """Loads the waveform files named by paths, directories (all *_spikes.h5),
    or neurobank ids"""
    out = []
    for name in names:
        path = Path(name)
        if path.is_dir():
            out.extend(
                Waveforms.load(p, str(p)) for p in sorted(path.glob("*_spikes.h5"))
            )
        elif path.exists():
            out.append(Waveforms.load(path, str(path)))
        else:
            found = nbank.find_resource(name, registry_url=registry_url)
            source = (
                nbank.registry.full_url(registry_url, name) if registry_url else name
            )
            out.append(Waveforms.load(found, source))
    return out


def script(argv=None):
    p = argparse.ArgumentParser(
        prog=PROG,
        description="Regenerate the pprox files of units from their waveform files "
        "(_spikes.h5) and a trial table. Writes the files to a directory; deposit "
        "them by hand.",
    )
    p.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    nbank.add_registry_argument(p)
    p.add_argument(
        "--units",
        required=True,
        action="extend",
        type=lambda s: [x for x in s.split(",") if x],
        help="the waveform files: paths, directories, or neurobank ids "
        "(comma-separated or repeated)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="directory for the regenerated pprox files",
    )
    which = p.add_mutually_exclusive_group()
    which.add_argument(
        "--trials",
        help="take the trials from this pprox (path or neurobank id), from the "
        "same recording and run",
    )
    which.add_argument(
        "--from-arf",
        action="store_true",
        help="make the trials from the ARF file with the current version of "
        "group-kilo-spikes (when no pprox from the same run exists)",
    )
    arf = p.add_argument_group("options for --from-arf")
    arf.add_argument("--sync", help="sync track (default: from the waveform files)")
    arf.add_argument(
        "--prepad", type=float, help="prepad, in s (default: from the waveform files)"
    )
    arf.add_argument("--sync-thresh", type=float, help="absolute sync threshold")
    arf.add_argument("--oeaudio-log", type=Path, help="open-ephys-audio log")
    arf.add_argument(
        "--local-stim-dir", type=Path, help="fallback directory for stimulus files"
    )
    p.add_argument("recording", help="the recording: neurobank id or ARF file")
    args = p.parse_args(argv)
    setup_log(args.debug)

    recording = Path(args.recording).stem
    try:
        units = load_waveforms(args.units, args.registry_url)
        if not units:
            raise RuntimeError("no waveform files")
        args.output.mkdir(parents=True, exist_ok=True)
        if args.from_arf:
            shared = arf_trials(args, units)
        elif args.trials is not None:
            path = (
                Path(args.trials)
                if Path(args.trials).exists()
                else nbank.find_resource(args.trials, registry_url=args.registry_url)
            )
            with open(path) as fp:
                shared = TrialSource(Path(args.trials).stem, json.load(fp))
        else:
            shared = None
            local = {Path(u.source).parent for u in units if Path(u.source).exists()}
            siblings = (
                local_siblings(recording, local)
                if local
                else registry_siblings(recording, args.registry_url)
            )
    except (OSError, RuntimeError, ValueError, KeyError) as err:
        log.error("%s: %s", PROG, err)
        sys.exit(1)

    failed = 0
    for waveforms in units:
        log.info("- %s:", waveforms.unit)
        outfile = args.output / f"{waveforms.unit}.pprox"
        try:
            m = _re_unit.fullmatch(waveforms.unit)
            if m is None or m["recording"] != recording:
                raise RuntimeError(f"not a unit of {recording}")
            if outfile.exists():
                raise RuntimeError(f"{outfile} exists")
            source = shared or choose_trials(waveforms, siblings)
            doc = regenerate(waveforms, source)
        except (OSError, RuntimeError, ValueError, KeyError) as err:
            log.error("    - unable to regenerate: %s", err)
            failed += 1
            continue
        with open(outfile, "w") as fp:
            json.dump(doc, fp, default=json_serializable)
        log.info(
            "    - %d spikes in %d trials -> %s",
            sum(len(t["events"]) for t in doc["pprox"]),
            len(doc["pprox"]),
            outfile,
        )
    log.info("- regenerated %d of %d units", len(units) - failed, len(units))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    script()
