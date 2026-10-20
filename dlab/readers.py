# -*- mode: python -*-
"""Metadata readers for neurobank deposit

Each reader is registered as an entry point in the `nbank.metadata_readers`
group, under the name of the dtype it reads. It takes the path of a file,
checks that the file is of that dtype (raising ValueError with a message for
the user if it isn't), and returns the metadata for the registry as
JSON-serializable values. Readers don't depend on neurobank, don't change the
file, and don't use the network.

"""

import json
import math
from pathlib import Path
from typing import Any

import h5py as h5
import numpy as np

from dlab import pprox

# names used by older producers, and the agreed names
LEGACY_NAMES = {"bird": "animal", "experiment": "protocol"}
# the metadata registered for a unit's files (the fields that would be searched
# on); everything else stays in the files
REGISTERED_KEYS = (
    "animal",
    "experimenter",
    "recording",
    "protocol",
    "processed_by",
    "sorted_by",
    "presenter",
)


def rename_legacy(metadata: dict[str, Any]) -> dict[str, Any]:
    """Renames legacy keys in metadata to the agreed names. Raises ValueError
    if a legacy key and its replacement are both present with different
    values."""
    out = dict(metadata)
    for old, new in LEGACY_NAMES.items():
        if old not in out:
            continue
        value = out.pop(old)
        if new in out and out[new] != value:
            raise ValueError(f"{old!r} ({value!r}) and {new!r} ({out[new]!r}) disagree")
        out[new] = value
    return out


def registered(metadata: dict[str, Any]) -> dict[str, Any]:
    """The REGISTERED_KEYS in metadata, after renaming legacy keys"""
    metadata = rename_legacy(metadata)
    return {k: metadata[k] for k in REGISTERED_KEYS if k in metadata}


def _is_scalar(value: Any) -> bool:
    """True for strings, booleans, and finite numbers"""
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, str | int)


def _attr_to_json(value: Any) -> Any:
    """An hdf5 attribute as a JSON scalar (numpy scalars converted, bytes
    decoded), or None if it isn't one"""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    elif isinstance(value, np.generic):
        value = value.item()
    return value if _is_scalar(value) else None


def read_pprox_metadata(path: Path) -> dict[str, Any]:
    """Reader for spikes-pprox: a pprox or stimtrial file. Returns the
    REGISTERED_KEYS among the top-level scalars, with legacy names replaced;
    processed_by, a list, is joined with '; '. Null and non-finite numbers are
    left out. Only the top level is checked, not the trials."""
    try:
        doc = json.loads(Path(path).read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ValueError(f"not a JSON file ({err})") from err
    if not isinstance(doc, dict):
        raise ValueError("not a pprox file (the top level isn't an object)")
    known = (pprox._base_schema, pprox._stimtrial_schema)
    schema = doc.get("$schema")
    if not isinstance(schema, str) or schema.rstrip("#") not in {
        uri.rstrip("#") for uri in known
    }:
        raise ValueError(
            f"not a pprox file ($schema is {schema!r}, not one of {', '.join(known)})"
        )
    trials = doc.get("pprox")
    if not isinstance(trials, list):
        raise ValueError("not a pprox file (no 'pprox' list of trials)")
    if not all(isinstance(trial, dict) for trial in trials):
        raise ValueError("not a pprox file (a trial in 'pprox' isn't an object)")
    processed_by = doc.get("processed_by")
    if isinstance(processed_by, list) and all(isinstance(p, str) for p in processed_by):
        doc["processed_by"] = "; ".join(processed_by)
    return registered({k: v for k, v in doc.items() if _is_scalar(v)})


# the waveform file format written by dlab.spikes.save_waveforms
WAVEFORMS_SCHEMA = "meliza-org.spikewaveforms"
WAVEFORMS_SCHEMA_VERSIONS = (1, 2)


def _check_waveforms(fp: h5.File) -> None:
    """Raises ValueError if fp isn't a waveform file"""
    schema = _attr_to_json(fp.attrs.get("schema"))
    if schema != WAVEFORMS_SCHEMA:
        raise ValueError(
            f"not a waveform file (schema is {schema!r}, not {WAVEFORMS_SCHEMA!r})"
        )
    version = _attr_to_json(fp.attrs.get("schema_ver"))
    if version not in WAVEFORMS_SCHEMA_VERSIONS:
        raise ValueError(f"unknown version of {WAVEFORMS_SCHEMA} ({version!r})")
    for name, ndim, attrs in (
        ("waveforms", 2, ("sampling_rate", "peak_index")),
        ("times", 1, ("sampling_rate",)),
    ):
        dset = fp.get(name)
        if not isinstance(dset, h5.Dataset) or dset.ndim != ndim:
            raise ValueError(f"no {ndim}-D dataset {name!r}")
        missing = [attr for attr in attrs if attr not in dset.attrs]
        if missing:
            raise ValueError(f"dataset {name!r} has no {', '.join(missing)}")
    n_spikes = fp["waveforms"].shape[0]
    if fp["times"].shape[0] != n_spikes:
        raise ValueError(f"{n_spikes} waveforms but {fp['times'].shape[0]} spike times")
    if version >= 2 and n_spikes > 0:
        for name in ("mean_waveform", "sd_waveform"):
            if not isinstance(fp.get(name), h5.Dataset):
                raise ValueError(f"no dataset {name!r}")


def read_spikes_hdf5_metadata(path: Path) -> dict[str, Any]:
    """Reader for spikes-hdf5: a waveform file written by save_waveforms.
    Returns the REGISTERED_KEYS among the root attributes, as JSON scalars,
    with legacy names replaced. Arrays, non-finite numbers, and strings that
    aren't UTF-8 are left out. Only the datasets' shapes and attributes are
    read, not the data."""
    try:
        fp = h5.File(path, "r", locking=False)
    except OSError as err:
        raise ValueError(f"not an HDF5 file ({err})") from err
    with fp:
        _check_waveforms(fp)
        metadata = {}
        for key, value in fp.attrs.items():
            value = _attr_to_json(value)
            if value is not None:
                metadata[key] = value
    return registered(metadata)
