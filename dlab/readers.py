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

from dlab import pprox

# names used by older producers, and the agreed names
LEGACY_NAMES = {"bird": "animal", "experiment": "protocol"}


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


def _is_scalar(value: Any) -> bool:
    """True for strings, booleans, and finite numbers"""
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, str | int)


def read_pprox_metadata(path: Path) -> dict[str, Any]:
    """Reader for spikes-pprox: a pprox or stimtrial file. Returns the
    top-level scalars except $schema, with legacy names replaced. Lists and
    objects, null, and non-finite numbers are left out. Only the top level is
    checked, not the trials."""
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
    metadata = {k: v for k, v in doc.items() if k != "$schema" and _is_scalar(v)}
    return rename_legacy(metadata)
