# -*- mode: python -*-
"""The pprox files group-kilo-spikes writes conform to the published stimtrial
schema (bundled in dlab/schemas)."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError
from jsonschema.validators import validator_for

from dlab.pprox import SCHEMAS, validate

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("path", sorted(SCHEMAS.glob("*.json")), ids=lambda p: p.name)
def test_schemas_are_valid(path):
    schema = json.loads(path.read_text())
    cls = validator_for(schema, default=None)
    assert cls is not None, f"unrecognized $schema {schema['$schema']}"
    cls.check_schema(schema)


@pytest.mark.parametrize(
    "path", sorted((DATA / "E36_excerpt_golden").glob("*.pprox")), ids=lambda p: p.name
)
def test_golden_outputs_are_valid(path):
    validate(json.loads(path.read_text()))


def test_aux_outputs_are_valid(tmp_path):
    """Outputs with aux pulses (--aux led=ADC4) conform, including aux_tracks."""
    from test_group_spikes_excerpt import run_excerpt

    out = run_excerpt(tmp_path, extra=("--aux", "led=ADC4"))
    for path in sorted(out.glob("*.pprox")):
        doc = json.loads(path.read_text())
        assert doc["aux_tracks"] and any(t["aux"] for t in doc["pprox"])
        validate(doc)


def test_validation_is_active():
    """A trial without the interval stimtrial requires is rejected."""
    doc = json.loads(next((DATA / "E36_excerpt_golden").glob("*.pprox")).read_text())
    broken = copy.deepcopy(doc)
    del broken["pprox"][0]["interval"]
    with pytest.raises(ValidationError, match="'interval' is a required property"):
        validate(broken)
