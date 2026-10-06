# -*- mode: python -*-
"""Validation against the pprox and stimtrial JSON schemas (copies in
tests/data/schemas)."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SCHEMAS = Path(__file__).parent / "data" / "schemas"


def _registry() -> Registry:
    resources = []
    for path in SCHEMAS.glob("*.json"):
        contents = json.loads(path.read_text())
        resources.append((contents["$id"], Resource.from_contents(contents)))
    return Registry().with_resources(resources)


REGISTRY = _registry()


def validate(doc: dict) -> None:
    """Validates doc against the schema named in its $schema (each schema is
    interpreted in its own dialect). Raises jsonschema.ValidationError."""
    Draft202012Validator({"$ref": doc["$schema"]}, registry=REGISTRY).validate(doc)
