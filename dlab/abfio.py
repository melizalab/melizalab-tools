# -*- mode: python -*-
"""Read the ABF files in abfzip archives in place, without unpacking them

ZippedAxonIO and ZippedAxonRawIO are neo's AxonIO and AxonRawIO for an ABF file
in an abfzip archive (see dlab.abfzip), such as an intracellular-abfzip
resource. The signals are memory-mapped from the zip file. For example:

    path = neurobank.find_resource("6519d899")
    for member in abf_members(path):
        block = ZippedAxonIO(path, member).read_block()

The members must be stored without compression, as abf-zip writes them. For
libraries that only read files by path (e.g. pyabf), extract a member with
nbank.zipped.extract_member.

neo's header parser only takes a file name, so while a header is parsed, the
names that neo.rawio.axonrawio uses to read the file are replaced with ones
that read the member. Zipped files are parsed one at a time, but a plain ABF
file shouldn't be parsed with neo in another thread at the same moment.

"""

import os
import threading
from pathlib import Path
from unittest import mock

import numpy as np
from nbank.zipped import member_extent, members, open_member
from neo.io.axonio import AxonIO
from neo.rawio import axonrawio
from neo.rawio.axonrawio import AxonRawIO

from dlab.abfzip import is_abf, read_header

# one header parse at a time, as each replaces names in neo.rawio.axonrawio
_parse_lock = threading.Lock()


def abf_members(path: Path | str) -> list[str]:
    """The ABF files in the zip file at path, in the order they're stored"""
    return [name for name in members(Path(path)) if is_abf(name)]


class _Numpy:
    """numpy, with memmap offsets counted from where the member starts"""

    def __init__(self, start: int):
        self._start = start

    def __getattr__(self, name):
        return getattr(np, name)

    def memmap(self, filename, dtype, mode, shape=None, offset=0):
        return np.memmap(
            filename, dtype, mode, shape=shape, offset=self._start + offset
        )


class _OSPath:
    """os.path, with getsize giving the member's size"""

    def __init__(self, size: int):
        self._size = size

    def __getattr__(self, name):
        return getattr(os.path, name)

    def getsize(self, filename):
        return self._size


class _OS:
    """os, with path.getsize giving the member's size"""

    def __init__(self, size: int):
        self.path = _OSPath(size)

    def __getattr__(self, name):
        return getattr(os, name)


class ZippedAxonRawIO(AxonRawIO):
    """neo's AxonRawIO for the ABF file member of the zip file at path. Raises
    KeyError if there's no such member, and ValueError if it's compressed."""

    def __init__(self, path: Path | str, member: str):
        self.member = member
        self._start, self._size = member_extent(Path(path), member)
        super().__init__(filename=str(path))

    def _parse_header(self):
        path = Path(self.filename)

        def parse_axon_soup(filename):
            with open_member(path, self.member) as fp:
                return read_header(fp)

        with (
            _parse_lock,
            mock.patch.object(axonrawio, "parse_axon_soup", parse_axon_soup),
            mock.patch.object(axonrawio, "np", _Numpy(self._start)),
            mock.patch.object(axonrawio, "os", _OS(self._size)),
        ):
            super()._parse_header()
        # the signals are read from the zip file, where the member starts
        for segments in self._buffer_descriptions.values():
            for buffers in segments.values():
                for description in buffers.values():
                    description["file_offset"] += self._start

    def _source_name(self):
        return f"{self.filename}:{self.member}"


class ZippedAxonIO(ZippedAxonRawIO, AxonIO):
    """neo's AxonIO (read_block, read_protocol, ...) for the ABF file member of
    the zip file at path. Raises KeyError if there's no such member, and
    ValueError if it's compressed."""
