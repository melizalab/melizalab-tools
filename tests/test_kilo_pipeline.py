# -*- mode: python -*-
"""Requirements for splitting recordings into trials, for each supported
combination of stimulus presenter and sync method:

- oeaudio-present with old-style clicks (like examples/E69_1_1.arf)
- oeaudio-present with sustained pulses (no example recording yet)
- jpresent with sustained pulses (like examples/P352_1_1.arf)

Unlike the other test modules, these state what *should* happen. Cells that
don't work yet are marked xfail(strict=True) with the reason; when a fix makes
one pass, the strict marker turns it into a failure, so remove the marker. See
TODO.md for the underlying problems.

Each recording has four 2 s stimuli with 0.9 s gaps, matching the timing of
the example recordings, and messages that precede their sync events as they do
in the examples. oeaudio-present stimuli can be listed either by the message
dataset or by an --oeaudio-log file; jpresent only has the message dataset.
"""

import arf
import pytest
from conftest import (
    SAMPLING_RATE,
    SYNC,
    StubFinder,
    jpresent_messages,
    oeaudio_log_text,
    oeaudio_messages,
)

from dlab import kilo

STIMULI = [
    (name, 30000 + i * 87000, 30000 + i * 87000 + 60000)
    for i, name in enumerate("abcd")
]
NSAMPLES = STIMULI[-1][2] + 60000
DURATIONS = {name: 2.0 for name, _, _ in STIMULI}
# where a real pulse's flat top first dips (P352: 40-222 samples after onset)
PULSE_FIRST_DIP = 100


def build(make_arf, tmp_path, combo, source, missing=()):
    """Write a recording for `combo` and return (arf path, oeaudio log or None).

    missing: names of stimuli whose sync event was not recorded
    """
    presenter, sync = combo.split("-")
    if presenter == "oeaudio":
        messages = oeaudio_messages(STIMULI, metadata={"animal": "P1"})
    else:
        messages = jpresent_messages(STIMULI, conditions={"b", "d"})
    present = [s for s in STIMULI if s[0] not in missing]
    if sync == "clicks":
        sync_args = dict(clicks=[on for _, on, _ in present], click_samples=60)
    else:
        sync_args = dict(
            pulses=[(on, off) for _, on, off in present], dips=(PULSE_FIRST_DIP,)
        )
    log = None
    message_dset = "MessageCenter"
    if source == "network-events":  # GUI < 0.6
        message_dset = "Network_Events-104.0_TEXT_group_1"
    if source == "log":
        log = tmp_path / "oeaudio.log"
        log.write_text(oeaudio_log_text(messages))
        messages = None
    path = make_arf(
        dict(
            name="entry_0",
            timestamp=1000.0,
            nsamples=NSAMPLES,
            messages=messages,
            message_dset=message_dset,
            **sync_args,
        )
    )
    return path, log


def split(path, log):
    with arf.open_file(path, "r") as fp:
        return kilo.arf_to_trials(fp, StubFinder(DURATIONS), SYNC, oeaudio_log=log)


CASES = [
    pytest.param("oeaudio-clicks", "messages", id="oeaudio-clicks"),
    pytest.param("oeaudio-clicks", "network-events", id="oeaudio-clicks-pre0.6"),
    pytest.param("oeaudio-clicks", "log", id="oeaudio-clicks-log"),
    pytest.param("oeaudio-pulses", "messages", id="oeaudio-pulses"),
    pytest.param("oeaudio-pulses", "log", id="oeaudio-pulses-log"),
    pytest.param("jpresent-pulses", "messages", id="jpresent-pulses"),
]


@pytest.mark.parametrize("combo,source", CASES)
def test_one_trial_per_stimulus_at_sync_onset(make_arf, tmp_path, combo, source):
    """At the default threshold, there is one trial per stimulus, in
    presentation order, starting within 2 samples of the sync onset and lasting
    the stimulus duration.
    """
    result = split(*build(make_arf, tmp_path, combo, source))
    assert [t.stimulus_name for t in result] == ["a", "b", "c", "d"]
    for t, (_, onset, _) in zip(result, STIMULI, strict=True):
        assert 0 <= t.stimulus_start - onset <= 2, "trial starts at the sync onset"
        assert t.stimulus_end - t.stimulus_start == 60000, "lasts the duration"


@pytest.mark.parametrize(
    "combo,source",
    [
        pytest.param("oeaudio-clicks", "messages", id="oeaudio"),
        pytest.param("oeaudio-clicks", "log", id="oeaudio-log"),
        pytest.param("jpresent-pulses", "messages", id="jpresent"),
    ],
)
def test_entry_to_metadata_has_sampling_rate(make_arf, tmp_path, combo, source):
    """Entry metadata is a dict with the sampling rate, which pprox consumers
    (pprox.trial_iterator) need. The sync method doesn't affect this.
    """
    path, _ = build(make_arf, tmp_path, combo, source)
    with arf.open_file(path, "r") as fp:
        meta = kilo.entry_to_metadata(fp["entry_0"])
    assert isinstance(meta, dict), "metadata should be a dict"
    assert meta["sampling_rate"] == SAMPLING_RATE


@pytest.mark.parametrize(
    "combo,source",
    CASES,
)
def test_missed_sync_event_drops_only_that_stimulus(make_arf, tmp_path, combo, source):
    """If one sync event is missed, that stimulus is dropped and the others
    keep their own names and onsets.
    """
    result = split(*build(make_arf, tmp_path, combo, source, missing={"b"}))
    assert [t.stimulus_name for t in result] == ["a", "c", "d"]
    expected = [onset for name, onset, _ in STIMULI if name != "b"]
    for t, onset in zip(result, expected, strict=True):
        assert 0 <= t.stimulus_start - onset <= 2, f"{t.stimulus_name} keeps its onset"
