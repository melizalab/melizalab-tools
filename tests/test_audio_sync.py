# -*- mode: python -*-
"""Tests of dlab.audio_sync: stimulus onsets from an analog copy of the audio.

The stimuli are made of noise bursts with different amplitude patterns, and the
channel is a recording of them presented one after another, each a lag after
its start message, with background noise. Onsets are compared with where the
stimuli were placed.
"""

import arf
import numpy as np
import pandas as pd
import pytest
from conftest import FIRST_SAMPLE, StubFinder
from scipy.signal import butter, sosfilt

from dlab import audio_sync, kilo

RATE = 30000


def burst_stimulus(rng, length=None):
    """A stimulus of noise bursts with random lengths, gaps and levels"""
    n = round((length or rng.uniform(0.3, 0.6)) * RATE)
    out = np.zeros(n)
    i = 0
    while i < n:
        burst = round(rng.uniform(0.01, 0.06) * RATE)
        out[i : i + burst] = rng.uniform(0.2, 1.0) * rng.normal(0, 1, burst)[: n - i]
        i += burst + round(rng.uniform(0.005, 0.04) * RATE)
    return out


class Presentation:
    """Stimuli presented in order, each `lag` s (plus uniform jitter of up to
    `jitter` s) after its message, with `gap` s between the end of one stimulus
    and the next message. channel is the recording; onsets are where each
    stimulus starts on it (None if it wasn't played)."""

    def __init__(
        self,
        order,
        waveforms,
        *,
        lag=0.4,
        jitter=0.02,
        gap=0.8,
        noise=0.01,
        seed=1,
        played=None,
    ):
        rng = np.random.default_rng(seed)
        played = dict(played or {})  # index -> the waveform actually played
        self.waveforms = waveforms
        self.stimuli, self.onsets = [], []
        t = round(0.5 * RATE)
        pieces = []
        for i, name in enumerate(order):
            onset = t + round((lag + rng.uniform(0, jitter)) * RATE)
            self.stimuli.append(kilo.Stimulus(name, t))
            sound = played.get(i, waveforms[name])
            self.onsets.append(onset if sound is not None else None)
            if sound is not None:
                pieces.append((onset, sound))
            t = onset + waveforms[name].size + round(gap * RATE)
        self.channel = rng.normal(0, noise, t + RATE)
        for onset, sound in pieces:
            self.channel[onset : onset + sound.size] += sound

    def read(self, channel=None):
        x = self.channel if channel is None else channel
        return lambda a, b: x[a:b]

    def match(self, channel=None, **kwargs):
        return audio_sync.audio_sync_onsets(
            self.read(channel), self.stimuli, self.waveforms, RATE, **kwargs
        )


@pytest.fixture(scope="module")
def stimulus_set():
    rng = np.random.default_rng(0)
    return {f"s{i}": burst_stimulus(rng) for i in range(12)}


def standard_order(n=40):
    return [f"s{i % 12}" for i in range(n)]


def test_hardwired_copy_onsets_exact(stimulus_set):
    """On a clean copy, every stimulus is found at the sample it was played."""
    pres = Presentation(standard_order(), stimulus_set)
    matches = pres.match()
    assert [m.method for m in matches] == ["audio"] * len(matches)
    assert [m.onset for m in matches] == pres.onsets


def test_inverted_noisy_copy(stimulus_set):
    """The polarity of the channel doesn't matter, and noise doesn't move the
    onsets by more than a sample."""
    pres = Presentation(standard_order(), stimulus_set, noise=0.1)
    matches = pres.match(-pres.channel)
    assert all(m.method == "audio" for m in matches)
    assert np.abs(np.array([m.onset for m in matches]) - pres.onsets).max() <= 1


def test_microphone_like_copy(stimulus_set):
    """A filtered, delayed, inverted and noisy copy (like a microphone) is
    matched on the envelope, to within a few ms of the delay."""
    pres = Presentation(standard_order(), stimulus_set, noise=0.05)
    sos = butter(4, [500, 8000], "bandpass", fs=RATE, output="sos")
    delay = 45  # 1.5 ms
    mic = -0.3 * np.concatenate([np.zeros(delay), sosfilt(sos, pres.channel)])
    matches = pres.match(mic, refine=False)
    assert all(m.method == "audio" for m in matches)
    error = np.array([m.onset for m in matches]) - np.array(pres.onsets) - delay
    assert np.abs(error).max() <= 0.003 * RATE


def test_jittered_messages(stimulus_set):
    """Messages whose lag varies over 0.2 s (as with a large audio buffer) are
    calibrated for."""
    pres = Presentation(standard_order(), stimulus_set, lag=0.55, jitter=0.215)
    matches = pres.match()
    assert [m.onset for m in matches] == pres.onsets


def test_repeated_motif_not_matched_twice(stimulus_set):
    """A stimulus made of a motif played twice is found at its own onset, not
    at its second half or the next presentation of it."""
    rng = np.random.default_rng(3)
    motif = burst_stimulus(rng, 0.25)
    waveforms = dict(stimulus_set, rep=np.concatenate([motif, motif]))
    order = standard_order()
    order[10:13] = ["rep", "rep", "rep"]
    pres = Presentation(order, waveforms, gap=0.3)
    matches = pres.match()
    assert [m.onset for m in matches] == pres.onsets


def test_level_variants_are_listed(stimulus_set):
    """Variants of one stimulus (the same scene with a faint target at another
    level) can't be told apart; the onset is found and the variants listed.
    A variant with a louder target can be told apart on a clean copy."""
    rng = np.random.default_rng(4)
    scene = burst_stimulus(rng, 0.8)
    target = burst_stimulus(rng, 0.8)
    waveforms = dict(stimulus_set)
    for db in (-20, -45, -50, -55):
        waveforms[f"scene{db}"] = scene + 10 ** (db / 20) * target
    order = standard_order()
    order[20] = "scene-50"
    pres = Presentation(order, waveforms)
    m = pres.match()[20]
    assert (m.method, m.onset) == ("audio", pres.onsets[20])
    assert set(m.variants) == {"scene-45", "scene-55"}


def test_wrong_stimulus_played(stimulus_set):
    """If another stimulus of the same length was played, and it matches
    clearly better, the trial doesn't match."""
    rng = np.random.default_rng(5)
    waveforms = dict(stimulus_set)
    waveforms["a"] = burst_stimulus(rng, 0.5)
    waveforms["b"] = burst_stimulus(rng, 0.5)
    order = standard_order()
    order[15] = "a"
    pres = Presentation(order, waveforms, played={15: waveforms["b"]})
    m = pres.match()[15]
    assert m.method == "unmatched"
    assert m.reason.startswith("b matches better")


def test_silence_is_estimated(stimulus_set):
    """A silent stimulus has no onset to find; it's placed at its message plus
    the typical lag."""
    waveforms = dict(stimulus_set, silence=np.zeros(RATE // 2))
    order = standard_order()
    order[7] = "silence"
    pres = Presentation(order, waveforms, jitter=0.0)
    m = pres.match()[7]
    assert m.method == "estimated"
    assert abs(m.onset - (pres.stimuli[7].start + 0.4 * RATE)) <= 1


def test_stimulus_not_played(stimulus_set):
    """A stimulus missing from the channel doesn't match; the others do."""
    pres = Presentation(standard_order(), stimulus_set, played={9: None})
    matches = pres.match()
    assert matches[9].method == "unmatched"
    assert all(m.method == "audio" for i, m in enumerate(matches) if i != 9)


def test_channel_without_the_audio(stimulus_set):
    """A channel that didn't record the audio can't be calibrated."""
    pres = Presentation(standard_order(), stimulus_set)
    noise = np.random.default_rng(6).normal(0, 1, pres.channel.size)
    with pytest.raises(RuntimeError, match="were found on the audio channel"):
        pres.match(noise)


def test_relatives_by_length():
    """Stimuli within 50 ms (or 2%) of each other's length are compared."""
    lengths = {"a": 15000, "b": 15600, "c": 15000 + 1600, "d": 300000, "e": 305000}
    rel = audio_sync.relatives(lengths, RATE)
    assert rel["a"] == ["b"] and rel["d"] == ["e"] and rel["c"] == ["b"]


def test_contrast():
    """The contrast is the peak's excess over the best correlation outside the
    exclusion zone."""
    r = np.array([0.1, 0.2, 0.9, 0.85, 0.3, 0.6, 0.1])
    k, best, contrast = audio_sync.peak(r, exclusion=1)
    assert (k, best) == (2, 0.9) and contrast == pytest.approx(0.3)


# --- arf_to_trials with --audio-sync


def audio_entry(pres, channel="stim"):
    """add_entry arguments for a recording of a Presentation, on `channel`,
    with no sync track, and its start and stop messages"""
    messages = []
    for stim in pres.stimuli:
        messages.append((FIRST_SAMPLE + stim.start, f"start {stim.name}"))
        length = pres.waveforms[stim.name].size
        messages.append((FIRST_SAMPLE + stim.start + length, f"stop {stim.name}"))
    return dict(
        name="entry_0",
        timestamp=1000.0,
        nsamples=pres.channel.size,
        sync=None,
        messages=messages,
        channels={channel: np.round(pres.channel * 8000)},
    )


def audio_trials(path, pres, **kwargs):
    finder = StubFinder(
        {name: w.size / RATE for name, w in pres.waveforms.items()}, pres.waveforms
    )
    with arf.open_file(path, "r") as fp:
        return kilo.arf_to_trials(
            fp,
            finder,
            "ADC3",
            prepad=0.5,
            oeaudio_log=None,
            audio_sync="stim",
            **kwargs,
        )


def test_trials_from_audio_channel(make_arf, stimulus_set):
    """With an audio channel and no sync track, each trial's onset is where its
    stimulus was played, and the trial records how it was found."""
    pres = Presentation(standard_order(), stimulus_set)
    trials = audio_trials(make_arf(audio_entry(pres)), pres)
    assert [t.stimulus_start for t in trials] == pres.onsets
    assert [t.stimulus_name for t in trials] == [s.name for s in pres.stimuli]
    record = trials[0].audio_sync
    assert record["method"] == "audio" and record["r"] > 0.9
    (pproc,) = list(
        kilo.trials_to_pprox(pd.DataFrame(trials[:1]).assign(events=np.nan), RATE)
    )
    assert pproc["audio_sync"] == record


def test_trials_from_audio_channel_silence(make_arf, stimulus_set, caplog):
    """A silent stimulus is kept, with an estimated onset and a warning."""
    waveforms = dict(stimulus_set, silence=np.zeros(RATE // 2))
    order = standard_order()
    order[7] = "silence"
    pres = Presentation(order, waveforms, jitter=0.0)
    with caplog.at_level("WARNING", logger="dlab.kilo"):
        trials = audio_trials(make_arf(audio_entry(pres)), pres)
    assert trials[7].audio_sync["method"] == "estimated"
    assert "stimulus 7 (silence) is silent" in caplog.text


def test_trials_from_audio_channel_too_many_missing(make_arf, stimulus_set):
    """More than 1% of the stimuli (at least one) not found is an error."""
    pres = Presentation(standard_order(), stimulus_set, played={5: None, 9: None})
    with pytest.raises(RuntimeError, match="2 of 40 stimuli weren't found"):
        audio_trials(make_arf(audio_entry(pres)), pres)
