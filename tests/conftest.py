# -*- mode: python -*-
"""Shared test fixtures: synthetic ARF recordings shaped like arfx-oephys output.

dlab.kilo reads ARF files made by arfx-oephys from oeaudio-present recordings.
Each entry holds one int16 dataset per recorded channel, including the sync
track that carries a click at each stimulus onset, and an EVENT dataset of the
network messages oeaudio-present sends ("start <file>", "stop <file>", and one
"metadata: {...}" message). The layout here follows arfx/oephys.py for GUI
versions >= 0.6, where the message dataset is named "MessageCenter":

- continuous channels: int16, attrs `sampling_rate` and `offset`, where offset
  is the open-ephys sample number of the first sample, in seconds
- messages: compound (start: int64, message: S513), units ("samples", ""),
  attr `sampling_rate`. `start` is an open-ephys sample number, so it includes
  the recording's first sample number, unlike indices into the sync track.

The sync track is modeled on the two example recordings in examples/ (see
test_kilo_examples.py for the measurements):

- old style (E69, GUI 0.5): a 2 ms click (60 samples) at about 21000, SD ~300,
  at each stimulus onset. One-sample impulses are also available; quickspikes
  reports those at exactly the click sample, which keeps expected trial
  boundaries exact.
- new style (P352, GUI 1.0): a pulse that is high for the whole stimulus. The
  top is flat at the ADC ceiling (30083, about 4.6 V), with only rare dips of a
  few counts. The flat top matters: quickspikes reports a pulse at the last
  sample before its first dip, so dips are placed explicitly.
"""

import json
import os

import arf
import numpy as np
import pytest

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SAMPLING_RATE = 30000
SYNC = "ADC3"
MESSAGES = "MessageCenter"
# first sample number of a real recording (taken from the arfx test fixtures)
FIRST_SAMPLE = 152832


BASELINE = 330  # sync channel at rest
PULSE_LEVEL = 30083  # flat top of a sustained pulse
CLICK_LEVEL = 21000  # old-style click
CLICK_SAMPLES = 60  # length of an old-style click (2 ms)


def sync_track(nsamples, clicks=(), pulses=(), *, click_samples=1, dips=(), seed=0):
    """A sync channel: baseline noise plus clicks and/or sustained pulses.

    clicks: onset samples. With click_samples=1 each click is a one-sample
    impulse; otherwise it lasts click_samples at CLICK_LEVEL with SD 300 noise.
    pulses: (onset, offset) pairs. Each pulse is flat at PULSE_LEVEL except for
    a one-count dip at onset + d for each d in dips.
    """
    rng = np.random.default_rng(seed)
    data = BASELINE + rng.normal(0, 5, nsamples)
    for onset in clicks:
        if click_samples == 1:
            data[onset] = CLICK_LEVEL
        else:
            data[onset : onset + click_samples] = CLICK_LEVEL + rng.normal(
                0, 300, click_samples
            )
    for onset, offset in pulses:
        data[onset:offset] = PULSE_LEVEL
        for d in dips:
            data[onset + d] = PULSE_LEVEL - 1
    return np.round(data).astype("int16")


def message_table(messages):
    """EVENT records from (sample, text) pairs, as arfx-oephys stores them"""
    starts = np.array([sample for sample, _ in messages], dtype="int64")
    texts = np.array([text.encode("utf-8") for _, text in messages], dtype="S513")
    return np.rec.fromarrays([starts, texts], names=("start", "message"))


# How long the sync event lags its start message, in samples. Both presenters
# send the message before the sound (and its sync event) comes out of the audio
# buffer. Measured in examples/: ~0.39 s in E69, ~0.25 s in P352.
OEAUDIO_LEAD = 11700
JPRESENT_LEAD = 7500


def oeaudio_messages(
    stimuli, *, metadata=None, first_sample=FIRST_SAMPLE, lead=OEAUDIO_LEAD
):
    """The messages oeaudio-present sends, modeled on examples/E69_1_1.arf.

    stimuli: (name, onset, offset), with onset and offset as indices into the
    sync track. Each start/stop message is logged `lead` samples before the
    corresponding sync edge, and stored as an open-ephys sample number (so
    first_sample is added). The metadata message is sent only if metadata is
    given.
    """
    out = [
        (
            first_sample + 2,
            "StartRecord RecDir=/home/melizalab/open-ephys/ PrependText=P1 "
            "AppendText=expt",
        ),
        (first_sample + 2, "GetRecordingPath"),
    ]
    if metadata is not None:
        out.append((first_sample + 2, "metadata: " + json.dumps(metadata)))
    for name, onset, offset in stimuli:
        path = f"/home/melizalab/stimuli/expt/{name}.wav"
        out.append((first_sample + onset - lead, f"start {path}"))
        out.append((first_sample + offset - lead, f"stop {path}"))
    return out


def jpresent_messages(
    stimuli, *, conditions=(), first_sample=FIRST_SAMPLE, lead=JPRESENT_LEAD
):
    """The messages jpresent (jill) sends, modeled on examples/P352_1_1.arf.

    As for oeaudio_messages, but names have no directory or extension, there is
    no metadata message, and the stimuli named in `conditions` also get
    condition_start/condition_stop messages. jrelay's connect message predates
    the recording.
    """
    out = [(first_sample - 690, "jrelay connected")]
    for name, onset, offset in stimuli:
        start = first_sample + onset - lead
        stop = first_sample + offset - lead
        out.append((start, f"start {name}"))
        if name in conditions:
            out.append((start + 768, f"condition_start {name}"))
        out.append((stop, f"stop {name}"))
        if name in conditions:
            out.append((stop, f"condition_stop {name}"))
    return out


def oeaudio_log_text(messages, sampling_rate=SAMPLING_RATE):
    """The open-ephys-audio log file for a list of (sample, text) messages.

    Assumes open-ephys sample numbers count from StartAcquisition, which is
    logged at sample 0. Log lines have wall-clock timestamps.
    """
    import datetime

    t0 = datetime.datetime(2026, 6, 17, 13, 0, 0)
    lines = [f'{t0:%Y-%m-%d %H:%M:%S.%f},"StartAcquisition"']
    for sample, text in messages:
        ts = t0 + datetime.timedelta(seconds=sample / sampling_rate)
        lines.append(f'{ts:%Y-%m-%d %H:%M:%S.%f},"{text}"')
    return "\n".join(lines) + "\n"


class StubFinder:
    """Stands in for kilo.StimulusFinder with fixed durations (in s)"""

    def __init__(self, durations):
        self.durations = durations

    def get_durations(self, names):
        try:
            return {name: self.durations[name] for name in names}
        except KeyError as err:
            raise FileNotFoundError(err.args[0]) from err


@pytest.fixture
def make_arf(tmp_path):
    """Returns a function that writes entries to a new ARF file and returns its path.

    Each argument is a dict of keyword arguments for add_entry.
    """

    def make(*entries):
        path = tmp_path / "recording.arf"
        with arf.open_file(path, "w") as fp:
            for spec in entries:
                add_entry(fp, **spec)
        return path

    return make


def add_entry(
    fp,
    name,
    timestamp,
    *,
    nsamples,
    clicks=(),
    pulses=(),
    click_samples=1,
    dips=(),
    messages=None,
    first_sample=FIRST_SAMPLE,
    sampling_rate=SAMPLING_RATE,
    sync=SYNC,
    message_dset=MESSAGES,
    **attrs,
):
    """Add an entry with a sync channel (unless sync is None) and a message
    dataset (unless messages is None). Returns the entry."""
    entry = arf.create_entry(fp, name, timestamp, **attrs)
    if sync is not None:
        arf.create_dataset(
            entry,
            sync,
            sync_track(
                nsamples, clicks, pulses, click_samples=click_samples, dips=dips
            ),
            sampling_rate=sampling_rate,
            offset=first_sample / sampling_rate,
            channel_name=sync,
        )
    if messages is not None:
        arf.create_dataset(
            entry,
            message_dset,
            message_table(messages),
            units=("samples", ""),
            datatype=arf.DataTypes.EVENT,
            sampling_rate=sampling_rate,
        )
    return entry


# --- kilosort output

SPIKE = -1000 * np.exp(-0.5 * ((np.arange(60) - 20) / 4) ** 2)  # 2 ms, peak at 20


def make_kilosort_dir(
    path, clusters, *, nsamples, nchannels=4, sampling_rate=SAMPLING_RATE, seed=0
):
    """Write the kilosort/phy output group-kilo-spikes reads, and return path.

    clusters: {cluster_id: dict(times=spike samples, group="good"|"mua"|"noise",
    ch=channel, amplitude=scale of the spike shape, default 1)}. Each spike is
    added to the whitened data (temp_wh.dat) on its cluster's channel, with
    its trough SPIKE.argmin() samples after the spike time. Columns of
    cluster_info.tsv that the script reads are filled in.
    """
    import pandas as pd

    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 20, (nsamples, nchannels))
    times, ids, rows = [], [], []
    for cid, spec in clusters.items():
        t = np.asarray(spec["times"], dtype="int64")
        for s in t:
            data[s - SPIKE.argmin() : s - SPIKE.argmin() + SPIKE.size, spec["ch"]] += (
                spec.get("amplitude", 1) * SPIKE
            )
        times.append(t)
        ids.append(np.full(t.size, cid))
        rows.append(
            dict(
                cluster_id=cid,
                Amplitude=50.0 + cid,
                ContamPct=1.0 * cid,
                ch=spec["ch"],
                depth=100.0 * spec["ch"],
                group=spec["group"],
                n_spikes=t.size,
            )
        )
    order = np.argsort(np.concatenate(times), kind="stable")
    np.save(path / "spike_times.npy", np.concatenate(times)[order][:, None])
    np.save(path / "spike_clusters.npy", np.concatenate(ids)[order])
    pd.DataFrame(rows).set_index("cluster_id").to_csv(
        path / "cluster_info.tsv", sep="\t"
    )
    np.round(data).astype("int16").tofile(path / "temp_wh.dat")
    (path / "params.py").write_text(
        f"dat_path = 'temp_wh.dat'\n"
        f"n_channels_dat = {nchannels}\n"
        f"dtype = 'int16'\n"
        f"offset = 0\n"
        f"sample_rate = {sampling_rate:.1f}\n"
        f"hp_filtered = True\n"
    )
    return path
