# -*- mode: python -*-
"""Tests for dlab.pprox.

The first group (make_pprox through split_trial_empty) uses the shared fixture
data below. Later tests use small hand-built trials so that every expected
value can be checked by hand. Tests marked PINNED record current behavior that
looks like a bug; see TODO.md.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from dlab import pprox

log = logging.getLogger("dlab")

trials = (
    {
        "events": [1, 2, 3, 4],
        "interval": [0.0, 2.3],
        "offset": 0,
        "stimulus": {"name": "stim1", "interval": [1.0, 1.5]},
    },
    {
        "events": [4, 5, 6],
        "interval": [0.0, 2.3],
        "offset": 4.0,
        "stimulus": {"name": "stim2", "interval": [1.0, 1.5]},
    },
    {
        "events": [1.1, 2.1, 2.9, 4.01],
        "interval": [0.0, 2.3],
        "offset": 4.0,
        "stimulus": {"name": "stim1", "interval": [1.0, 1.5]},
    },
)

empty_trial = {
    "events": [],
    "recording": {"entry": 0, "start": 36989, "stop": 989947},
    "index": 0,
    "offset": 1.2329666666666668,
    "stimulus": {
        "name": "igmi8fxa-p1mrfhop-0oq8ifcb-l1a3ltpy-vekibwgj-9ex2k0dy-c95zqjxq-ztqee46x-g29wxi4q-jkexyrd5-30_btwmt59w-50",
        "interval": [1.0, 30.254149659863945],
    },
    "interval": [0.0, 31.765266666666665],
}

unit = {
    "$schema": "https://meliza.org/spec:2/pprox.json#",
    "pprox": [
        {
            "events": [
                0.1489,
                0.1717,
                0.22346666666666667,
                0.5972333333333333,
                0.6348,
                1.0875666666666666,
                3.541933333333333,
                7.3800333333333334,
                7.432066666666667,
                7.460066666666667,
                7.8529,
                8.1841,
                8.7151,
                16.006466666666668,
                18.276933333333332,
                19.355333333333334,
                20.1928,
                20.8047,
                21.667766666666665,
                22.336766666666666,
                23.334633333333333,
                23.6505,
                24.37473333333333,
                25.661833333333334,
                27.4715,
                28.974266666666665,
                29.2809,
                30.1314,
                30.724566666666668,
            ],
            "offset": 2.226566666666667,
            "index": 0,
            "interval": [-1.0, 30.765266666666665],
            "stimulus": {
                "name": "igmi8fxa-p1mrfhop-0oq8ifcb-l1a3ltpy-vekibwgj-9ex2k0dy-c95zqjxq-ztqee46x-g29wxi4q-jkexyrd5-30_btwmt59w-50",
                "interval": [0.0, 29.254133333333332],
            },
            "recording": {"entry": 0, "start": 36797, "end": 989755},
        },
        {
            "events": [
                -0.0037,
                0.4285,
                1.2156666666666667,
                1.7268333333333334,
                2.164433333333333,
                2.5422,
                2.7259333333333333,
                4.4635,
                6.153833333333333,
                8.7661,
                9.892333333333333,
                10.826433333333334,
                10.8462,
                12.428366666666667,
                14.692666666666666,
                15.734233333333334,
                16.900133333333333,
                20.7051,
                23.8364,
                26.699633333333335,
            ],
            "offset": 33.99183333333333,
            "index": 1,
            "interval": [-1.0, 30.765166666666666],
            "stimulus": {
                "name": "g29wxi4q-c95zqjxq-jkexyrd5-vekibwgj-ztqee46x-0oq8ifcb-9ex2k0dy-igmi8fxa-l1a3ltpy-p1mrfhop-30_btwmt59w-60",
                "interval": [0.0, 29.254133333333332],
            },
            "recording": {"entry": 0, "start": 989755, "end": 1942710},
        },
    ],
    "recording": "https://gracula.psyc.virginia.edu/neurobank/resources/C24_3_1/",
    "processed_by": ["group-kilo-spikes 2022.10.11"],
    "kilosort_amplitude": 2888.8,
    "kilosort_contam_pct": 0.0,
    "kilosort_source_channel": 103,
    "kilosort_probe_depth": 725.0,
    "kilosort_n_spikes": 3883,
    "entry_metadata": [
        {
            "animal": "C24",
            "experimenter": "smm3rc",
            "experiment": "msyn-chorus",
            "hemisphere": "R",
            "pen": 3,
            "site": 1,
            "x": 1876.8,
            "y": 2085.7,
            "z": -2498.9,
            "name": "/C24_2021-09-27_17-51-57_msyn-chorus_Record Node 104_experiment1_recording1",
            "sampling_rate": 30000,
        }
    ],
    "pen": 3,
    "bird": "3fe04228-347b-4884-bc02-83d56bafb861",
    "site": 1,
    "protocol": "chorus",
    "experimenter": "smm3rc",
}

stims = {
    "igmi8fxa-p1mrfhop-0oq8ifcb-l1a3ltpy-vekibwgj-9ex2k0dy-c95zqjxq-ztqee46x-g29wxi4q-jkexyrd5-30_btwmt59w-50": {
        "foreground": "igmi8fxa-p1mrfhop-0oq8ifcb-l1a3ltpy-vekibwgj-9ex2k0dy-c95zqjxq-ztqee46x-g29wxi4q-jkexyrd5",
        "background": "btwmt59w",
        "background-dBFS": -50,
        "foreground-dBFS": -30,
        "stim_begin": [
            2.0,
            3.926984126984127,
            6.8179818594104304,
            9.29718820861678,
            12.169183673469387,
            14.489183673469388,
            16.934172335600906,
            19.386167800453514,
            22.03816326530612,
            24.63315192743764,
        ],
        "stim_end": [
            3.426984126984127,
            6.3179818594104304,
            8.79718820861678,
            11.669183673469387,
            13.989183673469388,
            16.434172335600906,
            18.886167800453514,
            21.53816326530612,
            24.13315192743764,
            26.754149659863945,
        ],
    },
    "g29wxi4q-c95zqjxq-jkexyrd5-vekibwgj-ztqee46x-0oq8ifcb-9ex2k0dy-igmi8fxa-l1a3ltpy-p1mrfhop-30_btwmt59w-60": {
        "foreground": "g29wxi4q-c95zqjxq-jkexyrd5-vekibwgj-ztqee46x-0oq8ifcb-9ex2k0dy-igmi8fxa-l1a3ltpy-p1mrfhop",
        "background": "btwmt59w",
        "background-dBFS": -60,
        "foreground-dBFS": -30,
        "stim_begin": [
            2.0,
            4.594988662131519,
            7.046984126984127,
            9.66798185941043,
            11.98798185941043,
            14.639977324263038,
            17.119183673469387,
            19.564172335600908,
            21.491156462585035,
            24.36315192743764,
        ],
        "stim_end": [
            4.094988662131519,
            6.546984126984127,
            9.16798185941043,
            11.48798185941043,
            14.139977324263038,
            16.619183673469387,
            19.064172335600908,
            20.991156462585035,
            23.86315192743764,
            26.754149659863945,
        ],
    },
}


def split_fun(name):
    info = stims[name].copy()
    info["foreground"] = info["foreground"].split("-")
    return pd.DataFrame(info).rename(lambda s: s.replace("-", "_"), axis="columns")


def test_make_pprox():
    pp = pprox.from_trials(trials, test_attribute="blank")
    assert pp["$schema"] == pprox._base_schema
    assert pp["test_attribute"] == "blank"
    assert pp["pprox"] == trials


def test_group_by_stim():
    pp = pprox.from_trials(trials)
    for stim, group in pprox.groupby(pp, lambda trial: trial["stimulus"]["name"]):
        if stim == "stim1":
            assert list(group) == [trials[0], trials[2]]
        elif stim == "stim2":
            assert list(group) == [trials[1]]
        else:
            raise ValueError("unexpected stimulus name")


def test_aggregate_events_simple():
    all_events = pprox.aggregate_events(pprox.from_trials(trials))
    assert all_events.size == sum(len(t["events"]) for t in trials)


def test_aggregate_events_complex():
    all_events = pprox.aggregate_events(unit)
    assert all_events.size == sum(len(t["events"]) for t in unit["pprox"])


def test_split_trial():
    for trial in unit["pprox"]:
        stim = trial["stimulus"]["name"]
        split = pprox.split_trial(trial, split_fun)
        assert split.shape[0] == len(stims[stim]["stim_end"])
        trial_spikes = np.asarray(trial["events"]) + trial["offset"]
        split_spikes = (
            split.apply(lambda x: x.events + x.offset, axis=1).dropna().explode()
        )
        # round to avoid floating point imprecision
        assert np.all(
            np.isin(np.floor(split_spikes * 1000), np.floor(trial_spikes * 1000))
        )


def test_split_trial_empty():
    split = pprox.split_trial(empty_trial, split_fun)
    stim = empty_trial["stimulus"]["name"]
    split_spikes = split.apply(lambda x: x.events + x.offset, axis=1).dropna().explode()
    assert split.shape[0] == len(stims[stim]["stim_end"])
    assert len(split_spikes) == 0


# --- constructors and helpers


def test_empty():
    """empty() is a pprox collection with the base schema and no trials."""
    pp = pprox.empty()
    assert pp == {"$schema": pprox._base_schema, "pprox": ()}


def test_from_trials_consumes_iterators_and_returns_tuple():
    """from_trials accepts any iterable (e.g. a generator) and stores the trials as
    a tuple.
    """
    pp = pprox.from_trials(iter(trials))
    assert isinstance(pp["pprox"], tuple), "trials should be stored as a tuple"
    assert pp["pprox"] == trials


def test_from_trials_custom_schema():
    """The $schema can be overridden."""
    pp = pprox.from_trials(trials, schema=pprox._stimtrial_schema)
    assert pp["$schema"] == pprox._stimtrial_schema


def test_wrap_uuid_str_and_bytes():
    """Both str and bytes UUIDs become a urn:uuid: string."""
    raw = "123e4567-e89b-12d3-a456-426614174000"
    assert pprox.wrap_uuid(raw) == f"urn:uuid:{raw}"
    assert pprox.wrap_uuid(raw.encode("ascii")) == f"urn:uuid:{raw}"


def test_wrap_uuid_rejects_garbage():
    """A string that is not a UUID raises ValueError."""
    with pytest.raises(ValueError):
        pprox.wrap_uuid("not a uuid")


def test_groupby_orders_groups_by_key():
    """groupby sorts trials by the key first, so groups come out in key order."""
    pp = pprox.from_trials(trials)
    keys = [k for k, _ in pprox.groupby(pp, lambda t: t["stimulus"]["name"])]
    assert keys == ["stim1", "stim2"], "groups should come out in key order"


def test_trial_iterator_annotates_sampling_rate_from_entry_metadata():
    """A trial whose recording block has no sampling_rate gets one from the
    collection's entry_metadata, looked up by the trial's entry. The trial dict is
    modified in place.
    """
    trial = {"events": [], "recording": {"entry": 1}}
    pp = {
        "pprox": [trial],
        "entry_metadata": [{"sampling_rate": 20000}, {"sampling_rate": 30000}],
    }
    ((i, out),) = list(pprox.trial_iterator(pp))
    assert i == 0
    assert out["recording"]["sampling_rate"] == 30000, (
        "rate should come from entry_metadata[entry]"
    )
    # NB: the annotation is made in place on the input trial
    assert trial["recording"]["sampling_rate"] == 30000, "annotation is made in place"


def test_trial_iterator_keeps_existing_sampling_rate():
    """A sampling_rate already on the trial is not overwritten."""
    trial = {"recording": {"entry": 0, "sampling_rate": 1000}}
    pp = {"pprox": [trial], "entry_metadata": [{"sampling_rate": 30000}]}
    ((_, out),) = list(pprox.trial_iterator(pp))
    assert out["recording"]["sampling_rate"] == 1000, (
        "existing sampling_rate should not be overwritten"
    )


def test_trial_iterator_without_metadata_raises():
    """Without entry_metadata and without a rate on the trial, KeyError is raised."""
    pp = {"pprox": [{"recording": {"entry": 0}}]}
    with pytest.raises(KeyError):
        list(pprox.trial_iterator(pp))


def test_aggregate_events_applies_offsets_in_trial_order():
    """Events from all trials are concatenated in trial order, each shifted by its
    trial's offset. Trials with no events contribute nothing.
    """
    pp = pprox.from_trials(
        [
            {"events": [1.0, 2.0], "offset": 0.0},
            {"events": [], "offset": 5.0},
            {"events": [0.5], "offset": 10.0},
        ]
    )
    assert pprox.aggregate_events(pp).tolist() == [1.0, 2.0, 10.5], (
        "events should be shifted by offset, in trial order"
    )


def test_aggregate_events_empty_collection():
    """A collection with no trials aggregates to an empty array."""
    out = pprox.aggregate_events(pprox.empty())
    assert out.size == 0 and out.dtype == float


def test_unimplemented_stubs_raise():
    """validate and combine_recordings are not implemented yet, and say so
    rather than silently doing nothing."""
    with pytest.raises(NotImplementedError):
        pprox.validate(pprox.empty())
    with pytest.raises(NotImplementedError):
        pprox.combine_recordings()


# --- split_trial, with hand-computed expectations


def make_trial(events, name="s", interval=(1.0, 3.0), offset=10.0, index=7):
    """A minimal trial dict with the fields split_trial reads."""
    return {
        "events": events,
        "interval": [0.0, 4.0],
        "offset": offset,
        "index": index,
        "stimulus": {"name": name, "interval": list(interval)},
    }


def fixed_splits(begin, end):
    """A split_fun that ignores the stimulus name and returns the given split table."""
    names = [f"split{i}" for i in range(len(begin))]
    frame = pd.DataFrame({"stim_begin": begin, "stim_end": end, "name": names})
    return lambda _name: frame.copy()


two_splits = fixed_splits([0.0, 1.0], [0.5, 1.5])


def split_events(df):
    """events per split as plain lists (nan -> None)"""
    return [None if isinstance(e, float) else list(e) for e in df.events]


def test_split_trial_values():
    """Hand-computed example. The stimulus starts at 1.0 s and is split at 0 and 1.0
    s (stimulus time); events are 1.2, 1.4, 2.5 and 3.5 s in the trial.

    Expected: events are re-referenced to the start of each split (0.2 and 0.4 in
    the first; 0.5 in the second), the 3.5 s event falls after the last split and
    is dropped, offsets are trial offset + split start + stimulus onset (11 and
    12), and each split's end and interval end are relative to its own start.
    """
    df = pprox.split_trial(make_trial([1.2, 1.4, 2.5, 3.5]), two_splits)
    assert list(df.columns) == [
        "interval",
        "stim_end",
        "name",
        "interval_end",
        "events",
        "offset",
        "source_trial",
    ]
    assert df.interval.tolist() == [0, 1]
    assert df.name.tolist() == ["split0", "split1"]
    # times are relative to the start of each split; 3.5 is past the end of the
    # last split (stim-relative 2.5 >= 2.0) and is dropped
    assert split_events(df)[0] == pytest.approx([0.2, 0.4], abs=1e-6), (
        "events should be relative to split start"
    )
    assert split_events(df)[1] == pytest.approx([0.5], abs=1e-6), (
        "events should be relative to split start; 3.5 dropped"
    )
    # trial offset + split start + stimulus onset
    assert df.offset.tolist() == [11.0, 12.0], (
        "offset = trial offset + split start + stimulus onset"
    )
    assert df.stim_end.tolist() == [0.5, 0.5], (
        "stim_end should be relative to split start"
    )
    assert df.interval_end.tolist() == [1.0, 1.0], (
        "interval_end should be relative to split start"
    )
    assert df.source_trial.tolist() == [7, 7], (
        "source_trial should come from trial['index']"
    )


def test_split_trial_split_without_events_is_nan():
    """A split that contains no events has NaN (a float, not an array) in its events
    column.
    """
    df = pprox.split_trial(make_trial([1.2]), two_splits)
    assert split_events(df)[0] == pytest.approx([0.2], abs=1e-6)
    assert split_events(df)[1] is None, "split without events should be NaN"


def test_split_trial_events_are_float32():
    """Events come back as float32. This is deliberate: trial events are relative
    to the trial (kilo writes them relative to stimulus onset), so they are a few
    seconds at most and float32 resolves them to ~1e-7 s, far below one sample
    at 30 kHz. It would not be enough for times referenced to the start of an
    hours-long recording (~0.5 ms at 2 h).
    """
    df = pprox.split_trial(make_trial([1.2]), two_splits)
    assert df.events.iloc[0].dtype == np.float32


def test_split_trial_pads_last_interval_by_mean_gap():
    """The last split has no following stimulus, so its interval is extended by the
    mean of the gaps between the other splits (0.5 and 1.0 here, so 0.75).
    """
    splits = fixed_splits([0.0, 1.0, 2.5], [0.5, 1.5, 3.0])
    df = pprox.split_trial(make_trial([1.2]), splits)
    # gaps after splits 0 and 1 are 0.5 and 1.0, so the last is padded by 0.75
    assert df.interval_end.tolist() == [1.0, 1.5, 1.25], (
        "last interval should be padded by the mean gap (0.75)"
    )
    assert df.offset.tolist() == [11.0, 12.0, 13.5], (
        "offset = trial offset + split start + stimulus onset"
    )


def test_split_trial_single_split_has_no_padding():
    """With only one split there are no gaps to average, so the interval ends where
    the stimulus ends and later events are dropped.
    """
    df = pprox.split_trial(make_trial([1.2, 2.4, 2.6]), fixed_splits([0.0], [1.5]))
    assert df.interval_end.tolist() == [1.5], "a single split should not be padded"
    # 2.6 - 1.0 = 1.6 is after the end of the interval
    assert split_events(df)[0] == pytest.approx([0.2, 1.4], abs=1e-6), (
        "events past the interval end should be dropped"
    )


def test_split_trial_passes_stimulus_name_to_split_fun():
    """split_fun is called with the name of the trial's stimulus."""
    seen = []

    def split_fun(name):
        seen.append(name)
        return two_splits(name)

    pprox.split_trial(make_trial([], name="song7"), split_fun)
    assert seen == ["song7"], "split_fun should get the stimulus name"


def test_split_trial_extra_split_columns_are_kept():
    """Extra columns in the split table (metadata about each split) pass through to
    the output.
    """
    frame = pd.DataFrame(
        {
            "stim_begin": [0.0, 1.0],
            "stim_end": [0.5, 1.5],
            "name": ["a", "b"],
            "syllable_type": ["x", "y"],
        }
    )
    df = pprox.split_trial(make_trial([1.2]), lambda _name: frame.copy())
    assert df.syllable_type.tolist() == ["x", "y"], (
        "extra split columns should pass through"
    )


def test_split_trial_requires_trial_index():
    """split_trial reads trial["index"] for the source_trial column, so a trial
    without one raises KeyError. kilo.trials_to_pprox always provides it.
    """
    trial = make_trial([1.2])
    del trial["index"]
    with pytest.raises(KeyError):
        pprox.split_trial(trial, two_splits)


def test_split_trial_drops_events_before_stimulus_onset():
    """Events before the stimulus starts (prepad) are not assigned to any split."""
    # prepad events are not assigned to any split
    df = pprox.split_trial(make_trial([0.5, 1.2]), two_splits)
    assert sum(len(e) for e in split_events(df) if e is not None) == 1, (
        "prepad event should be dropped"
    )


def test_split_trial_event_exactly_at_first_split_start_is_kept():
    """Splits are half-open: an event exactly at the start of the first split
    belongs to it, at time 0."""
    df = pprox.split_trial(make_trial([1.0, 1.2]), two_splits)
    assert split_events(df)[0] == pytest.approx([0.0, 0.2], abs=1e-6), (
        "event at the first split start is kept"
    )


def test_split_trial_event_exactly_at_later_split_start_goes_to_that_split():
    """Splits are half-open: an event exactly at a later split's start belongs
    to that split, at time 0, not to the end of the previous one."""
    df = pprox.split_trial(make_trial([1.2, 2.0]), two_splits)
    assert split_events(df)[0] == pytest.approx([0.2], abs=1e-6)
    assert split_events(df)[1] == pytest.approx([0.0], abs=1e-6), (
        "event at a later split start goes to that split"
    )
