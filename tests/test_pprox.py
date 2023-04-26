# -*- coding: utf-8 -*-
# -*- mode: python -*-
import pytest
import logging
import numpy as np
import pandas as pd
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


async def split_fun(name):
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


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_split_trial(anyio_backend):
    for trial in unit["pprox"]:
        stim = trial["stimulus"]["name"]
        split = await pprox.split_trial(trial, split_fun)
        assert split.shape[0] == len(stims[stim]["stim_end"])
        trial_spikes = np.asarray(trial["events"]) + trial["offset"]
        split_spikes = (
            split.apply(lambda x: x.events + x.offset, axis=1).dropna().explode()
        )
        # round to avoid floating point imprecision
        assert np.all(
            np.isin(np.floor(split_spikes * 1000), np.floor(trial_spikes * 1000))
        )


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_split_trial_empty(anyio_backend):
    split = await pprox.split_trial(empty_trial, split_fun)
    stim = empty_trial["stimulus"]["name"]
    split_spikes = split.apply(lambda x: x.events + x.offset, axis=1).dropna().explode()
    assert split.shape[0] == len(stims[stim]["stim_end"])
    assert len(split_spikes) == 0
