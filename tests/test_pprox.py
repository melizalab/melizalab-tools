# -*- coding: utf-8 -*-
# -*- mode: python -*-
import unittest
import logging
from dlab.pprox import _schema, from_trials, groupby, aggregate_events

log = logging.getLogger("dlab")

trials = [
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
]


class TestPprox(unittest.TestCase):
    def test_make_pprox(self):
        pprox = from_trials(trials, test_attribute="blank")
        self.assertEqual(pprox["$schema"], _schema)
        self.assertEqual(pprox["test_attribute"], "blank")
        self.assertSequenceEqual(pprox["pprox"], trials)

    def test_group_by_stim(self):
        pprox = from_trials(trials)
        for stim, group in groupby(pprox, lambda trial: trial["stimulus"]["name"]):
            if stim == "stim1":
                self.assertSequenceEqual(tuple(group), [trials[0], trials[2]])
            elif stim == "stim2":
                self.assertSequenceEqual(tuple(group), [trials[1]])
            else:
                raise ValueError("unexpected stimulus name")

    def test_aggregate_events(self):
        pprox = from_trials(trials)
        all_events = aggregate_events(pprox)
        self.assertEqual(all_events.size, sum(len(t["events"]) for t in trials))
