# -*- coding: utf-8 -*-
# -*- mode: python -*-
import unittest
import logging
from dlab.pprox import from_trials, _schema

log = logging.getLogger("dlab")


class TestPprox(unittest.TestCase):

    def test_make_pprox(self):
        pprox = from_trials([], test_attribute="blank")
        self.assertEqual(pprox["$schema"], _schema)
        self.assertEqual(pprox["test_attribute"], "blank")
