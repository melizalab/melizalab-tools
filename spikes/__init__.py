#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
modules in this package are for extracting and sorting spike data.
Two main tasks:

1. Convert raw waveform data into event times. Spikes
are located with a threshold discriminator; the data points around
each peak are extracted. PCA is used to compress the information
in these raw waveforms, and the feature data can be exported for
analysis in a clustering program, like Klusters or GGobi.

2. Sort spike events by which stimulus was being played at the time.
This stage takes .clu and .fet files from klusters and generates
toe_lis files for each stimulus

Both functions rely heavily on dlab.explog for metadata about stimuli

"""

__all__ = ['extractor','klusters']
