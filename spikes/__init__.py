#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
modules in this package are for extracting and sorting spike data.

saber generates pcm_seq2 files with raw data and an explog file that
tells what went on during each episode of the acquisition. For multi-
channel acquisition there is a pcm_seq2 file for each channel. In
each episode (i.e. rtrig event in saber, generally one per stimulus)
the data from each channel is stored in a separate entry in the
associated pcm_seq2 file.

What we want to get to are toe_lis files with spike times for each
presentation of the stimulus. Files should be grouped by recording site
(pen/site commands in saber) and by unit. This is a multi-stage process.

Many of the commands have been made available on the command line
with the program mspikes.py

1. Parse the explog file into a better format, and sort pcm_seq2 files
   according to site. Run:
   mspikes.py --sort <explog>

2. Eliminate unusable episodes. For instance, the animal might
   have moved.  Single events can usually be sorted out; prolonged wiggling
   generates big clusters of noise that can dramatically increase the
   RMS of the signal.  Calculate the RMS of each entry and plot it
   as a time series.  The bad episodes should pop out, even if the amount
   of cellular activity varies greatly.  Use the -t or --start/--stop flags
   in subsequent commands to exclude bad episodes.
   
   mspikes.py --stats -p <pen> -s <site> <explog.h5>

3. Extract spike times and waveforms. This is done with a threshold discriminator.
   It usually makes sense to set the threshold in terms of RMS units above
   the mean, since the DC offset and RMS noise level can shift over the course
   of acquisition.  Setting the threshold is kind of tricky. For single
   units this should be as high as possible; if trying to sort out multiple
   units a lower value is more appropriate. Use the following command
   to examine waveforms:
   mspikes.py --inspect -p <pen> -s <site> <explog.h5>

4. Compute principal components of waveforms. Waveforms need to be upsampled
   and realigned before doing this.  Then calculate the projections of the
   waveforms on the top N principal components.
   mspikes.py --extract ...

5. Sort spikes. There's good software to do this, so I export to the file
   formats used by Klusters

6. Aggregate events into episodes (based on the start/stop times of episodes)
   and combine episodes based on the stimulus.  Output toe_lis files for each
   stimulus and each unit.
   groupevents.py ... <explog.h5>

"""

__all__ = ['extractor','klusters','stat']
