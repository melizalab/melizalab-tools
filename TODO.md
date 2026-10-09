# TODO

Suspected bugs found while writing tests. Current behavior is pinned by tests
whose docstrings start with `PINNED`; when fixing one, update that test.

`tests/test_kilo_pipeline.py` states what should work for each supported
presenter/sync combination (oeaudio-present with clicks or pulses, jpresent
with pulses). Cells that don't work yet are strict xfails; remove the marker
when a fix makes one pass.

Completed items are moved to COMPLETED (not tracked), for release notes.

## Trial splitting and sync (group-kilo-spikes)

- [ ] The audit's `stimulus-mislabeled` still compares each trial's label
  with the last message before its onset, which assumes messages arrive
  before their sync events; group-kilo-spikes now pairs in order (see
  `match_sync_events`). Messages that jittered by more than the gap between
  stimuli would be reported as mislabeling, though `stimulus-duration-mismatch`
  is the dispositive check. None of the archive comes close (largest spread
  ~0.2 s, gaps >= 0.68 s).
- [ ] Spikes are assigned to trials with
  `trials.recording_start.searchsorted(events.time)`, which assumes trial
  starts increase through the file and share the kilosort sample frame. With
  more than one entry, trial positions restart at 0 for each entry. No such
  files exist yet, but they may; fixing it won't change single-entry output.
  Needs: how the kilosort input (temp_wh.dat) is assembled from several
  entries (order, and any gaps between them).
- [ ] Rescue stimulus onsets in recordings without a sync track by
  cross-correlating the stimulus files with the ADC channel that records an
  analog copy of the audio sent to the speaker. Example: examples/C401_1_1b
  (no sync line connected; ADC1 or ADC2 look like audio). Its reference
  outputs can't be used for A/B comparison: their onsets are exactly 31.765 s
  apart, drift to 41.5 s before the start messages by the end, and don't
  match any channel in the ARF.

## Provenance in the outputs (group-kilo-spikes)

- [ ] Sorting part of a recording: merge in the code the student used to sort
  a time window (deliberate exclusions, confirmed 2026-10-09), and record the
  window actually sorted as a top-level annotation in the pprox and waveform
  files, with enough to replicate the exclusion from the ARF file alone: the
  entry (or entries), the start and end in samples from the entry's first
  sample (half-open, [start, end)) as well as in s, the open-ephys sample
  number of the entry's first sample (so message times can be placed), how
  the window was applied (data cut before sorting, or kilosort's
  `tmin`/`tmax`) and the program and version that applied it. Check against
  the student's code which of these it already knows. Deposited examples:
  P388_3_1 (700 s to ~4500 s) and P390_3_1 (from 500 s), whose trials and
  spikes are timed from the window's start; C110_1_1, C122_1_1, P388_4_1,
  E82_1_1, E82_2_1, which end early and keep the recording's origin. Decide
  whether output is timed from the recording's start (consistent with the
  ARF; needs the spike times shifted) or the window's start (with the
  annotation, consumers can convert). The audit and regenerate-pprox
  `--from-arf` should then use the annotation: messages outside the window
  are expected to have no trial, and a shifted origin is not a problem (now
  inferred by the audit's `clock-shift`).
- [ ] Store the other parameters that determine group-kilo-spikes output in
  the pprox (and waveform files where they apply), so a unit can be
  reproduced and audited without the command line. Not recorded now:
  - `artifact_reject_thresh`, which decides which spikes are dropped as
    artifacts, and the counts of spikes dropped (as artifacts, too close to
    the ends of temp_wh.dat for a waveform, as duplicate times, and before
    the first trial), so a unit's spike total can be reconciled with
    `kilosort_n_spikes`;
  - `waveform_pre_peak`/`waveform_post_peak` in the pprox (implicit in the
    waveform file's shape and `peak_index`, but they also decide which
    spikes are too close to the ends);
  - the cluster id and its phy group (`good`, or `mua` with `--mua`), now
    only implied by the file name and the option;
  - the sort: the sort directory's name, `params.py` (dtype, channel count,
    sampling rate) and the number of samples in temp_wh.dat. The sample
    count also shows whether a window was sorted (see the item above);
  - `--local-stim-dir`, when stimulus durations came from local files
    instead of neurobank.
  The trial options already recorded (`sync_track`, `prepad`, `sync_thresh`,
  `oeaudio_log`, `aux_tracks`) stay as they are; the schema allows extra
  top-level fields.

## Deposited data and the registry

- [ ] The neurobank resource "trials" is the only stimtrial file without
  `interval`; probably a dummy that should be removed.
- [ ] Existing klopto outputs (e.g. E36's) have placeholder `led_start` /
  `led_end` values in trials without the LED (sample 0 relative to the
  stimulus, e.g. -3.27 s); analyses must use `led` first.

## Recording practice

- [ ] Remind experimenters to shorten the JACK period and audio driver/card
  buffers: TTL pulses and sync events follow the jrelay messages by 0.2-0.3 s
  (up to 1 s on some setups), which loosens message-to-pulse matching. The
  audit's `message-jitter` finds sessions where the lags were spread over
  ~0.2 s (summer 2025 on the kiaa0319 rig; 2026-01-06 to -12 and 2026-04-09
  to 05-11 on the induction rig).

## pprox.py

- [ ] `combine_recordings` is not implemented; it raises `NotImplementedError`
  (it was a silent `pass` stub). Implement if needed.
