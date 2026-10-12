# TODO

Suspected bugs found while writing tests. Current behavior is pinned by tests
whose docstrings start with `PINNED`; when fixing one, update that test.

`tests/test_kilo_pipeline.py` states what should work for each supported
presenter/sync combination (oeaudio-present with clicks or pulses, jpresent
with pulses). Cells that don't work yet are strict xfails; remove the marker
when a fix makes one pass.

Completed items are moved to COMPLETED (not tracked), for release notes.

## Trial splitting and sync (group-kilo-spikes)

- [ ] Spikes are assigned to trials with
  `trials.recording_start.searchsorted(events.time)`, which assumes trial
  starts increase through the file and share the kilosort sample frame. With
  more than one entry, trial positions restart at 0 for each entry. No such
  files exist yet, but they may; fixing it won't change single-entry output.
  Needs: how the kilosort input (temp_wh.dat) is assembled from several
  entries (order, and any gaps between them).
- [ ] Find stimulus onsets on an audio channel without messages (an exhaustive
  search for every stimulus anywhere in the recording), for recordings with
  neither a sync track nor messages (none known). `--audio-sync` needs the
  messages, or a log, to know where to look.

## Outputs (group-kilo-spikes)

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

## Audit

- [ ] Know which errors each version of group-kilo-spikes (and
  group-klopto-spikes) fixed, and grade findings by version: a finding that
  matches a known error of the version that produced the output is explained
  (as now for pulse onsets reported at the pulse's end, via
  `SYNC_FIX_VERSION`), while the same finding in output from a version after
  the fix is unexpected (a new bug, or a file changed after processing) and
  should be more severe. Needs a table of fixes by version in one place,
  instead of the single `SYNC_FIX_VERSION` constant: e.g. 2026.10.07 (pulse
  onsets at the pulse's end; trials after a missed sync event mislabeled;
  spikes before the first trial kept in waveform files) and the next release
  (in-order pairing; stimulus length check; duplicate jpresent start
  messages). Also check the window annotation, once recorded (see Provenance).
- [ ] The audit's `stimulus-mislabeled` still compares each trial's label
  with the last message before its onset, which assumes messages arrive
  before their sync events; group-kilo-spikes now pairs in order (see
  `match_sync_events`). Messages that jittered by more than the gap between
  stimuli would be reported as mislabeling, though `stimulus-duration-mismatch`
  is the dispositive check. None of the archive comes close (largest spread
  ~0.2 s, gaps >= 0.68 s).

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
