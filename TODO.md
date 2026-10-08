# TODO

Suspected bugs found while writing tests. Current behavior is pinned by tests
whose docstrings start with `PINNED`; when fixing one, update that test.

`tests/test_kilo_pipeline.py` states what should work for each supported
presenter/sync combination (oeaudio-present with clicks or pulses, jpresent
with pulses). Cells that don't work yet are strict xfails; remove the marker
when a fix makes one pass.

## neurobank.py

- [x] `find_resources` downloaded over https without credentials, so
  resources that need a login failed with 403; its client now uses
  `default_auth` (from ~/.netrc).

## kilo.py

- [x] `group_spikes_script` memory-mapped temp_wh.dat copy-on-write, which
  fails with ENOMEM for a 28-43 GB sort on a machine with less memory. Now
  read-only, with waveform windows taken by indexing (same values as
  `qs.peaks`, which needs a writable array; checked identical on P397).
  quickspikes is no longer used by dlab but is still a dependency.

- [x] `oeaudio_log_to_stimuli`: a `start` line before `StartAcquisition` raised
  `TypeError`; now a `ValueError` naming the line.
- [x] `entry_metadata` returned `None` when the message dataset has no
  `metadata:` message, as in every jpresent recording (its relay converts jack
  MIDI to zmq messages and sends none). Now returns the entry name and
  sampling rate.
- [x] Missed-sync repair in `match_sync_events` was wrong three ways: message
  times (open-ephys sample numbers, which include the recording's first
  sample number: 48114176 in E69, 2048 in P352) were compared with sync-track
  indices; each sync event was assumed to precede its message, but it follows
  it (~0.4 s in E69, ~0.25 s in P352, from audio buffering), so trials after
  a missed sync event were silently mislabeled; and a message before the
  first sync event wrapped to the last one. Now message times are converted
  to sync-track samples, and each sync event is matched to the last message
  at or before it; this runs even when the counts agree. A stimulus with no
  sync event is dropped with a warning; a sync event before any message, or
  two after the same message, is a ValueError.
- [x] Related: when `match_sync_events` returned fewer stimuli than clicks, the
  loop crashed with `AttributeError`. `match_sync_events` now returns one stimulus
  per sync event or raises.
- [x] Sync detection did not work for sustained pulses (the z-scored
  quickspikes detector found none in P352 at the default threshold, and found
  them 1.3-7.4 ms late at 0.5). Replaced by `detect_sync_onsets`: rising
  edges through a threshold set as a fraction (default 0.5) of the way from
  baseline to peak, rejecting tracks whose peak is < 20x the baseline noise.
  (`--sync-thresh` was later changed to an absolute override; see below.)
  Old-style click onsets move by 0-2 samples (now the first sample over the
  midpoint, rather than the peak). No sync events is now a clear error.
  Baseline and noise are estimated from ~1e6 samples: estimating them from
  the whole track needed several GB for C180's 1.5 h recording.
- Note: `--oeaudio-log` is only needed for the few GUI 0.6+ recordings made
  without MessageCenter logging (both presenters now require it). Earlier GUI
  versions kept the messages in the Network Events dataset.
- [x] The `--oeaudio-log` route assumes open-ephys sample numbers count from
  StartAcquisition, so log times can be converted like message times.
  Confirmed with E79_1_1b.arf and its log (`TestPairedLog`): the log's
  messages match the recording's MessageCenter exactly, log times are 40-70 ms
  after the corresponding sample numbers, clicks follow logged starts by
  ~0.36 s, and both routes give identical trials.
- [x] `find_message_dset` only matched `MessageCenter`, the dataset name for GUI
  >= 0.6, so earlier recordings (messages in `Network_Events-..._TEXT_...`)
  needed `--oeaudio-log`. Now matches both, skipping empty datasets (E69 has
  an empty `Message_Center-904.0_TEXT_group_1`).
- [x] `arf_to_trials` was annotated `-> Iterator[Trial]` but returns a
  list, and never closed the oeaudio log file.

## group-kilo-spikes (end to end)

- [x] Optogenetic stimulation, in place of group-klopto-spikes
  (github.com/bpqle/melizalab-tools, branch patch): `--aux NAME=CHANNEL`
  records the pulses on an auxiliary channel in each trial's `aux` list
  (`{"name", "interval"}` relative to stimulus onset, assigned to the trial
  where they start, unclipped) and the channel in `aux_tracks`. On E36 this
  matches klopto's `opto` field on all 1300 trials, to within one sample.
- [x] Publish the specifications from melizalab/lab_specs (branch
  aux-and-2020-12): pprox 1.0 as published (interval optional; schema in JSON
  Schema 2020-12, with fixes that don't change which documents are valid) and
  stimtrial 1.0 (requires `interval` and `stimulus`; optional `aux` and
  `aux_tracks`), matching the ~37,500 stimtrial files in neurobank. Copies of
  the schemas are in tests/data/schemas, and test_stimtrial_schema.py checks
  that group-kilo-spikes output conforms; update them if the specs change.
- [ ] The neurobank resource "trials" is the only stimtrial file without
  `interval`; probably a dummy that should be removed.
- [x] Cross-check aux pulses against jrelay messages: start/stop messages on
  every stream (stimulus, `trial_`, `condition_`, `channelN_`) are parsed by
  `messages_to_events`. With `--aux NAME=CHANNEL:STREAM`, each pulse is checked
  against that stream's messages (`check_aux_pulses`): a message without a
  pulse, a pulse outside every message's window, or a pulse whose lag is out of
  line is logged. On all of E36 the 650 condition messages match the 650 LED
  pulses (lag 0.266 s).
- [ ] Remind experimenters to shorten the JACK period and audio driver/card
  buffers: TTL pulses and sync events follow the jrelay messages by 0.2-0.3 s
  (up to 1 s on some setups), which loosens message-to-pulse matching.
- [ ] Existing klopto outputs (e.g. E36's) have placeholder `led_start` /
  `led_end` values in trials without the LED (sample 0 relative to the
  stimulus, e.g. -3.27 s); analyses must use `led` first.
- [x] Compare the end-to-end output of `group-kilo-spikes` (pprox and waveform
  files) against known-good results (`test_group_spikes_examples.py`):
  - P397_1_1 (oeaudio-present, clicks on ADC3, `--oeaudio-log`; reference from
    version 2026.06.22): all 99 reference pprox files match apart from onsets
    moving 0-3 samples earlier and 14 spikes, each within 2 samples of a trial
    boundary, moving to the adjacent trial as a result; waveforms identical.
    But the new run writes 20 more clusters, all 'good' in a cluster_info.tsv
    older than the reference. Were only some clusters deposited?
  - E36_5_1 (jpresent, pulses on ADC3, clicks on ADC5): the reference in
    output/ (from group-klopto-spikes) was made from a different sort, though
    its trial structure matches (1299 of 1300 onsets identical). Instead, the
    version at a20b62a (patched only to map temp_wh.dat read-only) was run on
    the same sort with --sync-thresh 0.5, the only old threshold that finds
    the pulses. Same 66 clusters; waveforms identical; spike assignment
    identical except in the trials around 3 stimuli (0, 3, 12), whose pulses
    the old detector reported at their end (1.2-1.5 s late). All other old
    onsets were 38-300 samples late (typically ~50). A one-off check: the old
    outputs are not kept in examples/.
- [x] The artifact check calls `input()` ("Press any key to continue") when
  more than half a cluster's spikes look like artifacts. Kept deliberately.
- [x] When the recording isn't a local file, it was looked up with
  `nbank.default_registry`, ignoring `--registry`; now uses `--registry`.
- [x] `processed_by` (and get-songs' `created_by`) used argparse's `prog`,
  which depended on how the script was invoked; `prog` is now set explicitly.
- [x] Spikes within 2 ms of the start or 5 ms of the end of the recording are
  dropped from the pprox as well as the waveforms: intended, to keep the two
  in sync. (`test_spikes_too_close_to_edges_are_dropped`)
- [x] Spikes before the first trial were dropped from the pprox but kept in
  the waveforms; now dropped from both, to keep them in sync.
- [ ] Spikes are assigned to trials with
  `trials.recording_start.searchsorted(events.time)`, which assumes trial
  starts increase through the file and share the kilosort sample frame. With
  more than one entry, trial positions restart at 0 for each entry. No such
  files exist yet, but they may; fixing it won't change single-entry output.
  Needs: how the kilosort input (temp_wh.dat) is assembled from several
  entries (order, and any gaps between them).

- [x] Record the options that determine the trials (`sync_track`, `prepad`,
  and `sync_thresh`/`oeaudio_log` when given) in the pprox and waveform files,
  so trials can be reconstructed without the command line.
- [ ] Rescue stimulus onsets in recordings without a sync track by
  cross-correlating the stimulus files with the ADC channel that records an
  analog copy of the audio sent to the speaker. Example: examples/C401_1_1b
  (no sync line connected; ADC1 or ADC2 look like audio). Its reference
  outputs can't be used for A/B comparison: their onsets are exactly 31.765 s
  apart, drift to 41.5 s before the start messages by the end, and don't
  match any channel in the ARF.
- [x] Sync detection accepted channels without a sync signal (e.g. C401's
  floating ADC5/ADC6 gave >120,000 events). Now, by default, the threshold is
  set automatically (halfway from baseline to peak, with the noise check),
  and it is an error if there are more sync events than stimuli or more than
  1% of stimuli (at least one) have none: this catches the wrong channel, a
  recording without a sync signal, or a bad threshold. `--sync-thresh` now
  overrides the automatic threshold with an absolute level, in the channel's
  units (it skips the noise check, but not the count checks). Aux channels
  always use the automatic threshold.

## Audit and regeneration (planned)

Goal: find likely errors in deposited data, without changing the registry. A
re-sort or re-sync must be strongly justified, as deposited resources have
permanent identifiers and may already have been analyzed. Registry dtypes:
`spikes-pprox`, `spikes-hdf5` (waveforms; deposited for all but very old
recordings, which are out of scope).

- [ ] Shared function: rebuild a unit's pprox events from its `_spikes.h5`
  times and a trial table (spikes at a trial's start go to the previous trial,
  as in group-kilo-spikes). Checked by hand on all 306 example units (P397,
  C401, E36 old and klopto runs): every trial's events are reproduced exactly
  from the trial table of the unit's own pprox. Pre-trial spikes in old
  waveform files fall outside the trials and are ignored.
- [ ] `regenerate-pprox RECORDING --units ...`: one recording. Writes pprox
  files for units whose pprox is missing, to a directory, for manual deposit.
  Trial table from another pprox of the same recording and run (exact), or
  else from the ARF with the current pipeline (needs the sync track and
  prepad unless recorded in the waveform file; onsets may differ from the
  original version). Metadata: `kilosort_*`, `recording` and the original
  `processed_by` from the waveform file; cluster id from the name;
  `entry_metadata` from the ARF; unit metadata from the registry. Adds a
  `derived_from` field naming the waveform resource and appends its own
  `processed_by`.
- [ ] `audit-kilo-spikes RECORDING --units ...`: one recording; writes a JSON
  report. Exit code 0 if the audit ran, whatever it found; non-zero only if it
  couldn't run. Findings are graded by their effect on analyses already done:
  info (no effect, e.g. pre-trial spikes in waveform files, onset shifts <= 3
  samples, an old version), warn (specific trials unreliable, listed so they
  can be excluded, e.g. a few lag outliers), fail (the unit is unreliable,
  e.g. stimulus labels that don't match the messages, most trials out of line
  as in C401). Only a fail would make a case for re-syncing; re-sorting is
  out of scope. Checks:
  - pprox alone: stimtrial schema, trials ordered and non-overlapping, events
    within intervals, unique indexes, event total <= `kilosort_n_spikes`.
  - pprox vs waveforms: events rebuilt from the waveform file (see above)
    match.
  - pprox vs registry: the pprox's unit metadata (bird, pen, site, protocol,
    experimenter, ...) against the registry metadata for the pprox and its
    recording. Usually the registry is wrong; reported for a case-by-case
    decision.
  - pprox vs ARF messages (cheap): stimulus names an in-order subsequence of
    the start messages, at most 1% dropped, `sync_lag_outliers` (see
    `pprox_lag_outliers` in test_kilo_examples.py, which flags E36's
    end-of-pulse trials in the old version and C401's drift), `recording`
    start/end consistent with offset and interval, aux pulses against their
    stream.
  - across units of a recording: identical trial tables and versions.
  - opt-in `--resync`: re-detect onsets on the sync track (from `sync_track`,
    or found by trying channels for older files) and compare per trial.
  Each finding carries the unit's `processed_by` version, so known errors of
  old versions are labeled as such.
- [ ] Selection script (separate): queries the registry and writes a control
  file, one line per recording (`RECORDING<TAB>UNIT,UNIT,...`), plus orphans
  (waveform files without a pprox, for regenerate-pprox; pprox without a
  waveform file). Run with e.g. `parallel --colsep '\t' --joblog audit.log
  --resume -a control.tsv 'audit-kilo-spikes {1} --units {2} -o
  reports/{1}.json'` on the archive host, where the ARF files are local.

## pprox.py

- [x] `split_trial` boundary handling was inconsistent (an event exactly at the
  first split's start was dropped; one at a later split's start went to the
  previous split). Splits are now half-open: an event at a split's start
  belongs to that split.
- [x] `split_trial` casts events to float32 (`dtype="f"`). Decided to keep:
  trial events are relative to the trial (seconds), where float32 resolves
  ~1e-7 s, far below one 30 kHz sample. Not suitable for times referenced to
  the start of a long recording (~0.5 ms at 2 h).
- [x] `split_trial` emitted a pandas `ChainedAssignmentError` FutureWarning
  under some pandas 2.x versions. A false positive: results are the same under
  pandas 3.0.6. Rewritten without augmented assignment anyway.
- [x] `aggregate_events` on an empty collection raised `ValueError`; now
  returns an empty array.
- [ ] `validate` and `combine_recordings` are not implemented; they now raise
  `NotImplementedError` (they were silent `pass` stubs). Implement if needed.

## spikes.py

- [x] `psth` dropped the last bin of the interval, and spikes on bin edges
  (common, since spike times are multiples of the sampling interval) could
  land in the wrong bin through floating-point error in `np.arange` edges:
  for a spike at every 30 kHz sample over 1 s in 1 ms bins, it returned 999
  bins with counts of 29-31 and lost 29 spikes. Bins are now assigned by
  index with a 1e-9 bin tolerance, all half-open, as many whole bins as fit
  in [start, stop).
- [x] The `exponential` kernel (in `signal.smoothing_kernel`) was nonzero
  only for t < 0, so with `rate` each spike raised the rate *before* it
  (anti-causal). It is now causal by default: zero for t <= 0, peaking at
  +bandwidth.

- [x] `SpikeWaveforms` docstring said `waveforms` is `(npoints, nspikes)`, but
  `save_waveforms` requires `(nspikes, npoints)` (and the script produces that).
- [x] `rate` docstring said `stop` defaults to the last spike; it is the last
  spike plus one bin (inherited from `psth`).

## get_songs.py

- [x] `get_interval` converted int16 samples to float32 without scaling to
  +/-1, so the script logged meaningless dBFS values before rescaling (output
  levels were right); now scaled with `ewave.rescale` (float samples, what our
  software stores, are unchanged). An interval past the end of the data was
  silently truncated; now an interval that is empty or not entirely inside
  the dataset raises ValueError.

## plotting.py

- [x] `simple_axes` docstring said "only bottom and right lines shown"; it
  shows bottom and left.

## signal.py

- [x] `ramp_signal` raised `ValueError` when the ramp rounds to 0 samples
  (`s[-0:]` selects the whole array).
- [x] `ABC_weighting` validated with `curve not in "ABC"`, a substring test, so
  `""` and `"AB"` are accepted and return a meaningless gain.

## util.py

- [x] `ParseKeyVal` updated the parser's default dict in place, so values
  leaked between parses; now copies it. Badly formed arguments are an
  argparse usage error, and `__call__` matches `argparse.Action` (ty
  invalid-method-override).
