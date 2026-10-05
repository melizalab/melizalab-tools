# TODO

Suspected bugs found while writing tests. Current behavior is pinned by tests
whose docstrings start with `PINNED`; when fixing one, update that test.

`tests/test_kilo_pipeline.py` states what should work for each supported
presenter/sync combination (oeaudio-present with clicks or pulses, jpresent
with pulses). Cells that don't work yet are strict xfails; remove the marker
when a fix makes one pass.

## kilo.py

- [ ] `match_clicks`: a stimulus logged before the first click gets
  `stim_onsets[-1]` (the last click), because `searchsorted` returns 0 and
  `idx - 1` wraps. The wrong stimulus is then dropped.
  (`test_match_clicks_stimulus_before_first_click_matches_last_click`)
- [x] `oeaudio_log_stims`: a `start` line before `StartAcquisition` raised
  `TypeError`; now a `ValueError` naming the line.
- [x] `entry_metadata` returned `None` when the message dataset has no
  `metadata:` message, as in every jpresent recording (its relay converts jack
  MIDI to zmq messages and sends none). Now returns the entry name and
  sampling rate.
- [ ] `oeaudio_to_trials` computes `stim_sample_offset` (the recording's first
  sample number) but never uses it. Message `start` values are open-ephys
  sample numbers; click times are sync-track indices. When the counts match
  this doesn't matter, but when `match_clicks` has to repair a missed click,
  every stimulus is compared with the wrong click.
  (`test_trials_missing_click_with_real_sample_numbering_is_an_error`)
  Confirmed in both example recordings: message times include the first
  sample number (48114176 in E69, 2048 in P352).
- [x] Related: when `match_clicks` returned fewer stimuli than clicks, the
  loop crashed with `AttributeError`; now a `RuntimeError`.
- [x] Sync detection did not work for sustained pulses (the z-scored
  quickspikes detector found none in P352 at the default threshold, and found
  them 1.3-7.4 ms late at 0.5). Replaced by `detect_sync_onsets`: rising
  edges through a threshold set as a fraction (default 0.5) of the way from
  baseline to peak, rejecting tracks whose peak is < 20x the baseline noise.
  `--sync-thresh` now takes that fraction; old z-score values are rejected.
  Old-style click onsets move by 0-2 samples (now the first sample over the
  midpoint, rather than the peak). No sync events is now a clear error.
- [ ] `match_clicks` assumes each click comes *before* its start message. In
  both example recordings it comes after: ~0.4 s in E69, ~0.25 s in P352
  (audio buffering). So when a sync event is missed, the wrong stimulus is
  dropped and later trials are silently mislabeled, even with the first sample
  number subtracted (`test_trials_missing_click_mislabels_trials`). Matching
  each sync event to the last message before it would fit the data.
  (`test_clicks_follow_start_messages`, `test_pulses_follow_start_messages`)
- Note: both presenters now require MessageCenter logging, so
  `--oeaudio-log` is only needed for pre-0.6 recordings and a few early 0.6+
  ones recorded without it. The log route assumes open-ephys sample numbers
  count from StartAcquisition (only matters for missed-sync repair).
- [ ] Question: `oeaudio_log_stims` offsets are relative to StartAcquisition,
  which is a third time origin. Does the repair path work with `--oeaudio-log`?
- [ ] Question: `find_stim_dset` only matches `MessageCenter*`, the dataset
  name arfx-oephys uses for GUI >= 0.6. Pre-0.6 recordings
  (`Network_Events-..._TEXT_group_1`) need `--oeaudio-log`. Intended? E69
  also has an empty `Message_Center-904.0_TEXT_group_1`, which doesn't match
  either.
  (`test_find_stim_dset_ignores_pre_0_6_dataset_name`)
- [x] `oeaudio_to_trials` was annotated `-> Iterator[Trial]` but returns a
  list, and never closed the oeaudio log file.

## group-kilo-spikes (end to end)

- [ ] Handle optogenetic stimulation when checking the full pipeline:
  jpresent's `condition_start` / `condition_stop` messages (half of P352's
  stimuli; trials ignore them now) and the separate opto pulse track. Start
  from the existing patch (not yet in the repo).
- [ ] Compare the end-to-end output of `group-kilo-spikes` (pprox and waveform
  files) against known-good results for some example recordings, to be copied
  to examples/.

## pprox.py

- [x] `split_trial` boundary handling was inconsistent (an event exactly at the
  first split's start was dropped; one at a later split's start went to the
  previous split). Splits are now half-open: an event at a split's start
  belongs to that split.
- [ ] `split_trial` casts events to float32 (`dtype="f"`). Fine within a few
  seconds of stimulus onset (~1e-7 s), but worth a decision.
- [x] `split_trial` emitted a pandas `ChainedAssignmentError` FutureWarning
  under some pandas 2.x versions. A false positive: results are the same under
  pandas 3.0.6. Rewritten without augmented assignment anyway.
- [x] `aggregate_events` on an empty collection raised `ValueError`; now
  returns an empty array.
- [ ] `validate` and `combine_recordings` are `pass` stubs. Implement or remove.

## spikes.py

- [x] `SpikeWaveforms` docstring said `waveforms` is `(npoints, nspikes)`, but
  `save_waveforms` requires `(nspikes, npoints)` (and the script produces that).
- [x] `rate` docstring said `stop` defaults to the last spike; it is the last
  spike plus one bin (inherited from `psth`).

## signal.py

- [x] `ramp_signal` raised `ValueError` when the ramp rounds to 0 samples
  (`s[-0:]` selects the whole array).
- [x] `ABC_weighting` validated with `curve not in "ABC"`, a substring test, so
  `""` and `"AB"` are accepted and return a meaningless gain.

## util.py

- [x] `all_same([])` raised `StopIteration`; now returns None.
