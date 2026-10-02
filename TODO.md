# TODO

Suspected bugs found while writing tests. Current behavior is pinned by tests
whose docstrings start with `PINNED`; when fixing one, update that test.

## kilo.py

- [ ] `match_clicks`: a stimulus logged before the first click gets
  `stim_onsets[-1]` (the last click), because `searchsorted` returns 0 and
  `idx - 1` wraps. The wrong stimulus is then dropped.
  (`test_match_clicks_stimulus_before_first_click_matches_last_click`)
- [ ] `oeaudio_log_stims`: a `start` line before `StartAcquisition` raises
  `TypeError` (`start_acq_time` is None) instead of a useful error.
  (`test_oeaudio_log_start_before_acquisition_raises_typeerror`)
- [ ] `entry_metadata` returns `None` if the stim dataset has no `metadata:`
  message; `pprox.trial_iterator` would later fail on `None["sampling_rate"]`.
  (not yet tested; tier 2)
- [ ] `oeaudio_to_trials` is annotated `-> Iterator[Trial]` but returns a list,
  and `open(oeaudio_log)` is never closed.

## pprox.py

- [ ] `split_trial` boundary handling is inconsistent: an event exactly at the
  start of the first split is dropped, but one exactly at a later split's start
  is assigned to the *previous* split.
  (`test_split_trial_event_exactly_at_*`)
- [ ] `split_trial` casts events to float32 (`dtype="f"`). Fine within a few
  seconds of stimulus onset (~1e-7 s), but worth a decision.
- [ ] `split_trial` emits a pandas `ChainedAssignmentError` FutureWarning on
  pandas 2.3.x (`df["interval_end"] -= df.stim_begin`). Not checked whether it
  is harmful or a false positive.
- [ ] `aggregate_events` on an empty collection raises `ValueError` from
  `np.concatenate`. Decide whether it should return an empty array.
- [ ] `validate` and `combine_recordings` are `pass` stubs. Implement or remove.

## spikes.py

- [ ] `SpikeWaveforms` docstring says `waveforms` is `(npoints, nspikes)`, but
  `save_waveforms` requires `(nspikes, npoints)` (and the script produces that).
- [ ] `rate` docstring says `stop` defaults to the last spike; it is the last
  spike plus one bin (inherited from `psth`).

## signal.py

- [ ] `ramp_signal` raises `ValueError` when the ramp rounds to 0 samples
  (`s[-0:]` selects the whole array).
- [ ] `ABC_weighting` validates with `curve not in "ABC"`, a substring test, so
  `""` and `"AB"` are accepted and return a meaningless gain.

## util.py

- [ ] `all_same([])` raises `StopIteration`.
