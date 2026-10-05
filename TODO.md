# TODO

Suspected bugs found while writing tests. Current behavior is pinned by tests
whose docstrings start with `PINNED`; when fixing one, update that test.

`tests/test_kilo_pipeline.py` states what should work for each supported
presenter/sync combination (oeaudio-present with clicks or pulses, jpresent
with pulses). Cells that don't work yet are strict xfails; remove the marker
when a fix makes one pass.

## kilo.py

- [x] `oeaudio_log_stims`: a `start` line before `StartAcquisition` raised
  `TypeError`; now a `ValueError` naming the line.
- [x] `entry_metadata` returned `None` when the message dataset has no
  `metadata:` message, as in every jpresent recording (its relay converts jack
  MIDI to zmq messages and sends none). Now returns the entry name and
  sampling rate.
- [x] Missed-sync repair in `match_clicks` was wrong three ways: message
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
- [x] Related: when `match_clicks` returned fewer stimuli than clicks, the
  loop crashed with `AttributeError`. `match_clicks` now returns one stimulus
  per sync event or raises.
- [x] Sync detection did not work for sustained pulses (the z-scored
  quickspikes detector found none in P352 at the default threshold, and found
  them 1.3-7.4 ms late at 0.5). Replaced by `detect_sync_onsets`: rising
  edges through a threshold set as a fraction (default 0.5) of the way from
  baseline to peak, rejecting tracks whose peak is < 20x the baseline noise.
  `--sync-thresh` now takes that fraction; old z-score values are rejected.
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
- [x] `find_stim_dset` only matched `MessageCenter`, the dataset name for GUI
  >= 0.6, so earlier recordings (messages in `Network_Events-..._TEXT_...`)
  needed `--oeaudio-log`. Now matches both, skipping empty datasets (E69 has
  an empty `Message_Center-904.0_TEXT_group_1`).
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

- [ ] `psth` drops the last bin of the interval: `np.arange(start, stop,
  binwidth)` gives the *left* edges, but they are used as `np.histogram` bin
  edges, so with start=0, stop=1 and 0.1 s bins there are 9 bins covering
  0-0.9 s and spikes in 0.9-1.0 s are lost (and a spike exactly at the last
  edge is counted, since numpy closes the final bin). `rate` inherits this.
  (`test_psth_drops_last_bin_of_interval`)
- [ ] Question: with the `exponential` kernel (nonzero only for t < 0), `rate`
  puts each spike's contribution *before* the spike: for a spike at 1.0 s the
  rate is nonzero from 0.53 to 0.99 s and peaks at 0.9 s. A causal smoother
  would spread it after. Flipped kernel? (`test_rate_with_exponential_kernel_precedes_spike`)

- [x] `SpikeWaveforms` docstring said `waveforms` is `(npoints, nspikes)`, but
  `save_waveforms` requires `(nspikes, npoints)` (and the script produces that).
- [x] `rate` docstring said `stop` defaults to the last spike; it is the last
  spike plus one bin (inherited from `psth`).

## get_songs.py

- [ ] `get_interval` converts int16 samples to float32 without scaling to
  +/-1, so the script logs meaningless dBFS values ("RMS 78 dBFS") until it
  rescales. The output level is right. An interval past the end of the data
  is silently truncated.

## plotting.py

- [ ] `simple_axes` docstring says "only bottom and right lines shown"; it
  shows bottom and left.

## signal.py

- [x] `ramp_signal` raised `ValueError` when the ramp rounds to 0 samples
  (`s[-0:]` selects the whole array).
- [x] `ABC_weighting` validated with `curve not in "ABC"`, a substring test, so
  `""` and `"AB"` are accepted and return a meaningless gain.

## util.py

- [ ] `ParseKeyVal` updates the parser's default dict in place, so with
  `default=dict()` (as its docstring suggests) values leak into later parses.
  It should copy. Badly formed arguments also raise `ValueError` instead of an
  argparse usage error. (`test_parse_key_val_shares_mutable_default`)

- [x] `all_same([])` raised `StopIteration`; now returns None.
