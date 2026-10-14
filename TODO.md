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

- [ ] Sorting part of a recording (`--sort-start`/`--sort-stop`, from the
  student's fork, 410767a and 0a9bbce at github.com/bpqle/melizalab-tools):
  done for single-entry recordings, timed from the recording's start, with
  the `sorted_window` annotation and the checks in dlab.kilo_window. Still
  open:
  - how kilosort lays out a sort of a window is found from the data, not
    assumed: temp_wh.dat may start at the window's start or the
    recording's, and so may the spike times (`sorted_data_from`,
    `spike_times_from` in the annotation). What's known (2026-10-14): the
    student used kilosort 2.5's `tmin`/`tmax`, and recalls temp_wh.dat
    being full length with spike times from `tmin` (and patched 2.5 to
    write them from the recording's start). The deposited P388_3_1 and
    P390_3_1 show temp_wh.dat starting at `tmin` too: their waveform
    snippets match the recording only at spike time + `tmin` (median r 0.94
    and 0.91, against 0.00 at the times as given). Kilosort 4 writes
    spike_times.npy from the recording's start (MouseLand/Kilosort#1024;
    only its plots were wrong, fixed in 4.1.6; `save_to_phy` adds `imin`),
    and its temp_wh.dat, written batch by batch from `imin`, probably starts
    at `tmin` (from the 4.1.7 source). Tested only on synthetic sorts in each layout; check
    against a real sort directory from the patched 2.5 or kilosort 4 when
    there is one.
  - `MAX_SORT_PADDING` assumes kilosort 2-3 write temp_wh.dat in whole
    batches of 65600 samples and kilosort 4 no more; not verified against
    kilosort's source or a real sort directory.
  - the checks map temp_wh.dat's columns to probe channels with
    channel_map.npy, and probe channel i to the ARF dataset CH{i+1}. Whether
    phy's `ch` in cluster_info.tsv is a column or a channel id (they differ
    if kilosort dropped channels) is not settled; group-kilo-spikes has
    always used it as a column.
  - the audit should use the annotation: messages outside the window are
    expected to have no trial. For deposited units without it, the audit
    could find the window from the spike times: threshold crossings on the
    unit's channel against its spike times locate the start, and the
    average of the band-passed recording at the spikes confirms it to the
    sample (as kilo_window's spike-average check does). This found P388_3_1
    starting at exactly 700 s (sample 21000000, last spike at 4501.7 s) and
    P390_3_1 at exactly 500 s (to the end of the recording), both timed
    from the window's start; the other deposited examples (C110_1_1,
    C122_1_1, P388_4_1, E82_1_1, E82_2_1) end early and keep the
    recording's origin.
- [ ] The log misstates spike counts (pinned by
  `test_spike_count_log_messages`): "N artifact spike(s) excluded" counts the
  spikes too close to the ends of temp_wh.dat as artifacts too, and "a total
  of N spikes were assigned" counts each unit's spikes before the near-end and
  artifact exclusions. The outputs' `spike_counts` are right.
- [ ] Kilosort 4 writes temp_wh.dat only with `save_preprocessed_copy=True`
  (default False; from the 4.1.7 source), so a default kilosort 4 sort can't
  be run through group-kilo-spikes, which needs it for the waveforms. Its copy
  is drift-corrected, and probably has only the channels in chanMap while
  params.py gives `n_channels_dat` as all the channels in the binary file
  (not checked against a real sort). Check with a real kilosort 4 sort.

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
