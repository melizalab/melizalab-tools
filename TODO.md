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
    The new run writes 20 more clusters, 'good' in this copy of the sort but
    never deposited (no pprox or waveform files in the registry). The sort
    directory (phy's only session, 6-17 15:35-15:53) and the 6-22 version of
    the script give no reason to leave them out, so the deposit was probably
    made from a later copy of the sort. Accepted; test_group_spikes_examples.py
    lists them as expected extras.
  - E36_5_1 (jpresent, pulses on ADC3, clicks on ADC5): the reference in
    output/ (from group-klopto-spikes) was made from a different sort, though
    its trial structure matches (1299 of 1300 onsets identical). Instead, the
    version at a20b62a (patched only to map temp_wh.dat read-only) was run on
    the same sort with --sync-thresh 0.5, the only old threshold that finds
    the pulses. Same 66 clusters; waveforms identical; spike assignment
    identical except in the trials around 3 stimuli (0, 3, 12), whose pulses
    the old detector reported at their end (1.2-1.5 s late). All other old
    onsets were 39-294 samples late (typically ~59). The old outputs are kept
    in examples/E36_5_1/output-a20b62a, which test_group_spikes_examples.py
    uses as E36's reference, skipping trials 0, 2, 3, 11 and 12.
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
- [x] Check stimulus lengths against the sync track when splitting trials
  (`kilo.check_stimulus_lengths`, in `arf_to_trials`): the gaps before the
  next onsets, and pulse widths for a pulse track (`kilo.is_pulse_track`:
  median width > 50 ms); see `stimulus-durations` in the audit. Logs each
  trial that doesn't fit, and stops if more than 1% of the stimuli (at least
  one) don't, since the trials would probably be mislabeled. Applies to
  `--oeaudio-log` too, where it checks a log paired in order. Tests in
  tests/test_kilo_arf.py; the excerpt tests now use the real stimulus lengths.
- [ ] Pair sync events with stimuli by order, checked by stimulus length, with
  message times only to locate a missed or spurious sync event.
  `match_sync_events` pairs each sync event with the last message before it,
  which assumes messages arrive before their sync event; messages that
  jittered by more than the gap between stimuli would mislabel trials (or
  raise). The same applies to the audit's `stimulus-labels` check.
- [ ] Sorting part of a recording: merge in the code the student used to sort
  a time window (deliberate exclusions, confirmed 2026-10-09), and record the
  window actually sorted as a top-level annotation in the pprox and waveform
  files, with enough to replicate the exclusion from the ARF file alone: the
  entry (or entries), the start and end in samples from the entry's first
  sample (half-open, [start, end)) as well as in s, the open-ephys sample
  number of the entry's first sample (so message times can be placed), how
  the window was applied (data cut before sorting, or kilosort's
  `tmin`/`tmax`) and the program and version that applied it. Check against
  the student's code which of these it already knows. Deposited examples: P388_3_1 (700 s to ~4500 s) and P390_3_1 (from
  500 s), whose trials and spikes are timed from the window's start; C110_1_1,
  C122_1_1, P388_4_1, E82_1_1, E82_2_1, which end early and keep the
  recording's origin. Decide whether output is timed from the recording's
  start (consistent with the ARF; needs the spike times shifted) or the
  window's start (with the annotation, consumers can convert). The audit and
  regenerate-pprox `--from-arf` should then use the annotation: messages
  outside the window are expected to have no trial, and a shifted origin is
  not a failure (see the clock-shift item under the audit).
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

- [x] Shared function: rebuild a unit's pprox events from its `_spikes.h5`
  times and a trial table (spikes at a trial's start go to the previous trial,
  as in group-kilo-spikes). Checked by hand on all 306 example units (P397,
  C401, E36 old and klopto runs): every trial's events are reproduced exactly
  from the trial table of the unit's own pprox. Pre-trial spikes in old
  waveform files fall outside the trials and are ignored. Done as
  `kilo.waveforms_to_events`, with `kilo.assign_spikes` for the boundary rule
  (also used by group-kilo-spikes).
- [x] `regenerate-pprox RECORDING --units ...` (`dlab/kilo_regenerate.py`):
  rebuilds pprox files from waveform files into a directory, for manual
  deposit. Trials from another unit's pprox of the same recording (checked to
  reproduce that unit's own events; by version if the units disagree), from
  `--trials`, or `--from-arf` (current pipeline; sync track and prepad from
  the waveform file or options; no aux). Adds `derived_from`, `trials_from`
  and its own `processed_by` entry. Three P397 units regenerated from
  siblings are identical to the deposited originals apart from provenance.
- [x] `audit-kilo-spikes RECORDING --units ...` (`dlab/kilo_audit.py`): one
  recording; writes a JSON report. Exit code 0 if the audit ran, whatever it
  found; 1 only if it couldn't run. Findings are graded by their effect on
  analyses already done: info (no effect, e.g. pre-trial spikes in waveform
  files), warn (specific trials unreliable, listed so they can be excluded,
  e.g. a few lag outliers), fail (the unit is unreliable, e.g. stimulus labels
  that don't match the messages, most trials out of line as in C401). Checks:
  - pprox alone: the fields stimtrial requires, trial order, unique indexes,
    events within intervals, event total <= `kilosort_n_spikes`, trials not
    overlapping, `recording` start/end consistent with offset and interval.
  - pprox vs waveforms: events rebuilt from the waveform file match; the
    waveform file names the same recording.
  - pprox vs ARF messages (or `--oeaudio-log`): each trial's onset follows a
    start message for its stimulus, no two trials share a message, lags are
    consistent (`sync_lag_outliers`; noted as expected for versions before
    2026.10.07), at most 1% of messages without a trial.
  - across units of a recording: identical trial tables and versions.
  Results on the examples: P397 (with its log) and E36's klopto output only
  have pre-trial spikes; E36's old a20b62a output warns on trials 0, 3, 12 in
  every unit; C401 fails (mislabeled trials, lag outliers in 108 of 110
  trials, out-of-interval events in the last trial, and waveform files whose
  `recording` is E76_1_1b, although their spikes match the pprox).
- [x] audit-kilo-spikes metadata checks: the recording metadata (bird, pen,
  site, hemisphere, protocol, experimenter) in the registry, the ARF entry
  attributes (arfx-oephys >= 2.8.0) and oeaudio-present's metadata message
  are compared (`metadata-arf`); the bird in the message or entry name against
  the recording id (`metadata-name`); each pprox against the recording's
  registry record (`metadata-registry`) and its own resources' records
  (`metadata-unit`). Registry checks only with a registry. Found: C180_1_1's
  metadata message says experimenter uac6qw, its entry attributes bple.
- [x] audit-kilo-spikes aux checks (units processed with `--aux`): fields
  (`aux-tracks`, `aux-fields`), the pprox pulses against those detected on
  their channel (`aux-pulses`), and the channel's pulses against the stream
  in `aux_tracks` (`aux-stream`; `kilo.match_aux_pulses`, shared with
  group-kilo-spikes). Clean on all 650 E36 LED pulses.
- [x] audit-kilo-spikes schema check (`schema`): each pprox against the
  schema in its `$schema`, using copies of the published schemas bundled in
  dlab/schemas (`pprox.validate`, `pprox.validation_errors`; jsonschema is now
  a runtime dependency).
- [ ] audit-kilo-spikes, still to do:
  - opt-in `--resync`: re-detect onsets on the sync track (from `sync_track`,
    or found by trying channels for older files) and compare per trial.
- Audit of the `induction` archive (2026-10-08; 170 recordings with their ARF
  files on the VM, recorded 2026-01-06 to 2026-09-21; see
  `audit-findings-2026-10.md`). Checked against the analog stimulus copy
  (`stim`, or ADC2 on the 1.0.2 rig): every deposited label is right and every
  onset within ~1 ms. The failures and warnings came from the audit, apart
  from E36_2_1 trial 0. Audit bugs found, now fixed (tests in
  tests/test_kilo_audit.py; the 8 recordings below re-audited as expected):
  - [x] `sync-lag`'s fixed 0.1 s tolerance flagged the tails of message
    jitter. In 47 recordings (open-ephys 0.5.3.1, 2026-01-06 to -12 and
    2026-04-09 to 05-11) the lag is spread evenly over ~0.215 s (median
    0.57-0.67 s, against 0.369 s and a 28 ms spread otherwise), probably from
    a larger audio buffer on the presentation machine; the onsets come from
    the clicks and are right. Every one of the 47 `sync-lag` warnings in
    problems.txt was this. Now `kilo.sync_lag_outliers` (shared with
    group-kilo-spikes) flags lags more than 0.1 s outside the 5th-95th
    percentile range (`kilo.sync_lag_range`), or of the median if that range
    is wider than 0.3 s or there are fewer than 20 lags. C165_3_1, C361_1_1,
    E92_4_1 and P399_4_1 now have only an info finding.
    `test_wide_message_jitter_is_info`, `test_late_onset_with_wide_jitter`.
    Then (2026-10-09) `sync-lag` was made info only, since the lags only
    describe the message timing: the neural data and sync track share a
    clock, so an onset that is the stimulus's sync event is right whatever
    its lag. It reports the median lag and 5th-95th percentile when that
    range is wider than 0.1 s or trials are out of line (listed), to flag
    problems with the presentation setup. The onsets and labels are checked
    by `stimulus-durations` (below) and `stimulus-labels`.
  - [x] `stimulus-durations` (2026-10-09): the messages give the order of the
    stimuli, so the trials are right if each sync event has the right
    stimulus, which the stimulus lengths check. From the pprox alone, the gap
    from the end of each stimulus to the next onset must not be short against
    the recording's other gaps (`kilo.stimulus_length_mismatches`: below the
    lower Tukey fence less 50 ms, or negative; the gap is exactly 1.100 s for
    jpresent and within ~0.35 s for oeaudio-present in the archive, and a
    one-trial label shift breaks 28-96% of trials). With a pulse sync track in
    the ARF (`sync_track`, or the pulse channel rising at the most onsets, at
    least half), each onset must be the rise of a pulse within 10 ms as long as
    its stimulus (to the sample in E36_2_1); this catches the old pulse-end
    error. Warn for at most 1% of trials (at least one), fail above. On 12
    archive recordings only E36_2_1 trial 0 is listed (clicks in the 0.5 and
    1.0 rigs, P397 with labels from the log, E1 clicks, P352 and E36 pulses).
    Tests in tests/test_kilo_audit.py (`test_late_onset`,
    `test_pulse_end_onset_in_old_version`,
    `test_pulse_track_found_without_sync_track`,
    `test_stimulus_durations_*`).
  - [x] A constant clock shift was reported as mislabeling. P388_3_1 and
    P390_3_1 were sorted from 700 s and 500 s into the recording, and
    group-kilo-spikes 2025.09.03 timed their trials (and spikes) from the
    start of the sort: trial i is message i+403 (i+282), with onsets exactly
    700 s (500 s) early; labels right. Now, if the trials don't fit the
    messages as they are, `clock_shift` looks for an offset k at which the
    labels match (95% of trials) and the lags are consistent but far from
    the usual 0-2 s; it is reported as `clock-shift` (warn), and the other
    checks use the messages it pairs the trials with. Both recordings now
    have `clock-shift` and `trials-dropped` (the messages outside the sort).
    The sorted-window annotation (see group-kilo-spikes) would make this
    explicit for new output. `test_clock_shift`,
    `test_clock_shift_with_trials_before_messages`.
  - [x] `messages-before` returned before the other message checks, so
    P390_3_1's summary listed one trial though 2897 of 2908 trials don't
    match their messages. Now the other checks run on the remaining trials.
    `test_messages_before_with_other_checks`.
  - [x] `messages-before` failed a unit for a single trial (E36_2_1 trial 0:
    the pulse track was high when recording started, and
    group-klopto-spikes 2026.07.15 put the onset at sample 0), with an
    explanation that didn't fit. Now warn if at most half the trials are
    listed, fail if more, as for `sync-lag`; the docs give both causes.
    `test_trial_zero_at_recording_start`,
    `test_most_trials_before_messages_fail`.
  - [x] jpresent sent each `start` message twice in E1 (2026-08-12), so
    `trials-dropped` warned that half the messages have no trial. The pairs
    name the same stimulus, usually at the same sample but up to 948 samples
    (32 ms) apart, and one pair is 30 samples out of order, which
    `match_sync_events` would have rejected if E1 were rerun. Now
    `messages_to_events` drops a start message for the still-open event of
    the same name within 1500 samples (50 ms at 30 kHz), keeping the earlier
    time, so group-kilo-spikes is fixed too. E1_1_1 now audits clean.
    `test_duplicate_start_messages`.
  - [x] find-kilo-units stopped with a PermissionError traceback when the
    archive's directories weren't readable (nbank's `resolve_extension`, via
    `local_copy`). Now `local_copy` skips the copy and `unavailable_reason`
    says "in an archive here that this user can't read (check its
    permissions)". `test_local_copy_unreadable_archive`.
  - [ ] Not checked: 311 kilo recordings in cold storage (2024-01 to 2025-12;
    group-kilo-spikes 2023.08.25, 2024.01.29, 2025.09.03). Whether the wide
    jitter or start-trimmed sorts occur there needs their ARF files on this
    host.
- [x] Selection script, `find-kilo-units`: searches the registry for
  `spikes-pprox` and `spikes-hdf5` resources (optionally by `--name`
  fragment, or for the recordings listed in a file or on stdin, e.g. piped
  from `nbank search`), groups them by recording using group-kilo-spikes's names
  (`<recording>_c<N>`, `<recording>_c<N>_spikes`), drops recordings that
  aren't registered, and writes a control file, one line per recording
  (`RECORDING<TAB>UNIT,UNIT,...`), plus (`--orphans`) one of waveform files
  without a pprox, for regenerate-pprox. With `--reports DIR`, recordings
  whose report covers the same units are skipped (incremental runs; the
  registry has no date filter). Run with e.g. `parallel --colsep '\t' -a
  audit.tsv 'audit-kilo-spikes {1} --units {2} -o reports/{1}.json'` on the
  archive host, where the ARF files are local. Since units are grouped by
  name, the audit checks that each pprox names the recording
  (`recording-name`).
  Requires a selection (a list, `--name` or `--all`). Recordings whose ARF isn't in
  a neurobank archive on this host (e.g. cold storage) are skipped, or listed
  with `--unavailable`; their orphans are still listed. The audit scripts
  never download (`find_local`, `no_download=True`).
- [x] `collect-kilo-audit REPORTS...`: summarizes the reports (recordings and
  units by status, findings by check, units by version, and the recordings at
  or above `--level`, worst first). `--tsv` writes one row per finding;
  `--control` lists recordings in the control file without a report (the
  audit couldn't run).

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
- [x] `validate` is implemented, against the bundled schemas (dlab/schemas).
- [ ] `combine_recordings` is not implemented; it raises `NotImplementedError`
  (it was a silent `pass` stub). Implement if needed.

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
