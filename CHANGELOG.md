# Changelog

## 2026.10.14

### group-kilo-spikes

Fixes:

- Sync events are paired with the stimuli in order. Message times are only
  used to place missed sync events, so messages that arrive after their sync
  events, or out of order, no longer cause errors or mislabeled trials.
- The stimuli are checked against the sync track: each stimulus must fit the
  gap to the next onset and, for pulse sync, the width of its pulse. Trials
  that don't fit are dropped with a warning, along with the spikes and aux
  pulses in their intervals. These are usually presentations that jpresent
  stopped early because of a jack xrun; 2026.10.07 kept them, labeled with
  the full stimulus. More than 1% of trials not fitting is an error.
- Recordings whose messages are on another clock from the data (C182_4_1,
  where the acquisition board's sample counter reset during the recording)
  are paired in order with a warning, and the aux checks use the messages
  shifted to line up with the sync events. 2026.10.07 failed with "more than
  one sync event after stimulus".
- Spikes after the end of their trial (in a dropped trial) are dropped from
  the pprox and waveform files.
- The error for a stimulus that can't be found says why (not registered, or
  registered but not downloadable).

New options:

- `--audio-sync CHANNEL` finds the stimulus onsets on a copy of the audio, for
  recordings without a sync track (also `python -m dlab.audio_sync`).
- `--sort-start` and `--sort-stop` for sorts of part of a recording (e.g.
  kilosort's `tmin`/`tmax`). Where temp_wh.dat and the spike times start is
  found from the data and checked against the neural channels; output is
  timed from the recording's start, with only the trials in the sorted part
  (`sorted_window` in the outputs).
- `--alt-stim-names YAML` maps the names of stimuli in the messages to the
  names they were deposited under in neurobank (`stimulus_names` in the
  outputs).
- `--sorter` gives the spike sorter and its version; read from
  `kilosort4.log` for kilosort 4.
- `--debug-http` shows the HTTP requests to the registry (all scripts).

Outputs (pprox and waveform files) now record:

- the options that determine the trials: `sync_track` (or
  `audio_sync_track`), `prepad`, `sync_thresh`, `oeaudio_log`,
  `local_stim_dir`, `aux_tracks`;
- the sort and the unit: `sort` (directory, samples, channels, dtype,
  sampling rate), `sorted_by`, `kilosort_cluster`, `kilosort_group`,
  `artifact_reject_thresh`, `waveform_pre_peak`, `waveform_post_peak`;
- `spike_counts`: the unit's spikes in the sort, and how many were dropped
  as duplicates, before the first trial, after the end of their trial, too
  close to the ends of temp_wh.dat, or as artifacts;
- the mean spike waveform, upsampled and realigned (waveform files).

### New scripts

- `audit-kilo-spikes` checks the output for a recording against its waveform
  files, the stimulus messages and sync track in the ARF file, the registry
  metadata, and the stimtrial schema, and writes a JSON report. It finds the
  errors of earlier versions (see docs/audit.md), including trials of
  presentations stopped early in 2026.10.07 output
  (`stimulus-duration-mismatch`).
- `find-kilo-units` finds the units of recordings in the registry, and
  `collect-kilo-audit` summarizes the reports.
- `regenerate-pprox` rebuilds missing pprox files from their waveform files,
  with the trials of another unit from the same run or from the ARF file.
- pprox files can be validated against the bundled pprox and stimtrial
  schemas (`dlab.pprox.validate`).
