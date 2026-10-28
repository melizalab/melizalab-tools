# Changelog

## Unreleased

- `prep-oephys` prepares an open-ephys recording for spike sorting, on the
  recording computer: it confirms the metadata with the user (starting from
  what the stimulus presenter logged and the session directory's name), looks
  up the animal in the bird colony, checks the values with the registry, runs
  arfx-oephys, and copies the ARF and the session directory to the sorting
  workstation's share with a manifest (`prep-oephys convert`). Values changed
  from the recording are listed in `revised_after_recording`. Once the ARF is
  deposited, `prep-oephys cleanup <manifest>` removes the ARF and the session
  directory from the recording computer. The continuous channels are stored
  as `EXTRAC_RAW` (wideband), or the type given with `-T`. Needs arfx 3.1.1.
- `dlab.abfio` reads the ABF files in an abfzip archive (an
  `intracellular-abfzip` resource) in place, without unpacking it:
  `abf_members(path)` lists them, and `ZippedAxonIO(path, member)` and
  `ZippedAxonRawIO` are neo's `AxonIO` and `AxonRawIO` for one of them.
- `abf-zip` catalogs every file in the recording's directory with its SHA-1
  hash (`sha1`), not just the ABF files, and records itself in `metadata` as
  `processed_by` (abfzip 1.0, melizalab-specs 0.1.1). `abf-zip build` refuses a
  file that has changed since the catalog was written.
- `group-kilo-spikes` creates the output directory if it doesn't exist.
- `group-kilo-spikes` refuses an ARF file with more than one entry, whose
  spikes it would assign to the wrong trials; run arfx-oephys again to make a
  file with one entry.
- `audit-kilo-spikes` fails every unit of a recording whose ARF file has more
  than one entry (`several-entries`), since group-kilo-spikes put their spikes
  in the wrong trials.
- `--debug` no longer shows h5py's internal debug messages ("Creating converter
  from 3 to 5").
- The log names the unit's channel as the recording does (e.g. `CH26`) for both
  the recorded amplitude and the waveforms; the waveform message used the
  sort's 0-based channel index.
- `get-songs --deposit` works again; it failed on every file with neurobank
  0.13.
- `dlab.neurobank.find_resources` uses neurobank's lookup
  (`nbank.core.find_resources`). Downloads are cached where `nbank` caches
  them (`NBANK_CACHE`, or `neurobank` in the user cache directory), so files
  in the old `dlab` cache are downloaded again once; that cache can be
  deleted. Downloads from a host other than the registry's use that host's
  `~/.netrc` entry. `fetch_resource` and `dlab.cache` are gone.
- `find-kilo-units` lists a recording whose archive can't be read here as "in
  an archive that is missing or not accessible on this host", with the others
  not on this host.

## 2026.10.22

### group-kilo-spikes

- `processed_by` records the provenance of the outputs, oldest first: the
  programs that wrote the ARF file's entries (`entry_creator`, e.g.
  `org.meliza.arfx/arfx-oephys 3.0.2`), then group-kilo-spikes. The waveform
  files hold the same list joined with "; ".
- Fixed a crash for recordings deposited with the `extracellular-arf` reader:
  `processed_by`, `recording` and `presenter` from the recording's registry
  record are no longer copied into the outputs, which have their own.

### regenerate-pprox and audit-kilo-spikes

- The group-kilo-spikes entry is found in `processed_by` by name rather than
  by position.

## 2026.10.21

### group-kilo-spikes

Changes:

- `processed_by` entries are prefixed with `org.meliza.melizalab-tools/`, as
  arfx does (`org.meliza.melizalab-tools/group-kilo-spikes 2026.10.21`).
  Output from earlier versions has no prefix; `audit-kilo-spikes` reads both.
- The outputs record `presenter` (`jpresent` or `oeaudio-present`), found from
  the messages in the ARF file or `--oeaudio-log`, when the messages say.
- The waveform files get the recording's metadata, as the pprox files do.
- `experimenter` and `animal` are taken from the recording's registry record
  itself, which no longer copies them into its metadata.

All spikes, trials and onsets are unchanged.

### regenerate-pprox

- The same changes: the `processed_by` prefix, `presenter` (`--from-arf`), and
  `experimenter` and `animal` from the registry record.

### audit-kilo-spikes

- pprox files are validated against the schemas in melizalab-specs; the copies
  bundled in `dlab/schemas` are gone.
- `processed_by` entries with or without the `org.meliza.melizalab-tools/`
  prefix are graded by version.

### get-songs

- `created_by` in the deposited metadata has the `org.meliza.melizalab-tools/`
  prefix.

### New

- `abf-zip` makes an abfzip archive of the ABF files of an intracellular
  recording (`abf-zip catalog`, then `abf-zip build`); see README.
- `nbank deposit` metadata readers for `spikes-pprox`, `spikes-hdf5` and
  `intracellular-abfzip` resources (entry point `nbank.metadata_readers`)
  register the searchable fields: animal, experimenter, recording, protocol,
  processed_by, sorted_by and presenter.
- `-k key=value` arguments are read as JSON (and `True`, `False`, `None`), as
  nbank reads them.
- neurobank 0.13 is required; melizalab-specs and neo are new dependencies.

## 2026.10.20

### group-kilo-spikes

Fixes:

- Kilosort 2.5 sorts that haven't been saved in phy stopped with "Per-column
  arrays must each be 1-dimensional". They now stop with a message to open
  the sort in phy and save it: phy writes cluster_info.tsv, which the script
  needs, only when the sort is saved. No output was written by the failing
  runs.
- spike_clusters.npy saved as a column vector, as kilosort 2.5 writes it, is
  read. Phy rewrites it as 1-D when the sort is saved, so this matters only
  for sorts whose cluster_info.tsv came from elsewhere.

Outputs are unchanged.

## 2026.10.17

### group-kilo-spikes

Fixes:

- Kilosort 2.5 sorts work again. 2026.10.07 and 2026.10.14 stopped at the
  first good cluster ("arrays used as indices must be of integer (or
  boolean) type"), because kilosort 2.5 saves the spike times as uint64.
  Kilosort 4 sorts weren't affected, and no output was written by the
  failing runs, so nothing deposited needs checking.

New:

- The outputs (pprox and waveform files) record `recorded_amplitude`: the
  unit's peak-to-trough amplitude and its channel's noise (robust SD) in the
  ARF file, band-passed as kilosort filters it, in uV, with the channel and
  the number of spikes averaged (up to 500). Unlike `kilosort_amplitude` and
  the waveforms, it doesn't depend on the sorter. It's measured only for
  recordings with one entry, and skipped with a warning if the unit's
  channel isn't in the ARF file. Measuring it reads the recording around
  each unit's spikes, a few seconds per unit; `--no-recorded-amplitude`
  skips it. All other fields are unchanged.

Note on existing outputs: `kilosort_amplitude` is the template amplitude in
each kilosort version's own scale (thousands for 2.5, tens for 4, for the same
units), and the waveforms are whitened data, scaled by each sorter in its own
way. Neither should be compared across sorters or read as uV. Outputs before
2026.10.14 don't record the sorter; the scale of `kilosort_amplitude` shows
which it was.

### regenerate-pprox

- `recorded_amplitude` is taken from the unit's own waveform file, never from
  the pprox whose trials are used.

### audit-kilo-spikes

- Findings are graded by the version that made the unit (`version_note`): a
  finding that a known error of that version explains says so; one in output
  from a version that fixed every known error for it is one level more
  severe ("unexpected"). See docs/audit.md.
- Onsets are paired with the nearest start message (less the recording's
  typical lag) for `stimulus-mislabeled`, `trials-share-message` and
  `trials-before-messages`, so a message that arrives after its onset no
  longer gives a false finding.
- Trials outside a sorted window (recorded since 2026.10.14) are counted
  separately in `trials-dropped`.
- `collect-kilo-audit` counts the unexpected findings.

### find-kilo-units

- Listing only recordings without units (e.g. after their units were purged)
  no longer fails with "400 Bad Request"; they're reported as having no
  units.

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
