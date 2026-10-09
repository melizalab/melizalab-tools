# Auditing group-kilo-spikes output

Although carefully following the spike sorting pipeline and depositing data in neurobank helps to avoid surprises with bad data, errors do occur, and it can be helpful to have a way of finding them before they create problems in an analysis or before the raw data is moved to cold storage. `melizalab-tools` includes some audit scripts to help with this process. Each unit's pprox file is checked against its waveform file, against the stimulus messages in the recording's ARF file, and against the other units from the same recording.

Auditing doesn't change any files or neurobank registry entries. Deposited
resources have permanent identifiers and may already have been analyzed, so
reprocessing should only be done when the issues are likely to have
significantly changed results. The [dictionary](#dictionary-of-findings) below
describes each problem and what can be done about it.

Running the audit on a collection of recordings proceeds in three steps, each with its own script:

1. `find-kilo-units`: finds the units of each recording in the registry and
   writes a control file.
2. `audit-kilo-spikes`: audits one recording and writes a JSON report. It is
   run once per line of the control file, usually with GNU parallel.
3. `collect-kilo-audit`: summarizes the reports.

## Requirements

- A version of melizalab-tools that includes these scripts (added after
  release 2026.10.07).
- Read access to the registry. Give it with `-r URL`, or set the
  `NBANK_REGISTRY` environment variable.
- Local access to the archive with the resources.
- GNU parallel, for batch runs.

## 1. Find the units

```bash
find-kilo-units --name P397 -o audit.tsv
```

This searches the registry for `spikes-pprox` and `spikes-hdf5` resources and
groups them by recording. The control file (`audit.tsv`) has one line per
recording: the recording id, a tab, and its units, comma-separated:

```
P397_1_1	P397_1_1_c114,P397_1_1_c357,P397_1_1_c373
```

There are three ways to choose recordings:

- **By name fragment:** `--name P397` matches every resource whose name
  contains `P397`.
- **From a list:** give a file of recording ids, one per line, or `-` to read
  them from standard input. This lets you use any `nbank search`:
  ```bash
  nbank search -d <arf dtype> -k experimenter=<name> | find-kilo-units - -o audit.tsv
  ```
  Pipe only recording names. Any other names in the list are logged as
  having no units.
- **Everything:** `--all` finds every unit in the registry. This is slow,
  so it must be asked for explicitly: with no list, `--name` or `--all`, the
  script exits with an error.

Units are grouped by the names `group-kilo-spikes` gives them:
`<recording>_c<N>` for the pprox and `<recording>_c<N>_spikes` for the
waveform file. Names that don't fit this pattern are logged and skipped.
Recordings that aren't in the registry are also logged and skipped.

Other options:

- `--reports DIR` skips recordings that already have a report in `DIR`
  covering the same units. Use this for incremental runs. A recording with a
  newly deposited unit is audited again.
- `--orphans FILE` writes a second control file, in the same format, listing
  waveform files that have no pprox. These are the candidates for regenerating
  pprox files (see [Regenerating missing pprox files](#regenerating-missing-pprox-files)).
  Recordings in cold storage are included, because regenerating a pprox from
  another unit's trials doesn't need the ARF file.
- Recordings whose ARF file is only in cold storage (e.g. on tape, with no
  `neurobank` or `http(s)` location in the registry) are left out of the
  control file, because their audits couldn't read the ARF file. The script
  logs how many were skipped. `--cold-storage FILE` writes them to a control
  file in the same format, to audit later if they are brought back.

## 2. Run the audits

```bash
mkdir -p reports
parallel --colsep '\t' -a audit.tsv --joblog audit.log --resume \
    'audit-kilo-spikes {1} --units {2} -o reports/{1}.json'
```

Each run writes one report, `reports/<recording>.json`. With `--joblog` and
`--resume`, an interrupted batch can be restarted where it left off.

- **Metadata:** the recording metadata are checked against the registry
  only when one is configured (`-r`, or `NBANK_REGISTRY`). Without one, only
  the ARF's own metadata are checked (see
  [metadata findings](#findings-about-the-recording-metadata)).
- **Exit status:** `audit-kilo-spikes` exits with 0 whenever the audit ran,
  whatever it found. A non-zero exit, shown in `audit.log`, means the audit
  couldn't run, for example because a file was missing or unreadable.
- **Single recordings:** a recording can be audited on its own. Units can be
  pprox files, directories of pprox files, or neurobank ids:
  ```bash
  audit-kilo-spikes P397_1_1 --units P397_1_1_c114,P397_1_1_c357
  audit-kilo-spikes ~/data/P397_1_1.arf --units ~/data/P397_1_1/
  ```
  A local ARF file must be named after its neurobank id, because the audit
  checks that each pprox correctly references that recording.
- **Recordings without messages:** some recordings made with open-ephys GUI
  0.6 or later don't have the stimulus messages in the ARF file. Their units
  get a `messages` finding (info), and their trials aren't checked against
  the stimuli. You can audit them individually with the open-ephys-audio log from the
  same session:
  ```bash
  audit-kilo-spikes P397_1_1 --units ... --oeaudio-log oeaudio_20260617-130556.log -o reports/P397_1_1.json
  ```
  To list the recordings that need a log:
  ```bash
  jq -r 'select(any(.units[].findings[]; .check == "messages")) | .recording' reports/*.json
  ```

### The report

Each report is a JSON object with these fields:

- `recording`: the recording's neurobank id.
- `arf`: the path of the ARF file that was used.
- `audited_by`: the script and its version.
- `registry`: the registry the metadata were checked against, or `null` if
  none was configured (only the ARF's metadata were checked).
- `status`: the worst finding for the recording.
- `findings`: findings about the recording as a whole.
- `units`: one object for each unit, with:
  - `name`, `pprox` and `waveforms` (the files that were used);
  - `processed_by`;
  - `status`;
  - `findings`.

Each finding has a `check` (see the [dictionary](#dictionary-of-findings)), a
`severity`, a `message`, and, for problems in specific trials, `trials`: the
indexes of the trials in the pprox's `pprox` array.

Findings are graded by their effect on analyses of the data:

| Severity | Meaning                                                                                                                                                              |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ok`     | No findings.                                                                                                                                                         |
| `info`   | No effect on analyses. No action needed.                                                                                                                             |
| `warn`   | Specific trials are unreliable, or the files are inconsistent in a way that doesn't change the spike times. The listed trials can usually be excluded from analyses. |
| `fail`   | The unit as a whole is unreliable. Don't use it until the problem is understood.                                                                                     |

## 3. Collect the results

```bash
collect-kilo-audit --control audit.tsv --tsv findings.tsv reports
```

This prints a summary:

- **Status:** recordings and units by status.
- **Checks:** each check and severity, with the number of recordings and
  units it was found in.
- **Versions:** units by the `group-kilo-spikes` version that processed them,
  and their status. Use this to separate the known errors of old versions
  from new problems.
- **Flagged recordings:** each recording with findings at or above `--level`
  (default `warn`), worst first. Each finding is followed by the number of
  units it was found in (`x38`). A finding about the recording as a whole has
  no count.

Options:

- `--tsv FILE` writes every finding as a tab-separated row: recording, unit,
  version, check, severity, trials and message. Use it for anything the
  summary doesn't answer, for example in pandas or `awk`.
- `--control FILE` lists recordings in the control file that have no report,
  meaning their audit couldn't run. See `audit.log` for the reason.
- Unreadable report files are logged and skipped.

To look at one recording's problems in detail:

```bash
jq '.units[] | select(.status == "fail") | {name, findings}' reports/C401_1_1b.json
```

## Fixing problems

None of the scripts modify deposited files, because neurobank resources are supposed to be immutable. Problems can be addressed in three ways, from least to most disruptive:

1. **Exclude trials or units in analyses.** Use this for `warn` findings, with the
   affected trials listed in the report.
2. **Correct the registry metadata** with `nbank modify -k KEY=VALUE <id>`.
   Use this when the registry is wrong but the files are right, or to note a
   known problem on a resource so later users see it.
3. **Reprocess and deposit a new resource.** Examples: regenerating a pprox from
   its waveform file, or rerunning `group-kilo-spikes` with the current version
   to fix the sync. This needs a strong justification, and should be coordinated
   with anyone who has analyzed the original. There is no convention yet for
   superceding old resources, so discuss with Dan if you feel a re-deposit after
   use is needed.

### Regenerating missing pprox files

If a recording is missing one or more pprox files, they can be rebuilt from the corresponding waveform files (`_spikes.h5`), if those exist. **Before you do this**, make sure
you understand why the pprox files are missing; a common pattern is when pprox files from a bad sort get purged but not the corresponding waveform files.

The `regenerate-pprox` script will attempt to regenerate the missing pprox files
using the times in the waveform files and information about the trial structure
in other pprox files from the same recording (if they exist) or from the ARF
file. Use the orphans file. You can set up a parallel job using the orphans file
from `find-kilo-units`:

```bash
find-kilo-units --name P397 -o audit.tsv --orphans orphans.tsv
parallel --colsep '\t' -a orphans.tsv \
    'regenerate-pprox {1} --units {2} -o regenerated'
```

Regenerated files are written to `regenerated/` for you to check and
deposit by hand. The script refuses to overwrite a file that already exists,
and exits with a non-zero status if any unit couldn't be regenerated.

The trial table comes from one of three places:

1. **Another unit of the same recording** (the default), if other units'
   pprox files are found next to local waveform files, or in the registry.
   - If the other units don't all have the same trials, only those processed
     by the same version as the waveform file are used.
   - A table is only used if it reproduces its own unit's events from that
     unit's waveform file.

   This reproduces the original pprox exactly. On P397_1_1, three deposited
   units regenerated this way were identical to the originals apart from
   their provenance.
2. **`--trials PPROX`:** a pprox from the same recording and run, given
   explicitly. Use this when the default can't choose.
3. **`--from-arf`:** the trials made from the ARF file by the current version
   of `group-kilo-spikes`, for recordings with no pprox from the same run.
   - The sync track and prepad are taken from the waveform file (files from
     after 2026.10.07 record them), or manually specified with `--sync` and `--prepad`. Add
     `--oeaudio-log` and `--local-stim-dir` as for `group-kilo-spikes`.
   - The onsets may differ by a few samples from what the original version
     made, or more for pulse sync before 2026.10.07.
   - Aux pulses are not included.

Each regenerated file records how it was generated:

- `derived_from`: the waveform file's neurobank URL, or its path.
- `trials_from`: the pprox the trials came from, or the ARF file.
- `processed_by`: the version that made the waveform file, followed by
  `regenerate-pprox`.

Check the regenerated files with `audit-kilo-spikes` before depositing them.

## Dictionary of findings

Each finding has a `check` name. A check can appear at different severities,
depending on how bad the problem is. In findings that list trials, the trial
numbers are indexes into the pprox's `pprox` array.

| Check                                                           | Severity   | Fixable?  | Remedy                                                  |
|-----------------------------------------------------------------|------------|-----------|---------------------------------------------------------|
| [`trial-tables`](#trial-tables)                                 | warn       | yes       | use the units of one run                                |
| [`versions`](#versions)                                         | info       | —         | none needed                                             |
| [`registry`](#registry)                                         | warn       | yes       | register the recording, or check its id                 |
| [`metadata-arf`](#metadata-arf)                                 | warn       | yes       | correct the registry metadata                           |
| [`metadata-name`](#metadata-name)                               | warn       | yes       | check which recording the file is                       |
| [`schema`](#schema)                                             | warn, info | sometimes | exclude the listed trials; regenerate the pprox         |
| [`pprox-fields`](#pprox-fields)                                 | fail       | sometimes | regenerate the pprox from its waveform file             |
| [`recording-name`](#recording-name)                             | warn, info | yes       | registry metadata, or audit against the right recording |
| [`trial-order`](#trial-order)                                   | warn       | yes       | sort trials when reading                                |
| [`trial-index`](#trial-index)                                   | warn       | yes       | identify trials by offset                               |
| [`events-in-interval`](#events-in-interval)                     | warn       | yes       | drop events outside the interval                        |
| [`spike-count`](#spike-count)                                   | warn       | unknown   | investigate                                             |
| [`recording-field`](#recording-field)                           | info       | —         | none needed                                             |
| [`trial-overlap`](#trial-overlap)                               | warn       | yes       | clip trial intervals                                    |
| [`recording-range`](#recording-range)                           | warn       | partly    | exclude the listed trials                               |
| [`waveforms`](#waveforms)                                       | info       | —         | none needed                                             |
| [`waveforms-recording`](#waveforms-recording)                   | fail       | yes       | registry metadata, or find the right file               |
| [`waveforms-events`](#waveforms-events)                         | fail       | sometimes | regenerate the pprox from its waveform file             |
| [`waveforms-before-first-trial`](#waveforms-before-first-trial) | info       | —         | none needed                                             |
| [`messages`](#messages)                                         | info       | yes       | audit with `--oeaudio-log`                              |
| [`clock-shift`](#clock-shift)                                   | warn       | yes       | note the shift; correct times if the ARF is needed      |
| [`messages-before`](#messages-before)                           | warn, fail | sometimes | exclude trials (warn), re-sync (fail)                   |
| [`stimulus-labels`](#stimulus-labels)                           | fail       | sometimes | re-sync                                                 |
| [`messages-shared`](#messages-shared)                           | fail       | sometimes | re-sync                                                 |
| [`sync-lag`](#sync-lag)                                         | info–fail  | yes       | exclude trials (warn), re-sync (fail); none (info)      |
| [`trials-dropped`](#trials-dropped)                             | info, warn | —         | none needed; check the sync track if many               |
| [`metadata-registry`](#metadata-registry)                       | warn, info | yes       | correct the registry metadata                           |
| [`metadata-unit`](#metadata-unit)                               | warn       | yes       | correct the registry metadata                           |
| [`aux-tracks`](#aux-tracks)                                     | warn       | partly    | the channel can't be checked                            |
| [`aux-fields`](#aux-fields)                                     | warn       | partly    | exclude the listed trials from aux analyses             |
| [`aux-pulses`](#aux-pulses)                                     | warn       | sometimes | exclude the listed trials, or reprocess                 |
| [`aux-stream`](#aux-stream)                                     | warn, info | —         | check the listed trials (warn)                          |

"Re-sync" means rerunning `group-kilo-spikes` with the current version on the
original sort, and depositing the output as new resources (see [Fixing
problems](#fixing-problems)). This needs the kilosort output and a usable sync
track.

### Findings about the recording

#### `trial-tables`

**Severity:** warn

**Problem:** The units don't all have the same trials (onsets, stimuli or
intervals); the message lists the groups of units. Usually the units come from
different runs of `group-kilo-spikes`, for example a rerun deposited next to
the original, or a file was edited.

**Fixable:** yes. Use the `versions` table and each unit's `processed_by` to
work out which run each group comes from. Analyses that compare or pool units
should use units from one run. No reprocessing is needed if one complete run
exists.

#### `versions`

**Severity:** info

**Problem:** The units were processed by different versions. This is often
harmless, but check whether `trial-tables` was also reported.

**Fixable:** nothing to fix on its own.

### Findings about the recording: metadata

These need a registry (`-r`, or `NBANK_REGISTRY`), except where noted. The
audit compares the metadata fields that describe a recording: `bird`, `pen`,
`site`, `hemisphere`, `protocol` and `experimenter`. They are kept in several
places:

- **The recording's registry record:** the current values.
- **The ARF entry attributes:** written by arfx-oephys 2.8.0 and later.
- **oeaudio-present's metadata message:** in the ARF file. It calls the
  protocol `experiment`, and also names the bird (`animal`).
- **Each pprox:** a copy of the registry record, made when the unit was
  processed.

None of these is automatically right: usually the registry is wrong, but
check each case. These findings don't affect spike times, but they do affect
analyses that select or group units by these fields.

#### `registry`

**Severity:** warn

**Problem:** The recording isn't in the registry, so the metadata couldn't be
checked against it. Either the recording id is wrong (the ARF file is named
after something other than its id), or the recording was never registered.

**Fixable:** yes. Audit with the right id, or register the recording.

#### `metadata-arf`

**Severity:** warn

**Problem:** The ARF file and the registry disagree on a field, or, without a
registry (or for a field the registry lacks), the ARF's entry attributes and
metadata message disagree with each other. The message lists each value and
its source. For example, C180_1_1's metadata message gives experimenter
`uac6qw`, but its entry attributes give `bple`.

**Fixable:** yes. Decide which value is right. If it's the ARF's, correct the
registry with `nbank modify -k FIELD=VALUE <recording>`. The ARF file itself
is never edited. If the registry is right, note the error in the recording's
registry metadata.

#### `metadata-name`

**Severity:** warn (checked without a registry too)

**Problem:** The bird named in the ARF file, in the metadata message
(`animal`) or the entry name (e.g. `E79_2026-06-23_...`), isn't the bird in
the recording's id (e.g. `C180_1_1`). Either the ARF file was deposited under
the wrong id, or it was named wrongly when it was recorded.

**Fixable:** yes, once the right recording is known. If the id is wrong, every
unit sorted from the file carries the wrong recording. Correct the registry
metadata, and note it on the units.

### Findings about a unit: the pprox file

#### `schema`

**Severity:** warn (trials listed), or info if the pprox has no `$schema`, or
one that isn't pprox or stimtrial

**Problem:** The pprox doesn't conform to the schema named in its `$schema`
(validated against copies of the published
[pprox](https://meliza.org/spec:2/pprox.json) and
[stimtrial](https://meliza.org/spec:2/stimtrial.json) schemas bundled with
melizalab-tools). The message gives the number of violations and the first
few, with their place in the file (e.g. `pprox[1].events[0]: 'x' is not of
type 'number'`). Tools that rely on the schema may fail on the file. As info:
the pprox isn't validated, because it names no schema or an unknown one.

**Fixable:** sometimes. Exclude the listed trials. If a trial is unusable,
`pprox-fields` is also reported; see there.

#### `pprox-fields`

**Severity:** fail

**Problem:** Some trials lack a field that stimtrial requires (`events`,
`offset`, `interval`, or `stimulus` with `name` and `interval`), or have
non-numeric times, and the unit isn't checked further. Possible causes are an old format, another pipeline, or
a truncated or edited file.

**Fixable:** sometimes. If the waveform file and another unit's pprox from the
same run exist, the pprox can be regenerated from the waveform file and
deposited as a new resource (see
[Regenerating missing pprox files](#regenerating-missing-pprox-files)). Otherwise the unit
needs reprocessing.

#### `recording-name`

**Severity:** warn, or info if the pprox names no recording

**Problem:** The pprox's `recording` field names a different recording from
the one it was grouped with by name. Either the resource is misnamed, or its
`recording` field is wrong. As info: the pprox names no recording (some older
files), so it was matched to the recording by name only.

**Fixable:** yes. If the message checks pass, the unit belongs to this
recording and its `recording` field is wrong: note this in the unit's registry
metadata. If they fail, audit the unit against the recording its pprox names.

#### `trial-order`

**Severity:** warn

**Problem:** The trials are not in time order. This affects analyses that rely
on the order of the array, for example of adaptation. The file was probably
edited or merged.

**Fixable:** yes. Sort the trials by `offset` when reading.

#### `trial-index`

**Severity:** warn

**Problem:** Trial indexes are not unique. The file was probably merged from
several.

**Fixable:** yes. Identify trials by `offset` instead of `index`.

#### `events-in-interval`

**Severity:** warn (trials listed)

**Problem:** The listed trials have events outside their `interval`, which
inflates rates computed over the interval. Seen in the last trial of
C401_1_1b (version 2026.07.15), which held every spike to the end of the
recording, though its interval ended earlier.

**Fixable:** yes, in analyses. Drop the events outside the interval, or
exclude the listed trials.

#### `spike-count`

**Severity:** warn

**Problem:** The pprox has more events than `kilosort_n_spikes`, the cluster's
size in the sort. That shouldn't happen. The likely causes are merged units,
or metadata copied from another cluster.

**Fixable:** unknown until the cause is found. Investigate before using the
unit.

#### `recording-field`

**Severity:** info

**Problem:** The trials have no `recording` sample ranges (files from older
versions), so the unit isn't checked against its waveform file.

**Fixable:** nothing to fix.

#### `trial-overlap`

**Severity:** warn (trials listed)

**Problem:** The listed trials end after the next trial starts.
`group-kilo-spikes` makes trials that abut exactly, so the file came from
elsewhere or was edited. Spikes in the overlap may be counted in both trials.

**Fixable:** yes, in analyses. Clip each trial's interval to the next trial's
start.

#### `recording-range`

**Severity:** warn (trials listed)

**Problem:** For the listed trials, the sample range in `recording` doesn't
match the `offset` and `interval`. Either the onset or the range is wrong,
and the file alone can't tell which.

**Fixable:** partly. Exclude the listed trials, and investigate if there are
many.

### Findings about a unit: the waveform file

#### `waveforms`

**Severity:** info

**Problem:** No waveform file was found. Either it was never deposited (very
old recordings), or it isn't named `<unit>_spikes`. The events can't be
verified, and the pprox can't be regenerated.

**Fixable:** nothing to fix.

#### `waveforms-recording`

**Severity:** fail

**Problem:** The waveform file's `recording` attribute names a different
recording from the pprox.
- If `waveforms-events` was not also reported, the spike times match and
  only the attribute is wrong. For example, C401_1_1b's waveform files say
  E76_1_1b. The data are fine.
- If `waveforms-events` was also reported, the waveform file probably
  belongs to another unit or recording.

**Fixable:** yes. In the first case, note the error in the waveform resource's
registry metadata. In the second, find the correct waveform file; the
registry may have the two resources mixed up.

#### `waveforms-events`

**Severity:** fail (trials listed)

**Problem:** The listed trials' events don't match the spike times in the
waveform file. One of the two files is wrong, or they come from different
runs or sorts. Compare their `kilosort_*` attributes and `processed_by`.

**Fixable:** sometimes. If the waveform file is the right one and the trial
table is good (no message findings), regenerate the pprox from it
(`regenerate-pprox --trials`) and deposit
it as a new resource.

#### `waveforms-before-first-trial`

**Severity:** info

**Problem:** The waveform file has spikes from before the first trial.
Versions before 2026.10.07 kept these in the waveform file, though not in the
pprox. This has no effect on analyses.

**Fixable:** nothing to fix.

### Findings about a unit: metadata

#### `metadata-registry`

**Severity:** warn, or info for a field in only one of the two

**Problem:** The pprox's copy of the recording metadata disagrees with the
recording's current registry record. Usually the registry was corrected after
the unit was processed, or the registry is wrong. As info: a field is in only
one of the two, typically one added to the registry later.

**Fixable:** yes. If the registry is right, analyses should take these fields
from the registry rather than the pprox; reprocessing isn't needed. If the
registry is wrong, correct it with `nbank modify`.

#### `metadata-unit`

**Severity:** warn

**Problem:** The registry record of the unit's own pprox or waveform resource
has a field that disagrees with the pprox. Only fields in both are compared.

**Fixable:** yes. Correct the unit resource's registry metadata with
`nbank modify`.

### Findings about a unit: the stimulus messages

These compare each trial's onset (its sync event) with the start message that
the presentation script sent for its stimulus. Sync events follow their
message by a lag from audio buffering. The lag is typically 0.25–1 s,
depending on the setup. Within a recording it varies by about 30 ms, or is
spread evenly over about 0.2 s with a larger audio buffer.

#### `messages`

**Severity:** info

**Problem:** The trials weren't checked against the messages. Either the ARF
file has no stimulus messages (some recordings made with open-ephys GUI 0.6
or later), or the unit's trials span more than one entry, which isn't
supported yet.

**Fixable:** yes, for missing messages. Audit again with `--oeaudio-log` and
the log from the same session.

#### `clock-shift`

**Severity:** warn

**Problem:** The trials don't fit the messages as they are, but do with a
constant shift: trial *i* follows message *i*+*k* throughout, with the right
label and consistent lags, but the onsets are far from their messages (here,
more than 2 s after or any time before). The trials, and probably the spikes,
are timed from another origin than the ARF file's, as when a recording was
sorted from some time after its start (P388_3_1 from 700 s and P390_3_1 from
500 s, processed by group-kilo-spikes 2025.09.03). The message gives *k* and
the median time from message to onset, which is the shift plus the usual
lag. The other message checks use the messages the shift pairs the trials
with, so `trials-dropped` lists the messages outside the sorted part.

**Fixable:** yes. Analyses locked to the stimulus are unaffected, since the
spikes and onsets share the origin. The absolute times (`offset`,
`recording.start` and `end`) are off relative to the ARF file, which matters
only for analyses that refer back to it (LFP, other channels, other
recordings); correct them by the shift there.

#### `messages-before`

**Severity:** warn if at most half of the trials are listed, fail if more

**Problem:** The listed trials start before any stimulus message, so their
onsets are wrong and their labels can't be checked. The other message checks
still run on the remaining trials.
- A pulse track already high when the recording started was reported as an
  onset at sample 0 by group-klopto-spikes 2026.07.15 (E36_2_1, trial 0).
- If most trials are listed, there is no sync signal or the wrong channel was
  used, or the unit is from another recording (a constant shift is reported
  as `clock-shift` instead).

**Fixable:** sometimes. For a warning, exclude the listed trials. For a
failure, check `recording-name` first; if the recording is right, re-sync.

#### `stimulus-labels`

**Severity:** fail (trials listed)

**Problem:** The listed trials are labeled with a different stimulus from the
last start message before their onset, so their responses are attributed to
the wrong stimulus.
- Versions before 2026.10.07 matched sync events to messages in order, so
  one missed sync event mislabeled every later trial.
- A recording without a working sync line has the same effect, as in
  C401_1_1b.

**Fixable:** sometimes. Re-sync if the recording has a usable sync track.
Mislabeled trials invalidate stimulus-specific analyses, which makes this the
strongest case for reprocessing. Excluding trials is only safe if a few
isolated trials are listed. Without a sync signal, the onsets may be
recoverable from the analog copy of the stimulus (planned).

#### `messages-shared`

**Severity:** fail (trials listed)

**Problem:** Each listed trial follows the same start message as the trial
before it. One stimulus presentation became two trials, from a spurious sync
event or a missing message. This usually comes with `stimulus-labels`.

**Fixable:** as for `stimulus-labels`.

#### `sync-lag`

**Severity:** warn if at most half of the trials are listed, fail if more;
info for a wide spread of lags (no trials listed)

**Problem:** The listed trials' onsets lag their start message by more than
0.1 s outside the range of the other lags, so spike times in those trials are
misaligned with the stimulus. The range is the 5th to 95th percentile of the
lags (given in the message), which allows for recordings whose lags are spread
evenly over ~0.2 s (47 recordings from 2026, e.g. C165_3_1, from a larger
audio buffer; their onsets are right). If that range is wider than 0.3 s, or
there are fewer than 20 trials, the median is used instead.

At info, the lags are spread over more than 0.1 s (5th to 95th percentile),
rather than the usual ~30 ms: the stimulus messages were delayed by varying
amounts, which is worth knowing about on a single machine (e.g. an audio
buffer set larger than needed). The onsets come from the sync track, so this
doesn't affect the trials; `stimulus-labels` still checks the labels.
- Versions before 2026.10.07 reported some pulse onsets at the end of the
  pulse, 1.2–1.5 s late; the message notes this case. For example, trials 0,
  3 and 12 of E36_5_1 were affected in the old run.
- Without a working sync line, the onsets drift (C401_1_1b).

**Fixable:** yes. For a warning, exclude the listed trials. For a failure, the
onsets are unreliable: re-sync if the recording has a usable sync track.

#### `trials-dropped`

**Severity:** info up to 1% of messages (at least one), warn above

**Problem:** Some stimulus messages have no trial. A few are expected:
`group-kilo-spikes` drops a trial whose sync event was missed. A block at the
start or end is a sort of part of the recording (deliberate in C110_1_1,
C122_1_1, P388_4_1, E82_1_1, E82_2_1, P388_3_1 and P390_3_1). Otherwise, many
missing trials suggest a problem with the sync track. Repeated start messages
for the same stimulus within 50 ms (jpresent sent each one twice in E1) are
counted once.

**Fixable:** nothing to fix, since the missing trials are simply absent. For a
warning, check the sync track before trusting the rest of the unit.

### Findings about a unit: aux pulses

These apply only to units processed with `--aux` (see `group-kilo-spikes
--help`): each trial lists the pulses on auxiliary channels, such as an
optogenetic light source, in its `aux` field, and `aux_tracks` names their
channels and, optionally, the jrelay message stream that drives them. The
pulse track is the ground truth: `aux` should record the pulses that were
delivered, whether or not they were commanded.

#### `aux-tracks`

**Severity:** warn (trials listed)

**Problem:** The listed trials have aux pulses, but the pprox has no
`aux_tracks` field naming their channels, so they can't be checked against
the recording.

**Fixable:** partly. The pulses may well be right; if the channel is known,
note it in the unit's registry metadata.

#### `aux-fields`

**Severity:** warn (trials listed)

**Problem:** The listed trials have no `aux` list (a trial without pulses
should have an empty one, so that "no pulses" can be told apart from "not
recorded"), or have pulses without a `name` and `interval`, with a name not
in `aux_tracks`, or that don't start in the trial (each pulse belongs to the
trial in which it starts). The file was probably made or edited by hand.

**Fixable:** partly. Exclude the listed trials from analyses of the aux
pulses.

#### `aux-pulses`

**Severity:** warn (trials listed)

**Problem:** The pulses in the pprox don't match the pulses detected on their
channel in the ARF file: pulses in the pprox that aren't on the channel,
pulses on the channel that are missing from the pprox, or pulses whose end
differs. Also reported if `aux_tracks` gives no channel, or names one that
isn't in the ARF file. Pulses before the first trial are expected to be
missing (group-kilo-spikes drops them).

**Fixable:** sometimes. Exclude the listed trials from analyses of the aux
pulses. If many trials are listed, rerunning group-kilo-spikes with `--aux`
on the original sort gives the right pulses.

#### `aux-stream`

**Severity:** warn for pulses without a message; info for messages without a
pulse, and for pulses out of line

**Problem:** The pulses on the channel don't match the messages on the jrelay
stream named in `aux_tracks` (e.g. `condition`). Each pulse should follow its
message by about the same lag as the sync pulses.

- *A pulse outside every message's window* (warn): a pulse nobody
  commanded, perhaps noise on the line. It is recorded in the trial's `aux`
  as if it were real.
- *A message without a pulse* (info): the device didn't fire, for example
  an LED that failed. `aux` correctly records no pulse, so analyses that use
  `aux` are right, but the trial didn't get the intended condition.
- *A pulse out of line* (info): its lag after the message differs from the
  others by more than 0.1 s.
- *No messages on the stream* (info): the recording has none to check
  against.

**Fixable:** nothing to fix in the files. For a warning, check the listed
trials' pulses on the channel before trusting them.

## Not yet checked

These checks are planned; see `TODO.md`.

- Re-detecting the onsets from the sync track (`--resync`).
- Detecting stimulus onsets when the sync track is missing by cross-correlating an audio copy with the stimuli.
