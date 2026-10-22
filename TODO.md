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

## Outputs (group-kilo-spikes)

- [ ] Open questions about sorts of part of a recording
  (`--sort-start`/`--sort-stop`):
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
    there is one (C182_4_1, the only kilosort 4 sort, is of the whole
    recording).
  - `MAX_SORT_PADDING` (65600) assumes kilosort 2-3 write temp_wh.dat in
    whole batches of 65600 samples; not verified. Kilosort 4 pads too, to
    whole batches of `batch_size` (60000 by default; C182_4_1).
  - the checks map temp_wh.dat's columns to probe channels with
    channel_map.npy, and probe channel i to the ARF dataset CH{i+1}. Whether
    phy's `ch` in cluster_info.tsv is a column or a channel id (they differ
    if kilosort dropped channels) is not settled; group-kilo-spikes has
    always used it as a column. C182_4_1 doesn't settle it: its channel map
    is 0-127 in order, so columns and channel ids are the same.
  - the audit uses the annotation's counts of trials outside the window
    (`trials-dropped`). For deposited units without it, the audit could
    find the window from the spike times: threshold crossings on the
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
  is drift-corrected. It has all the channels in the binary file
  (`n_chan_bin`), with the channels not in chanMap all zeros: C182_4_1 (4.1.2)
  has 136 columns, the 128 in its channel map filtered and 8 zero.
- [ ] Reading the waveforms from temp_wh.dat is limited by the disk's random
  reads, not the code. On the VM (C182_4_1: 136 channels, ~35 GB, 16 GB of
  RAM, so nothing stays cached between units) it takes ~7 ms per spike, one
  seek per spike window: 115-145 s for units of 17,000-21,000 spikes, against
  0.1 s for the same read once cached and 0.3 s for `save_waveforms`.
  `MADV_RANDOM` (no readahead) is slower, ~12 ms per spike. It's much faster
  on the bare-metal machine. If it matters on slow storage, read all units'
  windows in one sequential pass (~8.5 min for the whole file at the VM's
  69 MB/s, whatever the number of units); on fast disks that would be slower
  than now.

## Deposited data and the registry

- [ ] Existing klopto outputs (e.g. E36's) have placeholder `led_start` /
  `led_end` values in trials without the LED (sample 0 relative to the
  stimulus, e.g. -3.27 s); analyses must use `led` first.
- [ ] `kilosort_amplitude` in the outputs is the template amplitude in each
  kilosort version's own scale, so it can't be compared across sorters
  (E76_2_1b: median 2128 from 2.5, 27.7 from 4). The whitened waveforms are
  only roughly comparable (both scale whitened noise to SD 200; 13% apart on
  the same neurons). Use `recorded_amplitude` (since 2026.10.17), or
  measure the band-passed ARF: on E76_2_1b, the same neurons have the same
  uV amplitude in both sorts (median ratio 1.00), and the site's "smaller"
  KS4 units were this field. Outputs before 2026.10.14 don't record the
  sorter; the amplitude's scale tells (thousands for 2.5, tens for 4).

## Spike sorting

- [ ] Kilosort 4 against 2.5, on six sites sorted with both (E76_2_1b,
  E76_4_1b, E77_1_1b, E77_3_1b, E79_2_1b, E79_3_1b; KS4 units as deposited,
  KS2.5 re-sorts by the same student, 2026-10-16; KS2.5 sort folders without
  temp_wh.dat in examples/). KS4 kept 427 good units, KS2.5 231. Paired by
  spikes within 0.33 ms:
  - KS2.5 detected the same spikes: 100% of the matched KS4 units' spikes
    are in some KS2.5 cluster (median 100% for the 29 unmatched).
  - 398 KS4 units have a KS2.5 cluster with at least half their spikes. Its
    phy label: good 126 (1.04x the spikes, ISI violations < 1.5 ms 0.00%),
    mua 105 (1.08x, 0.09%; KS2.5 ContamPct ~3%, a stricter curation
    threshold than the KS4 sort's), unlabeled 167 (2.17x, 0.45%; KS2.5's
    own label mua, ContamPct ~37%). The KS4 units have 0.00%.
  - So KS4's gain is mostly neurons KS2.5 merged with multiunit or noise
    spikes, which curation can't recover without splitting by hand. KS4
    rarely over-splits: 19 KS2.5 clusters hold more than one KS4 unit (28
    pairs); 17 pairs have near-coincidences at chance or above (distinct
    neurons), and only E79_3_1b's KS2.5 cluster 12 (KS4 450-452) is clearly
    one neuron split in three. 98 of the 107 KS2.5 good clusters without a
    KS4 partner overlap a KS4 unit already matched to another cluster
    (fragments or duplicates).
  - Units both kept have the same uV amplitude and isolation.
  Decision: the student's dataset stays on KS2.5 (most sites; re-sorting
  dozens isn't worth it). For new sorts, KS4 yields more isolated neurons;
  curate with one written rule (KSLabel, ContamPct threshold), review all
  KS-good clusters, and merge same-channel units whose spike trains never
  come within 1.5 ms. Scripts and tables (pair_sites.py, siblings.py) were
  in the session scratchpad, not kept.
- [ ] Scripts for the sorting workflow, once the new setup has been tried by
  hand a few times. The setup (2026-10-21): grafisia makes the ARF and copies
  it with the raw data to the sorting machine's share; sturnus mounts that
  share; nothing is deposited until the sort is curated. Then one command on
  sturnus: deposit the ARF from the share, run `group-kilo-spikes` against the
  deposited recording and the sort directory, deposit the units from a staging
  directory, and `nbank verify` before the raw copies are deleted. Dry run
  first; stop at the first failure. Possibly a helper on grafisia for the
  copy. Still open: whether to archive the curated sort output.

## Intracellular recordings (abf-zip)

- [ ] Identify the equipment behind the ABF header codes, from the rigs and
  the dates, before the catalog uses names for them (it stores none yet):
  - digitizer (`nDigitizerType`): 8 in every year; 6 alongside it since
    2023. Probably a Digidata 1550 and a 1440 (from memory, unconfirmed).
  - two rigs since 2023: digitizer 8 with channels `Vm_prime/Im_sec/T_bath/
    ValveLink/Camera`, digitizer 6 with `Vm1/Im1_sec/T`. Which rig is which?
    A `rig` field in the metadata needs this mapping.
  - telegraphed instrument (`nTelegraphInstrument`) 24 through 2023,
    probably a MultiClamp 700B; telegraphs off since 2023.
  - telegraph mode 2 (6 files) is cataloged as `"zero current"` (presumed:
    its primary channel is in mV, like current clamp); 0 and 1 agree with
    the channel units.
- [x] Normalize the intracellular-abfdir metadata in the registry before the
  migration (done 2026-10-21; `migrations/README.md`). 6edc858d and f5bbaa66,
  which had no ABF files, were deleted.
- [ ] bad08eac (rallapalli, `20250719_2_2`) holds the files of two sessions,
  2025-07-19 and 07-21 (the second recording wasn't given its own directory).
  Split it and deposit the 07-21 files as their own resource, after the
  migration.

## Recording practice

- [ ] Try a JACK period of 1024 frames (2 periods) on the C182 rig and count
  xruns in a full session (jill_error and the xrun lines in jpresent's log).
  At 512 every C182 session on 2026-10-12 had xruns (1-6 per 72 min; jill
  3.0.0), and jpresent stops the stimulus playing at an xrun (12 of 8520 cut
  short; group-kilo-spikes drops them); 4096 gave none in ~17.5 h (jill 2.2.2
  and 3.0.0rc1, a confound).

## pprox.py

- [ ] `combine_recordings` is not implemented; it raises `NotImplementedError`
  (it was a silent `pass` stub). Implement if needed.
