# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Commandline script to extract waveforms from extracellular recordings """
import json
import logging
import nbank
import numpy as np
import h5py as h5
import pandas as pd

log = logging.getLogger("dlab")


if __name__ == "__main__":
    import argparse
    from dlab.core import __version__, get_or_verify_datafile
    from dlab.util import setup_log

    script_version = "2022.06.24"

    p = argparse.ArgumentParser(
        description="extract waveforms from extracellular recording based "
        "on spike times in pprox files. A csv file is generated"
        "for each pprox file"
    )
    p.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {script_version} (melizalab-tools {__version__})",
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    p.add_argument(
        "--num-spikes",
        "-n",
        type=int,
        default=100,
        help="maximum number of spikes to extract",
    )
    p.add_argument(
        "--pre-spike",
        type=float,
        default=2.0,
        help="samples before the spike to keep (default %(default).1f ms)",
    )
    p.add_argument(
        "--post-spike",
        type=float,
        default=5.0,
        help="samples after the spike to keep (default %(default).1f ms)",
    )
    p.add_argument(
        "--upsample",
        type=float,
        default=3,
        help="factor to upsample spikes before aligning them (default %(default)0.1f)",
    )
    p.add_argument(
        "--seed", type=int, default=12345, help="random seed for selecting spikes"
    )
    p.add_argument("recording", help="ARF recording (local file, or neurobank id/URL)")
    p.add_argument(
        "pprox",
        nargs="+",
        type=argparse.FileType("r", encoding="utf-8"),
        help="pprox files with spike times",
    )
    args = p.parse_args()
    setup_log(log, args.debug)

    datafile, arf_info = get_or_verify_datafile(args.recording, args.debug)

    with h5.File(datafile, "r") as arf:
        for spikefile in args.pprox:
            log.info(" - processing spike times in '%s':", spikefile.name)
            spike_data = json.load(spikefile)
            # check that the recording matches the ARF file
            log.info("   - source recording: %s", spike_data["recording"])
            dset_channel = spike_data["kilosort_source_channel"]
            # TODO verify that that index is 1-based
            dset_name = f"CH{dset_channel + 1}"
            log.info("   - main channel: %s", dset_name)
            resource_info = nbank.describe(spike_data["recording"])
            if resource_info["sha1"] != arf_info["sha1"]:
                log.warning(
                    "   - warning: recording does not match the supplied ARF file, skipping"
                )
                continue
            # convert spike times to sample counts (relative to entry start)
            entry_metadata = spike_data["entry_metadata"]
            if len(set(e["sampling_rate"] for e in entry_metadata)) > 1:
                raise ValueError("Not all entries have the same sampling rate")
            sampling_rate = entry_metadata[0]["sampling_rate"]
            dsets = [arf[m["name"]][dset_name] for m in entry_metadata]
            n_before = int(args.pre_spike * sampling_rate / 1000)
            n_after = int(args.post_spike * sampling_rate / 1000)
            spikes = []
            for trial in spike_data["pprox"]:
                entry = trial["recording"]["entry"]
                sampling_rate = entry_metadata[entry]["sampling_rate"]
                for spike in trial["events"]:
                    peak_sample = (
                        int(spike * sampling_rate) + trial["recording"]["start"]
                    )
                    spike_start = peak_sample - n_before
                    spike_end = peak_sample + n_after
                    if spike_start > 0 and spike_end < trial["recording"]["stop"]:
                        spikes.append((entry, spike, spike_start, spike_end))
            spikes = pd.DataFrame.from_records(
                np.array(
                    spikes,
                    dtype=[
                        ("entry", "i4"),
                        ("peak", "f8"),
                        ("start", "i8"),
                        ("stop", "i8"),
                    ],
                )
            )
            nspikes, _ = spikes.shape
            selected = spikes.sample(
                min(args.num_spikes, nspikes), random_state=args.seed
            )
            nspikes, _ = selected.shape

            waveforms = np.zeros((nspikes, n_before + n_after), dtype="f8")
            for i, row in enumerate(selected.itertuples(index=False)):
                waveforms[i, :] = dsets[row.entry][row.start : row.stop]
            # __import__("IPython").embed()

            #     spike_idx = (np.asarray(trial["events"]) * sampling_rate).astype("int64") + trial["recording"]["start"]
            #     nspikes = spike_idx.size
            #     log.debug("    - trial %d: %d spikes from entry %d", trial["index"], nspikes, entry)
            #     spike_times.append(spike_idx)
            #     spike_entries.append(np.ones(nspikes, dtype="int32") * entry)
            # all_spike_entries = np.concatenate(spike_entries)
            # all_spike_times = np.concatenate(spike_times)
            # nspikes = all_spike_times.size

            # TODO sort by entry to avoid looking up the dataset on every spike
            # spike_tbl = pd.DataFrame({"entry":
            # spikes = []
            # for entry, peak in zip(all_spike_entries[selected], all_spike_times[selected]):
            #
            #     s_before = peak - args.pre_spike * entry_metadata[entry]["sampling_rate"]
            #     s_after = peak + args.post_spike * entry_metadata[entry]["sampling_rate"]
            #     spikes.append(dset[s_before:s_after])
