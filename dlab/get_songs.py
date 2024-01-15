# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Extract songs from the neurobank repository to use as stimuli.

This script uses a YAML file to control its behavior. The file needs to have the following format:

- name: O103
  source: O103_song
  dataset: entry_00004/pcm_000
  interval_ms: [90.0, 2185.0]
- name: O84
  source: O84_song
  dataset: entry_00004/pcm_000
  interval_ms: [2970.0, 4922.0]

For each item (denoted by a block starting with `-`), the script will open the
ARF file corresponding to `source` and extract the samples in the interval
`interval_ms` from the dataset denoted by `dataset`. The segments are resampled
to the target sampling rate, passed through a highpass filter, rescaled to the
target amplitude, and then saved in WAVE format. The parameters of the
resampling, filtering, and rescaling are set using commandline options. After
you're happy with the output, use the `--deposit` flag can be used to deposit
the WAVE files in neurobank along with metadata.

"""
import logging
import os
from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import ewave
from dlab import neurobank as nbank
from dlab.signal import Signal, resample, hp_filter, rescale, dBFS

# disable locking - neurobank archive is probably on an NFS share
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
log = logging.getLogger("dlab")


def get_interval(path: Path, dataset: str, interval_ms: Sequence[float]) -> Signal:
    import h5py as h5

    with h5.File(path, "r") as fp:
        dset = fp[dataset]
        sampling_rate = dset.attrs["sampling_rate"]
        start, stop, *rest = (int(t * sampling_rate / 1000) for t in interval_ms)
        data = dset[slice(start, stop)].astype("float32")
        return Signal(signal=data, sampling_rate=sampling_rate)


def script(argv=None):
    import argparse
    import yaml
    from dlab.util import setup_log

    from dlab.core import __version__

    script_version = "2024.01.15"

    p = argparse.ArgumentParser()
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    p.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {script_version} (melizalab-tools {__version__})",
    )
    nbank.add_registry_argument(p)
    p.add_argument(
        "--rate",
        type=int,
        default=44100,
        help="sampling rate for the output files",
    )
    p.add_argument(
        "--dBFS",
        type=float,
        default=-20,
        help="target level (dBFS) for the output files",
    )
    p.add_argument(
        "--highpass",
        type=float,
        help="cutoff frequency for a highpass butterworth filter to apply between resampling and rescaling",
    )
    p.add_argument(
        "--filter-order",
        type=int,
        default=10,
        help="order for the butterworth highpass filter (default %(default)d)",
    )
    p.add_argument(
        "--dtype",
        help="specify data type of the output sound file (defaults to the data type in the arf file",
    )
    p.add_argument(
        "--deposit",
        help="deposit files in neurobank archive (requires write access to registry)",
    )
    p.add_argument("songs", type=Path, help="YAML file with songs to extract")
    args = p.parse_args(argv)
    setup_log(log, args.debug)

    with open(args.songs) as fp:
        songs = yaml.safe_load(fp)

    for song in songs:
        log.info("%s: ", song["source"])
        try:
            path = nbank.find_resource(song["source"])
        except FileNotFoundError:
            path = Path(song["source"])
            if path.exists():
                log.info(" - using local path")
            else:
                log.info(" - unable to find resource")
                continue

        song_data = get_interval(path, song["dataset"], song["interval_ms"])
        log.info(
            f" - loaded {song['dataset']}: {song_data.signal.size} samples, RMS {song_data.dBFS:.2f} dBFS"
        )

        # resample(song_data, args.rate)
        # print(f" - adjusted sampling rate to {song_data['sampling_rate']}")
        # if args.highpass:
        #     hp_filter(song_data, args.highpass, args.filter_order)
        #     print(
        #         f" - highpass with cutoff of {args.highpass}. RMS is now {song_data['dBFS']:.2f} dBFS"
        #     )

        # rescale(song_data, args.dBFS)
        # absmax = np.amax(np.absolute(song_data["signal"]))
        # print(f" - adjusted RMS to {song_data['dBFS']:.2f} dBFS (peak is {absmax:.3f})")

        # out_file = song["name"] + ".wav"
        # dtype = args.dtype or song_data["signal"].dtype
        # with ewave.open(
        #     out_file, mode="w", sampling_rate=song_data["sampling_rate"], dtype=dtype
        # ) as fp:
        #     fp.write(song_data["signal"])
        # print(f" - wrote data to {out_file}")

        # if args.deposit:
        #     metadata = {
        #         "source_resource": song["source"],
        #         "source_dataset": song["dataset"],
        #         "source_interval_ms": song["interval_ms"],
        #         "dBFS": song_data["dBFS"],
        #     }
        #     if args.highpass:
        #         metadata.update(
        #             highpass_cutoff=args.highpass,
        #             highpass_order=args.filter_order,
        #             highpass_filter="butterworth",
        #         )
        #     for res in nbank.deposit(
        #         args.deposit,
        #         (out_file,),
        #         dtype="vocalization-wav",
        #         hash=True,
        #         auto_id=True,
        #         **metadata,
        #     ):
        #         print(f" - deposited in {args.deposit} as {res['id']}")


if __name__ == "__main__":
    script()
