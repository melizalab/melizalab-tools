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
from collections.abc import Sequence
from pathlib import Path

import ewave
import numpy as np

from dlab import neurobank as nbank
from dlab.signal import Signal, hp_filter, resample, rescale

# disable locking - neurobank archive is probably on an NFS share
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
log = logging.getLogger("dlab")


def get_interval(path: Path, dataset: str, interval_ms: Sequence[float]) -> Signal:
    import h5py as h5

    with h5.File(path, "r") as fp:
        dset = fp[dataset]
        sampling_rate = dset.attrs["sampling_rate"]
        start, stop, *_rest = (int(t * sampling_rate / 1000) for t in interval_ms)
        data = dset[slice(start, stop)].astype("float32")
        return Signal(samples=data, sampling_rate=sampling_rate)


def script(argv=None):
    import argparse

    import yaml

    from dlab import __version__
    from dlab.util import setup_log

    script_version = "2025.07.28"

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        help="sampling rate for the output files (default: %(default)d Hz)",
    )
    p.add_argument(
        "--dBFS",
        type=float,
        default=-20,
        help="target amplitude for the output files (default: %(default)s dBFS)",
    )
    p.add_argument(
        "--highpass",
        type=float,
        default=300,
        help="cutoff frequency for a highpass butterworth filter to apply between resampling and rescaling (default: %(default)s Hz)",
    )
    p.add_argument(
        "--filter-order",
        type=int,
        default=10,
        help="order for the butterworth highpass filter (default: %(default)d)",
    )
    p.add_argument(
        "--dtype",
        default="int16",
        help="specify data type of the output sound file (default: %(default)s)",
    )
    p.add_argument(
        "--deposit",
        type=Path,
        help="deposit files in neurobank archive (requires write access to registry)",
    )
    p.add_argument("songs", type=Path, help="YAML file with songs to extract")
    args = p.parse_args(argv)
    setup_log(args.debug)

    with open(args.songs) as fp:
        songs = yaml.safe_load(fp)

    for song in songs:
        log.info(f"{song['source']}:")
        try:
            path = nbank.find_resource(song["source"], registry_url=args.registry_url)
        except FileNotFoundError:
            path = Path(song["source"])
            if path.exists():
                log.debug(" - using local path")
            else:
                log.warning(" - unable to find resource, skipping")
                continue
        log.info(" - using %s", path)

        song_data = get_interval(path, song["dataset"], song["interval_ms"])
        song_data.name = song["name"]
        log.info(
            f" - read from dataset {song['dataset']}: {song_data.samples.size} samples, RMS {song_data.dBFS:.2f} dBFS"
        )

        song_data = resample(song_data, args.rate)
        log.info(f" - adjusted sampling rate to {song_data.sampling_rate}")
        if args.highpass:
            song_data = hp_filter(song_data, args.highpass, args.filter_order)
            log.info(
                f" - highpass with cutoff of {args.highpass} Hz. RMS is now {song_data.dBFS:.2f} dBFS"
            )

        song_data = rescale(song_data, args.dBFS)
        absmax = np.amax(np.absolute(song_data.samples))
        log.info(f" - adjusted RMS to {song_data.dBFS:.2f} dBFS (peak is {absmax:.3f})")

        out_file = Path(song_data.name + ".wav")
        with ewave.open(
            out_file,
            mode="w",
            sampling_rate=song_data.sampling_rate,
            dtype=args.dtype,
        ) as fp:
            fp.write(song_data.samples)
        log.info(f" - wrote data to {out_file}")

        if args.deposit:
            metadata = {
                "source_resource": song["source"],
                "source_dataset": song["dataset"],
                "source_interval_ms": song["interval_ms"],
                "dBFS": song_data.dBFS,
                "created_by": f"{p.prog} {script_version}",
            }
            if args.highpass:
                metadata.update(
                    highpass_cutoff=args.highpass,
                    highpass_order=args.filter_order,
                    highpass_filter="butterworth",
                )
            try:
                _res = next(
                    nbank.deposit(
                        args.deposit,
                        (out_file,),
                        dtype="vocalization-wav",
                        auth=nbank.default_auth,
                        hash=True,
                        auto_id=True,
                        **metadata,
                    )
                )
            except nbank.HTTPStatusError as err:
                log.warning(f" ✗ unable to deposit {out_file} to {args.deposit}")
                nbank.log_error(err)
        log.info("")


if __name__ == "__main__":
    script()
