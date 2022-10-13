# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Front-end scripts for song detector """
from pathlib import Path
import pickle
import argparse
import logging
import numpy as np

script_name = "quicksong"
script_version = "2022.09.21"

_example_config = dict(
    spectrogram={
        "window_ms": 20.0,
        "shift_ms": 10.0,
        "frequency_range": [0.7, 20.0],
    },
    features=["total_power", "wiener_entropy"],
    target_label="x",
    smoothing={"kernel_widths": [11, 21, 51]},
    classifier={"gamma": 2, "C": 1},
    testing={"test_size": 0.4, "random_state": 45},
    intervals={"max_gap_ms": 100, "min_interval_ms": 500},
    datasets=[],
)

log = logging.getLogger("")


def make_config(args):
    import yaml

    cfg = _example_config.copy()
    for lblfile in args.label_files:
        cfg["datasets"].append(lblfile.stem)
    with open(args.config, "wt") as fp:
        yaml.dump(cfg, fp)
        log.info("created new configuration file '%s'", args.config)
    if args.train:
        args.dataset_dir = lblfile.parent
        args.model = None
        train_classifier(args)


def make_extractor(cfg, sampling_rate_khz):
    from quicksong import features
    from quicksong.streaming import STFT, OverlapSaveConvolver
    from quicksong.core import FeatureExtractor, make_hanning_kernels

    speccr = STFT(
        sampling_rate_khz,
        cfg["spectrogram"]["window_ms"],
        cfg["spectrogram"]["shift_ms"],
        cfg["spectrogram"]["frequency_range"],
    )
    feats = tuple(getattr(features, f) for f in cfg["features"])
    conv = OverlapSaveConvolver(make_hanning_kernels(cfg["smoothing"]["kernel_widths"]))
    return FeatureExtractor(speccr, feats, conv)


def train_classifier(args):
    import yaml
    import ewave
    from quicksong import core, model
    from arfx import lblio

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)
    log.debug(cfg)
    label_ids = {cfg["target_label"]: 0}
    log.info("Training song classifier:")
    log.info("- version: %s", script_version)

    X_all = []
    y_all = []
    for dset in cfg["datasets"]:
        log.info("- dataset: %s", dset)
        path = args.dataset_dir / dset
        with ewave.open(path.with_suffix(".wav"), "r") as fp:
            Fs = fp.sampling_rate / 1000.0
            signal = ewave.rescale(fp.read(), "f")
            log.info(" - recording: %d samples @ %.2f kHz", signal.size, Fs)
        extractor = make_extractor(cfg, Fs)
        feats = extractor.process_all(signal)
        log.info("   - calculated features %s", feats.shape)

        with open(path.with_suffix(".lbl"), "rt") as fp:
            out = []
            for lbl in lblio.read(fp):
                n = lbl["name"]
                try:
                    id = label_ids[n]
                except KeyError:
                    id = len(label_ids)
                    label_ids[n] = id
                out.append((id, lbl["start"] * 1000, lbl["stop"] * 1000))
            labels = np.array(
                out, dtype=[("id", "i4"), ("start", "f8"), ("stop", "f8")]
            )
            log.info(" - labels: %d segments", labels.size)
            X, y = core.make_training_dataset(
                feats, labels, cfg["spectrogram"]["shift_ms"]
            )
            log.info("  - labeled frames: %d", y.size)
            X_all.append(X)
            y_all.append(y)
    X = np.row_stack(X_all)
    y = np.concatenate(y_all)
    assert X.shape[0] == y.size
    log.info("- training classifier...")
    classifier, score = model.train_classifier(X, y, cfg["classifier"], cfg["testing"])
    log.info("  - model score: %.3f", score)
    if args.model is None:
        args.model = args.config.with_suffix(".pkl")
    with open(args.model, "wb") as fp:
        pickle.dump(
            {"config": cfg, "classifier": classifier, "version": script_version}, fp
        )
        log.info("- wrote model to '%s'", args.model)


class ArfWriter:
    """Utility class that writes to sequentially numbered entries"""

    def __init__(self, arfp, template="song_{index:04}"):
        self.arfp = arfp
        self.entry_count = len(arfp.keys())
        self.template = template

    @property
    def filename(self):
        return self.arfp.filename

    def create_entry(self, timestamp, **attributes):
        import arf

        next_entry_name = self.template.format(index=self.entry_count)
        self.entry_count += 1
        return arf.create_entry(self.arfp, next_entry_name, timestamp, **attributes)


def extract_songs(args):
    import os
    import arf
    import h5py as h5
    from quicksong.core import IntervalFinder

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    log.info("Extracting songs:")
    log.info("- version: %s", script_version)
    with open(args.model, "rb") as fp:
        log.info("- loading classifier from '%s'", args.model)
        model = pickle.load(fp)
    # TODO check model version
    classifier = model["classifier"]
    dt = model["config"]["spectrogram"]["shift_ms"]
    finder = IntervalFinder(max_gap=args.max_gap, min_duration=args.min_duration, dt=dt)
    # TODO: more flexibility with output format
    with arf.open_file(args.output, mode="a") as ofp:
        writer = ArfWriter(ofp)
        log.info("- saving songs to '%s'", args.output)
        for fname in args.input:
            with h5.File(fname, "r") as ifp:
                log.info("- processing recordings in '%s'", ifp.filename)
                last_sample = None
                buffer = []
                intervals = []
                for entry in ifp.values():
                    if not isinstance(entry, h5.Group):
                        log.info("  ✗ %s: not an entry, skipping", entry.name)
                        continue
                    try:
                        dset = entry[args.dataset_name]
                    except KeyError:
                        log.info(
                            "  ✗ %s: does not have dataset `%s`, skipping",
                            entry.name,
                            args.dataset_name,
                        )
                        continue
                    sampling_rate = dset.attrs["sampling_rate"] / 1000.0
                    entry_start = entry.attrs["jack_frame"]
                    log.debug(
                        " - %s: start=%s, end=%s, sampling_rate=%.2f kHz",
                        entry.name,
                        entry_start,
                        entry_start + dset.size,
                        sampling_rate,
                    )
                    # is this a continuation of the previous entry?
                    if last_sample is not None and last_sample > entry_start:
                        log.debug("   - continues previous entry")
                    else:
                        # process the intervals
                        extract_intervals(
                            writer,
                            buffer,
                            intervals,
                            args.pad_before,
                            args.pad_after,
                        )
                        extractor = make_extractor(model["config"], sampling_rate)
                        finder.reset()
                        buffer = []
                        intervals = []
                    for block in extractor.process(dset):
                        pred = classifier.predict(block)
                        intervals.extend(
                            (start * dt, stop * dt)
                            for start, stop in finder.process(pred)
                        )
                    # jack_frame is a np.uint32 so it should overflow to allow
                    # comparison with next entry
                    last_sample = entry_start + dset.size
                    buffer.append(dset)


def extract_intervals(writer, dsets, intervals, pad_before, pad_after):
    import posixpath
    from datetime import timedelta
    from arf import timestamp_to_datetime, create_dataset, DataTypes, set_uuid
    from quicksong.core import pad_intervals
    from dlab.util import all_same

    fields_to_drop = (
        "entry_creator",
        "timestamp",
        "jack_frame",
        "jack_sampling_rate",
        "jack_usec",
        "trial_off",
    )

    if len(dsets) == 0:
        return
    dset_name = ", ".join(dset.parent.name for dset in dsets)
    if len(intervals) == 0:
        log.info("  ✗ %s: no song detected", dset_name)
        return
    sampling_rate = all_same(dset.attrs["sampling_rate"] for dset in dsets) / 1000.0
    assert sampling_rate is not None, "Entries don't have the same sampling rate!"
    parent_entry = dsets[0].parent
    entry_attrs = {
        k: v for k, v in parent_entry.attrs.items() if k not in fields_to_drop
    }
    entry_attrs.update(
        source_file=parent_entry.file.filename,
        source_uuid=b",".join(dset.parent.attrs["uuid"] for dset in dsets),
        entry_creator=f"org.meliza.dlab/{script_name} {script_version}",
    )
    entry_attrs.pop("uuid")
    dset_attrs = dict(dsets[0].attrs)
    dset_attrs["source_uuid"] = b",".join(dset.attrs["uuid"] for dset in dsets)
    dset_attrs["datatype"] = DataTypes.ACOUSTIC
    dset_attrs.pop("uuid")
    signal = np.concatenate(dsets)
    log.info("  ✓ %s: ", dset_name)
    for start, end in pad_intervals(intervals, pad_before, pad_after):
        start_sample = max(0, int(start * sampling_rate))
        end_sample = min(signal.size, int(end * sampling_rate))
        song_timestamp = timestamp_to_datetime(
            parent_entry.attrs["timestamp"]
        ) + timedelta(seconds=start_sample / sampling_rate / 1000)
        new_entry = writer.create_entry(song_timestamp, **entry_attrs)
        new_dset = create_dataset(
            new_entry,
            posixpath.basename(dsets[0].name),
            signal[start_sample:end_sample],
            compression="gzip",
            **dset_attrs,
        )
        set_uuid(new_dset)
        log.info(
            "    - samples %d–%d (%.1f ms) -> %s%s",
            start_sample,
            end_sample,
            (end_sample - start_sample) / sampling_rate,
            writer.filename,
            new_dset.name,
        )


def script(argv=None):
    from dlab.core import __version__
    from dlab.util import setup_log

    p = argparse.ArgumentParser(
        description="train the song detector on labeled recordings"
    )
    p.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {script_version} (melizalab-tools {__version__})",
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    sub = p.add_subparsers(title="subcommands")

    pp = sub.add_parser("init", help="generate a starting configuration file")
    pp.set_defaults(func=make_config)
    pp.add_argument(
        "--train",
        action="store_true",
        help="run training with the intialized config file",
    )
    pp.add_argument(
        "config", type=Path, help="path where the configuration file should be saved"
    )
    pp.add_argument(
        "label_files",
        type=Path,
        nargs="*",
        help="label files to add to config file as training dataset",
    )

    pp = sub.add_parser("train", help="train the classifier")
    pp.set_defaults(func=train_classifier)
    pp.add_argument(
        "--dataset-dir",
        "-d",
        type=Path,
        default=Path("."),
        help="directory where training data files are stored",
    )
    pp.add_argument("config", type=Path, help="path of the configuration file")
    pp.add_argument(
        "model",
        type=Path,
        nargs="?",
        help="path where the model should be saved (default is `config.pkl`)",
    )

    pp = sub.add_parser(
        "extract", help="use a trained model to extract songs from ARF files"
    )
    pp.set_defaults(func=extract_songs)
    pp.add_argument(
        "--dataset-name",
        default="pcm_000",
        help="the name of the dataset to process in each entry (default `%(default)s`)",
    )
    pp.add_argument(
        "--max-gap",
        type=float,
        default=100,
        help="merge segments separated by gaps less than this value (default %(default).1f ms)",
    )
    pp.add_argument(
        "--min-duration",
        type=float,
        default=500,
        help="segments must be at least this long (default %(default).1f ms)",
    )
    pp.add_argument(
        "--pad-before",
        type=float,
        default=1000,
        help="the amount of time before song onset to extract (default %(default).1f ms)",
    )
    pp.add_argument(
        "--pad-after",
        type=float,
        default=1000,
        help="the amount of time after song end to extract (default %(default).1f ms)",
    )
    pp.add_argument("model", help="trained classifier model (pkl file saved by `train`")
    pp.add_argument("input", help="the ARF file to process", nargs="+")
    pp.add_argument("output", help="output where the song segments should be saved")

    args = p.parse_args(argv)
    setup_log(log, args.debug)
    if not hasattr(args, "func"):
        p.print_usage()
        return 0

    args.func(args)


if __name__ == "__main__":
    script()
