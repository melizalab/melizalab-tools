# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Front-end scripts for song detector """
import os
import pickle
import argparse
import logging
import numpy as np

__version__ = "2022.09.06"

_example_config = """
spectrogram:
  window_ms: 20.0
  shift_ms: 10.0
  frequency_range: [0.5, 20.]
features:
  - TotalPower
  - WienerEntropy
target_label: x
smoothing:
  kernel_widths: [11, 21, 51]
classifier:
  gamma: 2
  C: 1
testing:
  test_size: 0.4
  random_state: 45
intervals:
  max_gap_ms: 100
  min_interval_ms: 500
# add your training data here, one line per labeled recording
datasets:
  - name_of_recording
"""

log = logging.getLogger("")


def make_config(args):
    with open(args.config, "wt") as fp:
        fp.writelines(_example_config)
        log.info("created new configuration file '%s'", args.config)
        log.info("edit this file to add training stimuli under `datasets`")


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
    feats = tuple(getattr(features, f)() for f in cfg["features"])
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
    log.info("- version: %s", __version__)

    X_all = []
    y_all = []
    for dset in cfg["datasets"]:
        log.info("- dataset: %s", dset)
        path = os.path.join(args.dataset_dir, dset)
        with ewave.open(path + ".wav", "r") as fp:
            Fs = fp.sampling_rate / 1000.0
            signal = ewave.rescale(fp.read(), "f")
            log.info(" - recording: %d samples @ %.2f kHz", signal.size, Fs)
        extractor = make_extractor(cfg, Fs)
        feats = extractor.process_all(signal)
        log.info("   - calculated features %s", feats.shape)

        with open(path + ".lbl", "rt") as fp:
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
        classifier, score = model.train_classifier(
            X, y, cfg["classifier"], cfg["testing"]
        )
        log.info("  - model score: %.3f", score)
        with open(args.model, "wb") as fp:
            pickle.dump(
                {"config": cfg, "classifier": classifier, "version": __version__}, fp
            )
            log.info("- wrote model to '%s'", args.model)


def extract_songs(args):
    import h5py as h5
    from quicksong.streaming import IntervalFinder

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    log.info("Training song classifier:")
    log.info("- version: %s", __version__)
    with open(args.model, "rb") as fp:
        log.info("- loading classifier from '%s'", args.model)
        model = pickle.load(fp)
    # TODO check model version
    classifier = model["classifier"]
    dt = model["config"]["spectrogram"]["shift_ms"]
    finder = IntervalFinder(max_gap=args.max_gap, min_duration=args.min_duration, dt=dt)
    with h5.File(args.input, "r") as ifp:
        log.info("- processing recordings in '%s'", args.input)
        last_entry = None
        for ename, entry in ifp.items():
            if not isinstance(entry, h5.Group):
                log.info(" - %s: not an entry, skipping", ename)
                continue
            dset = entry[args.dataset_name]
            sampling_rate = dset.attrs["sampling_rate"] / 1000.0
            # TODO check if this entry is a continuation of the previous entry
            extractor = make_extractor(model["config"], sampling_rate)
            log.debug(" - %s: %d samples @ %.2f kHz", ename, dset.size, sampling_rate)
            for block in extractor.process(dset[:]):
                pred = classifier.predict(block)
                for ival in finder.process(pred):
                    log.info(
                        " - %s: song detected between %d--%d ms",
                        entry.name,
                        ival[0] * dt,
                        ival[1] * dt,
                    )


def script(argv=None):
    from dlab.util import setup_log

    p = argparse.ArgumentParser(
        description="train the song detector on labeled recordings"
    )
    p.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    sub = p.add_subparsers(title="subcommands")

    pp = sub.add_parser("init", help="generate a starting configuration file")
    pp.set_defaults(func=make_config)
    pp.add_argument("config", help="path where the configuration file should be saved")

    pp = sub.add_parser("train", help="train the classifier")
    pp.set_defaults(func=train_classifier)
    pp.add_argument(
        "--dataset-dir",
        "-d",
        default=".",
        help="directory where training data files are stored",
    )
    pp.add_argument("config", help="path of the configuration file")
    pp.add_argument("model", help="path where the model should be saved")

    pp = sub.add_parser(
        "extract", help="use a trained model to extract songs from ARF files"
    )
    pp.set_defaults(func=extract_songs)
    pp.add_argument(
        "--dataset-name",
        "-d",
        default="pcm_000",
        help="the name of the dataset to process in each entry",
    )
    pp.add_argument(
        "--max-gap",
        type=float,
        default=100,
        help="merge segments separated by gaps less than this value (in ms)",
    )
    pp.add_argument(
        "--min-duration",
        type=float,
        default=500,
        help="segments must be at least this long (in ms)",
    )
    pp.add_argument("model", help="trained classifier model (pkl file saved by `train`")
    pp.add_argument("input", help="the ARF file to process")
    pp.add_argument("output", help="output where the song segments should be saved")

    args = p.parse_args(argv)
    setup_log(log, args.debug)
    if not hasattr(args, "func"):
        p.print_usage()
        return 0

    args.func(args)
