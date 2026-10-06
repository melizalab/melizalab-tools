# -*- mode: python -*-
"""Make the small real-data ARF excerpts in this directory from the example
recordings (which are too large to keep in the repository).

    python tests/data/make_excerpts.py examples/

Each excerpt is the start of a recording, up to the start message of the
sixth stimulus, so it holds five complete trials. Only the sync channels and
the stimulus messages are kept; datasets keep their attributes (including
offset, so message sample numbers stay valid), and are gzip-compressed.

- E36_excerpt.arf: jpresent, GUI 1.0.2. ADC3 has the sustained pulses, ADC5
  the clicks that drive the Schmitt trigger making them (positive at stimulus
  onset, negative at offset). The pulses of trials 0 and 3 have flat tops that
  never dip, which the earlier z-scored detector reported at their end.
- P397_excerpt.arf, P397_excerpt.log: oeaudio-present, GUI 1.0.2, recorded
  without MessageCenter logging (the dataset is empty), so stimuli come from
  the open-ephys-audio log. ADC3 has clicks, ADC4 low-amplitude pulses (about
  3000 counts above the negative rail).
"""

import sys
from pathlib import Path

import arf
import h5py

HERE = Path(__file__).parent
NSTIMULI = 5


def copy_entry(src_entry, dst_file, channels, end, messages):
    """Copy channels[:end] and the message rows to a new entry"""
    entry = arf.create_entry(
        dst_file,
        "entry",
        src_entry.attrs["timestamp"],
        **{k: v for k, v in src_entry.attrs.items() if k not in ("timestamp", "uuid")},
    )
    for name in channels:
        src = src_entry[name]
        dset = entry.create_dataset(
            name, data=src[:end], compression="gzip", compression_opts=9, shuffle=True
        )
        for k, v in src.attrs.items():
            dset.attrs[k] = v
    src = src_entry["MessageCenter"]
    dset = entry.create_dataset("MessageCenter", data=messages, maxshape=(None,))
    for k, v in src.attrs.items():
        dset.attrs[k] = v


def first_sample(dset):
    return round(dset.attrs["offset"] * dset.attrs["sampling_rate"])


def e36(examples: Path):
    with h5py.File(examples / "E36_5_1" / "E36_5_1.arf", "r") as fp:
        src = fp[next(iter(fp))]
        first = first_sample(src["ADC3"])
        rows = src["MessageCenter"][:]
        starts = [r["start"] for r in rows if r["message"].startswith(b"start ")]
        end = starts[NSTIMULI] - first
        with arf.open_file(HERE / "E36_excerpt.arf", "w") as out:
            copy_entry(
                src, out, ["ADC3", "ADC5"], end, rows[rows["start"] < starts[NSTIMULI]]
            )


def p397(examples: Path):
    import datetime

    log_src = examples / "P397_1_1" / "oeaudio_20260617-130556.log"
    lines = log_src.read_text().splitlines()
    t0 = datetime.datetime.strptime(lines[0].split(",")[0], "%Y-%m-%d %H:%M:%S.%f")
    starts = [i for i, line in enumerate(lines) if ',"start ' in line]
    (HERE / "P397_excerpt.log").write_text("\n".join(lines[: starts[NSTIMULI]]) + "\n")
    t = datetime.datetime.strptime(
        lines[starts[NSTIMULI]].split(",")[0], "%Y-%m-%d %H:%M:%S.%f"
    )
    with h5py.File(examples / "P397_1_1" / "P397_1_1.arf", "r") as fp:
        src = fp[next(iter(fp))]
        rate = src["ADC3"].attrs["sampling_rate"]
        end = round((t - t0).total_seconds() * rate) - first_sample(src["ADC3"])
        with arf.open_file(HERE / "P397_excerpt.arf", "w") as out:
            copy_entry(src, out, ["ADC3", "ADC4"], end, src["MessageCenter"][:])


if __name__ == "__main__":
    examples = Path(sys.argv[1] if len(sys.argv) > 1 else "examples")
    e36(examples)
    p397(examples)
    for path in sorted(HERE.glob("*_excerpt.*")):
        print(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB")
