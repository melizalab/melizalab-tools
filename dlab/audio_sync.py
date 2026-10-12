# -*- mode: python -*-
"""Stimulus onsets from an analog copy of the audio (group-kilo-spikes --audio-sync)

For recordings without a usable sync track, each stimulus is found in a channel
that recorded the audio sent to the speaker (a hardwired copy, or less
reliably a microphone) by cross-correlating it with the stimulus file. The
start messages give the order of the stimuli and roughly when each was
presented, so each stimulus is only looked for where it should be, and only
the stimulus that should be there.

Matching uses amplitude envelopes (the square root of the smoothed squared
signal), with a correlation coefficient computed at each lag. This doesn't
depend on the channel's polarity, and tolerates the small timing differences
between playback and recording that defeat a match on the raw waveform over
long stimuli. On a hardwired copy, the onset is then refined to the sample on
the raw waveform. See audio_sync_onsets.

Run as a script (python -m dlab.audio_sync --help) to find the stimuli in a
recording without the rest of group-kilo-spikes, and optionally compare them
with a sync track.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
from scipy.signal import correlate

log = logging.getLogger("dlab.audio_sync")

ENVELOPE_SMOOTHING = 0.001  # s
# where the stimuli are looked for to calibrate the lag (s after the message)
CALIBRATION_WINDOW = (0.0, 2.0)
CALIBRATION_STIMULI = 200
MIN_CALIBRATION_MATCHES = 10
# the search window extends this far beyond the calibrated range of lags (s)
WINDOW_MARGIN = 0.1
# a match must stand out from the rest of the window: its correlation must
# exceed the best more than PEAK_EXCLUSION s away by MIN_CONTRAST. Envelope
# correlations fall off slowly with lag, so a ratio of the two is less telling:
# on a microphone (P353_1_1) every match off by more than 5 ms had a contrast
# below 0.07, every right one above 0.12; on the hardwired copy, above 0.23
PEAK_EXCLUSION = 0.02
MIN_CONTRAST = 0.1
# and correlate at least this fraction of the recording's typical correlation
MIN_R_FRACTION = 0.5
# stimuli this close in length are compared with each other (s, or fraction)
RELATIVE_LENGTH = (0.05, 0.02)
# a relative that correlates this much better than the expected stimulus means
# the wrong stimulus was played (on a microphone, relatives beat the expected
# stimulus by up to 0.048 in P353_1_1; on the hardwired copy, never)
VARIANT_MARGIN = 0.05
# a relative whose misfit (1 - r) is less than this many times the expected
# stimulus's can't be told apart from it (on the hardwired copy, the ratio is
# typically 12-19, but 1.03 for one song in two backgrounds; on a microphone,
# about 1.1)
VARIANT_FACTOR = 2.0
# the raw waveform refines the onset within this distance of the envelope's (s)
REFINE_WINDOW = 0.002


def parameters() -> dict:
    """The parameters of the matching, for the record"""
    return {
        "envelope_smoothing": ENVELOPE_SMOOTHING,
        "calibration_window": list(CALIBRATION_WINDOW),
        "window_margin": WINDOW_MARGIN,
        "peak_exclusion": PEAK_EXCLUSION,
        "min_contrast": MIN_CONTRAST,
        "min_r_fraction": MIN_R_FRACTION,
        "relative_length": list(RELATIVE_LENGTH),
        "variant_margin": VARIANT_MARGIN,
        "variant_factor": VARIANT_FACTOR,
        "refine_window": REFINE_WINDOW,
    }


class AudioMatch(NamedTuple):
    """Where a stimulus was found on the audio channel.

    method is 'audio' (matched), 'estimated' (a stimulus with nothing to
    match, such as silence, placed at the message plus the typical lag), or
    'unmatched' (reason says why). onset is in samples on the channel; lag is
    the time from the message (s); r and contrast describe the match (see
    peak). rival is the stimulus of about the same length that matches best
    in the same place, with its correlation (rival_r); variants lists those
    that match about as well as the stimulus itself (VARIANT_FACTOR; level
    variants of one scene, say), which doesn't change the onset.
    """

    method: str
    onset: int | None = None
    lag: float | None = None
    r: float | None = None
    contrast: float | None = None
    variants: tuple[str, ...] = ()
    rival: str | None = None
    rival_r: float | None = None
    reason: str | None = None


def amplitude_envelope(
    x: np.ndarray, sampling_rate: float, smoothing: float = ENVELOPE_SMOOTHING
) -> np.ndarray:
    """The square root of the signal's power, smoothed over `smoothing` s"""
    n = max(1, round(smoothing * sampling_rate))
    power = np.convolve(np.asarray(x, dtype="d") ** 2, np.ones(n) / n, "same")
    return np.sqrt(np.maximum(power, 0.0))


def normalized_xcorr(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """The correlation coefficient of the template with each window of the
    signal of its length (one value per lag, as in 'valid' mode)"""
    signal = np.asarray(signal, dtype="d")
    template = np.asarray(template, dtype="d")
    n = template.size
    sd = template.std()
    if sd == 0 or signal.size < n:
        return np.zeros(max(signal.size - n + 1, 0))
    t = (template - template.mean()) / (sd * n)
    c = correlate(signal, t, "valid", method="fft")
    sums = np.concatenate([[0.0], np.cumsum(signal)])
    squares = np.concatenate([[0.0], np.cumsum(signal**2)])
    mean = (sums[n:] - sums[:-n]) / n
    var = (squares[n:] - squares[:-n]) / n - mean**2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = c / np.sqrt(var)
    r[~np.isfinite(r) | (var <= 1e-12 * max(var.max(), 1e-300))] = 0.0
    return r


def peak(r: np.ndarray, exclusion: int) -> tuple[int, float, float]:
    """The index and value of the highest correlation, and its contrast: how
    much it exceeds the highest more than `exclusion` samples away (or 0, if
    that is negative or there is none)"""
    k = int(np.argmax(r))
    away = np.abs(np.arange(r.size) - k) > exclusion
    other = max(float(r[away].max()), 0.0) if away.any() else 0.0
    return k, float(r[k]), float(r[k]) - other


def relatives(lengths: Mapping[str, int], sampling_rate: float) -> dict[str, list[str]]:
    """For each stimulus, the others of about the same length (within
    RELATIVE_LENGTH), which are checked against it: variants of one source
    (e.g. a scene with the target at different levels) have the same length."""
    tolerance_s, tolerance_frac = RELATIVE_LENGTH
    names = sorted(lengths)
    out = {}
    for name in names:
        n = lengths[name]
        tol = max(tolerance_s * sampling_rate, tolerance_frac * n)
        out[name] = [o for o in names if o != name and abs(lengths[o] - n) <= tol]
    return out


def _search(
    read: Callable[[int, int], np.ndarray],
    template_env: np.ndarray,
    start: int,
    stop: int,
    sampling_rate: float,
) -> tuple[int, float, float, np.ndarray] | None:
    """Looks for a template (an envelope) with its onset in [start, stop) of
    the channel. Returns the onset, r, contrast and the envelope of the
    searched stretch (for checking relatives), or None if the stretch is
    outside the channel."""
    pad = round(ENVELOPE_SMOOTHING * sampling_rate)
    a = max(start - pad, 0)
    raw = read(a, stop + template_env.size + pad)
    if raw.size < (stop - start) + template_env.size:
        return None
    env = amplitude_envelope(raw - np.median(raw), sampling_rate)[start - a :]
    r = normalized_xcorr(env, template_env)[: stop - start]
    if r.size == 0:
        return None
    k, best, contrast = peak(r, round(PEAK_EXCLUSION * sampling_rate))
    return start + k, best, contrast, env


def _refine(
    read: Callable[[int, int], np.ndarray],
    template: np.ndarray,
    onset: int,
    sampling_rate: float,
) -> int:
    """The onset, refined to the sample on the raw waveform: the peak of the
    magnitude of the correlation within REFINE_WINDOW of the envelope's"""
    w = round(REFINE_WINDOW * sampling_rate)
    a = max(onset - w, 0)
    raw = read(a, onset + w + template.size)
    r = np.abs(normalized_xcorr(raw - np.median(raw), template))
    if r.size == 0:
        return onset
    return a + int(np.argmax(r[: 2 * w + 1]))


def audio_sync_onsets(
    read: Callable[[int, int], np.ndarray],
    stimuli: Sequence,
    waveforms: Mapping[str, np.ndarray],
    sampling_rate: float,
    *,
    refine: bool = True,
) -> list[AudioMatch]:
    """Finds each stimulus on an audio channel, near where its message says it
    was presented. Returns an AudioMatch for each stimulus.

    read(a, b) returns samples [a, b) of the channel. stimuli have a name and
    a start (the message time, in samples on the channel), in the order they
    were presented; waveforms are the stimulus files at the channel's sampling
    rate.

    First, the lag from message to onset is calibrated on up to
    CALIBRATION_STIMULI stimuli, looked for in CALIBRATION_WINDOW: the matches
    that stand out (MIN_CONTRAST) give the typical lag, its range (5th to 95th
    percentile) and the typical correlation. Then each stimulus is looked for
    in that range of lags, widened by WINDOW_MARGIN, and before the next
    stimulus could start. A match must stand out from the rest of the window
    (MIN_CONTRAST) and correlate at least MIN_R_FRACTION of the typical
    correlation. Stimuli of about the same length (see relatives) are compared
    in the same window: if one matches better by more than VARIANT_MARGIN, the
    stimulus doesn't match (the wrong one was played); if one matches about as
    well (VARIANT_FACTOR), it is listed in `variants`. Stimuli whose files are silent are placed
    at the message plus the typical lag ('estimated'). With refine, matched
    onsets are refined to the sample on the raw waveform, which suits a
    hardwired copy of the audio.

    Raises RuntimeError if too few stimuli can be found to calibrate the lag.

    """
    rate = sampling_rate
    envelopes = {}
    for name, w in waveforms.items():
        env = amplitude_envelope(w, rate)
        if env.std() > 0:
            envelopes[name] = env
    related = relatives({name: w.size for name, w in waveforms.items()}, rate)
    starts = np.array([stim.start for stim in stimuli])

    # calibrate on stimuli spread through the recording
    candidates = [i for i, stim in enumerate(stimuli) if stim.name in envelopes]
    sample = candidates[:: max(1, len(candidates) // CALIBRATION_STIMULI)]
    lags, rs = [], []
    lo, hi = (round(x * rate) for x in CALIBRATION_WINDOW)
    for i in sample:
        env = envelopes[stimuli[i].name]
        found = _search(read, env, starts[i] + lo, starts[i] + hi, rate)
        if found is not None and found[2] >= MIN_CONTRAST:
            lags.append(found[0] - starts[i])
            rs.append(found[1])
    if len(lags) < MIN_CALIBRATION_MATCHES:
        raise RuntimeError(
            f"only {len(lags)} of {len(sample)} stimuli were found on the audio "
            "channel near their messages: it may not carry the audio, or the "
            "messages may not belong to the recording"
        )
    lags = np.array(lags)
    typical = int(np.median(lags))
    lag_lo, lag_hi = np.percentile(lags, [5, 95])
    margin = round(WINDOW_MARGIN * rate)
    min_r = MIN_R_FRACTION * float(np.median(rs))
    log.info(
        "    - audio lags %.3f s (%.3f-%.3f s) from %d stimuli; typical r %.3f",
        typical / rate,
        lag_lo / rate,
        lag_hi / rate,
        lags.size,
        np.median(rs),
    )

    out = []
    for i, stim in enumerate(stimuli):
        if stim.name not in envelopes:
            out.append(
                AudioMatch("estimated", int(stim.start + typical), typical / rate)
            )
            continue
        start = int(stim.start + lag_lo - margin)
        stop = int(stim.start + lag_hi + margin)
        if i + 1 < len(stimuli):
            stop = min(stop, int(starts[i + 1] + lag_lo - margin))
        if stop <= start:
            out.append(
                AudioMatch("unmatched", reason="no room before the next stimulus")
            )
            continue
        found = _search(read, envelopes[stim.name], start, stop, rate)
        if found is None:
            out.append(AudioMatch("unmatched", reason="outside the recording"))
            continue
        onset, r, contrast, env = found
        lag = (onset - stim.start) / rate
        rival, rival_r = None, None
        variants = []
        for other in related.get(stim.name, ()):
            if other not in envelopes:
                continue
            rel = normalized_xcorr(env, envelopes[other])[: stop - start]
            if rel.size == 0:
                continue
            r_other = float(rel.max())
            if rival_r is None or r_other > rival_r:
                rival, rival_r = other, r_other
            if 1 - r_other < VARIANT_FACTOR * (1 - r):
                variants.append(other)
        match = AudioMatch(
            "audio", int(onset), lag, r, contrast, tuple(variants), rival, rival_r
        )
        if rival_r is not None and rival_r > r + VARIANT_MARGIN:
            out.append(
                match._replace(
                    method="unmatched",
                    onset=None,
                    reason=f"{rival} matches better (r {rival_r:.3f} vs {r:.3f})",
                )
            )
            continue
        if r < min_r or contrast < MIN_CONTRAST:
            out.append(
                match._replace(
                    method="unmatched",
                    onset=None,
                    reason=f"r {r:.3f} (minimum {min_r:.3f}), contrast "
                    f"{contrast:.3f} (minimum {MIN_CONTRAST})",
                )
            )
            continue
        if refine:
            onset = _refine(read, waveforms[stim.name], onset, rate)
            match = match._replace(onset=onset, lag=(onset - stim.start) / rate)
        out.append(match)
    return out


def main(argv=None):
    """python -m dlab.audio_sync: find the stimuli of a recording on an audio
    channel, and write a table of the matches (see audio_sync_onsets)"""
    import argparse
    import csv
    import sys
    from pathlib import Path

    import h5py

    from dlab import kilo
    from dlab import neurobank as nbank
    from dlab.util import add_log_arguments, setup_log

    p = argparse.ArgumentParser(
        prog="python -m dlab.audio_sync",
        description="Find the stimuli of a recording on a channel with a copy of "
        "the audio (a hardwired copy; a microphone works less reliably, with its "
        "onsets ~1.3-2.6 ms late from the acoustic path), near where the "
        "stimulus messages say they were presented. Writes one row per stimulus.",
    )
    add_log_arguments(p)
    nbank.add_registry_argument(p)
    p.add_argument("recording", type=Path, help="the ARF file")
    p.add_argument("channel", help="the channel with the copy of the audio")
    p.add_argument(
        "--oeaudio-log",
        type=Path,
        help="read the stimuli from this open-ephys-audio log, for recordings "
        "without the messages in the ARF file",
    )
    p.add_argument(
        "--local-stim-dir",
        type=Path,
        help="look for stimulus files here if they aren't in neurobank",
    )
    p.add_argument(
        "--no-refine",
        action="store_true",
        help="don't refine the onsets on the raw waveform (for a microphone)",
    )
    p.add_argument(
        "--compare",
        metavar="SYNC",
        help="compare the onsets with the sync events on this channel",
    )
    p.add_argument(
        "--output", "-o", type=Path, help="write the table here (default: stdout)"
    )
    args = p.parse_args(argv)
    setup_log(args.debug, args.debug_http)
    finder = kilo.StimulusFinder(args.registry_url, args.local_stim_dir)
    rows = []
    with h5py.File(args.recording, "r") as fp:
        for entry_num, entry in kilo.iter_entries(fp):
            dset = entry[args.channel]
            rate = float(dset.attrs["sampling_rate"])
            first = round(dset.attrs["offset"] * rate)
            if args.oeaudio_log is not None:
                with open(args.oeaudio_log) as lfp:
                    stimuli = list(kilo.oeaudio_log_to_stimuli(lfp, rate))
            else:
                messages = kilo.find_message_dset(entry)
                if messages is None:
                    p.exit(
                        1, f"{entry.name}: no stimulus messages; use --oeaudio-log\n"
                    )
                stimuli = kilo.messages_to_stimuli(messages)
            stimuli = [s._replace(start=s.start - first) for s in stimuli]
            log.info("- %s: %d stimuli on %s", entry.name, len(stimuli), args.channel)
            waveforms = finder.get_waveforms({s.name for s in stimuli}, rate)
            matches = audio_sync_onsets(
                lambda a, b, d=dset: d[a:b].astype("d"),
                stimuli,
                waveforms,
                rate,
                refine=not args.no_refine,
            )
            sync = [None] * len(stimuli)
            if args.compare:
                onsets = kilo.detect_sync_onsets(entry[args.compare][:])
                paired = kilo.match_sync_events(stimuli, onsets)
                i = 0
                for onset, stim in zip(onsets, paired, strict=True):
                    while stimuli[i] is not stim:
                        i += 1
                    sync[i] = int(onset)
            for i, (stim, m) in enumerate(zip(stimuli, matches, strict=True)):
                row = {
                    "entry": entry_num,
                    "index": i,
                    "stimulus": stim.name,
                    "message": stim.start / rate,
                    "method": m.method,
                    "onset": None if m.onset is None else m.onset / rate,
                    "lag": m.lag,
                    "r": m.r,
                    "contrast": m.contrast,
                    "variants": len(m.variants),
                    "rival": m.rival,
                    "rival_r": m.rival_r,
                    "reason": m.reason,
                }
                if args.compare:
                    row["sync"] = None if sync[i] is None else sync[i] / rate
                    row["difference_ms"] = (
                        None
                        if sync[i] is None or m.onset is None
                        else (m.onset - sync[i]) / rate * 1000
                    )
                rows.append(row)
            _summarize(matches, [r.get("difference_ms") for r in rows[-len(stimuli) :]])
    out = open(args.output, "w", newline="") if args.output else sys.stdout
    with out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()}
            )


def _summarize(matches: list[AudioMatch], differences: list) -> None:
    """Logs a summary of one entry's matches (and their differences from the
    sync events, ms, if compared)"""
    from collections import Counter

    counts = Counter(m.method for m in matches)
    log.info("  - %s", ", ".join(f"{n} {method}" for method, n in counts.items()))
    found = [m for m in matches if m.method == "audio"]
    if found:
        r = np.array([m.r for m in found])
        c = np.array([m.contrast for m in found])
        log.info(
            "  - r %.3f-%.3f (median %.3f); contrast min %.3f (median %.3f); "
            "%d with indistinguishable variants",
            r.min(),
            r.max(),
            np.median(r),
            c.min(),
            np.median(c),
            sum(bool(m.variants) for m in found),
        )
    for i, m in enumerate(matches):
        if m.method == "unmatched":
            log.info("  - unmatched: stimulus %d: %s", i, m.reason)
    d = np.array([x for x in differences if x is not None])
    if d.size:
        log.info(
            "  - onset - sync event (ms): median %.3f, range %.3f to %.3f, "
            "%d more than 1 ms from the median",
            np.median(d),
            d.min(),
            d.max(),
            np.sum(np.abs(d - np.median(d)) > 1),
        )


if __name__ == "__main__":
    main()
