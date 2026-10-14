# -*- mode: python -*-
"""Sorts of part of a recording (group-kilo-spikes --sort-start/--sort-stop)

A recording that starts badly, goes bad, or both may be sorted over only part
of its length, the window [start, stop) of the ARF entry. group-kilo-spikes is
told the window, times the spikes from the entry's start, and only keeps the
trials within the window.

How kilosort lays out a sort of a window depends on its version and options,
so two origins are found from the data rather than assumed: the recording
sample that temp_wh.dat starts at (the window's start, if it holds only the
window, as the deposited sorts made with kilosort 2.5's tmin/tmax do; or the
recording's, if it holds the whole recording), and the one the spike times
count from (the window's start, as kilosort 2.5 writes them; or the
recording's, as kilosort 4 and a patched 2.5 do). Each candidate is checked
against the data to the sample, and exactly one of each must fit (see
check_sort_window):

- temp_wh.dat's length must fit it (up to MAX_SORT_PADDING samples longer),
  and excerpts of it, which kilosort high-pass filtered and whitened, must
  match the band-passed recording on the same channels at lag 0;
- the spike times must point at spikes in temp_wh.dat: the average of
  temp_wh.dat around each of the largest units' spikes must have a peak many
  times its noise at the spike time (averaging at the wrong positions gives
  noise), and the average of the band-passed recording around them must
  match it at lag 0.

A wrong origin doesn't mislabel any trials, since they come from the ARF file,
but it misaligns every spike with its stimulus without any other sign.

temp_wh.dat's columns are probe channels (channel_map.npy, if the sort
directory has it, gives the channel of each column; otherwise column i is
channel i), and probe channel i is the ARF dataset CH{i+1}.
"""

import logging
from collections import Counter
from collections.abc import Sequence
from typing import NamedTuple

import h5py as h5
import numpy as np
from scipy.signal import butter, sosfiltfilt

from dlab.audio_sync import normalized_xcorr, peak

log = logging.getLogger("dlab.kilo")

# kilosort 2-3 write temp_wh.dat in whole batches (NT = 64 * 1024 + 64 samples
# by default), so it can extend past the end of the data by less than this.
# Not verified against kilosort's source; kilosort 4 may not pad at all.
MAX_SORT_PADDING = 65600
# the band the recording is filtered to before it's compared with temp_wh.dat
FILTER_BAND = (300.0, 6000.0)  # Hz
FILTER_MARGIN = 0.01  # s of data on each side, so filtering has no edge effects
# excerpt matches: EXCERPT s of temp_wh.dat at each of EXCERPT_POSITIONS (as a
# fraction of the window), on the channels of the largest units, are looked
# for within EXCERPT_SEARCH s of where an origin puts them. A match must
# exceed the best correlation more than PEAK_EXCLUSION s away by MIN_CONTRAST
# (unrelated data reaches about 0.01; a neighbouring channel about 0.2, at a
# lag of a few samples, which is why several channels must agree).
EXCERPT = 1.0
EXCERPT_POSITIONS = (0.1, 0.5, 0.9)
EXCERPT_SEARCH = 0.5
PEAK_EXCLUSION = 0.001
MIN_CONTRAST = 0.1
# units checked (the largest good ones, on different channels), and the
# spikes of each averaged
CHECK_UNITS = 4
AVERAGE_SPIKES = 500
# the unit's average in temp_wh.dat must peak within AVERAGE_SEARCH s of the
# spike time at MIN_SPIKE_Z times the noise of an average (the column's noise
# over the square root of the number of spikes); real units reach hundreds,
# an average at the wrong positions a few. The average in the recording must
# correlate with it by at least MIN_AVERAGE_R, best at lag 0.
MIN_SPIKE_Z = 20.0
MIN_AVERAGE_R = 0.5
AVERAGE_SEARCH = 0.001


class SortWindow(NamedTuple):
    """The part of an ARF entry that was sorted, [start, stop) in samples from
    the entry's first sample; the number of samples in the entry and in
    temp_wh.dat; and the recording samples that temp_wh.dat and the spike
    times start from (None until found by check_sort_window)."""

    start: int
    stop: int
    entry_samples: int
    sorted_samples: int
    data_origin: int | None = None
    spike_origin: int | None = None


class ExcerptMatch(NamedTuple):
    """Where an excerpt of temp_wh.dat (on `column`, put at recording sample
    `position` by an origin) matched best in the recording: `lag` samples from
    there, with correlation r and its contrast (see audio_sync.peak)."""

    position: int
    column: int
    lag: int
    r: float
    contrast: float


class AverageMatch(NamedTuple):
    """A unit's average waveform in temp_wh.dat: its peak at the spike time,
    in units of the noise of an average (z); and how its average in the
    recording matches it, the best lag (samples) and the correlation there."""

    cluster: int
    column: int
    n_spikes: int
    z: float
    lag: int
    r: float


def origin_name(window: SortWindow, origin: int) -> str:
    """'window' if origin is the window's start (and that isn't the
    recording's), otherwise 'recording'"""
    return "window" if origin == window.start and origin > 0 else "recording"


def data_origins(window: SortWindow) -> list[int]:
    """The recording samples temp_wh.dat could start at, given its length: the
    window's start (if it holds the window) or the recording's (if it holds the
    whole recording). It may be up to MAX_SORT_PADDING samples longer."""
    out = []
    for origin, length in (
        (window.start, window.stop - window.start),
        (0, window.entry_samples),
    ):
        extra = window.sorted_samples - length
        if 0 <= extra < MAX_SORT_PADDING and origin not in out:
            out.append(origin)
    return out


def sort_window(
    start: float | None,
    stop: float | None,
    entry_samples: int,
    sorted_samples: int,
    sampling_rate: float,
) -> SortWindow:
    """The sorted window, from its start and stop in s (None for the ends of
    the entry). Raises RuntimeError if it isn't within the entry, or if
    temp_wh.dat's length fits neither the window nor the entry. If the window
    starts at the entry's start, its origins are known (both 0)."""
    rate = sampling_rate
    first = 0 if start is None else round(start * rate)
    last = entry_samples if stop is None else round(stop * rate)
    duration = entry_samples / rate
    if not 0 <= first < last <= entry_samples:
        raise RuntimeError(
            f"the sorted window ({first / rate:.3f}-{last / rate:.3f} s) isn't "
            f"within the recording (0-{duration:.3f} s)"
        )
    window = SortWindow(first, last, entry_samples, sorted_samples)
    if not data_origins(window):
        raise RuntimeError(
            f"temp_wh.dat has {sorted_samples} samples "
            f"({sorted_samples / rate:.3f} s), which fits neither the sorted window "
            f"({first / rate:.3f}-{last / rate:.3f} s, {last - first} samples) nor "
            f"the whole recording ({entry_samples} samples). If only part of the "
            "recording was sorted, give its start and stop with --sort-start and "
            "--sort-stop; otherwise, is it the sort of this recording?"
        )
    if first == 0:
        window = window._replace(data_origin=0, spike_origin=0)
    return window


def recording_channel(entry: h5.Group, channel: int) -> h5.Dataset:
    """The ARF dataset of probe channel `channel` (counting from 0)"""
    name = f"CH{channel + 1}"
    try:
        return entry[name]
    except KeyError as err:
        raise RuntimeError(
            f"the recording has no dataset {name} to check the sorted window against"
        ) from err


def filtered(
    dset: h5.Dataset, start: int, stop: int, sampling_rate: float
) -> np.ndarray:
    """Samples [start, stop) of dset (clipped to its extent), band-passed to
    FILTER_BAND with a zero-phase filter, as kilosort's filter is"""
    margin = round(FILTER_MARGIN * sampling_rate)
    a = max(0, start - margin)
    b = min(dset.size, stop + margin)
    sos = butter(3, FILTER_BAND, "bandpass", fs=sampling_rate, output="sos")
    x = sosfiltfilt(sos, dset[a:b].astype("d"))
    first = max(start, 0) - a
    return x[first : first + min(stop, dset.size) - max(start, 0)]


def excerpt_matches(
    sorted_data: np.ndarray,
    entry: h5.Group,
    window: SortWindow,
    data_origin: int,
    columns: Sequence[int],
    channel_map: np.ndarray,
    sampling_rate: float,
) -> list[ExcerptMatch]:
    """Where excerpts of temp_wh.dat (sorted_data, samples x columns) from
    within the window match the recording, on each of columns, relative to
    where they would be if temp_wh.dat starts at recording sample data_origin"""
    n = round(EXCERPT * sampling_rate)
    search = round(EXCERPT_SEARCH * sampling_rate)
    exclusion = round(PEAK_EXCLUSION * sampling_rate)
    length = window.stop - window.start
    n = min(n, length)
    out = []
    for fraction in EXCERPT_POSITIONS:
        position = window.start + int(fraction * (length - n))
        at = position - data_origin
        for column in columns:
            dset = recording_channel(entry, int(channel_map[column]))
            first = max(0, position - search)
            signal = filtered(dset, first, position + n + search, sampling_rate)
            template = sorted_data[at : at + n, column]
            k, r, contrast = peak(normalized_xcorr(signal, template), exclusion)
            out.append(
                ExcerptMatch(position, column, first + k - position, r, contrast)
            )
    return out


def excerpt_problem(
    excerpts: list[ExcerptMatch],
    data_origin: int,
    channel_map: np.ndarray,
    sampling_rate: float,
) -> str | None:
    """Why the excerpts don't show temp_wh.dat starting at data_origin, or None
    if they do: every position needs a clear match at lag 0, and no clear match
    may be at another lag."""
    rate = sampling_rate
    clear = [m for m in excerpts if m.contrast >= MIN_CONTRAST]
    lags = Counter(m.lag for m in clear)
    if not clear:
        channels = sorted({int(channel_map[m.column]) for m in excerpts})
        return (
            f"no excerpt matches the recording within {EXCERPT_SEARCH} s of where it "
            f"should be (channels {', '.join(map(str, channels))})"
        )
    lag, count = lags.most_common(1)[0]
    if lag != 0 and count > len(clear) / 2:
        start = data_origin + lag
        return (
            f"the excerpts match the recording {lag:+d} samples ({lag / rate:+.4f} "
            "s) from where they should be: the sorted data probably start at "
            f"recording sample {start} ({start / rate:.4f} s)"
        )
    matched = {m.position for m in clear if m.lag == 0}
    if len(lags) > 1 or matched != {m.position for m in excerpts}:
        return (
            "the excerpts don't match consistently ("
            + "; ".join(
                f"at {m.position / rate:.1f} s, channel {channel_map[m.column]}: "
                f"lag {m.lag:+d}, contrast {m.contrast:.2f}"
                for m in excerpts
            )
            + "); the sorted data may not be a continuous slice of the recording"
        )
    return None


def average_match(
    sorted_data: np.ndarray,
    dset: h5.Dataset,
    times: np.ndarray,
    data_origin: int,
    spike_origin: int,
    column: int,
    n_before: int,
    n_after: int,
    sampling_rate: float,
) -> tuple[int, float, int, float]:
    """The average of temp_wh.dat (which starts at recording sample
    data_origin) on `column` around spikes at `times` (counted from recording
    sample spike_origin): its largest absolute value within AVERAGE_SEARCH of
    the spike time, over the noise of an average (z); and the lag (samples) at
    which the average of the band-passed recording around the spikes best
    matches it, and the correlation there. Uses up to AVERAGE_SPIKES spikes,
    spread through the recording. Returns (number of spikes, z, lag, r)."""
    search = round(AVERAGE_SEARCH * sampling_rate)
    recorded = np.sort(np.asarray(times, dtype="int64")) + spike_origin
    in_sort = recorded - data_origin
    ok = (
        (in_sort >= n_before + search)
        & (in_sort < sorted_data.shape[0] - n_after - search)
        & (recorded >= n_before + search)
        & (recorded < dset.size - n_after - search)
    )
    recorded, in_sort = recorded[ok], in_sort[ok]
    if recorded.size > AVERAGE_SPIKES:
        keep = np.linspace(0, recorded.size - 1, AVERAGE_SPIKES).astype(int)
        recorded, in_sort = recorded[keep], in_sort[keep]
    if recorded.size == 0:
        return 0, 0.0, 0, 0.0
    offsets = np.arange(-n_before, n_after)
    template = sorted_data[in_sort[:, None] + offsets, column].astype("d").mean(0)
    # the column's noise, from the median absolute deviation of a stretch of it
    middle = int(np.median(in_sort))
    stretch = sorted_data[middle : middle + round(sampling_rate), column].astype("d")
    noise = np.median(np.abs(stretch - np.median(stretch))) / 0.6745
    near = template[n_before - search : n_before + search + 1]
    z = float(np.abs(near).max() / (noise / np.sqrt(recorded.size))) if noise else 0.0
    average = np.zeros(n_before + n_after + 2 * search)
    for at in recorded:
        average += filtered(
            dset, at - n_before - search, at + n_after + search, sampling_rate
        )
    k, r, _ = peak(normalized_xcorr(average / recorded.size, template), 0)
    return int(recorded.size), z, k - search, r


def average_problem(averages: list[AverageMatch]) -> str | None:
    """Why the unit averages don't fit, or None if each unit with spikes to
    average (there must be one) has a spike at the spike time (z of at least
    MIN_SPIKE_Z) that the recording matches at lag 0 with r of at least
    MIN_AVERAGE_R"""
    if not any(a.n_spikes for a in averages):
        return "no spikes fall within the sorted data"
    bad = [
        a
        for a in averages
        if a.n_spikes and (a.z < MIN_SPIKE_Z or a.lag != 0 or a.r < MIN_AVERAGE_R)
    ]
    if not bad:
        return None
    return (
        "; ".join(
            f"cluster {a.cluster} averages to {a.z:.0f} times the noise at its "
            f"spike times, and matches the recording best at lag {a.lag:+d} "
            f"(r {a.r:.2f})"
            for a in bad
        )
        + f", instead of at least {MIN_SPIKE_Z:.0f} times, at lag 0, with r of "
        f"at least {MIN_AVERAGE_R}"
    )


def only_fit(window: SortWindow, what: str, problems: dict[int, str | None]) -> int:
    """The one origin without a problem; raises RuntimeError if there isn't
    exactly one"""
    fits = [origin for origin, problem in problems.items() if problem is None]
    if len(fits) == 1:
        return fits[0]
    if fits:
        raise RuntimeError(
            f"{what} fit counted from either the window's or the recording's start"
        )
    raise RuntimeError(
        f"{what} don't fit the sorted window, counted "
        + "; or ".join(
            f"from the {origin_name(window, origin)}'s start: {problem}"
            for origin, problem in problems.items()
        )
        + ". Check the window, the recording, and channel_map.npy."
    )


def check_sort_window(
    sorted_data: np.ndarray,
    entry: h5.Group,
    window: SortWindow,
    units: Sequence[tuple[int, int, np.ndarray]],
    channel_map: np.ndarray,
    n_before: int,
    n_after: int,
    sampling_rate: float,
) -> tuple[SortWindow, dict]:
    """Finds the recording samples that temp_wh.dat (sorted_data) and the
    spike times start from, and checks them to the sample. units are (cluster,
    column, spike times as kilosort wrote them) for the units to check,
    largest first; their columns are also used for the excerpts. Returns the
    window with its origins, and a summary for the output files; raises
    RuntimeError unless exactly one origin of each kind fits."""
    rate = sampling_rate
    columns = list(dict.fromkeys(column for _, column, _ in units))
    if window.data_origin is not None:
        candidates = [window.data_origin]
    else:
        candidates = data_origins(window)
    excerpts, problems = {}, {}
    for origin in candidates:
        excerpts[origin] = excerpt_matches(
            sorted_data, entry, window, origin, columns, channel_map, rate
        )
        for m in excerpts[origin]:
            log.info(
                "    - temp_wh.dat from the %s's start: excerpt at %.1f s, "
                "channel %d: lag %+d, r %.3f, contrast %.3f",
                origin_name(window, origin),
                m.position / rate,
                channel_map[m.column],
                m.lag,
                m.r,
                m.contrast,
            )
        problems[origin] = excerpt_problem(excerpts[origin], origin, channel_map, rate)
    data_origin = only_fit(window, "the sorted data (temp_wh.dat)", problems)

    if window.spike_origin is not None:
        candidates = [window.spike_origin]
    else:
        candidates = list(dict.fromkeys((window.start, 0)))
    averages, problems = {}, {}
    for origin in candidates:
        averages[origin] = []
        for cluster, column, times in units:
            dset = recording_channel(entry, int(channel_map[column]))
            n, z, lag, r = average_match(
                sorted_data,
                dset,
                times,
                data_origin,
                origin,
                column,
                n_before,
                n_after,
                rate,
            )
            log.info(
                "    - spike times from the %s's start: cluster %d, channel %d: "
                "average of %d spikes %.0f times the noise, recording lag %+d, r %.3f",
                origin_name(window, origin),
                cluster,
                channel_map[column],
                n,
                z,
                lag,
                r,
            )
            averages[origin].append(AverageMatch(cluster, column, n, z, lag, r))
        problems[origin] = average_problem(averages[origin])
    spike_origin = only_fit(window, "the spike times", problems)

    window = window._replace(data_origin=data_origin, spike_origin=spike_origin)
    clear = [m for m in excerpts[data_origin] if m.contrast >= MIN_CONTRAST]
    return window, {
        "sorted_data_from": origin_name(window, data_origin),
        "spike_times_from": origin_name(window, spike_origin),
        "excerpt_min_contrast": round(min(m.contrast for m in clear), 3),
        "spike_average_min_z": round(
            min(a.z for a in averages[spike_origin] if a.n_spikes), 1
        ),
        "spike_average_min_r": round(
            min(a.r for a in averages[spike_origin] if a.n_spikes), 3
        ),
    }
