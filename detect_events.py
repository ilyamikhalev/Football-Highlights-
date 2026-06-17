"""
Audio-based key moment detector.

Crowd noise energy spikes reliably correlate with shots, saves, near-misses,
and other exciting moments in football — even when the scoreboard doesn't change.
"""

import numpy as np
from moviepy.editor import VideoFileClip


def _energy_series(samples, window_size):
    """RMS energy per window."""
    n_windows = len(samples) // window_size
    trimmed = samples[: n_windows * window_size]
    frames = trimmed.reshape(n_windows, window_size)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def _smooth(arr, radius):
    kernel = np.ones(radius * 2 + 1) / (radius * 2 + 1)
    return np.convolve(arr, kernel, mode='same')


def detect_audio_events(match_video, min_gap_seconds=20.0, top_n=20):
    """
    Return timestamps (seconds) of the top crowd-noise peaks in the video.

    Parameters
    ----------
    match_video     : path to the match video
    min_gap_seconds : minimum seconds between two returned events
    top_n           : maximum number of events to return
    """
    clip = VideoFileClip(match_video)
    if clip.audio is None:
        print("No audio track found — skipping audio event detection.")
        clip.close()
        return []

    audio_fps = 22050
    energy_fps = 10  # analyse energy at 10 Hz (one value per 0.1 s)
    window = audio_fps // energy_fps

    samples = clip.audio.set_fps(audio_fps).to_soundarray(fps=audio_fps)
    clip.close()

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    energy = _energy_series(samples.astype(np.float32), window)
    smoothed = _smooth(energy, radius=energy_fps * 2)  # 2-second smoothing

    threshold = smoothed.mean() + smoothed.std()

    # Find local maxima above threshold with minimum separation
    min_dist = int(min_gap_seconds * energy_fps)
    peaks = []
    last_peak = -min_dist
    for idx in np.argsort(smoothed)[::-1]:
        if smoothed[idx] < threshold:
            break
        if idx - last_peak >= min_dist:
            peaks.append((idx, smoothed[idx]))
            last_peak = idx
        if len(peaks) >= top_n:
            break

    # Return sorted by time
    timestamps = sorted(idx / energy_fps for idx, _ in peaks)
    return timestamps
