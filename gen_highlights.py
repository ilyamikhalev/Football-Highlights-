"""
Football Highlights Generator

Usage
-----
  python gen_highlights.py <match_video.mp4> [options]

Options
-------
  -o, --output     Output file path (default: highlights.mp4)
  --before         Seconds of footage before each event (default: 30)
  --after          Seconds of footage after each event (default: 30)
  --shots          Also include audio-detected key moments (shots / near-misses)
  --max-shots      Max number of audio events to include (default: 10)
"""

import argparse
import pandas as pd
from moviepy.editor import VideoFileClip, concatenate_videoclips

from get_dataframe import get_dataframe
from detect_events import detect_audio_events

CLIP_BEFORE = 30.0
CLIP_AFTER = 30.0


def _goal_timestamps(df):
    df = df.copy()
    df["Score1"] = df["Score1"].replace(["O"], "0")
    df["Score2"] = df["Score2"].replace(["O"], "0")

    prev1 = df["Score1"].shift(1, fill_value=df["Score1"].iloc[0])
    prev2 = df["Score2"].shift(1, fill_value=df["Score2"].iloc[0])

    goals = df.loc[prev1 != df["Score1"], "Timestamp"].tolist() + \
            df.loc[prev2 != df["Score2"], "Timestamp"].tolist()
    return sorted(goals)


def _merge_timestamps(timestamps, min_gap=5.0):
    """Remove timestamps that are within min_gap seconds of an earlier one."""
    merged = []
    for t in sorted(timestamps):
        if not merged or t - merged[-1] >= min_gap:
            merged.append(t)
    return merged


def _make_clip(clip, timestamp, clip_before, clip_after):
    start = max(0.0, timestamp - clip_before)
    end = min(clip.duration, timestamp + clip_after)
    return clip.subclip(start, end)


def generate_highlights(
    match_video,
    output_path="highlights.mp4",
    clip_before=CLIP_BEFORE,
    clip_after=CLIP_AFTER,
    include_shots=False,
    max_shots=10,
):
    print(f"Processing: {match_video}")

    csv_path = get_dataframe(match_video)
    df = pd.read_csv(csv_path)

    goal_times = _goal_timestamps(df)
    print(f"Goals detected ({len(goal_times)}): {goal_times}")

    event_times = list(goal_times)

    if include_shots:
        audio_times = detect_audio_events(match_video, top_n=max_shots + len(goal_times))
        # Keep audio events not already covered by a goal clip
        new_events = [
            t for t in audio_times
            if not any(abs(t - g) < clip_before + clip_after for g in goal_times)
        ][:max_shots]
        print(f"Additional audio events detected ({len(new_events)}): {new_events}")
        event_times = sorted(event_times + new_events)

    event_times = _merge_timestamps(event_times)

    if not event_times:
        print("No events detected — no highlights to produce.")
        return

    clip = VideoFileClip(match_video)
    clips = [_make_clip(clip, t, clip_before, clip_after) for t in event_times]

    final = concatenate_videoclips(clips)
    final.write_videofile(output_path)
    clip.close()
    print(f"Highlights saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate football match highlights")
    parser.add_argument("video", help="Path to the full match video (MP4)")
    parser.add_argument("-o", "--output", default="highlights.mp4",
                        help="Output highlights file (default: highlights.mp4)")
    parser.add_argument("--before", type=float, default=CLIP_BEFORE,
                        help="Seconds before each event (default: 30)")
    parser.add_argument("--after", type=float, default=CLIP_AFTER,
                        help="Seconds after each event (default: 30)")
    parser.add_argument("--shots", action="store_true",
                        help="Include audio-detected shots / near-misses")
    parser.add_argument("--max-shots", type=int, default=10,
                        help="Max audio events to include (default: 10)")
    args = parser.parse_args()

    generate_highlights(
        args.video,
        output_path=args.output,
        clip_before=args.before,
        clip_after=args.after,
        include_shots=args.shots,
        max_shots=args.max_shots,
    )
