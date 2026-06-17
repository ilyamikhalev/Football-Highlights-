"""
Scoreboard cropper — uses YOLOv8 to track and crop the scoreboard region
across all frames, producing a processed video for OCR.

Usage
-----
  python gen_reels.py <input_video.mp4> [output_video.mp4]
"""

import argparse
import math
import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")


def box_center(coords):
    left, top, right, bottom = coords
    return [(left + right) / 2, (top + bottom) / 2]


def closest_box(boxes, coords):
    center = box_center(coords)
    distances = [math.dist(box_center(b.xyxy[0].numpy().astype(int)), center) for b in boxes]
    return boxes[distances.index(min(distances))]


def adjust_box_size(coords, box_width, box_height):
    cx, cy = box_center(coords)
    return [cx - box_width / 2, cy - box_height / 2, cx + box_width / 2, cy + box_height / 2]


def adjust_boundaries(coords, screen_w, screen_h):
    left, top, right, bottom = coords
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > screen_w:
        left -= right - screen_w
        right = screen_w
    if bottom > screen_h:
        top -= bottom - screen_h
        bottom = screen_h
    return [round(left), round(top), round(right), round(bottom)]


def crop_scoreboard(file_source, file_target, crop_coords=None):
    vid = cv2.VideoCapture(file_source)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if crop_coords:
        left, top, right, bottom = crop_coords
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
    else:
        left, top, right, bottom = 0, 0, width, height

    box_width = right - left
    box_height = bottom - top
    last_coords = [left, top, right, bottom]

    writer = cv2.VideoWriter(
        file_target, cv2.VideoWriter_fourcc(*'MPEG'), fps, (box_width, box_height)
    )

    frame_n = 1
    while True:
        ret, frame = vid.read()
        if not ret:
            print("Done.")
            break
        print(f"Frame: {frame_n}")
        frame_n += 1

        results = model.predict(source=frame, conf=0.3, iou=0.2, device='0')
        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            box = closest_box(boxes, last_coords)
            last_coords = box.xyxy[0].numpy().astype(int)
            new_coords = adjust_box_size(last_coords, box_width, box_height)
            left, top, right, bottom = adjust_boundaries(new_coords, width, height)

        cropped = frame[top:bottom, left:right]
        writer.write(cropped)

    vid.release()
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop scoreboard region from match video")
    parser.add_argument("input", help="Input match video")
    parser.add_argument("output", nargs="?", default="processed.mp4", help="Output cropped video")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="Initial crop box (pixels)")
    args = parser.parse_args()

    crop_scoreboard(args.input, args.output, args.crop)
