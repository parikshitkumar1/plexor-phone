#!/usr/bin/env python3
"""
Detect active phone usage in videos.
- Accepts MP4/AVI/MOV
- Preserves original FPS/resolution, and remuxes original audio if ffmpeg is available
- Draws bounding boxes and labels with optional confidences
- Filters out static phones via motion gating and uses proximity to hands/face for "active" decision
- Logs usage timestamps and a summary report

Usage:
  python detect_phone_usage.py \
    --input path/to/input.mp4 \
    --weights path/to/best.pt \
    --output annotated.mp4 \
    --log usage_log.csv \
    --summary usage_summary.txt \
    --device mps  # or 0 for CUDA, or cpu

Dependencies:
  pip install ultralytics opencv-python mediapipe numpy
Optional:
  ffmpeg in PATH for audio remux
"""

import argparse
import os
import sys
import tempfile
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# MediaPipe
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    HAS_MP = True
except Exception:
    mp = None
    mp_hands = None
    mp_face = None
    HAS_MP = False


@dataclass
class UsageEvent:
    start_frame: int
    end_frame: int
    confidence: float


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)


def center_distance(a, b):
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return np.hypot(ax - bx, ay - by)


def bbox_from_landmarks(landmarks, w, h, pad=10):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = max(0, int(min(xs) - pad)), min(w - 1, int(max(xs) + pad))
    y1, y2 = max(0, int(min(ys) - pad)), min(h - 1, int(max(ys) + pad))
    return [x1, y1, x2, y2]


def merge_segments(events: List[UsageEvent], fps: float, gap_sec=1.0, min_dur_sec=0.3) -> List[UsageEvent]:
    if not events:
        return []
    merged = []
    cur = events[0]
    for e in events[1:]:
        gap_frames = e.start_frame - cur.end_frame
        if gap_frames <= int(gap_sec * fps):
            cur.end_frame = max(cur.end_frame, e.end_frame)
            cur.confidence = max(cur.confidence, e.confidence)
        else:
            if (cur.end_frame - cur.start_frame + 1) >= int(min_dur_sec * fps):
                merged.append(cur)
            cur = e
    if (cur.end_frame - cur.start_frame + 1) >= int(min_dur_sec * fps):
        merged.append(cur)
    return merged


def remux_original_audio(input_path: str, annotated_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", annotated_path,
        "-i", input_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input video file (MP4/AVI/MOV)")
    ap.add_argument("--weights", required=True, help="Path to YOLO weights (pt)")
    ap.add_argument("--output", default="annotated.mp4", help="Output annotated mp4")
    ap.add_argument("--log", default="usage_log.csv", help="CSV log of usage intervals")
    ap.add_argument("--summary", default="usage_summary.txt", help="Text summary report")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold")
    ap.add_argument("--device", default=None, help="'0' for first GPU, 'mps' for Apple, 'cpu' for CPU")
    ap.add_argument("--coco_phone", action="store_true", help="If using COCO weights, filter class 67 (cell phone)")
    ap.add_argument("--min_move_px", type=float, default=4.0, help="Min centroid movement (px) to consider non-static")
    ap.add_argument("--proximity_px", type=float, default=120.0, help="Distance (px) threshold for hand/face proximity")
    ap.add_argument("--draw_conf", action="store_true", help="Draw numeric confidences on boxes")
    ap.add_argument("--only_active", action="store_true", help="Render only active-use phone boxes")
    ap.add_argument("--gap_sec", type=float, default=1.0, help="Max gap (sec) to merge adjacent usage segments")
    ap.add_argument("--min_usage_sec", type=float, default=0.3, help="Minimum duration (sec) to keep a usage segment")
    return ap.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_out = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (w, h))

    model = YOLO(args.weights)
    coco_phone_id = 67

    previous_centers = deque(maxlen=8)

    usage_frames: List[UsageEvent] = []
    active_run: Optional[UsageEvent] = None
    frame_idx = 0

    hands = None
    face = None
    if HAS_MP:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        face = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_res = None
            face_res = None
            if HAS_MP and hands is not None:
                hand_res = hands.process(rgb)
            if HAS_MP and face is not None:
                face_res = face.process(rgb)

            hand_bboxes = []
            if HAS_MP and hand_res and hand_res.multi_hand_landmarks:
                for lm in hand_res.multi_hand_landmarks:
                    hand_bboxes.append(bbox_from_landmarks(lm.landmark, w, h, pad=12))

            face_bboxes = []
            if HAS_MP and face_res and face_res.detections:
                for det in face_res.detections:
                    rel = det.location_data.relative_bounding_box
                    x1 = max(0, int(rel.xmin * w))
                    y1 = max(0, int(rel.ymin * h))
                    x2 = min(w - 1, int((rel.xmin + rel.width) * w))
                    y2 = min(h - 1, int((rel.ymin + rel.height) * h))
                    face_bboxes.append([x1, y1, x2, y2])

            phone_boxes: List[Tuple[List[int], float]] = []
            for r in results:
                for b in r.boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    if args.coco_phone and cls_id != coco_phone_id:
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    phone_boxes.append(([x1, y1, x2, y2], conf))

            centers = []
            for bb, _ in phone_boxes:
                cx = (bb[0] + bb[2]) / 2.0
                cy = (bb[1] + bb[3]) / 2.0
                centers.append((cx, cy))
            previous_centers.append(centers)

            phone_active = False
            strongest_conf = 0.0

            for bb, conf in phone_boxes:
                move_ok = False
                if len(previous_centers) >= 2:
                    last_centers = previous_centers[-2]
                    if last_centers:
                        cx = (bb[0] + bb[2]) / 2.0
                        cy = (bb[1] + bb[3]) / 2.0
                        dists = [np.hypot(cx - pcx, cy - pcy) for pcx, pcy in last_centers]
                        if dists and min(dists) >= args.min_move_px:
                            move_ok = True

                prox_ok = False
                for hb in hand_bboxes:
                    if iou_xyxy(bb, hb) > 0.02 or center_distance(bb, hb) < args.proximity_px:
                        prox_ok = True
                        break
                if not prox_ok:
                    for fb in face_bboxes:
                        if center_distance(bb, fb) < args.proximity_px:
                            prox_ok = True
                            break

                is_active = prox_ok or move_ok

                if is_active or not args.only_active:
                    color = (0, 255, 0) if is_active else (80, 80, 80)
                    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    label = "phone_use" if is_active else "phone_static"
                    if args.draw_conf:
                        label += f" {conf:.2f}"
                    cv2.putText(frame, label, (bb[0], max(0, bb[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if is_active:
                    phone_active = True
                    strongest_conf = max(strongest_conf, conf)

            if phone_active and active_run is None:
                active_run = UsageEvent(start_frame=frame_idx, end_frame=frame_idx, confidence=strongest_conf)
            elif phone_active and active_run is not None:
                active_run.end_frame = frame_idx
                active_run.confidence = max(active_run.confidence, strongest_conf)
            elif not phone_active and active_run is not None:
                usage_frames.append(active_run)
                active_run = None

            writer.write(frame)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        if HAS_MP and hands is not None:
            hands.close()
        if HAS_MP and face is not None:
            face.close()

    merged = merge_segments(usage_frames, fps=fps, gap_sec=args.gap_sec, min_dur_sec=args.min_usage_sec)
    with open(args.log, "w") as f:
        f.write("start_time_s,end_time_s,duration_s,confidence\n")
        for e in merged:
            st = e.start_frame / fps
            et = e.end_frame / fps
            du = et - st
            f.write(f"{st:.3f},{et:.3f},{du:.3f},{e.confidence:.3f}\n")

    total = sum((e.end_frame - e.start_frame + 1) / fps for e in merged)
    with open(args.summary, "w") as f:
        f.write("Phone Usage Summary\n")
        f.write(f"Video: {os.path.basename(args.input)}\n")
        f.write(f"FPS: {fps:.3f}, Resolution: {w}x{h}\n")
        f.write(f"Total usage time: {total:.2f} seconds\n")
        f.write(f"Occurrences: {len(merged)}\n")
        for i, e in enumerate(merged, 1):
            st = e.start_frame / fps
            et = e.end_frame / fps
            f.write(f"{i}. {st:.2f}s - {et:.2f}s ({et - st:.2f}s), conf={e.confidence:.2f}\n")

    ok = remux_original_audio(args.input, tmp_out, args.output)
    if not ok:
        import shutil
        shutil.move(tmp_out, args.output)
        print("Warning: ffmpeg not available or failed; delivered video without original audio.")

    print(f"Annotated video saved to: {args.output}")
    print(f"Usage log saved to: {args.log}")
    print(f"Summary saved to: {args.summary}")


if __name__ == "__main__":
    main()
