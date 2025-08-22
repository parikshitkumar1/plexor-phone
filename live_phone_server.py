#!/usr/bin/env python3
import argparse
import time
from threading import Thread, Lock

import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Phone Usage Live Viewer</title>
  <style>
    body { background:#111; color:#eee; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
    .wrap { max-width: 1080px; margin: 24px auto; }
    h1 { font-size: 18px; font-weight: 600; }
    img { width: 100%; height: auto; background:#000; }
    .meta { opacity: .7; font-size: 13px; margin: 8px 0 16px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Phone Usage Detector — Live Stream</h1>
    <div class="meta">Source: {{ video_source }} | Model: {{ model }}</div>
    <img src="/video_feed" />
    <p class="meta">Press Ctrl+C in the terminal to stop the server.</p>
  </div>
</body>
</html>
"""

app = Flask(__name__)

# OpenCV perf tweaks
try:
    cv2.setNumThreads(1)
    cv2.useOptimized(True)
except Exception:
    pass

lock = Lock()
latest_jpeg = None
latest_frame_np = None  # last decoded BGR frame (original resolution)
last_boxes_conf = []    # last detections as (x1,y1,x2,y2,conf)


def draw_detections(frame, boxes_conf):
    for (x1, y1, x2, y2, conf) in boxes_conf:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"phone {conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
    return frame


def detect_loop(model, args):
    """Runs YOLO on the most recent frame asynchronously and updates last_boxes_conf."""
    global latest_frame_np, last_boxes_conf
    while True:
        # snapshot the latest frame
        with lock:
            frame = None if latest_frame_np is None else latest_frame_np.copy()
        if frame is None:
            time.sleep(0.002)
            continue
        # resize for inference
        h, w = frame.shape[:2]
        rsz = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        sx = w / float(args.imgsz)
        sy = h / float(args.imgsz)
        # infer
        results = model.predict(source=rsz, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
        cur_boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                conf = float(b.conf[0].item())
                cur_boxes.append((x1, y1, x2, y2, conf))
        with lock:
            last_boxes_conf = cur_boxes
        # small sleep to avoid 100% busy loop
        time.sleep(0.0005)


def streamer(args):
    global latest_jpeg, latest_frame_np, last_boxes_conf
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    # set a reasonable read size
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    model = YOLO(args.weights)
    try:
        model.fuse()
    except Exception:
        pass
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_fps = args.target_fps if args.target_fps > 0 else src_fps
    frame_interval = 1.0 / max(1e-6, target_fps)

    # start async detector thread
    t_det = Thread(target=detect_loop, args=(model, args), daemon=True)
    t_det.start()

    while True:
        loop_start = time.time()
        ok, frame = cap.read()
        if not ok:
            if args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        # publish latest frame for detector and draw last results
        with lock:
            latest_frame_np = frame.copy()
            boxes = list(last_boxes_conf)
        frame = draw_detections(frame, boxes)

        # Display or encode depending on mode
        if args.opencv_view:
            cv2.imshow('Phone Usage Live', frame)
            # calculate remaining time for exact pacing
            elapsed = time.time() - loop_start
            delay_ms = max(1, int(max(0.0, frame_interval - elapsed) * 1000))
            if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                break
        else:
            # encode JPEG for MJPEG streaming
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if not ret:
                continue
            with lock:
                latest_jpeg = jpeg.tobytes()
        # pacing to target FPS; optionally drop frames if lagging to keep smoothness
        elapsed = time.time() - loop_start
        if not args.opencv_view:
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            else:
                if args.drop_if_lag:
                    to_drop = int(elapsed / frame_interval) - 1
                    for _ in range(max(0, to_drop)):
                        cap.grab()

    cap.release()
    if args.opencv_view:
        cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template_string(HTML, video_source=args.source, model=args.weights)


@app.route('/video_feed')
def video_feed():
    def gen():
        boundary = b'--frame\r\n'
        while True:
            with lock:
                frame = latest_jpeg
            if frame is None:
                time.sleep(0.01)
                continue
            yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='phone.mp4', help='Video path or webcam index (0)')
    p.add_argument('--weights', default='runs/detect/best.pt', help='YOLO weights path')
    p.add_argument('--device', default='mps', help='mps, 0 (cuda), or cpu')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--width', type=int, default=0)
    p.add_argument('--height', type=int, default=0)
    p.add_argument('--loop', action='store_true', help='Loop the video when it ends')
    p.add_argument('--target_fps', type=float, default=0.0, help='Target output FPS (0 = source fps)')
    p.add_argument('--detect_every', type=int, default=2, help='Run detection every N frames (reuse last boxes)')
    p.add_argument('--jpeg_quality', type=int, default=75, help='MJPEG quality (lower = faster)')
    p.add_argument('--drop_if_lag', action='store_true', help='Drop frames when behind target FPS for smoothness')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--opencv_view', action='store_true', help='Render with OpenCV window instead of Flask/MJPEG')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.opencv_view:
        # Run viewer directly (blocking), no Flask
        streamer(args)
    else:
        t = Thread(target=streamer, args=(args,), daemon=True)
        t.start()
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
