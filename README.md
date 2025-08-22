# Phone Usage Detection — Training, Inference, and Live Viewer

This repository contains scripts to train and run a YOLO-based detector for handheld phone usage in videos, preserve video properties (resolution/FPS and audio), apply motion filtering to suppress static phones, and visualize results in near real-time.

- Dataset prep (COCO → phone-only): `scripts/download_coco_phone_fiftyone.py`
- Offline inference and logging: `detect_phone_usage.py`
- Live viewer (browser or OpenCV window): `live_phone_server.py`


## 0) Download Required Assets (from Google Drive)

Since large artifacts are not included in this repo, please download the prepared assets from Google Drive (https://drive.google.com/file/d/1nqyg7qD6fXMl8T8SicEL2FxfHLmEJQeV/view?usp=sharing):

- `datasets/coco_phone_yolo/` (cleaned COCO → phone-only YOLO dataset)
- `runs/detect/` (trained model experiments; includes `weights/best.pt`)

After downloading, place the folders into the project root so the structure looks like:

```
.
├── datasets/
│   └── coco_phone_yolo/
│       ├── train/
│       ├── val/
│       └── data.yaml
├── runs/
│   └── detect/
│       └── <experiment_name>/
│           └── weights/
│               └── best.pt
└── ... (scripts and code in this repo)
```

Example commands (after extracting the Drive zip files):

```bash
# Create directories if absent
mkdir -p datasets runs

# Copy the dataset folder
cp -R /path/to/downloads/coco_phone_yolo  datasets/

# Copy the training runs (contains best.pt)
cp -R /path/to/downloads/detect  runs/

# Verify expected files
ls datasets/coco_phone_yolo && ls runs/detect/*/weights/best.pt
```

If you only want to run inference and not training, you only need the `runs/detect/*/weights/best.pt` file. If you plan to retrain on Kaggle, having `datasets/coco_phone_yolo/` locally is optional, as you can upload it separately to Kaggle.


## 1) Setup

- Python 3.10/3.11 recommended (Apple Silicon supported)
- Create venv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- Optional (for audio remux on annotated outputs):
  - macOS: `brew install ffmpeg`

- Optional (for proximity filtering with hands/face):
  - Install MediaPipe if your environment supports it (commented in `requirements.txt`).


## 2) Dataset: COCO → Phone-only (FiftyOne)

This script filters COCO 2017 to the "cell phone" class and exports a clean YOLO dataset.

```bash
# Downloads COCO, filters to cell phone class, exports to datasets/coco_phone_yolo/
.venv/bin/python scripts/download_coco_phone_fiftyone.py \
  --out_dir datasets/coco_phone_yolo
```

- Output structure: `datasets/coco_phone_yolo/{train,val}/` and `data.yaml`
- You can zip `datasets/coco_phone_yolo/` and upload to Kaggle for training.


## 3) Training (Kaggle)

In a Kaggle Notebook:

```python
!pip install ultralytics fiftyone

from ultralytics import YOLO

# Path to your uploaded dataset data.yaml
DATA = '/kaggle/input/coco-phone-yolo/data.yaml'

model = YOLO('yolov8n.pt')  # or 'yolo11n.pt'
model.train(data=DATA, epochs=50, imgsz=640, batch=-1, project='runs', name='coco_phone')
# Best weights saved under /kaggle/working/runs/detect/coco_phone/weights/best.pt
```

Tips for accuracy: try `yolov8s/yolo11s`, more epochs, higher `imgsz`, and more curated negatives (static phones).


## 4) Offline Inference and Logging

Run detection on a video, preserve original resolution/FPS, and remux audio (if `ffmpeg` is available). Motion filtering helps avoid static-phone false positives.

```bash
.venv/bin/python detect_phone_usage.py \
  --source phone.mp4 \
  --weights runs/detect/best.pt \
  --device mps \
  --conf 0.35 \
  --only_active \
  --gap_sec 0.5 --min_usage_sec 0.5
```

- Outputs:
  - Annotated video: `annotated_<source_basename>.mp4`
  - Usage log CSV: `usage_log_<source_basename>.csv`
  - Summary TXT: `usage_summary_<source_basename>.txt`

Flags you may tune:
- `--conf`, `--iou`: detector thresholds
- `--only_active`: display only filtered active-use boxes
- `--gap_sec`, `--min_usage_sec`: merge/threshold usage segments


## 5) Live Viewer (Real-time)

Two modes are available:

A) Browser (Flask/MJPEG)
```bash
.venv/bin/python live_phone_server.py \
  --source phone.mp4 \
  --weights runs/detect/best.pt \
  --device mps \
  --imgsz 416 \
  --conf 0.4 \
  --target_fps 0 \
  --jpeg_quality 65 \
  --drop_if_lag \
  --loop
# Open http://127.0.0.1:8000
```

B) OpenCV Window (lowest latency)
```bash
.venv/bin/python live_phone_server.py \
  --source phone.mp4 \
  --weights runs/detect/best.pt \
  --device mps \
  --imgsz 352 \
  --conf 0.5 \
  --target_fps 0 \
  --opencv_view \
  --loop
# Press 'q' to quit the window
```

Notes:
- Original resolution is preserved for rendering; inference runs on a resized copy for speed.
- OpenCV optimizations and asynchronous detection are enabled for smooth playback.


## 6) Packaging & Submission Checklist

- Artifacts to include:
  - Annotated videos: `annotated*.mp4`
  - Usage logs: `usage_log*.csv`
  - Usage summaries: `usage_summary*.txt`
  - Trained weights: `runs/detect/<exp>/weights/best.pt`
  - Dataset (optional): `datasets/coco_phone_yolo/`
  - This repository: scripts + README + requirements.txt

- Example packaging commands:
```bash
# Zip dataset (optional)
zip -r datasets.zip datasets/coco_phone_yolo

# Zip runs (trained weights and metrics)
zip -r runs.zip runs/detect

# Collect selected outputs
zip -r submission_artifacts.zip \
  annotated*.mp4 usage_log*.csv usage_summary*.txt runs/detect/*/weights/best.pt
```


## 7) Troubleshooting

- MediaPipe install issues:
  - Keep it optional. The pipeline runs without proximity filtering.
- Audio missing in annotated video:
  - Ensure `ffmpeg` is installed and accessible in PATH.
- Live viewer not smooth:
  - Use `--opencv_view` mode; lower `--imgsz` (e.g., 352/320), raise `--conf`, or enable `--drop_if_lag` (Flask mode).
- Apple MPS:
  - `ultralytics` uses MPS on Apple Silicon when available; otherwise CPU or CUDA.


## 8) Repository Layout

```
.
├── detect_phone_usage.py
├── live_phone_server.py
├── scripts/
│   └── download_coco_phone_fiftyone.py
├── datasets/
│   └── coco_phone_yolo/ (generated)
├── runs/ (trained outputs)
├── requirements.txt
├── README.md
└── ...
```
