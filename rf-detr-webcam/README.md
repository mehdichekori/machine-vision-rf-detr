# RF-DETR Webcam Detection

Real-time object detection in the browser using your webcam and Roboflow's RF-DETR vision transformer. Low-latency binary WebSocket streaming with a pure JavaScript frontend — no base64 parsing or HTTP polling overhead.

## Setup

Ensure you are inside the `rf-detr-webcam` directory:

```bash
cd rf-detr-webcam
```

If a virtual environment exists, activate it:
```bash
source .venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

## Run

If your virtual environment is active:
```bash
python server.py
```

Otherwise, you can run the server directly using `python3` or the virtualenv python:
```bash
python3 server.py
# or
./.venv/bin/python3 server.py
```

Open http://localhost:7861 in your browser, grant webcam permission, then click **Camera On**. Use the **Detection On/Off** toggle to pause and resume inference without stopping the camera feed.

## Controls

| Control | Description |
|---------|-------------|
| **Camera On/Off** | Starts or stops the webcam stream |
| **Detection On/Off** | Sends frames to the server for inference (camera keeps running) |
| **Model** | Choose inference engine: Small (43M), Medium (160M), Large (227M), or Segment Small (129M) |
| **Confidence** | Filter detections by minimum confidence threshold |
| **Box Outline Style** | Configure object boundaries (Standard Box, Corners Only, or None) |
| **Active Overlays** | Toggle additional overlays: Labels, Color Fill, Filled Mask (segmentation), Center Dot, Triangle Pointer, and Confidence Bar |
| **Trigger Alarm Center** | Select a target class (from the 80 COCO classes dropdown) to activate warning banners, Audio Sirens, or Speech Synthesized announcements |

## Architecture

```
Browser (getUserMedia)
  → Canvas frame blob (raw JPEG binary bytes)
  → WebSocket → Flask server (port 7861)
    → RF-DETR inference at 320px max
    → supervision BoxAnnotator + LabelAnnotator
    → annotated JPEG bytes (raw binary) returned over WebSocket
  → Canvas overlay rendered from Object URL
```

- All 3 model sizes are pre-loaded and JIT-compiled on server startup (float16, `compile=True`)
- Frames are captured at ~15 FPS max (66ms throttle) to stay well within inference budget
- Isolated inference runs at ~35 FPS at 320px on M4 Max

## Requirements

- Python 3.10+
- Mac M4 Max / 48GB RAM (Large model runs fine; smaller models work on any hardware)
- Webcam access via browser

## Model Sizes

| Model | Params | Recommended for |
|-------|--------|----------------|
| Small (43M) | 43M | FPS-focused, CPU |
| Medium (160M) | 160M | Balanced |
| Large (227M) | 227M | Best accuracy, M4 Max |

Models are loaded from and cached locally in the repository's `rf_models/` directory (created at the repository root). If they are not present, they will be automatically downloaded and saved to that folder on first run.
