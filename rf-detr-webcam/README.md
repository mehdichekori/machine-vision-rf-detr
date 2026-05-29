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

- All 4 model sizes are pre-loaded and JIT-compiled on server startup (float16, `compile=True`)
- Frames are captured at ~15 FPS max (66ms throttle) to stay well within inference budget
- Isolated inference runs at ~35 FPS at 320px on Apple Silicon (MPS) / modern GPUs

## Requirements

- Python 3.10+
- **Hardware Acceleration:** Recommended to have a GPU/MPS-capable device (NVIDIA CUDA or Apple Silicon MPS) to run the models efficiently.
- **System Memory:** 
  - **16GB+ RAM** is recommended when loading all models simultaneously (especially Large & Segment).
  - Small and Medium models can run on CPUs and lower-resource environments.
- Webcam access via browser

## Model Sizes & Downloads

| Model | Params | Filename | Recommended for |
|-------|--------|----------|----------------|
| **Small (43M)** | 43M | `rf-detr-small.pth` | FPS-focused, CPU / Edge devices |
| **Segment Small (129M)** | 129M | `rf-detr-seg-small.pt` | Real-time instance segmentation (Requires GPU/MPS) |
| **Medium (160M)** | 160M | `rf-detr-medium.pth` | Balanced, CPU / Mid-range GPU |
| **Large (227M)** | 227M | `rf-detr-large-2026.pth` | Best accuracy, requires GPU/MPS acceleration |

### How Models are Downloaded & Stored

The server is configured to load these models locally:
1. **Cache Location:** On startup, the server creates (if not already present) an `rf_models/` directory at the **root of the workspace repository** (`computer-vision/rf_models/`).
2. **Automatic Download:** When `server.py` initializes, it checks if the model weight files (`.pth` or `.pt`) exist in the `rf_models/` folder.
3. **Download Process:** If any file is missing, the underlying `rfdetr` package automatically fetches the official pre-trained weights from Roboflow's public Google Cloud Storage bucket and saves them directly to `computer-vision/rf_models/`. 
4. **Manual Placement (Optional):** If you prefer to download them manually, you can download the weights from their respective URLs (defined in `rfdetr.assets.model_weights`) and place them directly in the `rf_models/` directory before starting the server.
