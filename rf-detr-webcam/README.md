# RF-DETR Webcam Detection

Real-time object detection in the browser using your webcam and Roboflow's RF-DETR vision transformer.

## Setup

```bash
cd rf-detr-webcam
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open http://localhost:7860 in your browser, grant webcam permission, select a model size (Large recommended for M4 Max), adjust the confidence threshold, and click Enable Camera.

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

Models download automatically on first use from Roboflow's hosted ONNX weights.
