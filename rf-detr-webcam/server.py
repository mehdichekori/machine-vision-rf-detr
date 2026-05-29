from flask import Flask, request, send_from_directory
from flask_sock import Sock
import threading
import numpy as np
import cv2
import base64
import json
import torch
import asyncio
import time

app = Flask(__name__)
sock = Sock(app)

# Global model state
_models = {}
_model_lock = threading.Lock()
_active_model_name = "Small (43M)"
_confidence_threshold = 0.5


def load_models():
    import os
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge
    from rfdetr.assets.coco_classes import COCO_CLASSES

    # Get absolute path to the local rf_models directory
    rf_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rf_models"))
    os.makedirs(rf_models_dir, exist_ok=True)

    global _models
    MODEL_MAP = {
        "Small (43M)": (RFDETRSmall, "rf-detr-small.pth"),
        "Medium (160M)": (RFDETRMedium, "rf-detr-medium.pth"),
        "Large (227M)": (RFDETRLarge, "rf-detr-large-2026.pth"),
    }

    for name, (cls, filename) in MODEL_MAP.items():
        print(f"Loading {name}...")
        weights_path = os.path.join(rf_models_dir, filename)
        model = cls(pretrain_weights=weights_path)
        model.optimize_for_inference(compile=True, dtype=torch.float16)
        _models[name] = model
    print("All models loaded.")


def get_model(name: str):
    if name in _models:
        return _models[name]
    return _models.get("Small (43M)", None)


# Load models in background thread
_thread = threading.Thread(target=load_models, daemon=True)
_thread.start()


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@sock.route("/ws")
def websocket(ws):
    from rfdetr.assets.coco_classes import COCO_CLASSES
    import supervision as sv

    box_ann = sv.BoxAnnotator()
    label_ann = sv.LabelAnnotator()
    _models_local = {}
    active_model_name = "Small (43M)"
    threshold = 0.5

    def get_m(name):
        if name not in _models_local or _models_local[name] is None:
            _models_local[name] = get_model(name)
        return _models_local[name]

    while True:
        try:
            data = ws.receive()
            if data is None:
                break

            if isinstance(data, bytes):
                # Decode raw JPEG bytes
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Downscale for inference
                h, w = frame.shape[:2]
                scale = 320 / max(h, w)
                if scale < 1.0:
                    frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                else:
                    frame_small = frame.copy()
                    scale = 1.0

                # Run detection
                model = get_m(active_model_name)
                if model is None:
                    ws.send(json.dumps({
                        "type": "status",
                        "message": f"Model '{active_model_name}' is still loading on the server. Please wait...",
                        "statusClass": "error"
                      }))
                    continue

                t_start = time.perf_counter()
                detections = model.predict(frame_small, threshold=threshold)
                inference_ms = (time.perf_counter() - t_start) * 1000.0

                # Scale boxes back to original
                if scale < 1.0:
                    detections.xyxy = (detections.xyxy / scale).astype(np.int64)

                # Build serialized detections
                objects_list = []
                if len(detections) > 0:
                    for cls_id, conf, bbox in zip(detections.class_id, detections.confidence, detections.xyxy):
                        objects_list.append({
                            "class": COCO_CLASSES[int(cls_id)],
                            "confidence": float(conf),
                            "box": [int(x) for x in bbox]
                        })

                # Send telemetry JSON
                ws.send(json.dumps({
                    "type": "detections",
                    "objects": objects_list,
                    "inference_ms": inference_ms
                }))

                # Annotate
                if len(detections) > 0:
                    labels = [
                        f"{COCO_CLASSES[int(cls_id)]} {float(conf):.2f}"
                        for cls_id, conf in zip(detections.class_id, detections.confidence)
                    ]
                    annotated = box_ann.annotate(frame.copy(), detections)
                    annotated = label_ann.annotate(annotated, detections, labels=labels)
                else:
                    annotated = frame.copy()

                # Encode annotated frame as binary JPEG
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                ws.send(buf.tobytes())

            elif isinstance(data, str):
                msg = json.loads(data)

                if msg.get("type") == "init":
                    ws.send(json.dumps({"type": "ready"}))

                elif msg.get("type") == "config":
                    new_model_name = msg.get("model", "Small (43M)")
                    if new_model_name != active_model_name:
                        active_model_name = new_model_name
                        _models_local.clear()  # force reload on next frame
                    threshold = float(msg.get("threshold", 0.5))
                    ws.send(json.dumps({"type": "config_ok", "model": active_model_name}))

        except Exception as e:
            print(f"WS error: {e}")
            break


if __name__ == "__main__":
    print("Starting RF-DETR WebSocket server at http://localhost:7861")
    print("Open http://localhost:7861 in your browser")
    app.run(host="0.0.0.0", port=7861, threaded=True, debug=False)
