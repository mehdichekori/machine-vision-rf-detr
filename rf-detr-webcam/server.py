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
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRSegSmall
    from rfdetr.assets.coco_classes import COCO_CLASSES

    # Get absolute path to the local rf_models directory
    rf_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rf_models"))
    os.makedirs(rf_models_dir, exist_ok=True)

    global _models
    MODEL_MAP = {
        "Small (43M)": (RFDETRSmall, "rf-detr-small.pth"),
        "Medium (160M)": (RFDETRMedium, "rf-detr-medium.pth"),
        "Large (227M)": (RFDETRLarge, "rf-detr-large-2026.pth"),
        "Segment Small (129M)": (RFDETRSegSmall, "rf-detr-seg-small.pt"),
    }

    for name, (cls, filename) in MODEL_MAP.items():
        weights_path = os.path.join(rf_models_dir, filename)
        if not os.path.exists(weights_path):
            print(f"{name} weights not found locally at '{weights_path}'. Downloading weights now (this may take a few minutes)...")
        else:
            print(f"Loading {name} from local cache...")
            
        model = cls(pretrain_weights=weights_path)
        print(f"Optimizing {name} for inference...")
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
    corner_ann = sv.BoxCornerAnnotator()
    color_ann = sv.ColorAnnotator()
    dot_ann = sv.DotAnnotator()
    triangle_ann = sv.TriangleAnnotator()
    percentage_bar_ann = sv.PercentageBarAnnotator()
    label_ann = sv.LabelAnnotator()
    mask_ann = sv.MaskAnnotator()

    annotations_config = {
        "boxType": "standard",
        "showLabels": True,
        "showColorFill": False,
        "showCenterDot": False,
        "showTriangle": False,
        "showConfidenceBar": False,
        "showMask": False
    }

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

                    # Also upscale segmentation masks to original frame resolution
                    if getattr(detections, 'mask', None) is not None and detections.mask is not None:
                        orig_h, orig_w = frame.shape[:2]
                        resized_masks = []
                        for m in detections.mask:
                            m_uint8 = (m.astype(np.uint8)) * 255
                            m_resized = cv2.resize(m_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                            resized_masks.append(m_resized > 127)
                        detections.mask = np.array(resized_masks)

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
                annotated = frame.copy()
                if len(detections) > 0:
                    # 0. Filled Mask (bottom layer, requires segment model)
                    if annotations_config.get("showMask", False) and getattr(detections, 'mask', None) is not None:
                        try:
                            annotated = mask_ann.annotate(annotated, detections)
                        except Exception as mask_err:
                            print(f"[mask] Annotation error: {mask_err}")

                    # 1. Color Fill (bottom layer)
                    if annotations_config.get("showColorFill", False):
                        annotated = color_ann.annotate(annotated, detections)

                    # 2. Box outline
                    box_type = annotations_config.get("boxType", "standard")
                    if box_type == "standard":
                        annotated = box_ann.annotate(annotated, detections)
                    elif box_type == "corners":
                        annotated = corner_ann.annotate(annotated, detections)

                    # 3. Center Dot
                    if annotations_config.get("showCenterDot", False):
                        annotated = dot_ann.annotate(annotated, detections)

                    # 4. Triangle Pointer
                    if annotations_config.get("showTriangle", False):
                        annotated = triangle_ann.annotate(annotated, detections)

                    # 5. Confidence Bar
                    if annotations_config.get("showConfidenceBar", False):
                        annotated = percentage_bar_ann.annotate(annotated, detections)

                    # 6. Labels (top layer)
                    if annotations_config.get("showLabels", True):
                        labels = [
                            f"{COCO_CLASSES[int(cls_id)]} {float(conf):.2f}"
                            for cls_id, conf in zip(detections.class_id, detections.confidence)
                        ]
                        annotated = label_ann.annotate(annotated, detections, labels=labels)

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
                    
                    # Update annotations config
                    if "annotations" in msg:
                        annotations_config.update(msg["annotations"])
                        
                    ws.send(json.dumps({"type": "config_ok", "model": active_model_name}))

        except Exception as e:
            print(f"WS error: {e}")
            break


if __name__ == "__main__":
    print("Starting RF-DETR WebSocket server at http://localhost:7861")
    print("Open http://localhost:7861 in your browser")
    app.run(host="0.0.0.0", port=7861, threaded=True, debug=False)
