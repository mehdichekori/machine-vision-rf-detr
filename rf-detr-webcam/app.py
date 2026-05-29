import gradio as gr
from PIL import Image
import numpy as np

from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge
from rfdetr.assets.coco_classes import COCO_CLASSES
import supervision as sv

MODEL_MAP = {
    "Small (43M)": RFDETRSmall,
    "Medium (160M)": RFDETRMedium,
    "Large (227M)": RFDETRLarge,
}

# Pre-load and cache all models at startup
_models = {}

def _ensure_cached():
    """Download and cache all model weights at startup."""
    for name, cls in MODEL_MAP.items():
        _models[name] = cls()

def get_model(name: str):
    if name not in _models:
        _models[name] = MODEL_MAP[name]()
    return _models[name]


def detect(frame, model_name, threshold):
    if frame is None:
        return np.zeros((360, 640, 3), dtype=np.uint8)

    # Ensure RGB
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)

    model = get_model(model_name)
    detections = model.predict(frame, threshold=threshold)

    if len(detections) == 0:
        return frame  # no detections — return clean frame

    labels = [
        f"{COCO_CLASSES[int(cls_id)]} {float(conf):.2f}"
        for cls_id, conf in zip(detections.class_id, detections.confidence)
    ]

    box_ann = sv.BoxAnnotator()
    label_ann = sv.LabelAnnotator()

    annotated = box_ann.annotate(frame.copy(), detections)
    annotated = label_ann.annotate(annotated, detections, labels=labels)

    return annotated


with gr.Blocks(title="RF-DETR Webcam Detection") as demo:
    gr.Markdown(
        "# RF-DETR Webcam Object Detection\n\n"
        "Real-time object detection using Roboflow's RF-DETR vision transformer.\n\n"
        "**Instructions:**\n"
        "1. Select a model size from the dropdown below\n"
        "2. Adjust the confidence threshold (lower = more detections)\n"
        "3. Click **Enable Camera** to start — allow browser access when prompted\n"
        "4. Point your webcam at objects; bounding boxes appear automatically\n\n"
        "Models are pre-downloaded for instant switching between sizes."
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_select = gr.Dropdown(
                choices=list(MODEL_MAP.keys()),
                value="Large (227M)",
                label="Model Size",
            )
            confidence = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="Confidence Threshold",
            )
            gr.Markdown("**Tip:** Lower the threshold to see more objects. Raise it to filter weak detections.")

        with gr.Column(scale=2):
            webcam = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="Webcam (click Enable Camera)",
                height=360,
            )
            output = gr.Image(label="Detections", show_label=True, height=360)

    # Run detect each time the webcam changes (frame update)
    webcam.stream(
        fn=detect,
        inputs=[webcam, model_select, confidence],
        outputs=[output],
    )

    model_select.change(
        fn=detect,
        inputs=[webcam, model_select, confidence],
        outputs=[output],
    )
    confidence.change(
        fn=detect,
        inputs=[webcam, model_select, confidence],
        outputs=[output],
    )


if __name__ == "__main__":
    print("Pre-downloading model weights (one-time, ~650MB total)...")
    _ensure_cached()
    print("All models cached. Starting Gradio server at http://localhost:7860")
    demo.launch()
