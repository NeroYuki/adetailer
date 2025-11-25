from __future__ import annotations

import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from adetailer import PredictOutput
from adetailer.common import create_mask_from_bbox


def groundingdino_predict(
    inp_model_id: str,
    image: Image.Image,
    text_labels: list[str],
    confidence: float = 0.3,
) -> PredictOutput:
    """
    GroundingDINO detection workflow:
      1. Loads a GroundingDINO model via Hugging Face.
      2. Processes an image with specified text labels.
      3. Gathers bounding boxes above the given confidence threshold.
      4. Returns a PredictOutput containing bounding boxes, masks, and a preview image.
    """
    mapping = {
        "groundingdino": "IDEA-Research/grounding-dino-tiny",
    }
    model_id = mapping.get(inp_model_id, inp_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    except Exception as e:
        print(f"[-] ADetailer: Failed to load GroundingDINO model: {e}")
        return PredictOutput()

    # Validate text_labels
    if not text_labels or not any(text_labels):
        print("[-] ADetailer: No text labels provided for GroundingDINO")
        return PredictOutput()

    print(f"[+] ADetailer: GroundingDINO detecting: {text_labels}")

    # Process the image with text labels
    inputs = processor(
        images=image,
        text=text_labels,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to get boxes and scores
    img_width, img_height = image.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=confidence,
        target_sizes=torch.tensor([[img_height, img_width]])
    )
    
    result = results[0] if results else None

    if not result or "boxes" not in result or len(result["boxes"]) == 0:
        return PredictOutput()

    # Collect all valid detections with their scores and areas
    detections = []
    
    for box, score in zip(result["boxes"], result["scores"]):
        if score.item() >= confidence:
            # Boxes are in format [x_min, y_min, x_max, y_max]
            x1, y1, x2, y2 = box.tolist()
            # Clip to image size
            x1 = max(0, min(img_width, x1))
            y1 = max(0, min(img_height, y1))
            x2 = max(0, min(img_width, x2))
            y2 = max(0, min(img_height, y2))
            
            # Ensure valid bounding box
            if x2 > x1 and y2 > y1:
                area = (x2 - x1) * (y2 - y1)
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": score.item(),
                    "area": area
                })

    if not detections:
        return PredictOutput()

    # Sort by area (largest first) and keep all detections
    detections.sort(key=lambda d: d["area"], reverse=True)
    
    # Extract bboxes in sorted order
    all_bboxes = [d["bbox"] for d in detections]

    # Create masks from bounding boxes
    masks = create_mask_from_bbox(all_bboxes, image.size)

    # Generate preview image with bounding boxes
    preview = image.copy().convert("RGB")
    preview_draw = ImageDraw.Draw(preview)
    for bbox in all_bboxes:
        preview_draw.rectangle(bbox, outline="red", width=2)

    return PredictOutput(
        bboxes=all_bboxes,
        masks=masks,
        preview=preview
    )