import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from adetailer import PredictOutput
from adetailer.common import create_bbox_from_mask

def groundingdino_predict(
    inp_model_id: str,
    image: Image.Image,
    text_labels: list[str],
    confidence: float = 0.3
) -> PredictOutput:
    """
    Similar to the mediapipe detection workflow, this function:
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
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Convert the PIL Image to a torch-ready format
    print(text_labels)
    arr = np.array(image)
    inputs = processor(
        images=image,
        text=[["yellow ascot"]],  # text_labels should be a list of label strings
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Use post_process_grounded_object_detection from the processor to get boxes & scores
    results = processor.post_process_grounded_object_detection(outputs, threshold=confidence)
    result = results[0] if results else None

    if not result:
        return PredictOutput()

    all_bboxes = []
    img_width, img_height = image.size
    for box, score, label_idx in zip(result["boxes"], result["scores"], result["labels"]):
        if score.item() >= confidence:
            x1, y1, x2, y2 = box.tolist()
            # Clip to image size just in case
            x1, x2 = sorted([max(0, x1), min(img_width, x2)])
            y1, y2 = sorted([max(0, y1), min(img_height, y2)])
            all_bboxes.append([x1, y1, x2, y2])

    if not all_bboxes:
        return PredictOutput()

    # Create masks from bounding boxes
    masks = create_bbox_from_mask([Image.new("L", image.size, "black")], image.size)
    for i, bbox in enumerate(all_bboxes):
        # Overwrite each mask with a filled rectangle
        x1, y1, x2, y2 = bbox
        draw = ImageDraw.Draw(masks[0])
        draw.rectangle([x1, y1, x2, y2], fill="white")

    # Generate preview image
    preview = image.copy()
    preview_draw = ImageDraw.Draw(preview)
    for bbox in all_bboxes:
        preview_draw.rectangle(bbox, outline="red", width=2)

    return PredictOutput(
        bboxes=all_bboxes,
        masks=masks,
        preview=preview
    )