from __future__ import annotations

import re
import sys
import traceback
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM

from adetailer import PredictOutput
from adetailer.common import create_mask_from_bbox


def parse_florence2_coordinates(text: str, image_size: tuple[int, int]) -> list[list[float]]:
    """
    Parse Florence2 coordinate tokens into bounding boxes.
    Florence2 outputs coordinates in the format: <loc_x1><loc_y1><loc_x2><loc_y2>
    where coordinates are normalized to 1000.
    
    Parameters
    ----------
    text : str
        The text output from Florence2 containing coordinate tokens
    image_size : tuple[int, int]
        Image size (width, height) to denormalize coordinates
        
    Returns
    -------
    list[list[float]]
        List of bounding boxes in format [x1, y1, x2, y2]
    """
    width, height = image_size
    
    # Pattern to match Florence2 location tokens: <loc_XXX>
    pattern = r'<loc_(\d+)>'
    matches = re.findall(pattern, text)
    
    if not matches or len(matches) % 4 != 0:
        return []
    
    bboxes = []
    # Process coordinates in groups of 4
    for i in range(0, len(matches), 4):
        # Florence2 normalizes coordinates to 1000
        x1 = float(matches[i]) / 1000.0 * width
        y1 = float(matches[i + 1]) / 1000.0 * height
        x2 = float(matches[i + 2]) / 1000.0 * width
        y2 = float(matches[i + 3]) / 1000.0 * height
        
        # Clip to image bounds
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        
        # Ensure valid bounding box
        if x2 > x1 and y2 > y1:
            bboxes.append([x1, y1, x2, y2])
    
    return bboxes


def parse_florence2_polygons(polygons_data: list, image_size: tuple[int, int]) -> list[Image.Image]:
    """
    Parse Florence2 polygon data into binary masks.
    
    Parameters
    ----------
    polygons_data : list
        List of polygon coordinate lists from Florence2
    image_size : tuple[int, int]
        Image size (width, height)
        
    Returns
    -------
    list[Image.Image]
        List of binary mask images
    """
    width, height = image_size
    masks = []
    
    for polygon_list in polygons_data:
        if not polygon_list:
            continue
            
        mask = Image.new("L", image_size, 0)
        mask_draw = ImageDraw.Draw(mask)
        
        for polygon in polygon_list:
            if len(polygon) < 6:  # Need at least 3 points (x, y pairs)
                continue
                
            # Denormalize polygon coordinates
            points = []
            for j in range(0, len(polygon), 2):
                if j + 1 < len(polygon):
                    x = float(polygon[j]) / 1000.0 * width
                    y = float(polygon[j + 1]) / 1000.0 * height
                    points.append((x, y))
            
            if len(points) >= 3:
                mask_draw.polygon(points, fill=255)
        
        masks.append(mask)
    
    return masks


def florence2_predict(
    inp_model_id: str,
    image: Image.Image,
    task: str = "<CAPTION_TO_PHRASE_GROUNDING>",
    text_input: str | None = None,
    confidence: float = 0.3,
) -> PredictOutput:
    """
    Florence2 detection workflow for both grounding and segmentation.
    
    Parameters
    ----------
    inp_model_id : str
        Model identifier (florence-2-base, florence-2-large, etc.)
    image : Image.Image
        Input PIL Image
    task : str
        Task type: "<CAPTION_TO_PHRASE_GROUNDING>", "<OPEN_VOCABULARY_DETECTION>", or "<REFERRING_EXPRESSION_SEGMENTATION>"
    text_input : str | None
        Text prompt for the task (required for all tasks)
    confidence : float
        Confidence threshold (currently not used by Florence2 directly)
        
    Returns
    -------
    PredictOutput
        Contains bboxes, masks, and preview image
    """
    # Model mapping
    mapping = {
        "florence-2-base": "microsoft/Florence-2-base",
        "florence-2-large": "microsoft/Florence-2-large",
        "florence-2-base-ft": "microsoft/Florence-2-base-ft",
        "florence-2-large-ft": "microsoft/Florence-2-large-ft",
    }
    
    model_id = mapping.get(inp_model_id, inp_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(device)
    except Exception as e:
        print(f"[-] ADetailer: Failed to load Florence2 model: {e}")
        return PredictOutput()
    
    # Validate text input
    if not text_input or not text_input.strip():
        print("[-] ADetailer: No text prompt provided for Florence2")
        return PredictOutput()
    
    print(f"[+] ADetailer: Florence2 task: {task}, prompt: {text_input}")
    
    # For CAPTION_TO_PHRASE_GROUNDING, OPEN_VOCABULARY_DETECTION, and REFERRING_EXPRESSION_SEGMENTATION
    # we need to provide the text in a specific format
    if task in ["<CAPTION_TO_PHRASE_GROUNDING>", "<OPEN_VOCABULARY_DETECTION>"]:
        # For phrase grounding and open vocabulary detection, combine task and text_input
        prompt = f"{task} {text_input}"
    elif task == "<REFERRING_EXPRESSION_SEGMENTATION>":
        # For referring expression segmentation, combine task and text_input
        prompt = f"{task} {text_input}"
    else:
        prompt = task
    
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Process the image with text prompt
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )
    
    # Move inputs to device with proper dtype handling
    inputs = {k: v.to(device) if v is not None else v for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=True,
                num_beams=3,
                use_cache=False,
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[-] ADetailer: Florence2 inference failed: {e}", file=sys.stderr)
        print(error_trace, file=sys.stderr)
        return PredictOutput()
    
    # Extract results based on task type
    result = parsed_answer.get(task, {})
    
    if not result:
        print(f"[-] ADetailer: No results from Florence2 for task {task}")
        return PredictOutput()
    
    bboxes = []
    masks = []
    
    # Handle CAPTION_TO_PHRASE_GROUNDING and OPEN_VOCABULARY_DETECTION tasks (same output format)
    if task in ["<CAPTION_TO_PHRASE_GROUNDING>", "<OPEN_VOCABULARY_DETECTION>"]:
        # Result format: {'labels': [...], 'bboxes': [[x1, y1, x2, y2], ...]}
        if 'bboxes' in result and result['bboxes']:
            labels = result.get('labels', [])
            raw_bboxes = result['bboxes']
            
            # Filter by text_input if provided (match labels)
            if text_input and labels:
                text_input_lower = text_input.lower().strip()
                for i, label in enumerate(labels):
                    if i < len(raw_bboxes):
                        label_lower = label.lower().strip()
                        # Check if the label matches any word in text_input
                        if any(word in label_lower for word in text_input_lower.split()):
                            bbox = raw_bboxes[i]
                            x1, y1, x2, y2 = bbox
                            # Ensure valid bounding box
                            if x2 > x1 and y2 > y1:
                                bboxes.append([float(x1), float(y1), float(x2), float(y2)])
            else:
                # Use all bboxes
                for bbox in raw_bboxes:
                    x1, y1, x2, y2 = bbox
                    if x2 > x1 and y2 > y1:
                        bboxes.append([float(x1), float(y1), float(x2), float(y2)])
            
            # Create masks from bounding boxes
            if bboxes:
                masks = create_mask_from_bbox(bboxes, image.size)
    
    # Handle REFERRING_EXPRESSION_SEGMENTATION task
    elif task == "<REFERRING_EXPRESSION_SEGMENTATION>":
        # Result format: {'polygons': [[polygon1, polygon2, ...]], 'labels': [...]}
        if 'polygons' in result and result['polygons']:
            polygons_data = result['polygons']
            masks = parse_florence2_polygons(polygons_data, image.size)
            
            # Create bounding boxes from masks
            if masks:
                for mask in masks:
                    bbox = mask.getbbox()
                    if bbox is not None:
                        bboxes.append(list(bbox))
    
    if not bboxes and not masks:
        print(f"[-] ADetailer: No valid detections from Florence2")
        return PredictOutput()
    
    # Generate preview image
    preview = image.copy().convert("RGB")
    preview_draw = ImageDraw.Draw(preview)
    
    # Draw bounding boxes
    for bbox in bboxes:
        preview_draw.rectangle(bbox, outline="red", width=2)
    
    # Overlay masks if available (for segmentation task)
    if masks and task == "<REFERRING_EXPRESSION_SEGMENTATION>":
        from PIL import ImageEnhance
        # Create a semi-transparent red overlay for each mask
        for mask in masks:
            # Create a red layer
            red_layer = Image.new("RGB", image.size, (255, 0, 0))
            # Convert mask to RGBA for alpha blending
            mask_array = np.array(mask)
            # Make the overlay semi-transparent (50% alpha)
            alpha = (mask_array > 0).astype(np.uint8) * 128
            mask_rgba = Image.fromarray(alpha, mode="L")
            # Blend the red layer with preview using the mask
            preview = Image.composite(red_layer, preview, mask_rgba)
    
    print(f"[+] ADetailer: Florence2 detected {len(bboxes)} objects")
    
    return PredictOutput(
        bboxes=bboxes,
        masks=masks if masks else create_mask_from_bbox(bboxes, image.size),
        preview=preview
    )
