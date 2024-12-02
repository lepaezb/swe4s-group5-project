# Ignore CellPose warning due to future package updates
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 0. Import necessary libraries ---
import os
import cv2
import numpy as np
from cellpose import models as cp_models
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from matplotlib import cm

# --- 1. Check for GPU availability ---
use_gpu = torch.cuda.is_available()
print(f"Using GPU: {use_gpu}")

# --- 2. Image Loading Function ---
def load_images(directory):
    """
    Loads image file paths from the specified directory.
    Args:
        directory (str): Path to the directory containing images.
    Returns:
        List[str]: List of image paths.
    """
    image_paths = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            image_paths.append(img_path)
    return image_paths

# --- 3. Colony Segmentation with CellPose ---
class ColonySegmenter:
    """
    Segments colonies in images using the CellPose model.
    Args:
        model_type (str): Type of model to use ('cyto', 'nuclei', or path to custom model).
        use_gpu (bool): If True, uses GPU for CellPose (if available).
    """
    def __init__(self, model_type='cyto', use_gpu=False):
        self.model = cp_models.Cellpose(gpu=use_gpu, model_type=model_type)

    def segment_colonies(self, image, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0):
        """
        Segments colonies in the given image.
        Args:
            image (np.array): Input image (numpy array).
            diameter (float): Estimated diameter of the cells. If None, CellPose will estimate it.
            flow_threshold (float): Threshold for flows.
            cellprob_threshold (float): Threshold for cell probabilities.
        Returns:
            masks (np.array): Segmentation masks for colonies.
        """
        if len(image.shape) == 2: # Grayscale image
            channels = [0, 0]
        elif image.shape[2] == 3: # RGB image
            channels = [0, 0] 
        else:
            raise ValueError("Unsupported image format")
        
        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
        return masks

# --- 4. Colony Extraction Function ---
def extract_colonies(image, masks):
    """
    Extracts individual colonies from the image based on segmentation masks.
    Args:
        image (np.array): Original image.
        masks (np.array): Segmentation masks for colonies.
    Returns:
        List[np.array]: List of colony images.
        List[dict]: List of properties for each colony (e.g., bounding box coordinates).
    """
    colonies = []
    properties = []
    unique_ids = np.unique(masks)
    unique_ids = unique_ids[unique_ids != 0]  # Exclude background
    for colony_id in unique_ids:
        colony_mask = (masks == colony_id).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(colony_mask)
        colony_img = cv2.bitwise_and(image, image, mask=colony_mask)
        colony_crop = colony_img[y:y+h, x:x+w]
        colonies.append(colony_crop)
        properties.append({'id': colony_id, 'bbox': (x, y, w, h)})
    return colonies, properties

# --- 5. Function to Annotate and Save Segmented Image ---
def annotate_and_save(image, masks, properties, save_path):
    """
    Annotates the image with bounding boxes and saves it.
    Args:
        image (np.array): Original image.
        masks (np.array): Segmentation masks.
        properties (List[dict]): List of properties for each colony.
        save_path (str): Path to save the annotated image.
    """
    annotated_image = image.copy()
    colormap = cm.get_cmap('hsv', len(properties) + 1)
    overlay = np.zeros_like(image, dtype=np.uint8)
    for idx, prop in enumerate(properties):
        colony_id = prop['id']
        color = tuple(int(c * 255) for c in colormap(idx)[:3])
        colony_mask = (masks == colony_id).astype(np.uint8)
        overlay[colony_mask == 1] = color
        x, y, w, h = prop['bbox']
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated_image, f'ID:{colony_id}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    alpha = 0.5
    annotated_image = cv2.addWeighted(annotated_image, 1 - alpha, overlay, alpha, 0)
    cv2.imwrite(save_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"Annotated image saved to {save_path}")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Parameters ---
    image_directory = '/scratch/Users/lupa9404/swe4s/data'
    output_directory = '/scratch/Users/lupa9404/swe4s/cell_segmentation/results_cyto'
    os.makedirs(output_directory, exist_ok=True)
    model_type = 'cyto' 
    diameter = None 
    flow_threshold = 0.4
    cellprob_threshold = 0.0

    # --- Load Images ---
    image_paths = load_images(image_directory)
    print(f"Total images loaded: {len(image_paths)}")

    # --- Initialize Colony Segmenter ---
    segmenter = ColonySegmenter(model_type=model_type, use_gpu=use_gpu)

    # --- Process Each Image ---
    for img_path in image_paths:
        print(f"Processing image: {img_path}")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading image: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Segment Colonies ---
        masks = segmenter.segment_colonies(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )

        # --- Extract Colonies ---
        colonies, properties = extract_colonies(image, masks)
        print(f"Found {len(colonies)} colonies in image {os.path.basename(img_path)}")

        # --- Annotate and Save Image ---
        annotated_image_path = os.path.join(output_directory, f"annotated_{os.path.basename(img_path)}")
        annotate_and_save(image, masks, properties, annotated_image_path)

        # --- Save Individual Colonies ---
        colonies_output_dir = os.path.join(output_directory, f"colonies_{os.path.splitext(os.path.basename(img_path))[0]}")
        os.makedirs(colonies_output_dir, exist_ok=True)
        for idx, colony_img in enumerate(colonies):
            colony_img_path = os.path.join(colonies_output_dir, f"colony_{idx+1}.png")
            cv2.imwrite(colony_img_path, cv2.cvtColor(colony_img, cv2.COLOR_RGB2BGR))
        print(f"Saved individual colonies to {colonies_output_dir}")