"""
This docstring explains how to use this document
"""

# Establish the correct environment for the script
#pip install openpyxl
import subprocess
from cellpose import models 
from pathlib import Path
import numpy as np
import pandas as pd
import os
import tifffile as tiff
import matplotlib.pyplot as plt
import argparse


segmentation_cellpose_parse = argparse.ArgumentParser(
    description="Pass parameters into segmentation_cellpose", prog="segmentation_cellpose"
)

# Parse the arguments for the segmentation_cellpose.py script
segmentation_cellpose_parse.add_argument(
    '--parent_directory',
    type=str,
    help="Filepath of the merged images to be masked and counted.",
    required=True,
)

segmentation_cellpose_parse.add_argument(
    '--channel_number',
    type=int,
    help="Channel of the image to use for segmentation. 1 = red, 2 = green, 3 = blue.",
    default=True,
)

'''
segmentation_cellpose_parse = argparse.ArgumentParser(
    "--fiji_path",
    type=str,
    help="Filepath of the Fiji executable.",
    required=True,
)

segmentation_cellpose_parse = argparse.ArgumentParser(
    "--thresh_min",
    type=int.
    help="Minimum value for the FIJI mask thresholding.",
    required=False,
)

segmentation_cellpose_parse = argparse.ArgumentParser(
    "--thresh_max",
    type=int.
    help="Maximum value for the FIJI mask thresholding.",
    required=False,
)
'''

args = segmentation_cellpose_parse.parse_args()


# Define parent directory with all image reps
parent_directory = Path(args.parent_directory)
# Path("C:/Users/laure/OneDrive - UCB-O365/ROTATION1/DDX6 Pilot/Reps 1-3 LO/TEST_DIR/TEST_1/")

# Define the channel number (nuclear channel) 
channel_number = args.channel_number

# Initialize the CellPose model for nuclear segmentation
model = models.Cellpose(model_type='nuclei')

# Fiji location
fiji_path = "C:/Users/laure/OneDrive/Desktop/Fiji.app/ImageJ-win64.exe"
#args.fiji_path

macro_path = "./dapi_actin_merge_OG.js"

# "C:/Users/laure/OneDrive - UCB-O365/FALL 24 CLASSES/SOFTWARE/swe4s-group5-project/Tropho_segmentation/DAPI_ACTIN_MERGE.js"


'''
# Run the Fiji macro to split, mask, and merge the raw images
def run_fiji_macro(raw_directory, output_path):
    # Macro arguments 
    thresh_min = int(5500) #args.thresh_min
    thresh_max = int(10500) #args.thresh_max

    # Change the filepath for javascript 
    raw_directory = raw_directory.replace("\\", "/")
    output_path = output_path.replace("\\", "/")

    #Combine macro arguments
    macro_args = f"raw_directory={raw_directory},output_path={output_path}"
    # Ensure paths exist
    if not os.path.isfile(fiji_path):
    raise FileNotFoundError(f"Fiji not found at: {fiji_path}")

    if not os.path.isfile(macro_path):
    raise FileNotFoundError(f"Macro file not found at: {macro_path}")

    if not os.path.isdir(output_path):
    os.makedirs(output_path)

    subprocess.run([fiji_path, "--headless", "--console", "-macro", macro_path, macro_args], check=True)
    return ("Fiji executed successfully.")

'''

# Read all tiff files into memory and store them by frame number
def read_all_tiff_files(output_path, channel_number):
    tiff_images = {}
    for filename in os.listdir(output_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(output_path, filename)
            image = tiff.imread(file_path)
            tiff_images[filename] = image
    return tiff_images


def run_cellpose(direct, num, img, channel_number, diam):
    # Run the CellPose segmentation on the first frame
    masks, flows, styles, diams = model.eval(img, channels=[channel_number, 0], diameter= diam, flow_threshold =0.5) 
    # Plot and save masks 
    # Display the result for the first frame
    fig = plt.figure(figsize=(8, 8))
    fig = plt.imshow(img[0], cmap='gray')
    fig = plt.imshow(masks, alpha=0.5)
    fig = plt.title(f'Segmented frame')
    plt.draw()
    # Save the masks 
    plt.savefig(str(direct) + "/output" + str(num))
    plt.close()
    return(masks)


# Run cellpose on all files in the directory
def main(parent_directory, channel_number):

    # Run the cellpose segmentation on each new masked and merged output directory
    masked_subdirs = [d for d in parent_directory.iterdir() if (d.is_dir() and "MASKED" in str(d))]
    for masked_subdir in masked_subdirs:
        tiff_images = read_all_tiff_files(str(masked_subdir), channel_number)
        data = {}
        frame_number = list(tiff_images.keys())
        i = 0
        for img in frame_number:
            i +=1
                
            # Ensure the image is in the correct format (convert to float32 if necessary)
            img_iterate = tiff_images[img]
            if img_iterate.dtype != np.float32:
                img_iterate = img_iterate.astype(np.float32)
            
            # Run the CellPose segmentation on the frame
            masks = run_cellpose(parent_directory, str(i), img_iterate, channel_number, 100.0)
            num_masks = len(set(masks.flatten())) - (1 if 0 in masks else 0)


            # Add to dictionary
            data[img] = [num_masks]

        # Convert the mask data to a DataFrame
        df = pd.DataFrame(data)
        # df = df.transpose()
        output_excel_path = str(masked_subdir) + "\\MASK_DATA.xlsx"
        
        # Save the DataFrame to an Excel file
        df.to_excel(output_excel_path, index=False)
    
    return("Segmentation complete.")

# Run the main function when file is called
if __name__ == "__main__":
    main(parent_directory, channel_number)

