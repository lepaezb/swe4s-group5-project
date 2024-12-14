"""
This script automates the processing and segmentation of TIFF images using Fiji and CellPose.

Usage:
    bash run.sh--parent_directory <path_to_images> --channel_number <channel> --fiji_path <path_to_fiji> [--thresh_min <value>] [--thresh_max <value>]
"""

# Establish the correct environment for the script
from pathlib import Path
import numpy as np
import pandas as pd
import os
import subprocess
import tifffile as tiff
from argparse import ArgumentParser, Namespace
import psutil
import matplotlib.pyplot as plt
from cellpose import models

# Path to the Fiji macro in the current directory
macro_path = "./dapi_actin_merge_macro_ARGS.js"

def parse_args() -> Namespace:
# Argument parser to handle the user input arguments
    segmentation_cellpose_parse = ArgumentParser(
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

    segmentation_cellpose_parse.add_argument(
            "--fiji_path",
            type=str,
            help="Filepath of the Fiji executable.",
            required=True,
        )

    segmentation_cellpose_parse.add_argument(
            "--thresh_min",
            type=int,
            help="Minimum value for the FIJI mask thresholding.",
            required=False,
        )

    segmentation_cellpose_parse.add_argument(
            "--thresh_max",
            type=int,
            help="Maximum value for the FIJI mask thresholding.",
            required=False,
        )

    args = segmentation_cellpose_parse.parse_args()
    return args

""" 
Check if Fiji (ImageJ) is open in the background.
This function is necessary to continue the main() function after the user manually closes Fiji during image thresholding.
"""
def is_fiji_open():
    for process in psutil.process_iter(['name']):
        if 'ImageJ' in process.info['name']:
            return True
    return False


""" 
Run the Fiji macro to split, mask, and merge the raw images. 
Function takes in a parent directory with subdirectories that contain tiff images.
The subprocess to the FIJI executable will run the macro with the specified arguments.
The user must interact with the FIJI GUI as the macro is running; as images are split and merged, the user must manually close the 'save' message.
"""
def run_fiji_macro(fiji, parent_dir, min, max):
    #Combine macro arguments
    direct = str(parent_dir)
    direct = (direct + "\\")
    direct = direct.replace("\\", "/")
    # Run the macro through subprocesses with the specified arguments
    macro_args = f"parent_directory={direct},thresh_min={min},thresh_max={max}"
    print(macro_args)
    subprocess.run([fiji, "--console", "-macro", macro_path, macro_args], check=True)
    if not is_fiji_open():
        return ("Fiji executed successfully.")
    


"""
Read a directory with all tiff files in a directory memory and store them by frame number. 
Returns a dictionary with the tiff images.
"""
def read_all_tiff_files(output_path):
    tiff_images = {}
    for filename in os.listdir(output_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(output_path, filename)
            image = tiff.imread(file_path)
            tiff_images[filename] = image
    if tiff_images == {}:
        raise ValueError("Error: No TIFFs in directory")
    return tiff_images


"""
Run cellpose on an input image with specified channel_number to use for segmentation.
Function returns the mask for the input image save as a .png file. 
"""
# Run Cellpose on input images with specified parameters
def run_cellpose(direct, subdir, num, img, channel_number):
    from cellpose import models
    # Initialize the CellPose model for nuclear segmentation
    model = models.Cellpose(model_type='nuclei')

    # Run the CellPose segmentation on the first frame
    masks, flows, styles, diams = model.eval(img, channels=[channel_number, 0], diameter= 100.0, flow_threshold =0.5) 
   
    # Plot and save masks 
    # Display the result for the first frame
    fig = plt.figure(figsize=(8, 8))
    fig = plt.imshow(img[0], cmap='gray')
    fig = plt.imshow(masks, alpha=0.5)
    fig = plt.title(f'Segmented frame')
    plt.draw()

    # Save the masks 
    plt.savefig(str(subdir) + "/cellpose_mask" + str(num))
    plt.close()
    return(masks)

"""
Run the segmentation on all subdirectories within a parent directory.
Function segment the directory will run if the folder contains MASKED in the name.
Exctract the Cellpose parameter 'Masks' and save the number of masks corresponding to each frame to an Excel file.
"""
def subdir_segmentation(parent_directory, channel_number):
        # Determine the directories with the masked images by searching for "MASKED" in the title
        masked_subdirs = [d for d in parent_directory.iterdir() if (d.is_dir() and "MASKED" in str(d))]
        
        # Check for files in the directory
        if masked_subdirs:

            # Loop through all files
            for masked_subdir in masked_subdirs:
                tiff_images = read_all_tiff_files(str(masked_subdir))
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
                    masks = run_cellpose(parent_directory, masked_subdir, str(i), img_iterate, channel_number)
                    num_masks = len(set(masks.flatten())) - (1 if 0 in masks else 0)

                    # Add to dictionary
                    data[img] = [num_masks]

                # Convert the mask corresponding image name to a DataFrame
                df = pd.DataFrame(data)
                # df = df.transpose()
                output_excel_path = str(masked_subdir) + "\\MASKED.xlsx"
                
                # Save the DataFrame to an Excel file
                df.to_excel(output_excel_path, index=False)
                print(f"Segmentation complete for {masked_subdir}.")

            print("All segmentation complete for " + str(parent_directory))
        else:
            print("No folders with MASK in title. Try again with a different directory.")

def main(fiji_path, parent_directory, thresh_min, thresh_max, channel_number,):
    run_fiji_macro(fiji_path, parent_directory, thresh_min, thresh_max) # User must manually close
    subdir_segmentation(parent_directory, channel_number)
    return("Segmentation complete.")

# Run the main function when file is called
if __name__ == "__main__":
    args = parse_args()
    parent_directory = Path(args.parent_directory)
    main(args.fiji_path, parent_directory, args.thresh_min, args.thresh_max, args.channel_number)
