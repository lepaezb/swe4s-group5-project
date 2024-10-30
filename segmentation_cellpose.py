# main.py

from cellpose import models #not sure how to import this rn
import numpy as np
import pandas as pd
import os
import tifffile as tiff
import matplotlib.pyplot as plt

# Specify the directory where the tiff files are stored
directory_path = "C:/Users/laure/OneDrive - UCB-O365/ROTATION1/DDX6 Pilot/Reps 1-3 LO/DDX6+DOX1_COMPRESSED"

# Define the channel number (nuclear channel)
channel_number = 2

# Read all tiff files into memory and store them by frame number
def read_all_tiff_files(directory_path, channel_number):
    tiff_images = {}
    filename = "DDX6+DOX1-ACTIN1.tif_DAPI_MERGE.tif"
    for filename in os.listdir(directory_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(directory_path, filename)
            image = tiff.imread(file_path)
            tiff_images[filename] = image
            """
            if image.ndim > 2:
                image = image[channel_number]
            frame_number = int(filename.split('_')[-1].split('.')[0])  # Assuming frame number is in the filename
            tiff_images[frame_number] = image
            """
    return tiff_images

tiff_images = read_all_tiff_files(directory_path, channel_number)

# Initialize the CellPose model for nuclear segmentation
model = models.Cellpose(model_type='nuclei')

# Find the first frame to segment (smallest frame number) 
first_frame_number = min(tiff_images.keys())
first_image = tiff_images[first_frame_number]

# Ensure the image is in the correct format (convert to float32 if necessary)
if first_image.dtype != np.float32:
    first_image = first_image.astype(np.float32)

# Run the CellPose segmentation on the first frame
masks, flows, styles, diams = model.eval(first_image, channels=[0, 0], diameter= 100.0, flow_threshold =0.5)  # [0, 0] for grayscale

# Optionally: Display the result for the first frame
plt.figure(figsize=(8, 8))
plt.imshow(first_image, cmap='gray')
plt.imshow(masks, alpha=0.5)
plt.title(f'Segmented frame {first_frame_number}')
plt.show()

# Prepare the data to be saved
#read file name
image_data = tiff.imread(file_path)

data = {
    'image_name': [image_data],
    'masks': [masks],
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Specify the output Excel file path
output_excel_path = "segmentation_output.xlsx"

# Save the DataFrame to an Excel file
df.to_excel(output_excel_path, index=False)

print(f"Segmentation results saved to {output_excel_path}")

# Now you have segmented the first frame and visualized it