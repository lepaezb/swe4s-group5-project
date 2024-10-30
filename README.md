# Stem Cell Toolkit (SWE4S - Group 5 Project)

## Project Outline ideas
 
### 1.1 Segmentation of STB cells based on actin staining
 1. **Objective:** 
    - Use actin staining to segment the border of STB cells from an image.
1. **Approach:**
    - **Preprocessing:** Use image processing techniques to enhance actin staining regions, like Gaussian blurring or contrast enhancement.
    - **Segmentation Algorithm:** Try classical methods like thresholding (e.g., Otsu’s method), edge detection (e.g., Canny), or more advanced approaches like watershed segmentation.
2. **Libraries:** 
     - OpenCV or scikit-image for image processing and segmentation.
     - Maybe for deep learning-based approaches, Keras or PyTorch with models like U-Net could help with image segmentation.
3. **Suggested pipeline:**
   - Load image and preprocess it to enhance acting staining
   - Apply the segmentation algorithm to isolate STB cells
   - Return the segmented regions as masks or outlines

### 1.2 Nuclei and fusion index quantification in STBs
1. **Objective**: 
   - Use DAPI staining to count the number of nuclei in STBs and calculate the fusion index of the cell population.
2. **Approach**: 
   - **Preprocessing**: Convert the image to grayscale and enhance the nuclei regions using filtering.
   - **Nuclei detection**: Use blob detection (e.g., Laplacian of Gaussian) or another method to identify individual nuclei.
   - **Fusion Index Calculation**: After counting the nuclei, calculate the fusion index as a ratio of nuclei count per STB region.
3. **Libraries**:
   - scikit-image or OpenCV for preprocessing and nuclei detection.
   - NumPy for handling image arrays and calculating the fusion index
4. **Suggested pipeline**:
   - Detect individual nuclei using blob detection.
   - Count the number of nuclei within each STB region.
   - Calculate the fusion index by determining the number of nuclei per segmented cell.

### 2. SCs colony assessment
1. **Objective**: 
   - Train a machine learning (ML) model that takes in images of iPSC colonies and classifies their status based on contamination and differentiation.
2. **Approach**: 
   - **Dataset**: Label dataset of iPSC images categorized by health (contamination and differentiation) status.
   - **Preprocessing**: Standardize image sizes, normalize pixel values, and augment dataset to avoid overfitting.
   - **Model**: Maybe convolutional Neural Networks (CNNs) since they are good for image classification tasks. We could either build a custom CNN or fine-tune a pre-trained model like ResNet or VGG using transfer learning.
3. **Libraries**: 
   - TensorFlow or PyTorch for model building and training.
   - OpenCV or PIL for image preprocessing.
4. **Suggested pipeline**: 
   - Preprocess input images (resizing, normalization).
   - Train or fine-tune an ML model on your labeled dataset.
   - Develop a function that takes a new image, runs it through the trained model, and returns a health assessment (healthy, undifferentiated, etc.).
