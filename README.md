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
   - Publically availible data was used for training from: (Mamaeva et al 2022)
      
   - **Preprocessing**: Standardize image sizes, normalize pixel values, and augment dataset to avoid overfitting. Image inputs currently accepted are 256 x 256 pixel png images. 
   - The image preprocessing steps have been tested with this model: 
      - graylevel transform
      - binarization 
      - normalization 
      - histogram equalization

   - **Model**: 
    - An existing convolutional neural network was trained using publically availible data for the classification of iPSC colonies as healthy or unhealthy. Training was allowed to continue for ~ 80 epochs and 25% of data was witheld and used for validating the model. 
    - This model has been published: Mamaeva, A.; Krasnova, O.; Khvorova, I.; Kozlov, K.; Gursky, V.; Samsonova, M.; Tikhonova, O.; Neganova, I. Quality Control of Human Pluripotent Stem Cell Colonies by Computational Image Analysis Using Convolutional Neural Networks. Int. J. Mol. Sci. 2023, 24, 140. https://doi.org/10.3390/ijms24010140
    
    - Model metrics after initial training: 
      ![Accuracy, Precision, Recal by Epoch](Stemcell_classifier/metrics/Training_metrics.png)

      ![Loss function, Validation, training](Stemcell_classifier/metrics/Error_loss_Validation_training.png)

    - Fine tuning of model can be performed using the `training.ipynb` notebool  

3. **Libraries**: 

   - PyTorch for model building and training.
   - OpenCV, PIL, scikitimage for image preprocessing.
   - pandas and numpy for data manipulation
   - matplotlib and plotly 

   - An environment can be created using the StemCell.yml file included in the Stemcell_classifier directory by running 
```sh
mamba env create -f Stemcell.yml
```

4. **Suggested pipeline**: 
   - Preprocess input images (resizing, normalization).
   - Train or fine-tune an ML model on your labeled dataset.
   - Develop a function that takes a new image, runs it through the trained model, and returns a health assessment (healthy, undifferentiated, etc.).

   - Currently: a binary classifier exists to query the existing model. To run this from the command line cd to the Stemcell_classifier directory and select a pre-trained model. 
   - To classify a single image the following can be run in the command line: 
   ```sh
   python classifier_single.py path_to_your_model.pth path_to_your_image.png
   ```
   - To classify all images in a directory and get outputs in a csv file, the following can be run in the command line: 
   ```sh
   python classifier_multi.py path_to_your_model.pth path_to_your_image.jpg path_to_output_predictions.csv
   ```





