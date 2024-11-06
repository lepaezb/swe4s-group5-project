# Stem Cell Toolkit (SWE4S - Group 5 Project)

## Tool #1: Stem Cell Colony Classification

The goal is to train a machine learning (ML) model that takes in images of iPSC colonies and classifies their status based on their differentiation (type of cell they are)

### Approach
1. **Dataset**: Label dataset of iPSC images categorized by health (contamination and differentiation) status.
   - Publically availible data was used for training from: (Mamaeva et al 2022)
   - This data is in the repository directory entitled `model_data`
   - All data files are 256 x 256 png images and can be used for testing modules. 
      
2. **Preprocessing**: Standardize image sizes, normalize pixel values, and augment dataset to avoid overfitting. Image inputs currently accepted are 256 x 256 pixel png images. 
 - The image preprocessing steps have been tested with this model: 
    - graylevel transform
    - binarization 
    - normalization 
   - histogram equalization

3. **Model**: 
 - An existing convolutional neural network was trained using publically availible data for the classification of iPSC colonies as healthy or unhealthy. Training was allowed to continue for ~ 80 epochs and 25% of data was witheld and used for validating the model. 
 - This model has been published: Mamaeva, A.; Krasnova, O.; Khvorova, I.; Kozlov, K.; Gursky, V.; Samsonova, M.; Tikhonova, O.; Neganova, I. Quality Control of Human Pluripotent Stem Cell Colonies by Computational Image Analysis Using Convolutional Neural Networks. Int. J. Mol. Sci. 2023, 24, 140. https://doi.org/10.3390/ijms24010140
    
   - Model metrics after initial training: 
      ![Accuracy, Precision, Recal by Epoch](Stemcell_classifier/metrics/Training_metrics.png)

      ![Loss function, Validation, training](Stemcell_classifier/metrics/Error_loss_Validation_training.png)

   - Fine tuning of model can be performed using the `training.ipynb` notebook 

4. **Libraries**: 

- PyTorch for model building and training.
- OpenCV, PIL, scikitimage for image preprocessing.
- pandas and numpy for data manipulation
- matplotlib and plotly 

- An environment can be created using the StemCell.yml file included in the Stemcell_classifier directory by running 
   ```sh
   mamba env create -f Stemcell.yml
   ```

## Implementation
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

## Tool # 2: Cell Segmentation using CellPose

## Tool # 3: Stem Cell Segmentation and Classification (In progress)

This project provides a Python script to analyze images of iPSC (induced pluripotent stem cell) colonies, segment individual colonies, classify them based on their quality (good/bad), and annotate the images with the results.

## Features

- **Segmentation:** Uses CellPose to accurately segment iPSC colonies in microscopic images.
- **Feature Extraction:** Extracts relevant features (e.g., mean intensity, area) from each segmented colony. (Can be extended to include more sophisticated features.)
- **Classification:**  Trains a CNN (Convolutional Neural Network) model based on ResNet50 to classify colonies as "good" or "bad".
- **Annotation:** Annotates the original images with bounding boxes around each colony and labels them with the predicted class.
- **Detailed Output:**  Generates a text file for each image with comprehensive information about the classification results (including confidence scores).

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- CellPose (`cellpose`)
- TensorFlow (`tensorflow`)
- NumPy (`numpy`)

**Installation:**

```bash
pip install opencv-python cellpose tensorflow numpy


