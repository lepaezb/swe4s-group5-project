import cv2
from cellpose import models 
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import tensorflow as tf

# --- 1. Image Loading and Preprocessing ---
def load_images(directory):
    """Loads images and labels from the given directory."""
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            images.append(img)
            filenames.append(filename)

            # Extract label from filename
            if "good" in filename:
                labels.append(0)  # 0 for 'good'
            else:
                labels.append(1)  # 1 for 'bad'
    return images, labels, filenames

# --- 2. Colony Segmentation with CellPose ---
def segment_colonies(images):
    """Segments iPSC colonies using CellPose."""
    model = models.Cellpose(gpu=True, model_type='cyto')
    segmented_images = []
    for img in images:
        masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0, 0])
        segmented_images.append(masks)
    return segmented_images


# --- 3. Feature Extraction ---
def extract_features(segmented_images, original_images):
    """Extracts features from segmented colonies."""
    features = []
    for i, masks in enumerate(segmented_images):
        img = original_images[i]
        for colony_id in range(1, np.max(masks) + 1):
            colony_mask = (masks == colony_id).astype(np.uint8)
            colony_img = cv2.bitwise_and(img, img, mask=colony_mask)

            # Calculate features (example: mean intensity, area)
            mean_intensity = cv2.mean(colony_img, mask=colony_mask)[0]
            area = np.sum(colony_mask)
            # ... add more features (texture, morphology, etc.) ...

            features.append([mean_intensity, area])  # ... other features ...
    return features

# --- 4. Model Creation and Training ---
def create_classification_model():
    """Creates and compiles the CNN classification model."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification (good/bad)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 5. Annotation and Output ---
def annotate_image(image, masks, predictions):
    """Annotates the image with colony labels."""
    for colony_id in range(1, np.max(masks) + 1):
        colony_mask = (masks == colony_id).astype(np.uint8)
        contours, _ = cv2.findContours(colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

        # Get prediction label
        label = "Good" if predictions[colony_id - 1][0] < 0.5 else "Bad"

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def save_results(image, filename, predictions):
    """Saves the annotated image and a text file with detailed information."""
    cv2.imwrite(f"annotated_{filename}", image)

    with open(f"results_{filename[:-4]}.txt", "w") as f:
        f.write(f"Image: {filename}\n")
        for i, pred in enumerate(predictions):
            label = "Good" if pred[0] < 0.5 else "Bad"
            confidence = (1 - pred[0]) * 100 if label == "Good" else pred[0] * 100
            f.write(f"Colony {i+1}: {label} (Confidence: {confidence:.2f}%)\n")

# --- Main Execution ---
if __name__ == "__main__":
    image_directory = 'path/to/your/images'
    images, labels, filenames = load_images(image_directory)
    segmented_images = segment_colonies(images)

    # --- Prepare data for training ---
    all_features = []
    all_labels = []
    for i, img in enumerate(images):
        features = extract_features([segmented_images[i]], [img])
        all_features.extend(features)
        all_labels.extend([labels[i]] * len(features))  # Duplicate label for each colony in the image

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    # --- Create and train the model ---
    model = create_classification_model()

    # Normalize features (important for neural networks)
    all_features = tf.keras.utils.normalize(all_features, axis=1)

    model.fit(all_features, all_labels, epochs=10, batch_size=32)  # Adjust epochs and batch_size

    # --- Make predictions and annotate ---
    for i, img in enumerate(images):
        features = extract_features([segmented_images[i]], [img])
        features = tf.keras.utils.normalize(np.array(features), axis=1)  # Normalize features
        predictions = model.predict(features)
        annotated_image = annotate_image(img.copy(), segmented_images[i], predictions)
        save_results(annotated_image, filenames[i], predictions)