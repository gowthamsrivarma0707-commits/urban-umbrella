# urban-umbrella
A Machine Learning model that identifies and classifies blood cancer from image inputs.
# Blood Cancer AI Classification App

A Streamlit-based web application for classifying blood cell images as **Cancer** or **Normal** using a deep learning model (VGG16). Supports model training, image prediction, and post-training evaluation.

## ✨ Features

- **Training Mode**: Custom training of a VGG16-based model with progress tracking.
- **Inference Mode**: Upload an image to predict the most likely class and confidence score.
- **Evaluation Mode**: Post-training accuracy assessment on a dedicated test dataset.
- **User-Friendly Interface**: Clean Streamlit layout with sidebar navigation and responsive design.

## 🛠️ Prerequisites

- **Python** (≥ 3.7)
- **TensorFlow** (≥ 2.0)
- **Streamlit** (latest version)
- **Pillow** (for image handling)

Install the required packages:

```bash
pip install streamlit tensorflow pillow
```

## 📁 Data Preparation

The application expects images in a structured directory layout.

```
<data_root>
├── cancer/          # Contains cancer class images
├── normal/          # Contains normal class images
└── test/            # Contains test images for evaluation
```

- The training data is automatically split into 80% training and 20% validation sets.
- The test directory is used exclusively for unbiased evaluation.

## 🚀 How to Run

1.  **Locate the code file**: Find the script file (e.g., `blood_cancer_app.py`).
2.  **Adjust paths** (if needed): Ensure the `TRAIN_DIR` and `TEST_DIR` paths in the script match your data's location.
3.  **Execute the script**:

    ```bash
    streamlit run app.py
    ```

This will launch the Streamlit web interface.

## 📌 Usage Guide

### Training Mode
1. Navigate to **"Training Mode"** in the sidebar.
2. Adjust the number of epochs with the slider.
3. Click **"Start Training"** to begin and observe the training progress.
4. After completion, the model is saved as `blood_cancer_model.h5`.

### Inference Mode
1. Navigate to **"Inference (Predict)"** in the sidebar.
2. Select an image file (JPG, PNG) from the file uploader.
3. The app will display the image and provide:
    - The predicted class (Cancer/Normal)
    - Confidence score as a percentage
    - A bar chart of prediction probabilities.

### Evaluation Mode
After training, the app can automatically evaluate the model on the test dataset to measure accuracy.

## ⚙️ Customization

You can modify the following parameters in the code file:

- **Image Size**: `IMG_SIZE = (128, 128)`
- **Batch Size**: `BATCH_SIZE = 32`
- **Epochs**: Adjusted via the slider during training.
- **Model Architecture**: The base model is VGG16; you can modify the layers in the `build_model()` function.

## 📝 Notes

- **Data Quality**: Model performance depends on the quality and representativeness of your dataset.
- **Class Detection**: The number of classes is automatically detected from the dataset folders.
- **Security**: This project is for educational purposes. Ensure data privacy and compliance with relevant policies.

---

This content is all you need in the README. Keep the code in a separate file, and this documentation will be clear and sufficient for your project.
