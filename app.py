import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import os
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="Blood Cancer AI", layout="wide")

MODEL_PATH = "blood_cancer_model.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

TRAIN_DIR = r"C:\Users\hp\Downloads\bloodcancer\cancer\train"
TEST_DIR = r"C:\Users\hp\Downloads\bloodcancer\cancer\test"

# --- SIDEBAR NAVIGATION ---
page = st.sidebar.selectbox(
    "Select Mode",
    ["Inference (Predict)", "Training Mode"]
)

# --- HELPER FUNCTIONS ---
def build_model(num_classes):

    base_model = VGG16(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --- TRAINING PAGE ---
if page == "Training Mode":

    st.header("⚙️ Blood Cancer Model Training")

    st.info(f"Training Data Path: {TRAIN_DIR}")

    epochs = st.slider(
        "Number of Epochs",
        1, 20, 10
    )

    if st.button("Start Training"):

        with st.status(
            "Training in progress...",
            expanded=True
        ) as status:

            # Data Preparation
            datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )

            st.write("Loading datasets...")

            train_gen = datagen.flow_from_directory(
                TRAIN_DIR,
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                subset="training"
            )

            val_gen = datagen.flow_from_directory(
                TRAIN_DIR,
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                subset="validation"
            )

            # Automatically detect number of classes
            num_classes = train_gen.num_classes

            st.write(
                f"Detected classes: {train_gen.class_indices}"
            )

            # Model Training
            model = build_model(num_classes)

            # Callback
            class stCallback(tf.keras.callbacks.Callback):

                def on_epoch_end(
                    self,
                    epoch,
                    logs=None
                ):
                    st.write(
                        f"Epoch {epoch+1}: "
                        f"Accuracy={logs['accuracy']:.2f}, "
                        f"Val_Acc={logs['val_accuracy']:.2f}"
                    )

            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=[stCallback()]
            )

            model.save(MODEL_PATH)

            status.update(
                label="Training Complete! Model Saved.",
                state="complete",
                expanded=False
            )

        st.success(
            "Model trained and saved as blood_cancer_model.h5"
        )

        st.line_chart(
            history.history["accuracy"]
        )


# --- INFERENCE PAGE ---
elif page == "Inference (Predict)":

    st.header("🩸 Blood Cancer Classifier")

    if not os.path.exists(MODEL_PATH):

        st.error(
            "No trained model found! "
            "Please go to 'Training Mode' first."
        )

    else:

        class_names = sorted([
            folder
            for folder in os.listdir(TRAIN_DIR)
            if os.path.isdir(
                os.path.join(TRAIN_DIR, folder)
            )
        ])

        @st.cache_resource
        def load_trained_model():
            return tf.keras.models.load_model(
                MODEL_PATH
            )

        model = load_trained_model()

        uploaded_file = st.file_uploader(
            "Upload blood-cell image",
            type=["jpg", "png", "jpeg"]
        )

        if uploaded_file:

            img = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                img,
                width=300
            )

            # Preprocess
            img_processed = img.resize(
                IMG_SIZE
            )

            img_array = img_to_array(
                img_processed
            ) / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Predict
            preds = model.predict(
                img_array
            )

            result = class_names[
                np.argmax(preds)
            ]

            confidence = np.max(preds)

            st.metric(
                "Prediction",
                result,
                f"{confidence*100:.2f}% Confidence"
            )

            st.bar_chart(
                dict(
                    zip(
                        class_names,
                        preds[0].tolist()
                    )
                )
            )

        st.warning(
            "For educational/research use only. "
            "This is not a medical diagnosis."
        )