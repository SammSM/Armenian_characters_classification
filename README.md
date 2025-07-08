# ✍ Armenian_characters_classification
A deep learning project using CNN to recognize and classify handwritten Armenian alphabet characters.

## Models overview

## This project contains two neural network models trained to recognize handwritten Armenian letters:

### Model without data augmentation (arm_h5_model.h5 / arm_pkl_model.pkl)
- Trained only on the original dataset.
- No artificial noise or transformations applied.
- Serves as a baseline model.

### Model with data augmentation (aug_arm_h5_model.h5 / aug_arm_pkl_model.pkl)

- Trained on the original dataset plus additional augmented data.
- Augmentation includes techniques such as random noise and scaling.
- Shows improved performance and better generalization.

## Both models support classification of 78 Armenian characters (uppercase and lowercase) and can be used to predict the letter shown in a handwritten image.

---

## 📚 Libraries

- **TensorFlow / Keras** — Model building, training, saving, and loading.
- **scikit-learn** — Utilities for train/test splitting and shuffling data.
- **NumPy** — Data manipulation.
- **OpenCV** — Image preprocessing: resizing, color conversion, bitwise operations.
- **Matplotlib** — Visualizing images and plotting loss/accuracy graphs.
- **Pickle** — Saving/loading machine learning models (use cautiously for Keras models).
- Other Keras modules: layers, optimizers, callbacks (EarlyStopping).

---

## 🔄 Data Preprocessing and Augmentation

- Images are loaded as grayscale, reshaped to (64x64x1).
- Augmented data from the `augmented_dataset/` folder is merged with original data.
- Data is normalized and one-hot encoded for 78 classes.
- Training data is shuffled before training.

---

