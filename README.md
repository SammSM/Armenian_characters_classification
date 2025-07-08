# ✍ Armenian_characters_classification
A deep learning project using CNN to recognize and classify handwritten Armenian alphabet characters.

## Models overview

#### This project contains two neural network models trained to recognize handwritten Armenian letters:

### Model without data augmentation (arm_h5_model.h5 / arm_pkl_model.pkl)
- Trained only on the original dataset.
- No artificial noise or transformations applied.
- Serves as a baseline model.

### Model with data augmentation (aug_arm_h5_model.h5 / aug_arm_pkl_model.pkl)

- Trained on the original dataset plus additional augmented data.
- Augmentation includes techniques such as random noise and scaling.
- Shows improved performance and better generalization.

#### Both models support classification of 78 Armenian characters (uppercase and lowercase) and can be used to predict the letter shown in a handwritten image. The dataset used for training was loaded from Kaggle and contains labeled images of Armenian handwritten letters.

## 📁 Repository Structure

```
📂 arm_char_cnn/
│
├── models/
├── test_images/
├── aug_arm_char_cnn.ipynb
├── augmentation.ipynb
└── requirements.txt

📂 arm_char_cnn_without_augmentation/
│
├── models/
├── test_images/
├── arm_char_cnn.ipynb
└── requirements.txt
```

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

## 🔄 Data preprocessing + augmentation

- Images are loaded as grayscale, reshaped to (64x64x1).
- Augmented data from the `augmented_dataset/` folder is merged with original data.
- Data is normalized and one-hot encoded for 78 classes.
- Training data is shuffled before training.

---

## 🔧 Optimization and regularization techniques
### To improve training efficiency and model performance, the following techniques were applied:

- Dropout
- Batch Normalization
- Early Stopping
- Learning rate manually set for the Adam optimizer

#### Together, these methods optimize the training process, improve generalization, and help the model achieve higher accuracy with fewer epochs.

---

## ⚙️ How it works

#### Provide the path to an input image of a handwritten Armenian letter.

The image will be:

- Loaded from the specified path
- Converted to grayscale
- Resized to 64x64 pixels
- Inverted (if needed) to match the training format (white character on black background)
- Reshaped into the required input shape (1, 64, 64, 1)

#### The model will process the image and return the predicted class index and corresponding Armenian character.

---

## 🛠️ Setup guide

### 1. Clone the repository
Open your environment and clone the repository
```bash
https://github.com/SammSM/Armenian_characters_classification.git
```
### 2. Change to Armenian_characters_classification directory
```bash
cd Armenian_characters_classification
```

### 4. Create a virtual environment and activate it

- ### Create a virtual environment
On Windows:
```bash
python -m venv venv
```
On macOS / Linux:
```bash
python3 -m venv venv
```
- ### Activate the virtual environment
On Windows:
```bash
venv\Scripts\activate
```
On macOS / Linux:
```bash
source venv/bin/activate
```

### 5. Install PIP package manager for Python
```bash
py -m pip install --upgrade pip
```
### Or
```bash
python -m pip install --upgrade pip
```

### 6. Install requirements
```bash
pip install -r requirements.txt
```
