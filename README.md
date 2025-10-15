# 🧠 Brain Tumor Classification using CNN

This repository contains a Convolutional Neural Network (CNN) model for classifying MRI brain images into three tumor types: **Glioma**, **Meningioma**, and **Pituitary**. The model is built and trained on the public dataset available on Kaggle:

> [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

---

## 📂 Dataset

The dataset consists of MRI images categorized into four folders: `glioma`, `meningioma`, `pituitary`, and `no_tumor`. Each image is preprocessed before being fed into the CNN model, including resizing and normalization.

---

## ⚙️ Model Architecture

A standard **Convolutional Neural Network (CNN)** was implemented using TensorFlow/Keras. The architecture includes:

* Convolutional layers with ReLU activation
* MaxPooling layers for spatial reduction
* Fully connected dense layers
* Softmax output layer for 3-class classification

The network is designed for simplicity and efficiency — no transfer learning or pre-trained weights are used.

---

## 📊 Evaluation Metrics

The model is trained with **categorical cross-entropy loss** and evaluated using **accuracy** as the main performance metric.

Performance may vary depending on hyperparameters, augmentation strategy, and hardware setup.

---

## 🚀 Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/jeremysean/brain-tumor-classification.git
   cd brain-tumor-classification
   ```
2. Download the dataset from Kaggle and extract it into a `dataset/` folder.
3. Open and run the notebook:

   ```bash
   jupyter notebook dcnn_.ipynb
   ```

You can modify paths, batch sizes, or image dimensions inside the notebook as needed.

---

## 🧪 Results

After training for several epochs, the model achieves a reasonable classification accuracy on the test set. Accuracy typically ranges between **85–95%**, depending on tuning and preprocessing.

Confusion matrices and accuracy plots can be generated for visual evaluation.

---

## 📜 License

This project is licensed under the **MIT License**. You’re free to use, modify, and distribute it with proper attribution.

---

## 👨‍💻 Author

Developed by a data science enthusiast focusing on deep learning for medical imaging applications. Contributions, discussions, and improvements are welcome!

---

### 🧠 Future Work

* Integrate Grad-CAM for interpretability
* Experiment with Transfer Learning (VGG16, ResNet50)
* Deploy as a simple web or mobile app for real-time inference
