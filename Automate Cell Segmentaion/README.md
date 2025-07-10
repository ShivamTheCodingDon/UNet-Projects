# 🧬 Nucleus Segmentation with U-Net  
A deep learning approach to automate nucleus detection using biomedical images from the 2018 Data Science Bowl.

---

## 📌 Overview

This project addresses a critical medical imaging challenge automated detection of cell nuclei posed by the [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018). Accurate nucleus segmentation plays a key role in accelerating biomedical research by enabling high-throughput cellular analysis. Since nuclei carry genetic material, their detection helps in understanding disease progression, evaluating treatment responses, and expediting drug development.

---

## 🎯 Objective

To build a robust deep learning pipeline using the **U-Net architecture** for precise and efficient segmentation of cell nuclei in varied biomedical images. This repository contains a fully reproducible solution that includes preprocessing, model training, evaluation, and hyperparameter tuning.

---

## ⚙️ Methodology

The workflow follows a structured approach:

- **Data Preprocessing**: Normalize and standardize input images and masks.  
- **Model Architecture**: Implement a U-Net-based convolutional neural network for segmentation.  
- **Training & Validation**: Train the model on labeled data and validate on a separate subset.  
- **Evaluation**: Visually and quantitatively assess the model's segmentation performance.  
- **Hyperparameter Tuning**: Optimize model parameters using **Keras Tuner** to improve generalization.

---

## 📂 Dataset

The dataset comes from Kaggle’s 2018 Data Science Bowl. It contains labeled images of cell nuclei captured using different imaging modalities (e.g., brightfield, fluorescence) and magnifications.

### 🔽 Download Instructions

1. Create a Kaggle account.
2. Generate an API key from your account settings (downloads `kaggle.json`).
3. Upload the key in your Colab notebook:

```python
from google.colab import files
uploaded = files.upload()  # Upload your kaggle.json here
```

---

## 📊 Exploratory Data Analysis (EDA)

A sample of input images and corresponding masks was visualized to understand variations in:

- Cell shape and size  
- Imaging conditions  
- Mask complexity

This analysis guided decisions around preprocessing and model architecture.

---

## 📈 Results

### 🔍 Prediction Visualization

For sample test cases, we present:

- **Original Image**: Raw input image  
- **Ground Truth Mask**: Annotated mask showing nucleus locations  
- **Predicted Mask**: Output from the U-Net model

**Visual Alignment** between predictions and ground truth confirms strong segmentation performance.

### 📉 Training Curves

Plots of training/validation loss and accuracy over epochs demonstrate:

- Rapid convergence during early epochs  
- Minimal overfitting  
- Consistently high accuracy on both train and validation sets

---

## 🧪 Hyperparameter Tuning

After baseline training, the model was refined via hyperparameter tuning using `Keras Tuner`. Parameters like filter size, dropout rate, and learning rate were explored systematically.

### 🔄 Untuned vs. Tuned Model Comparison

| Metric              | Untuned Model | Tuned Model   |
|---------------------|---------------|---------------|
| Training Loss       | **0.0877**    | 0.0885        |
| Validation Loss     | **0.0794**    | 0.0933        |
| Training Accuracy   | 96.54%        | **96.61%**    |
| Validation Accuracy | **96.93%**    | 96.29%        |

The untuned model slightly outperformed the tuned version on unseen data, suggesting strong baseline hyperparameters.

---

## 💻 Environment Setup

To run this project locally:

1. Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Clone this repository.
3. Navigate to the repo directory.
4. Create the environment:

```bash
conda env create -f nucleus-segmentation-env.yml
conda activate nucleus-segmentation-env
```

---

## ✅ Conclusion

This project demonstrates the effective application of U-Net for biomedical image segmentation. The model successfully identifies cell nuclei across diverse image types, with strong accuracy and generalization. This work contributes toward scalable, automated medical image analysis, potentially accelerating disease research and therapeutic development.

---

## 📎 References

- [2018 Data Science Bowl - Kaggle](https://www.kaggle.com/c/data-science-bowl-2018)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
