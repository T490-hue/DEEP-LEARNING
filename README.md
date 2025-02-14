# Brain Tumor Detection and Classification using Custom CNN

## Overview
This project implements a deep learning model using a custom Convolutional Neural Network (CNN) in PyTorch to detect and classify brain tumors. The model achieves an accuracy of **96%** on the test dataset.

## Dataset
The dataset used for training and testing consists of MRI brain scans categorized into four classes:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

### Dataset Structure
```
BrainTumour_ResizedDataset/
│── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   ├── notumor/
│── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   ├── notumor/
```

## Model Architecture
The CNN is designed to effectively extract spatial features and classify images with high accuracy. The architecture consists of:

1. **Convolutional Layers:**
   - Three convolutional layers with kernel size 3x3 and ReLU activation.
   - Feature extraction is performed at each layer, progressively capturing more complex patterns.

2. **MaxPooling Layers:**
   - Applied after each convolutional block to downsample feature maps.
   - Reduces spatial dimensions while retaining important features.

3. **Fully Connected Layers:**
   - After flattening, a fully connected layer with 512 neurons and ReLU activation is used.
   - A final output layer with four neurons (one per class) and softmax activation for classification.

4. **Dropout Layer:**
   - A dropout rate of 0.5 is applied to prevent overfitting by randomly disabling neurons during training.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- PIL (Pillow)
- NumPy

### Install Dependencies
```sh
pip install torch torchvision scikit-learn numpy pillow
```

## Usage

###To perform preprocessing of images
Run the following command:
python3 opencv_imagepreprocessing.py
### Training the Model
Run the following command to train the model:
```sh
python3 training.py
```


## Results
- **Best Test Accuracy:** 96%
