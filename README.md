# Facial Attribute Detection Project

## Overview
This project focuses on detecting human faces in live video streams and identifying key facial features using a Multi-task Cascaded Convolutional Neural Network (MTCNN). Additionally, we compare our custom implementation with an alternative implementation for further analysis. The system supports real-time face detection with bounding boxes and evaluates facial attributes such as smiling, eyeglasses, and other characteristics.

## Directory Structure
### `celeba_data/`
- **Description:** Contains the CelebA dataset used for training and testing the model.
- **Contents:** 
  - `img_align_celeba/`: Directory containing aligned celebrity face images.
  - `list_attr_celeba.csv`: CSV file containing attribute labels for each image.

### `data_loading/`
- **Description:** Scripts to load and preprocess the CelebA dataset.
- **Files:**
  - `data_loader_original.py`: Original implementation of the data loader.
  - `data_loader_refactored.py`: Refactored data loader for second model.
  - `downloader.py`: Script to download the dataset from Kaggle.

### `stages/`
- **Description:** Core components of the model, including MTCNN implementation and attribute recognition.
- **Subdirectories:**
  - `main_stages/`: Contains the primary stages of the MTCNN and attribute recognition networks.
    - `MTCNN.py`: Custom implementation of MTCNN for face detection.
    - `AttributeRecognitionGlobal.py`: CNN for attribute recognition.
    - `AttributeRecognitionMMCNN.py`: Alternative CNN for attribute recognition.
  - `models/`: Encapsulates the complete facial attribute detection pipeline.
    - `FacialAttributeDetection.py`: Main model integrating MTCNN and attribute recognition.
    - `FacialAttributeDetection_Refactored.py`: Refactored implementation for testing.
  - `mtcnn_stages/`: Individual components of the MTCNN model.
    - `pnet.py`: Proposal network.
    - `rnet.py`: Refinement network.
    - `onet.py`: Output network.
  - `utils/`: Utility functions for bounding box generation and patch extraction.
    - `props.py`: Functions for manipulating bounding boxes and extracting patches.

### `tests/`
- **Description:** Scripts for testing the MTCNN implementation and data loaders.
- **Files:**
  - `mtcnn_test.py`: Unit tests for the MTCNN implementation.
  - `test_loader.py`: Tests for dataset loading and preprocessing.

### `train/`
- **Description:** Scripts for training the models.
- **Files:**
  - `train_original.py`: Script to train the original MTCNN model with attribute recognition.
  - `train_refactored.py`: Refactored training script for comparative analysis.

### `Face_Attribute_model.pth`
- **Description:** Pre-trained model weights for facial attribute detection.

### `face_detection.py`
- **Description:** Script for real-time face detection with bounding boxes and attribute visualization.

### `README.md`
- **Description:** This file provides an overview of the project.

### `requirements.txt`
- **Description:** Contains the list of required Python packages.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Rapcole12/Facial_Attribute_Detection/tree/main
   cd Facial_Attribute_Detection
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   
3. Install the dataset: (This step is optional since repo holds first 20,000 datapoints)
   ```bash
   python data_loading/downloader.py
   
## Training the Model
1. Train the model using our implementation:
   ```bash
   python train/train_original.py
2. Alternatively, train the model using the online example of MTCNN:
   ```bash
   python train/train_refactored.py
3. After training, the model weights will be saved as Face_Attribute_model.pth in the root directory.

## Running Real-Time Face Detection
1. Ensure Face_Attribute_model.pth is in the root directory.
2. Run the face_detection.py script:
   ```bash
   python face_detection.py
3. The script will:
* Open a live video stream.
* Detect faces and draw bounding boxes.
* Display facial attributes as percentage confidence levels.
4. Press q to exit the live video stream.

## Resources
The following paper gave us inspriation for this project and gave us important insight on the math behind an MTCNN
and was able to help guide us in our implementaion.
https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf
