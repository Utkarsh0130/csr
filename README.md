# CSR Detection Model using OCT Images
This repository contains code and resources for training and deploying a deep learning model to detect Choroidal neovascularization (CNV) and Sub-Retinal fluid (SRF) in Optical Coherence Tomography (OCT) images.
<h3>Overview</h3>
Choroidal neovascularization (CNV) and Sub-Retinal fluid (SRF) are pathological features often found in retinal diseases such as age-related macular degeneration (AMD) and diabetic retinopathy (DR). Early detection of these features is crucial for timely treatment and preservation of vision.
This project aims to develop a convolutional neural network (CNN) model to automatically detect CNV and SRF in OCT images. The model is trained on a labeled dataset consisting of OCT scans with annotations indicating the presence or absence of CNV and SRF.
<h3>Dataset</h3>
The dataset used for training and evaluation is located in the following directories:
- Training data: `train_dir`
- Validation data: `validation_dir`
- Test data: `test_dir`
The images are preprocessed and augmented using techniques such as rescaling, shearing, zooming, and horizontal flipping.
<h3>Model Architecture</h3>
The CNN model architecture consists of convolutional layers followed by max-pooling layers for feature extraction. It utilizes three convolutional layers with increasing filter sizes (32, 64, and 128) and max-pooling layers to downsample the feature maps. The final layer performs binary classification using the sigmoid activation function.
<h3>Training</h3>
The model is trained using the adam optimizer and binary cross-entropy loss function. Training is performed for 50 epochs with a batch size of 32. The training and validation accuracy are monitored to prevent overfitting.
<h3>Evaluation</h3>
The trained model is evaluated on a separate test set to assess its performance in detecting CNV and SRF. Evaluation metrics such as accuracy, precision, recall, and F1-score are calculated to evaluate the model's effectiveness.
<h3>Deployment</h3>
The trained model can be deployed for inference in various applications, including web-based platforms, mobile applications, and healthcare systems. Instructions for loading the model and making predictions are provided in the deployment section.
<h3>Usage</h3>
To train the model:
1. Clone this repository.
2. Organize the dataset into directories (`train_dir`, `validation_dir`, `test_dir`).
3. Run the training script (`train.py`) and specify the dataset directories.
To deploy the model:
1. Load the trained model using TensorFlow or another deep learning framework.
2. Preprocess input OCT images as required.
3. Make predictions using the loaded model.

<h3>Dataset and Trained Model</h3>

If you want to try the code, you can download the dataset and the trained model from the following Google Drive links:

- [Download Dataset](https://drive.google.com/drive/folders/1YCsY895YKWygrCrIoiUqqn55Nx4U4qic?usp=drive_link)
- [Download Trained Model](https://drive.google.com/file/d/1UqQrmffVXw2AU5Td1qpvWuXkumGTJXMR/view?usp=drive_link)

<h3>Contributing</h3>

Contributions to this project are welcome. If you have suggestions for improvements or encounter issues, please open an issue or submit a pull request.
