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
<h3>Flask App for Model Deployment</h3>
In addition to training the model, we have also developed a Flask web application for deploying the trained model for real-time predictions. The Flask app allows users to upload OCT images and obtain predictions for the presence of Choroidal neovascularization (CNV) and Sub-Retinal fluid (SRF) using the trained model.
<h3>Usage</h3>
1. Installation: Ensure you have Flask installed. You can install Flask using pip:
   pip install flask
2. Run the Flask App: To run the Flask app, navigate to the project directory containing the Flask application (app.py) and run the following command:
   python app.py
3. Accessing the Web Interface: Once the Flask app is running, you can access the web interface by opening a web browser and navigating to http://localhost:5000 or http://127.0.0.1:5000.
4. Making Predictions: On the web interface, users can upload OCT images using the provided form. Upon submission, the uploaded image is processed by the deployed model, and the predicted probabilities of CNV and SRF are displayed on the result page.
<h3>Repository Structure</h3>
- app.py: Contains the Flask application code for handling HTTP requests and rendering web pages.
- templates/: Directory containing HTML templates for rendering the web pages.
- static/: Directory for storing static files such as CSS stylesheets or JavaScript scripts (if any).
<h3>Dependencies</h3>
- Flask: Web framework for Python used to build the web application.
- TensorFlow: Deep learning framework used for loading the trained model and making predictions.
<h3>Customization</h3>
The Flask app can be customized according to specific requirements, such as adjusting the visual appearance using CSS or extending the functionality to support additional features.
<h3>Deployment Options</h3>
The Flask app can be deployed on various platforms, including local servers, cloud platforms (such as Heroku or AWS), or containerized environments (using Docker). Ensure to follow best practices for deploying web applications securely and efficiently.
<h3>Dataset and Trained Model</h3>

If you want to try the code, you can download the dataset and the trained model from the following Google Drive links:

- [Download Dataset](https://drive.google.com/drive/folders/1YCsY895YKWygrCrIoiUqqn55Nx4U4qic?usp=drive_link)
- [Download Trained Model](https://drive.google.com/file/d/1UqQrmffVXw2AU5Td1qpvWuXkumGTJXMR/view?usp=drive_link)

<h3>Contributing</h3>

Contributions to this project are welcome. If you have suggestions for improvements or encounter issues, please open an issue or submit a pull request.
