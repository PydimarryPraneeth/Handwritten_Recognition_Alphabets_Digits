Handwritten Digit and Alphabet Recognition using CNN
This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits (0-9) and capital letters (A-Z). The model is built using Python and PyTorch, leveraging GPU acceleration via CUDA for efficient training.
Problem Definition
The objective is to classify handwritten characters, including digits and capital alphabets. This is a multi-class image classification problem and is a subset of Optical Character Recognition (OCR). The model recognizes separate characters rather than cursive handwriting.

Datasets Used
MNIST Dataset - Contains 60,000 images of handwritten digits (0-9).
Kaggle A-Z Dataset - Contains 372,450 images of handwritten capital letters (A-Z), modified from the NIST Special Database 19.
Both datasets consist of grayscale images of size 28x28.

Model Architecture
The CNN model consists of:

Convolutional Layers - For feature extraction.

Pooling Layers - For down-sampling.

Dropout Layers - To prevent overfitting.

Fully Connected Layers - For classification.

Loss Function: Negative Log Likelihood Loss

Optimizer: Adam

Output: Log-probabilities for each class, with the maximum taken as the predicted class.

Technologies Used
Python
PyTorch - For building and training the CNN model.
CUDA - For GPU acceleration.
NumPy and Pandas - For data manipulation.
Matplotlib - For visualization.
Installation
Clone the repository:
sh
Copy
Edit
git clone https://github.com/your-username/Handwritten-Character-Recognition-CNN.git
cd Handwritten-Character-Recognition-CNN
Create a virtual environment and activate it:
sh
Copy
Edit
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows
Install dependencies:
sh
Copy
Edit
pip install -r requirements.txt
Usage
Download and prepare the datasets:

MNIST dataset is available in torchvision.datasets.
Kaggle A-Z dataset should be downloaded manually and placed in the appropriate directory.
Train the model:

sh
Copy
Edit
python train.py
Test the model:

sh
Copy
Edit
python test.py
Predict on custom images:

sh
Copy
Edit
python predict.py --image_path <path_to_image>
Results
The model achieved high accuracy on test data.
GPU acceleration using CUDA significantly reduced training time.
The model is limited to recognizing single characters (no cursive handwriting).
Contributing
Contributions are welcome! Please follow the steps below:

Fork the project.
Create a new branch (git checkout -b feature/AmazingFeature).
Commit your changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a Pull Request.
