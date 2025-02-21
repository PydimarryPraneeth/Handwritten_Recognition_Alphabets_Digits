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

Download and prepare the datasets:
MNIST dataset is available in torchvision.datasets.
Kaggle A-Z dataset should be downloaded manually and placed in the appropriate directory.

