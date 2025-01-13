Potato Disease Classification Using CNN (Convolutional Neural Networks)
Project Overview
In this project, I developed a deep learning model using Convolutional Neural Networks (CNN) to classify potato leaf diseases with 100% accuracy. The model identifies key potato diseases like Early Blight, Late Blight, and also distinguishes Healthy leaves.

This project is an important step toward enhancing agricultural practices, offering farmers an AI-powered tool for quick, accurate disease detection, leading to better crop management and improved yields.

Dataset
The dataset used for training and testing is sourced from Plant Village on Kaggle. It contains labeled images of potato leaves, including various disease categories:

Early Blight
Late Blight
Healthy Leaves
Key Features
Data Preprocessing:
Data Augmentation: Applied techniques like rotation, flipping, and zooming to create varied training data and improve model generalization.
Image Resizing: All images were resized to a fixed dimension to ensure consistent input to the model.
Normalization: Pixel values were normalized to improve model performance and stability.
Model Architecture:
Designed a CNN model using TensorFlow and Keras.
Key layers:
Convolutional Layers for automatic feature extraction from images.
Pooling Layers for dimensionality reduction and more efficient processing.
Fully Connected Layers to perform the final classification.
Dropout Layers to prevent overfitting and ensure better generalization.
Training and Optimization:
Optimizer: Adam optimizer was used to minimize the loss function.
Loss Function: Categorical cross-entropy loss was employed for multi-class classification.
Achieved Accuracy: 100% accuracy on both the training and testing datasets, reflecting exceptional model performance.
Evaluation:
Evaluated the model using key metrics such as:
Accuracy
Precision
Recall
F1-Score
The confusion matrix showed perfect classification across all categories.
Deployment (Optional)
For real-time use, I deployed the model using Streamlit, enabling users to upload potato leaf images and get disease predictions instantly.

Technologies Used
Programming Language: Python
Frameworks/Libraries:
TensorFlow & Keras (Deep Learning)
NumPy (Data manipulation)
Matplotlib (Visualization)
Tools: Jupyter Notebooks, Google Colab (for training)
Skills Demonstrated
Deep Learning with CNNs
Image Classification
Model Training & Optimization
Data Augmentation and Preprocessing
Python Programming and TensorFlow/Keras proficiency
Results & Impact
Achieved 100% accuracy on both training and testing datasets, ensuring highly reliable results.
The model provides farmers with an accurate, fast tool to detect diseases early, helping prevent crop loss and enhance agricultural productivity.
How to Use
Clone the repository.
Install required libraries (e.g., TensorFlow, Keras, etc.).
Run the Jupyter Notebook or Google Colab notebook to train the model.
Optionally, use the Streamlit app for real-time disease classification.
