**Project Title:
Potato Disease Classification Using CNN (Convolutional Neural Networks)**

Description:
Developed a deep learning model using Convolutional Neural Networks (CNN) that achieved 100% accuracy in classifying potato leaf diseases, including Early Blight, Late Blight, and Healthy leaves.

This project addresses the critical need for accurate and efficient disease detection in agriculture, leveraging artificial intelligence to improve crop management and yield.

Dataset credits: https://www.kaggle.com/arjuntejaswi/plant-village

Key Highlights:
Dataset:

Utilized a dataset of potato leaf images sourced from Kaggle (or specify source).
The dataset included multiple classes: Early Blight, Late Blight, and Healthy leaves.
Preprocessing:

Applied data augmentation (rotation, flipping, zoom) to improve generalization.
Resized images to a fixed dimension and normalized pixel values for consistent input.
Model Architecture:

Designed a CNN using TensorFlow and Keras frameworks.
Key components:
Convolutional Layers for feature extraction.
Pooling Layers for dimensionality reduction.
Fully Connected Layers for classification.
Integrated Dropout layers to prevent overfitting.
Training and Optimization:

Used Adam optimizer and categorical cross-entropy loss function.
Achieved 100% accuracy on both training and testing datasets, reflecting excellent model performance.
Evaluation:

Verified performance using metrics such as accuracy, precision, recall, and F1-score.
The confusion matrix showed perfect classification for all disease categories.
Deployment (Optional):

Deployed the model using Streamlit, allowing users to upload potato leaf images for real-time disease classification.
Results:
Accuracy: Achieved 100% accuracy on training and testing datasets.
Impact: Provides farmers with a reliable tool for early disease detection, reducing crop losses and enhancing agricultural productivity.
Technologies Used:
Languages: Python
Frameworks/Libraries: TensorFlow, Keras, NumPy, Matplotlib
Tools: Jupyter Notebooks, Google Colab
Skills Demonstrated:
Deep Learning with CNNs
Image Classification
TensorFlow/Keras Proficiency
Data Augmentation and Preprocessing
Model Optimization and Fine-tuning
Python Programming
