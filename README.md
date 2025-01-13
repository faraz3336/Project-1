# **Potato Disease Classification Using CNN (Convolutional Neural Networks)**

## **Project Overview**
In this project, I developed a deep learning model using **Convolutional Neural Networks (CNNs)** to classify potato leaf diseases with **100% accuracy**. The model efficiently identifies three key categories:
- **Early Blight**
- **Late Blight**
- **Healthy Leaves**

This tool is designed to aid farmers by enabling quick and accurate disease detection in potato crops, contributing to improved crop management, higher yield, and reduced crop losses.

---

## **Dataset**
The dataset used for training and testing comes from [Plant Village on Kaggle](https://www.kaggle.com/arjuntejaswi/plant-village). It includes labeled images of potato leaves categorized into:
- **Early Blight**
- **Late Blight**
- **Healthy Leaves**

The dataset provides the necessary data to train the CNN model and assess its performance in real-world conditions.

---

## **Key Features**

### **Data Preprocessing**:
Data preprocessing is a crucial step in the pipeline to enhance model performance and ensure accuracy:
- **Data Augmentation**: I applied various augmentation techniques such as **rotation**, **flipping**, and **zooming** to diversify the training set and improve generalization, allowing the model to better handle real-world variations in leaf appearances.
- **Image Resizing**: Images were resized to a consistent shape (e.g., 150x150 pixels) for uniform input to the CNN.
- **Normalization**: Pixel values were normalized to a range between 0 and 1 to stabilize training and speed up convergence.

### **Model Architecture**:
- **CNN Design**: The model architecture is built using **TensorFlow** and **Keras**. Key layers include:
  - **Convolutional Layers**: These layers automatically extract features from images, such as edges, textures, and shapes.
  - **Pooling Layers**: Used to downsample the feature maps, reducing computational complexity and controlling overfitting.
  - **Fully Connected Layers**: These layers perform the final classification after the feature extraction.
  - **Dropout Layers**: Implemented to reduce overfitting by randomly setting a fraction of input units to zero during training.

### **Training and Optimization**:
- **Optimizer**: The **Adam optimizer** was used for efficient gradient descent and quicker convergence.
- **Loss Function**: The **categorical cross-entropy** loss function was employed for multi-class classification.
- **Achieved Accuracy**: The model achieved **100% accuracy** on both training and testing datasets, confirming its high performance in disease classification.

### **Model Evaluation**:
After training, the model was evaluated using key performance metrics such as:
- **Accuracy**: Measures the overall classification performance.
- **Precision**: Evaluates the accuracy of positive predictions.
- **Recall**: Measures the ability to detect actual positives.
- **F1-Score**: A balance between precision and recall.
  
The **confusion matrix** confirmed the model's ability to classify all disease categories without error.

---

## **Technologies Used**
- **Programming Language**: Python
- **Frameworks & Libraries**:
  - **TensorFlow** & **Keras** (Deep Learning)
  - **NumPy** (Data Manipulation)
  - **Matplotlib** (Visualization)
- **Tools**: Jupyter Notebooks, Google Colab (for training and experimentation)

---

## **Skills Demonstrated**
- **Deep Learning** with **CNNs** for image classification.
- **Model Training & Optimization** using **TensorFlow** and **Keras**.
- **Data Augmentation** techniques to improve model performance.
- **Python Programming** and application of advanced machine learning concepts.

---

## **Results & Impact**
- The model has achieved **100% accuracy** on both training and testing datasets, which makes it a highly reliable tool for potato disease classification.
- This project can significantly improve agricultural practices by enabling **farmers** to detect potato diseases early, preventing crop losses, and optimizing productivity. Early disease detection is key to maintaining healthy crops and maximizing yield.

---

## **How to Use**  
### 1. **Clone the repository**:
   - You can clone the repository to your local machine using the command:
     ```bash
     git clone https://github.com/yourusername/potato-disease-classification.git
     ```

### 2. **Install required libraries**:
   - Ensure you have the necessary libraries installed:
     ```bash
     pip install -r requirements.txt
     ```

### 3. **Train the model**:
   - Open and run the **Jupyter Notebook** or **Google Colab** notebook to train the CNN model on the potato leaf dataset.

### 4. **Prediction**:
   - Once the model is trained, you can use it to predict new images of potato leaves for disease classification.

---

## **Screenshots**  
Below are relevant screenshots showing the model's functionality during training and testing:

### 1. **Potato Disease Classification Model Training**:
   Potato disease classification model in Jupyter Notebook.:
   ![Model Training Accuracy](https://github.com/user-attachments/assets/6f720d9c-095f-4a22-9d62-71c9434d2cae)

### 2. **Visualize some of the images from our dataset**:
   ![Model Loss During Training](https://github.com/user-attachments/assets/559c55d9-5e49-4923-9b1d-5178693e7970)

### 3. ** Model Architecture**:
   The confusion matrix highlights the perfect classification performance, with all diseases accurately identified:
   ![Confusion Matrix](https://github.com/user-attachments/assets/df2a311f-2e8f-468e-a8db-84f3c58e6053)

### 4. **Compiling the Model**:
   ![Predicted vs. Actual Classification](https://github.com/user-attachments/assets/fb369666-36f7-4c4e-a5cc-62e32bd0b98e)

### 5. **Plotting the Accuracy and Loss Curves**:
   ![Example Classification](https://github.com/user-attachments/assets/0118d72c-578f-4f59-b63d-cc7d08766abe)

### 6. **Final Classification Results**:
   Final output showing how the model classifies various leaf images:
   ![Final Classification Results](https://github.com/user-attachments/assets/19c113f4-b6fe-4830-aaa1-36db077b5abe)
