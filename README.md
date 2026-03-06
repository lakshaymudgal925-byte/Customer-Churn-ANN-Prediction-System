# 🏦 Customer Churn ANN Prediction System

An **AI-powered Customer Churn Prediction Web App** built using **Artificial Neural Networks (ANN)**, **TensorFlow**, and **Streamlit**.  
This system predicts whether a bank customer is likely to **leave (churn)** or **stay**, helping businesses make data-driven retention strategies.

---

# 🚀 Project Overview

Customer churn is a major problem in industries like banking, telecom, and SaaS.  
Losing customers leads to significant revenue loss.

This project uses **Deep Learning (ANN)** to analyze customer data and predict churn probability.

The application includes:

- Model training interface
- Interactive data visualizations
- Real-time churn prediction
- Model performance metrics
- Customer risk analysis dashboard

---

# 🧠 Model Architecture

Artificial Neural Network Structure:

- **Input Layer:** 11 Features
- **Hidden Layer 1:** 16 Neurons (ReLU) + Dropout (20%)
- **Hidden Layer 2:** 9 Neurons (ReLU) + Dropout (20%)
- **Output Layer:** 1 Neuron (Sigmoid)

Training Configuration:

- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss Function:** Binary Crossentropy
- **Epochs:** 50
- **Batch Size:** 32

---

# 📊 Features of the Application

### 1️⃣ ANN Model Training
- Train the neural network model directly from the interface
- Uses **Scikit-learn Pipeline**
- Automatic preprocessing

### 2️⃣ Data Visualizations
Interactive charts built with **Plotly**:

- Churn distribution
- Churn by geography
- Age distribution
- Gender analysis
- Financial metrics
- Feature correlation heatmap

### 3️⃣ Churn Prediction System
Enter customer details and get:

- Churn probability
- Stay probability
- Risk score visualization
- Instant prediction

### 4️⃣ Model Performance Metrics

Includes:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- False Positive Rate
- Classification Report

### 5️⃣ Dataset Analysis

Provides:

- Descriptive statistics
- Feature distribution
- Churn comparison analysis

---

# 📂 Dataset Features

The model predicts churn using the following customer attributes:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Active Member Status
- Estimated Salary

Target Variable:

Exited (1 = Churn, 0 = Stay)

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run churn.py

Author
Lakshy Mudgal
