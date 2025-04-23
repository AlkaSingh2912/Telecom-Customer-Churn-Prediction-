# 📊 Customer Churn Prediction 

This project focuses on predicting customer churn using machine learning models, leveraging an "Indianized" version of the popular churn dataset. The project includes preprocessing, model building, evaluation, and visualizations to better understand churn behavior.

---

## 🔍 Project Objectives

- Clean and preprocess customer churn data.
- Apply machine learning models (Logistic Regression, KNN, and SVM).
- Evaluate models using accuracy, confusion matrix, and classification report.
- Visualize patterns and churn behavior using Seaborn and Matplotlib.

---

## 📁 Dataset

- Dataset: `indianized_churn.csv`
- Contains telecom customer data including demographic info, service usage, and churn status.
- `Churn` column is the target variable (Yes/No).

---

## ⚙️ Technologies Used

- Python 3.x
- Libraries:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`

---

## 🧼 Preprocessing Steps

- Dropped `customerID`
- Converted `TotalCharges` to numeric and handled missing values
- Encoded categorical features using `LabelEncoder`
- Normalized features using `StandardScaler`
- Performed train-test split (80/20)

---

## 🤖 Models Used

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)** - `n_neighbors=5`
3. **Support Vector Machine (SVM)** - `kernel='rbf'`

Each model is evaluated using:
- Accuracy
- Confusion Matrix
- Classification Report

---

## 📈 Visualizations

- 🔹 Customer Churn Distribution
- 🔹 Gender vs Churn
- 🔹 Monthly Charges by Churn Status
- 🔹 Tenure Distribution by Churn Status
- 🔹 Contract Type vs Churn

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/indianized-churn-prediction.git
   cd indianized-churn-prediction
