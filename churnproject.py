import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load data
df = pd.read_csv("E:/mtech/SEM2/Python/churnproject/indianized_churn.csv")

# Preview the first few rows of the dataset
print(df.head())

df.info()  # Displays column names, types, and non-null values

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values
df.dropna(inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Preprocessing Complete ✅")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(confusion_matrix(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# Confusion Matrix - Logistic Regression
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
plt.figure(figsize=(5,4))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='YlGnBu', cbar=False)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['No Churn', 'Churned'], fontsize=10)
plt.yticks([0.5, 1.5], ['No Churn', 'Churned'], fontsize=10)
plt.tight_layout()
plt.show()

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Confusion Matrix - KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(5,4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Oranges', cbar=False)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['No Churn', 'Churned'], fontsize=10)
plt.yticks([0.5, 1.5], ['No Churn', 'Churned'], fontsize=10)
plt.tight_layout()
plt.show()

# Support Vector Machine (SVM)
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix - SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(5,4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='BuPu', cbar=False)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['No Churn', 'Churned'], fontsize=10)
plt.yticks([0.5, 1.5], ['No Churn', 'Churned'], fontsize=10)
plt.tight_layout()
plt.show()

# 1. Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('Customer Churn Distribution', fontsize=14)
plt.xlabel('Churn Status\n(0 = No Churn, 1 = Churned)', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.xticks([0, 1], ['No Churn', 'Churned'], fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# 2. Gender vs Churn
plt.figure(figsize=(6,4))
sns.countplot(x='gender', hue='Churn', data=df, palette='coolwarm')
plt.title('Gender-wise Churn Distribution', fontsize=14)
plt.xlabel('Gender (0 = Female, 1 = Male)', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.legend(title='Churn Status', labels=['No Churn', 'Churned'])
plt.xticks([0, 1], ['Female', 'Male'], fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# 3. Monthly Charges vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set3')
plt.title('Monthly Charges by Churn Status', fontsize=14)
plt.xlabel('Churn Status\n(0 = No Churn, 1 = Churned)', fontsize=12)
plt.ylabel('Monthly Charges (₹)', fontsize=12)
plt.xticks([0, 1], ['No Churn', 'Churned'], fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# 4. Tenure vs Churn
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, bins=30, palette='husl')
plt.title('Tenure Distribution by Churn Status', fontsize=14)
plt.xlabel('Tenure (Months with Company)', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.legend(title='Churn Status', labels=['No Churn', 'Churned'])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# 5. Contract Type vs Churn
plt.figure(figsize=(8,4))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Paired')
plt.title('Contract Type vs Churn', fontsize=14)
plt.xlabel('Type of Customer Contract', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.xticks(ticks=[0, 1, 2], labels=['Month-to-Month Plan', '1-Year Plan', '2-Year Plan'], fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Churn Status', labels=['No Churn', 'Churned'])
plt.tight_layout()
plt.show()
