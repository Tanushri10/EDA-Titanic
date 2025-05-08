import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('titanic dataset/train.csv')
print("Rows and Columns:", df.shape)
print("Column Data Types:\n", df.dtypes)
print("Null values:\n", df.isnull().sum())
print("First few rows:\n", df.head())

#data cleaning
print("Missing values before cleaning:")
print(df.isnull().sum())

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("Duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

print("Missing values after cleaning:")
print(df.isnull().sum())

print("Data cleaned ✅")

# 1. Countplot of the 'Survived' column
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=df)
plt.title('Survived vs Not Survived')
plt.show()

# 2. Age distribution of passengers
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution')
plt.show()

# 3. Sex vs Survived
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set1')
plt.title('Survival Count by Sex')
plt.show()

# 4. Embarked vs Survived
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', hue='Embarked', data=df, palette='Set3')
plt.title('Survival Count by Embarked')
plt.show()

# 5. Correlation heatmap of numerical features
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Heatmap')
plt.show()

# Model Building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Feature selection and target variable
X = df.drop('Survived', axis=1)
X = pd.get_dummies(X, drop_first=True)
print("Feature columns:", X.columns.tolist())

y = df['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")


from sklearn.metrics import classification_report

# Evaluate model performance
print(classification_report(y_test, y_pred))


import joblib
joblib.dump(model, 'model.pkl')

from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)