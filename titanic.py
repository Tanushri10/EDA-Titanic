import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load the Titanic dataset
df = pd.read_csv('titanic dataset/train.csv')
print("Rows and Columns:", df.shape)
print("Column Data Types:\n", df.dtypes)
print("Null values:\n", df.isnull().sum())
print("First few rows:\n", df.head())

# Data cleaning
print("Missing values before cleaning:")
print(df.isnull().sum())

# Dropping unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Dropping duplicates
print("Duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Check missing values after cleaning
print("Missing values after cleaning:")
print(df.isnull().sum())

# Data visualization
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

# Feature selection and target variable
X = df.drop('Survived', axis=1)

# One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)
print("Feature columns:", X.columns.tolist())

y = df['Survived']

# Scaling the features
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

# Evaluate model performance
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Flask app
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the user
        Pclass = int(request.form['Pclass'])
        Sex = 1 if request.form['Sex'].lower() == 'male' else 0
        Age = float(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])

        embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
        Embarked = embarked_dict[request.form['Embarked']]

        # Create dataframe to apply dummy encoding like in training
        input_df = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]],
                                columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

        # One-hot encode the categorical variables like in training
        input_df = pd.get_dummies(input_df, drop_first=True)

        # Ensure the input matches the model's expected columns
        missing_cols = set(model.feature_names_in_) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0  # Add missing columns with 0s

        input_df = input_df[model.feature_names_in_]  # Reorder columns to match training data

        # Scale the features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        result = 'Survived ✅' if prediction[0] == 1 else 'Did Not Survive ❌'

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
