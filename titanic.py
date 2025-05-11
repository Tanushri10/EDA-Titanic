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
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop_duplicates(inplace=True)

# Visualization (optional - good for notebook demo)
# sns.countplot(x='Survived', data=df)
# plt.show()

# Feature engineering
X = df.drop('Survived', axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['Survived']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

from flask import Flask, request, render_template
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user input
        Pclass = int(request.form['Pclass'])
        Sex = 1 if request.form['Sex'].lower() == 'male' else 0
        Age = float(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])
        Embarked = {'C': 0, 'Q': 1, 'S': 2}[request.form['Embarked']]

        # Build input DataFrame
        input_df = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]],
                                columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

        # One-hot encoding and column alignment
        input_df = pd.get_dummies(input_df, drop_first=True)
        for col in model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model.feature_names_in_]

        # Scale input and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        result = 'Survived ✅' if prediction[0] == 1 else 'Did Not Survive ❌'

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
