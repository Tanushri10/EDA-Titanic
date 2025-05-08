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

from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)

# Load the Titanic dataset and train the model
titanic_data = pd.read_csv('titanic dataset/train.csv')

# Preprocessing: Handle categorical features
label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'])
titanic_data['Embarked'] = label_encoder.fit_transform(titanic_data['Embarked'].fillna('S'))

# Select features and target
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = titanic_data['Survived']

# Train RandomForest model
model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        Pclass = int(request.form['Pclass'])
        Sex = request.form['Sex']
        Age = float(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])
        Embarked = request.form['Embarked']

        # Preprocess the input data
        Sex = label_encoder.transform([Sex])[0]
        Embarked = label_encoder.transform([Embarked])[0]

        # Make the prediction
        prediction = model.predict([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

        # Output the result
        if prediction == 1:
            result = "Survived"
        else:
            result = "Did not survive"
        
        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
