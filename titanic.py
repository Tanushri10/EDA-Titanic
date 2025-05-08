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
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

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