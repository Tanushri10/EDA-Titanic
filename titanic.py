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