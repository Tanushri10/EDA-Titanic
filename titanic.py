import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('titanic dataset/train.csv')
print("Rows and Columns:", df.shape)
print("Column Data Types:\n", df.dtypes)
print("Null values:\n", df.isnull().sum())
print("First few rows:\n", df.head())