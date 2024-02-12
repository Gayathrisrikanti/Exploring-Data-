# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:47:14 2023

@author: srika
"""

import pandas as pd
import os
path = "C:/COMP309/TEST2"
filename = 'dataset3.csv'
fullpath = os.path.join(path, filename)
df_gayathri = pd.read_csv(fullpath, sep=',')
print(df_gayathri)
print("\nColumn Names:", df_gayathri.columns.values)
print("DataFrame Shape:", df_gayathri.shape)
print("\nDescriptive Statistics:\n", df_gayathri.describe())
print("\nData Types:\n", df_gayathri.dtypes)
print("\nFirst 5 Records:\n", df_gayathri.head(5))
print("\nFirst 2 Records:\n", df_gayathri.head(2))
print("\nMissing Values Summary:\n", df_gayathri.isnull().sum())

#EXploring data 

import pandas as pd
print("Column Names:", df_gayathri.columns.values)
print("\nColumn Types:\n", df_gayathri.dtypes)
print("\nUnique Values in Each Column:")
for col in df_gayathri.columns:
    print(f"{col}: {df_gayathri[col].unique()}")
print("\nStatistics of Numeric Columns:\n", df_gayathri.describe())
print("\nFirst Four Records:\n", df_gayathri.head(4))
print("\nMissing Values Summary:\n", df_gayathri.isnull().sum())


# Virsualixing & plot


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df_gayathri.hist(figsize=(12, 10), bins=15, layout=(5, 3))
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(df_gayathri.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()


#heatmap & scatterplot 


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.figure(figsize=(10, 8))
sns.heatmap(df_gayathri.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Feature Correlations')
plt.show()
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='BodyFat', data=df_gayathri)
plt.title('Scatterplot of Body Fat vs Age')
plt.xlabel('Age')
plt.ylabel('Body Fat (%)')
plt.show()


#Plot  scatter matrix showing the relationship between all columns 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
scatter_matrix(df_gayathri, alpha=0.8, figsize=(15, 15), diagonal='kde')
plt.tight_layout()
plt.show()

#boxplot & bar chart 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.figure(figsize=(8, 6))
sns.boxplot(y='Weight', data=df_gayathri)
plt.title('Boxplot of Weight')
plt.show()


# bar chart 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
OBESITY_THRESHOLD = 25  # Example threshold
df_gayathri['Obesity_Category'] = df_gayathri['BodyFat'].apply(lambda x: 'Obese' if x >= OBESITY_THRESHOLD else 'Not Obese')
obesity_counts = df_gayathri['Obesity_Category'].value_counts()
plt.figure(figsize=(8, 6))
obesity_counts.plot(kind='bar', color=['red', 'blue'])
plt.title('Count of Obese vs Not Obese Individuals')
plt.xlabel('Obesity Category')
plt.ylabel('Count')
plt.show()


#correlations 
# Replace 'Other_Column1', 'Other_Column2', and ... with the actual column names
selected_columns = ['Density', 'Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']
correlation_matrix = df_gayathri[selected_columns].corr()
print(correlation_matrix)



# DataFrame
import pandas as pd
column_with_most_missing = df_gayathri.isna().sum().idxmax()
df_gayathri.drop(columns=[column_with_most_missing], inplace=True)
mean_age = df_gayathri['Age'].mean()
df_gayathri['Age'].fillna(mean_age, inplace=True)
missing_values = df_gayathri.isna().sum()
print("Missing Values:\n", missing_values)
column_types = df_gayathri.dtypes
if 'Weight' in df_gayathri.columns:
    correlation_age_weight = df_gayathri['Age'].corr(df_gayathri['Weight'])
    print("Correlation between Age and Weight:", correlation_age_weight)
else:
    print("The 'Weight' column was not found in the DataFrame.")
    
    
    
    #model 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
model_name = "dt_Gayathri"  # Replace 'yourfirstname' with your first name
selected_features = ['Density', 'Age',  'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm']
X = df_gayathri[selected_features]
y = df_gayathri['BodyFat']   # Target variable (BodyFat in this case)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual BodyFat")
plt.ylabel("Predicted BodyFat")
plt.title("Actual vs. Predicted BodyFat")
plt.show()
