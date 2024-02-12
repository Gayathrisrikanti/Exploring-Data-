# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:56:01 2023

@author: srika
"""

import pandas as pd
url = "https://e.centennialcollege.ca/content/enforced/1010633-COMP309001_2023F/customer_markham.txt?_&d2lSessionVal=aCAP2iEM0ncjZU10QJFUZ2ERa"
df2_gayathri = pd.read_csv(url, sep='\t')
print("Column Names:")
print(df2_gayathri.columns)
print("\nColumn Types:")
print(df2_gayathri.dtypes)
print("\nStatistics:")
print(df2_gayathri.describe())
print("\nFirst Three Records:")
print(df2_gayathri.head(3))
print("\nSummary of Missing Values:")
print(df2_gayathri.isnull().sum())

import pandas as pd

# URL of the data file
url = "https://e.centennialcollege.ca/content/enforced/1010633-COMP309001_2023F/customer_markham.txt?_&d2lSessionVal=aCAP2iEM0ncjZU10QJFUZ2ERa"

# Read the data from the URL
df = pd.read_csv(url, delimiter='\t')

# Save the data as a CSV file
df.to_csv("df2_gayathri_standardized.csv", index=False)

print("Data has been converted and saved as df2_gayathri_standardized.csv")

# prociding the data
import pandas as pd
from sklearn.preprocessing import StandardScaler
df2_gayathri_numeric = pd.get_dummies(df2_gayathri, drop_first=True)
missing_values = df2_gayathri_numeric.isnull().sum().sum()
if missing_values == 0:
    print("There are no missing values.")
else:
    print(f"There are {missing_values} missing values in the data.")
scaler = StandardScaler()
df2_gayathri_numeric_standardized = scaler.fit_transform(df2_gayathri_numeric)
df2_gayathri_numeric_standardized = pd.DataFrame(df2_gayathri_numeric_standardized, columns=df2_gayathri_numeric.columns)
numeric_stats = df2_gayathri_numeric_standardized.describe()
print(numeric_stats)
df2_gayathri_numeric_standardized.to_csv("df2_gayathri_standardized.csv", index=False)




#model 

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df2_gayathri_standardized = pd.read_csv("df2_gayathri_standardized.csv")
df2_gayathri_standardized.dropna(inplace=True)
kmeans = KMeans(n_clusters=6, random_state=42)
df2_gayathri_standardized['cluster_gayathri'] = kmeans.fit_predict(df2_gayathri_standardized)
cluster_counts = df2_gayathri_standardized['cluster_gayathri'].value_counts().sort_index()
print("Number of observations in each cluster:")
print(cluster_counts)
cluster_gayathri = "Gayathri_Srikanti"
df2_gayathri_standardized['cluster_' + cluster_gayathri] = df2_gayathri_standardized['cluster_gayathri']
plt.hist(df2_gayathri_standardized['cluster_' + cluster_gayathri], bins=6, edgecolor='k')
plt.xlabel('Cluster Number')
plt.ylabel('Number of Observations')
plt.title(f'{cluster_gayathri.capitalize()} Clusters')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
cluster_counts = df2_gayathri_standardized['cluster_' + cluster_gayathri].value_counts()
print(cluster_counts)
