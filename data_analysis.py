# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

except Exception as e:
    print(f"Error loading data: {e}")

# Task 2: Basic Data Analysis
print("\nDescriptive Statistics:")
print(df.describe())

# Group by species and get mean of features
print("\nMean of features grouped by species:")
grouped = df.groupby("species").mean()
print(grouped)

# Task 3: Data Visualization

# 1. Line chart - simulate a trend (not real time-series)
plt.figure(figsize=(8, 4))
df_sorted = df.sort_values(by='sepal length (cm)')
plt.plot(df_sorted['sepal length (cm)'].values, label='Sepal Length')
plt.title('Trend of Sepal Length (simulated)')
plt.xlabel('Index (sorted by length)')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar chart - average petal length per species
plt.figure(figsize=(6, 4))
grouped['petal length (cm)'].plot(kind='bar', color='skyblue')
plt.title('Average Petal Length by Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 3. Histogram - distribution of sepal width
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width (cm)'], bins=15, color='lightgreen', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 4. Scatter plot - sepal length vs petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='Set2')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Findings or Observations
print("\nObservations:")
print("- Setosa has noticeably smaller petal lengths compared to Versicolor and Virginica.")
print("- Petal length and sepal length show a positive correlation.")
print("- Sepal width has a roughly normal distribution.")
