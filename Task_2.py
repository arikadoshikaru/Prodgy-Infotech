import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('titanic.csv')  # Make sure this file is in your working directory
print(df.info())
print("\nMissing values summary:\n", df.isnull().sum())

# Data Cleaning
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("\nCleaned Data Sample:\n", df.head())

# EDA Visualizations
plt.figure(figsize=(12, 6))  

# Survival Count
plt.subplot(2, 3, 1)
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')

# Survival by Gender
plt.subplot(2, 3, 2)
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.xticks([0, 1], ['Male', 'Female'])

# Age Distribution
plt.subplot(2, 3, 3)
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')

# Survival by Passenger Class
plt.subplot(2, 3, 4)
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')

# Survival by Embarked Location
plt.subplot(2, 3, 5)
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title('Survival Rate by Embarked Port')
plt.xticks([0, 1, 2], ['S', 'C', 'Q'])

# Layout adjustment
plt.tight_layout()
plt.show()

# Summary of EDA findings

print("\nEDA Summary:")
print("1. Survival Count:")
print("   - Majority did not survive.")

print("\n2. Survival Rate by Gender:")
print("   - Females had a higher survival rate than males.")

print("\n3. Age Distribution:")
print("   - Most passengers were between 20 and 40 years old.")
print("   - Distribution is slightly right-skewed.")

print("\n4. Survival Rate by Passenger Class:")
print("   - Passengers in 1st class had the highest survival rate.")
print("   - 3rd class passengers had the lowest.")

print("\n5. Survival Rate by Embarked Port:")
print("   - Passengers who embarked from 'C' (Cherbourg) had better survival chances.")
print("   - 'S' (Southampton) had the most passengers but lower survival rate.")