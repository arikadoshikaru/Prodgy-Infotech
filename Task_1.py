import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# --- Plot 1: Bar Chart for Gender Distribution ---
sns.countplot(x='Sex', data=df, palette='viridis', ax=axes[0])
axes[0].set_title('Distribution of Genders on the Titanic')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Number of Passengers')

# --- Plot 2: Histogram for Age Distribution ---
sns.histplot(data=df, x='Age', bins=20, kde=True, color='skyblue', ax=axes[1])
axes[1].set_title('Distribution of Passenger Ages on the Titanic')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Frequency')

# Adjust layout to prevent titles and labels from overlapping
plt.tight_layout()

plt.show()
