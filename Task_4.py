import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading and Preparation ---

DATASET_FILE = 'twitter_training.csv'
COLUMNS = ['tweet_id', 'entity', 'sentiment', 'tweet_text']

df = pd.read_csv(DATASET_FILE, names=COLUMNS)
print("Dataset loaded successfully. First 5 rows:")
print(df.head())
print("\nDataset Info:")
df.info()

# --- Data Cleaning ---

df.dropna(subset=['sentiment', 'tweet_text'], inplace=True)
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print(f"Shape after cleaning: {df.shape}")

# --- Sentiment Visualization ---

sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1]})

# Updated sns.countplot to remove the FutureWarning
sns.countplot(x='sentiment', hue='sentiment', data=df, palette='viridis', order=df['sentiment'].value_counts().index, ax=axes[0], legend=False)
axes[0].set_title('Distribution of Sentiments (Bar Chart)', fontsize=10)
axes[0].set_xlabel('Sentiment', fontsize=10)
axes[0].set_ylabel('Number of Tweets', fontsize=10)
axes[0].tick_params(axis='x', rotation=0)

sentiment_counts = df['sentiment'].value_counts()
labels = sentiment_counts.index
sizes = sentiment_counts.values
colors = sns.color_palette('pastel')[0:len(labels)]

axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
axes[1].set_title('Proportion of Sentiments (Pie Chart)', fontsize=10)
axes[1].axis('equal')

plt.tight_layout()
plt.show()
