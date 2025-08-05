# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading and Preprocessing ---

file_path = 'bank.csv'
data = pd.read_csv(file_path, sep=';')
print("Initial data loaded successfully.")

data.drop('duration', axis=1, inplace=True)

X, y = data.drop('y', axis=1), data['y']

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

le = LabelEncoder()
y_encoded = le.fit_transform(y)

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('scaler', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(X)

# --- Model Training and Evaluation ---

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- Output Results and Visualization ---
print(f"\nAccuracy of the Decision Tree Classifier: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

plt.figure(figsize=(30, 20))
feature_names = preprocessor.get_feature_names_out()
plot_tree(
    dt_classifier,
    filled=True,
    rounded=True,
    class_names=le.classes_,
    feature_names=feature_names,
    max_depth=5
)
plt.title("Decision Tree Visualization (max_depth=5)")
plt.savefig('decision_tree.png', dpi=100)
plt.show()
