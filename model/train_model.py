import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- 1. LOAD AND PREPARE DATA ---
print("Loading dataset...")
# Load the dataset from the CSV file created by data_collection.py
df = pd.read_csv('gesture_dataset/dataset.csv')

# Separate the features (the 42 landmark coordinates) and the labels (the gesture names)
X = df.drop('label', axis=1) # All columns except for the 'label' column
y = df['label'] # Only the 'label' column

# Split the data into training (80%) and testing (20%) sets.
# 'stratify=y' ensures that the proportion of each gesture is the same in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data preparation complete.")

# --- 2. CHOOSE AND TRAIN THE MODEL ---
print("Training the Random Forest model...")
# We use a RandomForestClassifier. 'n_estimators' is the number of "trees" in the forest.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)
print("Model training complete.")

# --- 3. EVALUATE THE MODEL ---
print("Evaluating the model...")
# Use the trained model to make predictions on the unseen test data
y_pred = model.predict(X_test)

# Calculate the overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print a detailed report showing performance for each gesture
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 4. SAVE THE TRAINED MODEL ---
# This saves your trained model so your main application can use it.
model_filename = 'gesture_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"\nModel saved successfully as {model_filename}")
