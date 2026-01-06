import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# --- 1. Load the dataset ---
# We assume the file is downloaded to the same directory
df = pd.read_csv('parkinsons.csv')

# --- 2. Select features ---
# Based on the provided paper (Table 3), the combination of PPE and DFA 
# yields high accuracy (approx 88.2%).
# Input features:
selected_features = ['PPE', 'DFA']
# Output feature (Target):
target = 'status'

X = df[selected_features]
y = df[target]

# --- 3. Scale the data ---
# Initialize the scaler
scaler = MinMaxScaler()
# Fit and transform the input features to be between 0 and 1
X_scaled = scaler.fit_transform(X)

# --- 4. Split the data ---
# Split into training (80%) and validation/test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- 5. Choose a model ---
# The paper suggests using a Kernel Support Vector Machine (SVM)
# with a Gaussian radial basis function (rbf) kernel.
model = SVC(kernel='rbf', random_state=42)

# Train the model
model.fit(X_train, y_train)

# --- 6. Test the accuracy ---
# Generate predictions on the test set
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.4f}")

# Check if we met the requirement of > 0.8
if accuracy >= 0.8:
    print("Success! Accuracy is above 0.8.")
else:
    print("Warning: Accuracy is below 0.8. Check feature selection.")

# --- 7. Save the model ---
# Save the trained model to a file
model_filename = 'parkinsons_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")
