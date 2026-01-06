import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# --- 1. Load the dataset ---
df = pd.read_csv('parkinsons.csv')

# --- 2. Select features ---
# The test expects exactly 2 features. 
# Based on the paper, PPE and DFA are the best performing pair.
features = ['PPE', 'DFA']
target = 'status'

X = df[features]
y = df[target]

# --- 3. Scale the data ---
# The test data used by the autograder is pre-scaled to [0, 1].
# We must scale our training data to match this range.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Split the data ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- 5. Choose a model ---
# The paper recommends SVM with RBF kernel for this specific data.
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# --- 6. Test the accuracy ---
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")

# --- 7. Save the model ---
# The config.yaml path must match this filename exactly.
model_filename = 'parkinsons_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")
