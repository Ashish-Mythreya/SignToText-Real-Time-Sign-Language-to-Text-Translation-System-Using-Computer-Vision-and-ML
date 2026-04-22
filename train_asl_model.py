import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Change this line at the top of train_asl_model.py
DATA_PATH = "asl_dataset/asl_dataset_augmented.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop('label', axis=1)  
y = df['label']  

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data - using a stratify flag ensures each letter is equally represented in test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Robust Model configuration
model = RandomForestClassifier(
    n_estimators=500,     # Increased from 100 to 500 for better fine-detail capture
    max_depth=None,       # Allow trees to grow until nodes are pure
    min_samples_split=2, 
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"New Model accuracy: {accuracy*100:.2f}%")

# Save the model
with open('asl_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to asl_model.pkl")