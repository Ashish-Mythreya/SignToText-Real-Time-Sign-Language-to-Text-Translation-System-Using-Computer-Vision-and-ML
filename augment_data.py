import pandas as pd
import numpy as np
import os

# Load your original collected data
DATA_PATH = "asl_dataset/asl_dataset_a_z.csv"
df = pd.read_csv(DATA_PATH)

def augment_landmarks(landmarks, noise_level=0.01):
    """Adds tiny random shifts to landmarks to simulate shaky hands"""
    noise = np.random.normal(0, noise_level, landmarks.shape)
    return landmarks + noise

print(f"Original dataset size: {len(df)}")

augmented_data = []

# For every sample you recorded, create 10 'noisy' variations
for index, row in df.iterrows():
    label = row['label']
    features = row.drop('label').values
    
    # Keep the original
    augmented_data.append(row.values)
    
    # Create 10 variations
    for _ in range(10):
        new_features = augment_landmarks(features)
        augmented_data.append(np.append(new_features, label))

# Save the new, much larger dataset
columns = df.columns
new_df = pd.DataFrame(augmented_data, columns=columns)
new_df.to_csv("asl_dataset/asl_dataset_augmented.csv", index=False)

print(f"New augmented dataset size: {len(new_df)}")
print("Done! Now update your training script to use 'asl_dataset_augmented.csv'.")