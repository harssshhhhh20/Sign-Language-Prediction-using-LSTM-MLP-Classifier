import numpy as np
import os
from tensorflow.keras.utils import to_categorical

# ===== Actions =====
actions = ['hello', 'my', 'name', 'is']

# ===== Path =====
DATA_PATH = "MP_Data"

# ===== Label Map =====
label_map = {label:num for num, label in enumerate(actions)}

print(label_map)

# ===== Data Holders =====
sequences = []
labels = []

# ===== Loop Through Actions =====
for action in actions:

    action_path = os.path.join(DATA_PATH, action)

    # Loop Through Sequences
    for sequence in os.listdir(action_path):

        window = []

        # Loop Through Frames
        for frame_num in range(30):

            frame_path = os.path.join(
                action_path,
                sequence,
                f"{frame_num}.npy"
            )

            res = np.load(frame_path)

            window.append(res)

        sequences.append(window)

        labels.append(label_map[action])

# ===== Convert to Arrays =====
X = np.array(sequences)

y = to_categorical(labels).astype(int)

# ===== Save Dataset =====
np.save("X.npy", X)
np.save("y.npy", y)

print("\nDataset Created Successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)