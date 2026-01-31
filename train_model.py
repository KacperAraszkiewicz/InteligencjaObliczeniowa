import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# ======================
# KONFIGURACJA
# ======================
DATASET_DIR = "asl_alphabet_train/asl_alphabet_train"
IGNORE_CLASSES = {"del", "space"}
FEATURES = 63  # 21 punktów * (x,y,z)

# ======================
# MEDIAPIPE
# ======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

X = []
y = []

print("📥 Wczytywanie danych i ekstrakcja landmarków...")

for label in sorted(os.listdir(DATASET_DIR)):
    if label in IGNORE_CLASSES:
        continue

    path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(path):
        continue

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            continue

        landmarks = result.multi_hand_landmarks[0].landmark

        # normalizacja względem nadgarstka
        wrist = landmarks[0]
        data = []
        for lm in landmarks:
            data.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

        X.append(data)
        y.append(label)

hands.close()

X = np.array(X, dtype="float32")
y = np.array(y)

print(f"✅ Zebrano {len(X)} próbek")

if len(X) == 0:
    raise RuntimeError("❌ Brak danych treningowych")

# ======================
# LABEL ENCODER
# ======================
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ======================
# PODZIAŁ
# ======================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

# ======================
# MODEL MLP
# ======================
model = Sequential([
    Input(shape=(FEATURES,)),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(le.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# TRENING
# ======================
print("🚀 Trenowanie...")
es = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[es]
)

# ======================
# ZAPIS
# ======================
model.save("model_landmarks.h5")
np.save("classes.npy", le.classes_)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model zapisany (landmarki)")
