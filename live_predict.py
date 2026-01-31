import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# ======================
# KONFIGURACJA
# ======================
CONFIDENCE_THRESHOLD = 0.8
BUFFER_SIZE = 10

# ======================
# MODEL
# ======================
model = load_model("model_landmarks.h5")
classes = np.load("classes.npy")

# ======================
# MEDIAPIPE
# ======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
pred_buffer = deque(maxlen=BUFFER_SIZE)

print("✋ Naciśnij Q aby wyjść")

# ======================
# LOOP
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark

        wrist = landmarks[0]
        data = []

        for lm in landmarks:
            data.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

        X = np.array(data, dtype="float32").reshape(1, -1)

        pred = model.predict(X, verbose=0)[0]
        idx = np.argmax(pred)
        confidence = pred[idx]

        if confidence >= CONFIDENCE_THRESHOLD:
            pred_buffer.append(idx)
        else:
            pred_buffer.append(None)

        valid = [i for i in pred_buffer if i is not None]
        if valid:
            final_idx = max(set(valid), key=valid.count)
            letter = classes[final_idx]
            conf = pred[final_idx] * 100

            cv2.putText(
                frame,
                f"{letter} ({conf:.1f}%)",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                4
            )

    cv2.imshow("Tłumacz Migowy (LANDMARKS)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
