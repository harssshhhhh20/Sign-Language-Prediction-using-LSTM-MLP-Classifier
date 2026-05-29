import cv2
import numpy as np
import mediapipe as mp

from collections import Counter
from tensorflow.keras.models import load_model

model = load_model("action_model.keras")

actions = np.array(['hello', 'my', 'name', 'is'])

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):

    pose = np.array(
        [[res.x, res.y, res.z, res.visibility]
         for res in results.pose_landmarks.landmark]
    ).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    face = np.array(
        [[res.x, res.y, res.z]
         for res in results.face_landmarks.landmark]
    ).flatten() if results.face_landmarks else np.zeros(468 * 3)

    left_hand = np.array(
        [[res.x, res.y, res.z]
         for res in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    right_hand = np.array(
        [[res.x, res.y, res.z]
         for res in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([
        pose,
        face,
        left_hand,
        right_hand
    ])

sequence = []
predictions = []
sentence = []

threshold = 0.7

display_counter = 0

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            continue

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(image)

        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS
        )

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

        keypoints = extract_keypoints(results)

        sequence.append(keypoints)

        sequence = sequence[-30:]

        display_counter += 1

        if display_counter > 30:
            display_counter = 1

        if len(sequence) == 30:

            input_data = np.expand_dims(
                sequence,
                axis=0
            )

            prediction = model.predict(
                input_data,
                verbose=0
            )[0]

            predicted_class = np.argmax(prediction)

            confidence = prediction[predicted_class]

            predicted_action = actions[predicted_class]

            if confidence > threshold:

                predictions.append(predicted_action)

                predictions = predictions[-10:]

                most_common = Counter(
                    predictions
                ).most_common(1)[0][0]

                if len(sentence) == 0:

                    sentence.append(most_common)

                elif most_common != sentence[-1]:

                    sentence.append(most_common)

                sentence = sentence[-5:]

        panel_width = 400

        height, width, _ = frame.shape

        panel = np.zeros(
            (height, panel_width, 3),
            dtype=np.uint8
        )

        cv2.putText(
            panel,
            "Real-Time Sign Recognition",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            panel,
            f"Prediction:",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.putText(
            panel,
            ' '.join(sentence),
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            panel,
            f"Frames: {display_counter}/30",
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.putText(
            panel,
            "Model: LSTM",
            (20, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.putText(
            panel,
            "MediaPipe + TensorFlow",
            (20, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        combined = np.hstack((frame, panel))

        cv2.imshow(
            "Continuous Sign Prediction",
            combined
        )

        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break

        if key == 32:
            sentence = []

cap.release()
cv2.destroyAllWindows()