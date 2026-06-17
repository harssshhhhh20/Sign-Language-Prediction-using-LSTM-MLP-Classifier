import cv2
import numpy as np
import mediapipe as mp
import pickle

from collections import Counter
from tensorflow.keras.models import load_model

lstm_model = load_model("action_model.keras")

with open("landmark_sign_model.pkl", "rb") as f:
    mlp_data = pickle.load(f)

mlp_model = mlp_data["model"]
mlp_scaler = mlp_data["scaler"]

actions = np.array(['hello', 'my', 'name', 'is'])
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE"]

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


def extract_mlp_features(hand):

    coords = []

    wrist = hand.landmark[0]

    for lm in hand.landmark:
        coords.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    tips = [4, 8, 12, 16, 20]

    palm_x = np.mean(
        [hand.landmark[i].x for i in [0, 5, 9, 13, 17]]
    )

    palm_y = np.mean(
        [hand.landmark[i].y for i in [0, 5, 9, 13, 17]]
    )

    for tip in tips:

        lm = hand.landmark[tip]

        dist = np.sqrt(
            (lm.x - palm_x) ** 2 +
            (lm.y - palm_y) ** 2
        )

        coords.append(dist)

    thumb = hand.landmark[4]
    index = hand.landmark[8]

    thumb_index_dist = np.sqrt(
        (thumb.x - index.x) ** 2 +
        (thumb.y - index.y) ** 2
    )

    coords.append(thumb_index_dist)

    return np.array(coords)


sequence = []
predictions = []
sentence = []

threshold = 0.7

mode = "LSTM"

current_prediction = ""
current_confidence = 0

last_letter = ""
stable_count = 0

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

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

        if mode == "MLP":

            if results.right_hand_landmarks:

                features = extract_mlp_features(
                    results.right_hand_landmarks
                )

                features = mlp_scaler.transform([features])

                pred = mlp_model.predict(features)[0]

                proba = mlp_model.predict_proba(features)[0]

                current_prediction = LETTERS[pred]
                current_confidence = np.max(proba) * 100

                if current_confidence > 70:

                    if current_prediction == last_letter:
                        stable_count += 1
                    else:
                        stable_count = 0
                        last_letter = current_prediction

                    if stable_count >= 15:

                        if current_prediction == "SPACE":

                            if (
                                len(sentence) == 0 or
                                sentence[-1] != " "
                            ):
                                sentence.append(" ")

                        else:

                            sentence.append(current_prediction)

                        stable_count = 0

            else:

                current_prediction = "No Hand"
                current_confidence = 0
                stable_count = 0

        else:

            keypoints = extract_keypoints(results)

            sequence.append(keypoints)

            sequence = sequence[-30:]

            if len(sequence) == 30:

                input_data = np.expand_dims(
                    sequence,
                    axis=0
                )

                prediction = lstm_model.predict(
                    input_data,
                    verbose=0
                )[0]

                predicted_class = np.argmax(prediction)

                confidence = prediction[predicted_class]

                predicted_action = actions[predicted_class]

                current_prediction = predicted_action
                current_confidence = confidence * 100

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

                    sentence = sentence[-50:]

        panel_width = 450

        height, width, _ = frame.shape

        panel = np.zeros(
            (height, panel_width, 3),
            dtype=np.uint8
        )

        cv2.putText(
            panel,
            f"Mode: {mode}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            panel,
            "Prediction",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            panel,
            current_prediction,
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            panel,
            f"Confidence: {current_confidence:.1f}%",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            panel,
            "Sentence",
            (20, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        display_text = ''.join(sentence)

        cv2.putText(
            panel,
            display_text,
            (20, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            panel,
            "M = MLP",
            (20, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            panel,
            "L = LSTM",
            (20, 540),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            panel,
            "SPACE = Clear",
            (20, 580),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        combined = np.hstack((frame, panel))

        cv2.imshow(
            "Sign Language Recognition",
            combined
        )

        key = cv2.waitKey(10) & 0xFF

        if key == ord('m'):
            mode = "MLP"

        elif key == ord('l'):
            mode = "LSTM"

        elif key == ord('q'):
            break

        elif key == 32:
            sentence = []
            predictions = []
            stable_count = 0
            last_letter = ""

cap.release()
cv2.destroyAllWindows()