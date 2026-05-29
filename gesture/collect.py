import cv2
import numpy as np
import os
import mediapipe as mp

# ===== MediaPipe Setup =====
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ===== Actions =====
actions = ['hello', 'my', 'name', 'is']

# ===== Parameters =====
num_sequences = 30
sequence_length = 30

DATA_PATH = "MP_Data"

# ===== Create Folders =====
for action in actions:
    for sequence in range(num_sequences):
        os.makedirs(
            os.path.join(DATA_PATH, action, str(sequence)),
            exist_ok=True
        )

# ===== Extract Keypoints =====
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

    return np.concatenate([pose, face, left_hand, right_hand])

# ===== Webcam =====
cap = cv2.VideoCapture(0)

# ===== MediaPipe =====
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for action in actions:

        for sequence in range(num_sequences):

            print(f'Collecting {action} Video {sequence}')

            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                if not ret:
                    continue

                # BGR → RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detection
                results = holistic.process(image)

                # Draw landmarks
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

                # Text
                cv2.putText(
                    frame,
                    f'{action} | Video {sequence} | Frame {frame_num}',
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

                cv2.imshow('Data Collection', frame)

                # Extract keypoints
                keypoints = extract_keypoints(results)

                # Save keypoints
                npy_path = os.path.join(
                    DATA_PATH,
                    action,
                    str(sequence),
                    str(frame_num)
                )

                np.save(npy_path, keypoints)

                # Small delay before start
                if frame_num == 0:
                    cv2.waitKey(1000)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()