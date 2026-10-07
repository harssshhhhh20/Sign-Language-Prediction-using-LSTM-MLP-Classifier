import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE"]

MODEL_FILE = "landmark_sign_model.pkl"
SAMPLES_PER_LETTER = 150

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_features(hand):

    coords = []

    wrist = hand.landmark[0]

    for lm in hand.landmark:
        coords.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    tips = [4,8,12,16,20]

    palm_x = np.mean([hand.landmark[i].x for i in [0,5,9,13,17]])
    palm_y = np.mean([hand.landmark[i].y for i in [0,5,9,13,17]])

    for tip in tips:
        lm = hand.landmark[tip]
        dist = np.sqrt((lm.x-palm_x)**2 + (lm.y-palm_y)**2)
        coords.append(dist)

    thumb = hand.landmark[4]
    index = hand.landmark[8]

    thumb_index_dist = np.sqrt(
        (thumb.x-index.x)**2 +
        (thumb.y-index.y)**2
    )

    coords.append(thumb_index_dist)

    return np.array(coords)



def collect_data():

    print("\nDATA COLLECTION STARTED")

    X = []
    y = []

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        for idx, letter in enumerate(LETTERS):

            print("Collecting:", letter)

            samples = 0

            while samples < SAMPLES_PER_LETTER:

                ret, frame = cap.read()
                frame = cv2.flip(frame,1)

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:

                    hand = results.multi_hand_landmarks[0]

                    mp_draw.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )

                    features = extract_features(hand)

                    key = cv2.waitKey(1)

                    if key == ord(" "):
                        X.append(features)
                        y.append(idx)
                        samples += 1

                cv2.putText(frame,
                            f"{letter} {samples}/{SAMPLES_PER_LETTER}",
                            (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2)

                cv2.imshow("Collect Data",frame)

            print("Done:", letter)

    cap.release()
    cv2.destroyAllWindows()

    return np.array(X), np.array(y)



def train_model(X,y):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = MLPClassifier(
        hidden_layer_sizes=(256,128),
        max_iter=500,
        random_state=42
    )

    model.fit(X_scaled,y)

    with open(MODEL_FILE,"wb") as f:
        pickle.dump({"model":model,"scaler":scaler},f)

    print("Model saved!")

    return model, scaler


# -------------------------------
# LIVE PREDICTION
# -------------------------------

def live_prediction(model, scaler):

    cap = cv2.VideoCapture(0)

    history = []

    with mp_hands.Hands(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:

            ret, frame = cap.read()
            frame = cv2.flip(frame,1)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            prediction = ""
            confidence = 0

            if results.multi_hand_landmarks:

                hand = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                features = extract_features(hand)

                features = scaler.transform([features])

                pred = model.predict(features)
                proba = model.predict_proba(features)

                confidence = np.max(proba)*100
                letter = LETTERS[pred[0]]

                history.append(letter)

                if len(history) > 10:
                    history.pop(0)

                prediction = Counter(history).most_common(1)[0][0]

            color = (0,255,0) if confidence > 70 else (0,0,255)

            cv2.putText(frame,
                        f"{prediction} {confidence:.1f}%",
                        (20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        color,
                        3)

            cv2.imshow("Sign Recognition",frame)

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# MAIN MENU
# -------------------------------

if __name__ == "__main__":

    print("\nSign Language Recognition")
    print("1 Collect Data + Train + Predict")
    print("2 Predict using saved model")

    choice = input("Choose option: ")

    if choice == "1":

        X,y = collect_data()
        model,scaler = train_model(X,y)
        live_prediction(model,scaler)

    elif choice == "2":

        with open(MODEL_FILE,"rb") as f:
            data = pickle.load(f)

        live_prediction(data["model"],data["scaler"])

    else:
        print("Invalid choice")