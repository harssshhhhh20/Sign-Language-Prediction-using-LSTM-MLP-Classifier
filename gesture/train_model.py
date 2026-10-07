import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split

X = np.load("X.npy")
y = np.load("y.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

log_dir = "Logs"
tb_callback = TensorBoard(log_dir=log_dir)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=False,
        activation='relu',
        input_shape=(30, 1662)
    )
)

model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(y.shape[1], activation='softmax'))

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)


model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[tb_callback, early_stop],
    verbose=1
)


model.save("action_model.keras")

print("\nModel Saved Successfully!")


results = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", results[1])