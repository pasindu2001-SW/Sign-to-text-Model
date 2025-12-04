import numpy as np
import os
import tensorflow as Keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load and preprocess data
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'ayubowan','alright','how are you'])
no_sequences = 60
sequence_length = 60

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(30,1662)))  # no custom activation
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train model
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Save model
model.save('action.h5')

# Evaluate model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))