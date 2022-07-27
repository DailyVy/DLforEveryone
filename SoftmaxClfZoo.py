from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np

src = "./dataset/data-04-zoo.csv"
data = np.loadtxt(src, delimiter=",", dtype=np.float32)
# 101 row, 17 column

x_data = data[:, :-1]
y_data = data[:, -1:]

# print(data, len(data),len(data[0]))
# print(x_data)
# print(y_data)
# print(max(y_data), min(y_data)) # class가 7개 0~6
# print(x_data.shape, y_data.shape) # (101, 16) (101, 1)

## convert y_data to one_hot
nb_classes = 7 # 0 ~ 6
y_data_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)

# print("one_hot : ", y_data_one_hot)
"""
다음에는 컬럼 14번 (legs)를 정규화시켜서 해보자
"""

model = Sequential()
model.add(Dense(units=nb_classes, input_dim=16, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=0.1), metrics=["accuracy"])
model.summary()

history = model.fit(x_data, y_data_one_hot, epochs=1000)

if __name__ == "__main__":
    # Single Dataset
    test_data = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]]) # expected prediction == 3(feathers)
    print(model.predict(test_data), np.argmax(model.predict(test_data), axis=-1))

    # Full x_data test
    pred = np.argmax(model.predict(x_data), axis=-1)
    for p, y in zip(pred, y_data.flatten()):
        print(f"[{p == int(y)}] Prediction: {p}, True Y : {int(y)}")