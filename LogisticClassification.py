from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD


x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

model = Sequential()
model.add(Dense(units=1, input_dim=2, activation="sigmoid"))
# 왜 TensorFlow가 pytorch에 비해 추상적이라고 하는지 알겠다..

"""
better result with loss function == "binary_crossentropy", try "mse" for yourself
adding accuracy metric to get accuracy report during training
"""
model.compile(loss="binary_crossentropy", # 여태 loss함수를 mse로 했음 근데 sigmoid 때문에 non-convex한 형태로 나와..
              optimizer=SGD(learning_rate=0.01),
              metrics=["accuracy"])
model.summary()

history = model.fit(x_data, y_data, epochs=4000, batch_size=200)

if __name__ == "__main__":
    print(f"Accuracy: {history.history['accuracy'][-1]}")
