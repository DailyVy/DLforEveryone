from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import tensorflow as tf
"""
backend가 뭐지?
backend를 사용하면 variable을 만들거나 연산을 할 수 있다고 한다.
eval 은 변수 값을 평가
"""
import numpy as np

x_raw = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_raw = [[0, 0, 1], # 2
          [0, 0, 1], # 2
          [0, 0, 1], # 2
          [0, 1, 0], # 1
          [0, 1, 0], # 1
          [0, 1, 0], # 1
          [1, 0, 0], # 0
          [1, 0, 0]] # 0

x_data = np.array(x_raw, dtype=np.float32)
y_data = np.array(y_raw, dtype=np.float32) # y는 이미 one_hot_encoding이 되어있네

nb_classes = 3 # 클래스 갯수는 세 개

print(x_data)
print(y_data)

model = Sequential()
model.add(Dense(units=nb_classes, input_dim=4, use_bias=True, activation="softmax")) # use_bias is True, by default
model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=0.1), metrics=["accuracy"])
model.summary() # param 이 15개 인데 feature가 4개이고 bias까지 합치면 input은 5개, 클래스는 3개 이므로 5*3=15

history = model.fit(x_data, y_data, epochs=2000)

# Testing & One-hot encoding
print("=====================================")
a = model.predict(np.array([[1, 11, 7, 9]]))
print(a, K.eval(tf.argmax(a, axis = 1)))

print("=====================================")
b = model.predict(np.array([[1, 3, 4, 3]]))
print(b, K.eval(tf.argmax(b, axis = 1)))

print("=====================================")
c = model.predict(np.array([[1, 1, 0, 1]]))
c_onehot = np.argmax(c, axis=-1) # 얜 왜 -1인가
print(c, c_onehot)

print("=====================================")
all = model.predict(np.array([[1, 11, 7, 9]]))
all_onehot = np.argmax(all, axis=-1) # 얜 왜 -1인가
print(all, all_onehot)
