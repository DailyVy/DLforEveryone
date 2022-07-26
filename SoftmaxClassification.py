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
print(a, K.eval(tf.argmax(a, axis = 1))) # 1
print(a, tf.argmax(a, axis = 1)) # tf.Tensor([1], shape=(1,), dtype=int64)

print("=====================================")
b = model.predict(np.array([[1, 3, 4, 3]]))
print(b, K.eval(tf.argmax(b, axis = 1))) # 0

print("=====================================")
c = model.predict(np.array([[1, 1, 0, 1]]))
c_onehot = np.argmax(c, axis=-1) # 얜 왜 -1인가
print(c, c_onehot) # 2

print("=====================================")
all = model.predict(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]))
all_onehot = np.argmax(all, axis=-1) # 얜 왜 -1인가
print(all, all_onehot) # [1 0 2] 앞의 결과와 같다.
"""
np.argmax

1. axis = None, 즉 axis를 지정하지 않은 경우 
모든 원소를 순서대로 1차원 array에 편 상태를 가정하고 argmax를 적용한 결과를 반환
2. axis = 1, 각 가로축 원소들끼리 비교 => 한 행을 기준으로 비교
3. aixs = 0, 세로축 원소들끼리 비교 => 한 열을 기준으로 비교

# 2차원 array 예시
b = np.array([[4, 3, 2], [8, 5, 9], [7, 6, 1]])
np.argmax(b) # 5
np.argmax(b, axis = 1) # array([0, 2, 0])
np.argmax(b, axis = 0) # array([1, 2, 1])

4. axis = -1 뒤쪽 축부터 인덱싱 => 1번축
5. axis = -2 => 0번축
  
"""