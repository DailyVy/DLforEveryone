import tensorflow as tf
import numpy as np

# 아래는 내가 주로 import 하는방식
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

# train_x = np.array(x_data)
# train_y = np.array(y_data)

model = Sequential()
# layer를 쌓자
# model.add(Dense(1, input_dim=3)) # 참고로 actigvation default 는 linear라고 한다.
# model.add(Activation("linear")) # 이건 생략될 수 있다. linear activation이 default기 때문에
model.add(Dense(1, input_dim=3, activation="linear"))

# optimizer
sgd = SGD(learning_rate=1e-5)
# 어떻게 생겼는지 보자
model.summary()
# 컴파일
# model.compile(loss="mse", optimizer=SGD(learning_rate=1e-5))
model.compile(loss="mse", optimizer=sgd)

# 학습합시다
# history = model.fit(train_x, train_y, epochs=100) # numpy array로 넣어주면 안되는구나...
history = model.fit(x_data, y_data, epochs=100)

# 예측
y_predict = model.predict(np.array([[72., 93., 90.]]))

if __name__ == "__main__":
    print(y_predict)