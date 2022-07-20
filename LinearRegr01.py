import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# w, x는 각각 -1, 1 이면 되겠다

# tf.model = tf.keras.Sequential()
# # units == output shape, input_dim == input shape
# tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))
#
# sgd = tf.keras.optimizers.SGD(lr=0.1)  # SGD == standard gradient descendent, lr == learning rate
# tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig (y'-y)^2
#
# # prints summary of the model to the terminal
# tf.model.summary()
#
# # fit() executes training
# tf.model.fit(x_train, y_train, epochs=200)
#
# # predict() returns predicted value
# y_predict = tf.model.predict(np.array([5, 4]))
# print(y_predict) # [[-3.9975138] [-2.998721 ]] : -4, -3 과 아주 근사




############## 아래코드는 내가 다시 작성한 것 ####################

model = Sequential()
model.add(Dense(units=1, input_dim=1)) # units=1 대신 그냥 1(output space)
# sgd = SGD(learning_rate=0.1) # 이렇게 learning rate를 줄수도 있구나,
sgd = SGD(lr=0.1) # lr로도 가능

model.summary()
model.compile(loss="mse", optimizer=sgd)

model.fit(x_train, y_train, epochs=200)

# predict
y_predict = model.predict(np.array([5, 4]))
print(y_predict)

"""
model.compile(loss="mse", optimizer="sgd") 했을 때 보다
sgd 를 따로 SGD(learning_rate=0.1)로 설정하고
model.compile(loss="mse", optimizer=sgd) 로 넣어줬을 때 더 loss값이 적고
predict 값이 실제값과 더 근사하다. 

다음에 tensorflow 사용해서 학습할 때 이렇게 해도 좋을듯..? 다만 adam을 주로 쓰긴하지만..
"""