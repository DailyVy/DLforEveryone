import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

# sgd = tf.keras.optimizers.SGD(lr=0.1) # lr대신에 learning_rate를 쓰라고 합니당
sgd = tf.keras.optimizers.SGD(learning_rate=0.1) # lr대신에 learning_rate를 쓰라고 합니당
tf.model.compile(loss="mse", optimizer=sgd)

tf.model.summary()

# fit() trains the model and returns history of train
history = tf.model.fit(x_train, y_train, epochs=100)

y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)

# plot training & validation loss values
plt.plot(history.history["loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()


############## 아래코드는 내가 다시 작성한 것 ####################

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import SGD

# model = Sequential()
# model.add(Dense(1, input_dim=1))
#
# sgd = SGD(learning_rate=0.1)
#
# model.compile(loss="mse", optimizer=sgd)
#
# model.summary()
#
# history = model.fit(x_train, y_train, epochs=100)
#
# y_predict= model.predict(np.array([5, 4]))
# print(y_predict)
#
# plt.plot(history.history["loss"])
# plt.title("Model Loss")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(["Train", "Test"], loc="upper left") # qna. 이거 Test는 없는데 왜 해주는거람?
# plt.show()