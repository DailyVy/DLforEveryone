# data-01-test-score.csv 를 가져와보자
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

src = "./dataset/data-01-test-score.csv"
# file경로, ","로 seperate하겠다, 이 파일의 데이터 타입은 float32(==> 이럴 경우 데이터 타입이 같아야겠지)
xy = np.loadtxt(src, delimiter=",", dtype=np.float32)

x_data = xy[:, :-1] # (25, 3)
y_data = xy[:, [-1]] # [[]] (25, 1)
# y_data = xy[:, -1] # 이렇게 하면 2차원 배열 형태가 아님 [] # (25, ) 이렇게 됨

# 이제 이걸로 학습 해보자
model = Sequential()
model.add(Dense(1, input_dim=3, activation="linear"))
# model.add(Activation('linear'))
model.summary()

model.compile(loss='mse', optimizer=SGD(learning_rate=1e-5))
history = model.fit(x_data, y_data, epochs=2000)

"""
Queue Runners
파일이 커서 메모리에 한번에 올리기 어려운 경우(numpy로 불러오기 어려운 경우)
tensorflow에서 Queue Runners라는 시스템을 제공
여러 개의 file을 Queue에 쌓아서 Reader로 연결하여 데이터를 읽어와 decoding을 한 다음에 또 Queue에 쌓음

==> TensorFlow 2부터 사라진듯
"""

if __name__ == "__main__":
    # print(x_data, "\nx_data shape: ", x_data.shape, len(x_data)) # ... (25, 3) 25
    # print(y_data, "\ny_data shape: ", y_data.shape) # ... (25, 1)

    print("Your score will be ", model.predict([[100, 70, 101]]))
    print("Other score will be ", model.predict([[60, 70, 110], [90, 100, 80]]))
