import numpy as np

b = np.array([[4, 3, 5], [8, 5, 3], [7, 9, 1]])

# 뒤쪽 축부터 인덱싱
print(b.argmax()) # 7
print(b.argmax(axis = 1)) # array([2, 0, 1])
print(b.argmax(axis = 0)) # array([1, 2, 0])
print(b.argmax(axis = -1)) # array([2, 2, 0])
print(b.argmax(axis = -2)) # array([1, 2, 0])