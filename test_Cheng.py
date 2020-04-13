import numpy as np

a = np.array([[[1], [3]],[[1], [2]],[[1], [4]]])
b = np.array([[[1,1,1], [1,1,1]], [[1,1,1], [1,1,1]], [[1,1,1], [1,1,1]]])
print("a is \n" + str(a))
print("a shape: " + str(a.shape))
print("b is \n" + str(b))
print("b shape: " + str(b.shape))

print(np.multiply(a, b))