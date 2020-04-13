import numpy as np

a = np.array([[[1], [3]],[[1], [2]],[[1], [4]]])
b = np.array([[[1,1,1,1,1,1], [1,1,1,1,1,1]], [[1,1,1,1,1,1], [1,1,1,1,1,1]], [[1,1,1,1,1,1], [1,1,1,1,1,1]]])
print("a is \n" + str(a))
print("a shape: " + str(a.shape))
print("b is \n" + str(b))
print("b shape: " + str(b.shape))

print(np.multiply(a, b))
print("Sum is")
c = np.sum(np.multiply(a, b),axis=1)
print(c)
print(c.shape)

c = c.reshape((-1, 18))
print(c)

b = np.eye(3, 3)
p1, p2, p3 = b
print(p1)
print(p2)
print(p3)