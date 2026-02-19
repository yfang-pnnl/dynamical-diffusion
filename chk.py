import numpy as np
a = np.array([1,2]).reshape(2,1)
print(a)
b = np.array([3,4]).reshape(2,1)
print(b)
print(a.shape)
c = np.stack([a,b],axis=2)
print(c.shape)
print(c)
