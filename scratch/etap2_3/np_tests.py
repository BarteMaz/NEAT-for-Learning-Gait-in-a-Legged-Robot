import numpy as np
from datetime import datetime


x = [1, 2, 3, 4, 5, 6]
y = [2, 3, 4, 5, 6, 8]

res = np.abs(np.asarray(x) - np.asarray(y))



res2 = (np.asarray([0, 0, 0, 0, 0, 0]) * 2 + res) / 3

print(res2)




