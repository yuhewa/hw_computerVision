import numpy as np
import matplotlib.pyplot as plt

x = np.arange(8)
y = np.random.randint(low=1,high=20,size=8)
m = np.arange(8)
n = np.random.rand(8)

plt.plot(x,y,'--r') # r代表紅色、g代表綠色、b代表藍色 ， --代表虛線、s代表方塊、^代表三角形 順序沒差
# plt.plot(m,n,'--b')
# plt.bar(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()