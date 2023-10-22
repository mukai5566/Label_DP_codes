import numpy as np
import pandas as pd
from seaborn import lineplot,scatterplot
import matplotlib.pyplot as plt

D1_2000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_1_2000.npy'))
D1_4000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_1_4000.npy'))
D1_6000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_1_6000.npy'))
D1_8000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_1_8000.npy'))
D1_10000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_1_10000.npy'))

DF_1 = pd.concat([D1_2000,D1_4000,D1_6000,D1_8000,D1_10000],ignore_index=True)
lineplot(data=DF_1, x=1, y=0, color='blue',label='$\epsilon=1$',  marker= 'o', markersize=5)
plt.xlabel('Size of Training Dataset: n')
plt.ylabel('Testing Error')
plt.legend()
plt.grid()


D2_2000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_2_2000.npy'))
D2_4000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_2_4000.npy'))
D2_6000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_2_6000.npy'))
D2_8000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_2_8000.npy'))
D2_10000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_2_10000.npy'))
DF_2 = pd.concat([D2_2000,D2_4000,D2_6000,D2_8000,D2_10000],ignore_index=True)
lineplot(data=DF_2, x=1, y=0, color='red',label='$\epsilon=2$',  marker= 'o', markersize=5)
plt.xlabel('Size of Training Dataset: n')
plt.ylabel('Testing Error')
plt.legend()
plt.grid()


D100_2000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_100_2000.npy'))
D100_4000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_100_4000.npy'))
D100_6000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_100_6000.npy'))
D100_8000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_100_8000.npy'))
D100_10000 = pd.DataFrame(np.load('/Real_Applications/Res_1/MNIST_100_10000.npy'))
DF_100= pd.concat([D100_2000,D100_4000,D100_6000,D100_8000,D100_10000],ignore_index=True)
lineplot(data=DF_100, x=1, y=0, color='black',label='$\epsilon=\infty$',  marker= 'o', markersize=5)
plt.xlabel('Size of Training Dataset: n')
plt.ylabel('Testing Error')
plt.legend()
plt.grid()
