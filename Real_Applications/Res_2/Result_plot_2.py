import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import lineplot,scatterplot

Data_nlogn_2000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_nlogn_2_2000.npy'))
Data_nlogn_4000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_nlogn_2_4000.npy'))
Data_nlogn_6000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_nlogn_2_6000.npy'))
Data_nlogn_8000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_nlogn_2_8000.npy'))
Data_nlogn_10000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_nlogn_2_10000.npy'))
DF_nlogn = pd.concat([Data_nlogn_2000,Data_nlogn_4000,Data_nlogn_6000,Data_nlogn_8000,Data_nlogn_10000],ignore_index=True)
lineplot(data=DF_nlogn, x=1, y=0, color='red',label='$\epsilon \\asymp \\frac{\log(n)}{\sqrt{n}}$',  marker= 'o', markersize=5)

Data_n_2000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_nlogn_2_2000.npy'))
Data_n_4000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_n_2_4000.npy'))
Data_n_6000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_n_2_6000.npy'))
Data_n_8000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_n_2_8000.npy'))
Data_n_10000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_n_2_10000.npy'))
DF_n = pd.concat([Data_n_2000,Data_n_4000,Data_n_6000,Data_n_8000,Data_n_10000],ignore_index=True)
lineplot(data=DF_n, x=1, y=0, color='black',label='$\epsilon\\asymp \\frac{1}{\sqrt{n}}$',  marker= 'o', markersize=5)



Data_nlognToget_2000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_nlogn_2_2000.npy'))
Data_nlognToget_4000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_nlognToget_2_4000.npy'))
Data_nlognToget_6000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_nlognToget_2_6000.npy'))
Data_nlognToget_8000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_nlognToget_2_8000.npy'))
Data_nlognToget_10000 = pd.DataFrame(np.load('/Real_Applications/Res_2/MNIST_nlognToget_2_10000.npy'))
DF_nlognToget = pd.concat([Data_nlognToget_2000,Data_nlognToget_4000,Data_nlognToget_6000,Data_nlognToget_8000,Data_nlognToget_10000],ignore_index=True)
lineplot(data=DF_nlognToget, x=1, y=0, color='blue',label='$\epsilon\\asymp \\frac{1}{\log(n)\sqrt{n}}$',  marker= 'o', markersize=5)


plt.xlabel('Size of Training Dataset: n')
plt.ylabel('Testing Error')

plt.legend()
plt.ylim(0.04,0.15)
plt.grid()


