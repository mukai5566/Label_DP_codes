import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import lineplot,scatterplot

Data_2000_1 = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_2\Simu_5_2_EP1.npy'))
lineplot(data=Data_2000_1, x=3, y=1, color='red',label='$(n,\epsilon)=(2,000,1)$', marker= 'o', markersize=5)
Data_2000_2 = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_2\Simu_5_2_EP2.npy'))
lineplot(data=Data_2000_2, x=3, y=1, color='blue',label='$(n,\epsilon)=(2,000,2)$',  marker= 'o', markersize=5)
Data_2000_100 = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_2\Simu_5_2_EP100.npy'))
lineplot(data=Data_2000_100, x=3, y=1, color='black',label='$(n,\epsilon)=(2,000,\infty)$',  marker= 'o', markersize=5)
Data_4000_1 = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_2\Simu_5_2_EP40001.npy'))
lineplot(data=Data_4000_1, x=3, y=1, color='red',label='$(n,\epsilon)=(4,000,1)$',  marker= 'o', markersize=5,linestyle='--')
Data_4000_2 = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_2\Simu_5_2_EP40002.npy'))
lineplot(data=Data_4000_2, x=3, y=1, color='blue',label='$(n,\epsilon)=(4,000,2)$',  marker= 'o', markersize=5,linestyle='--')
Data_4000_100 = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_2\Simu_5_2_EP4000100.npy'))
lineplot(data=Data_4000_100, x=3, y=1, color='black',label='$(n,\epsilon)=(4,000,\infty)$', marker= 'o',markersize=5,linestyle='--')
plt.xlabel('Number of Hidden Units: h')
plt.ylabel('Classification Error: CE($\widetilde{s}_{nn}$)')
plt.ylim(0.15,0.4)
plt.legend()
plt.grid()





lineplot(data=Data_2000_1, x=3, y=0, color='red',label='$(n,\epsilon)=(2,000,1)$', marker= 'o', markersize=5)
lineplot(data=Data_2000_2, x=3, y=0, color='blue',label='$(n,\epsilon)=(2,000,2)$',  marker= 'o', markersize=5)
lineplot(data=Data_2000_100, x=3, y=0, color='black',label='$(n,\epsilon)=(2,000,\infty)$',  marker= 'o', markersize=5)
lineplot(data=Data_4000_1, x=3, y=0, color='red',label='$(n,\epsilon)=(4,000,1)$',  marker= 'o', markersize=5,linestyle='--')
lineplot(data=Data_4000_2, x=3, y=0, color='blue',label='$(n,\epsilon)=(4,000,2)$',  marker= 'o', markersize=5,linestyle='--')
lineplot(data=Data_4000_100, x=3, y=0, color='black',label='$(n,\epsilon)=(4,000,\infty)$', marker= 'o', markersize=5,linestyle='--')
plt.xlabel('Number of Hidden Units: h')
plt.ylabel('Empirical Excess Risk: $\widehat{E}(\widetilde{s}_{nn}$)')
plt.ylim(0.01,0.1)
plt.legend(loc=2)
plt.grid()












