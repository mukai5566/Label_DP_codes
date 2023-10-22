
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import lineplot,scatterplot

Data_1000_fix = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_fix_1000.npy'))
Data_2000_fix = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_fix_2000.npy'))
Data_4000_fix = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_fix_4000.npy'))
Data_8000_fix = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_fix_8000.npy'))
Data_16000_fix = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_fix_16000.npy'))

DF_fix = pd.concat([Data_1000_fix,Data_2000_fix,Data_4000_fix,Data_8000_fix,Data_16000_fix],ignore_index=True)
Data_1000_nlogn = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_nlogn_1000.npy'))
Data_2000_nlogn = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_nlogn_2000.npy'))
Data_4000_nlogn = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_nlogn_4000.npy'))
Data_8000_nlogn = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_nlogn_8000.npy'))
Data_16000_nlogn = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_nlogn_16000.npy'))
DF_nlogn = pd.concat([Data_1000_nlogn,Data_2000_nlogn,Data_4000_nlogn,Data_8000_nlogn,Data_16000_nlogn],ignore_index=True)
Data_1000_n = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_n_1000.npy'))
Data_2000_n = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_n_2000.npy'))
Data_4000_n = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_n_4000.npy'))
Data_8000_n = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_n_8000.npy'))
Data_16000_n = pd.DataFrame(np.load('C:\D_Disk\Label_DP_Final\Simulation_DNN\Res_5_3\Simu_5_3_n_16000.npy'))
DF_n = pd.concat([Data_1000_n,Data_2000_n,Data_4000_n,Data_8000_n,Data_16000_n],ignore_index=True)
lineplot(data=DF_fix, x=4, y=1, color='black',label='$\epsilon=4$',  marker= 'o', markersize=5)
lineplot(data=DF_nlogn, x=4, y=1, color='red',label='$\epsilon\\asymp \log(n)n^{-1/2}}$',  marker= 'o', markersize=5)
lineplot(data=DF_n, x=4, y=1, color='blue',label='$\epsilon \\asymp n^{-1/2}}$',  marker= 'o', markersize=5)
plt.xticks([1000,2000,4000,8000,16000],size=8)
plt.xlabel('Size of Training Dataset: n')
plt.ylabel('Classification Error: CE($\widetilde{s}_{nn}$)')
plt.ylim(0.1,0.3)
plt.legend()
plt.grid()



lineplot(data=DF_fix, x=4, y=0, color='black',label='$\epsilon=4$',  marker= 'o', markersize=5)
lineplot(data=DF_nlogn, x=4, y=0, color='red',label='$\epsilon\\asymp \log(n)n^{-1/2}}$',  marker= 'o', markersize=5)
lineplot(data=DF_n, x=4, y=0, color='blue',label='$\epsilon \\asymp n^{-1/2}}$',  marker= 'o', markersize=5)
plt.xticks([1000,2000,4000,8000,16000],size=8)
plt.xlabel('Size of Training Dataset: n')
plt.ylabel('Empirical Excess Risk: $\widehat{E}(\widetilde{s}_{nn}$)')
plt.grid()