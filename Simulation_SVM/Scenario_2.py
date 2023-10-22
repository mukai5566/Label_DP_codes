import numpy as np
import pandas as pd
from sklearn import svm
def Excess_risk(Model,Beta,sed=1):
    n_test = 50000
    np.random.seed(sed)
    X_test = np.random.uniform(-1,1,[n_test,len(Beta)])
    P_test = 1 / (1 + np.exp(-(X_test * Beta).sum(axis=1)))
    Y_star = np.sign(P_test-1/2)
    Y_test = Model.predict(X_test)
    ER_est = np.mean((Y_test!=Y_star) * np.abs(P_test * 2 -1))
    CE_est = np.mean(Y_test!=Y_star)
    return(ER_est,CE_est)

def Gene_Model(n_train,epsi,p=2,sed=1):
    np.random.seed(sed)
    Beta = np.random.uniform(-1, 1, p)
    #Beta = np.ones(p)
    X_train = np.random.uniform(-1, 1, [n_train, p])
    theta = np.exp(epsi) / (1 + np.exp(epsi))
    P = 1 / (1 + np.exp(-(X_train * Beta).sum(axis=1)))
    P_tilde = theta * P + (1 - theta) * (1 - P)
    Y_tilde = (P_tilde - np.random.uniform(0, 1, n_train) > 0) * 2 - 1
    Model = svm.LinearSVC(loss='hinge',tol=0.001,max_iter=10000)
    Model.fit(X_train,Y_tilde)
    return(Model,Beta)


Error = []
Size = [100 * 2 ** i for i in range(9)]
epset = [1/5,1/4,1/3,1/2,2/3,1,0]
for sp in epset:
    for i in range(9):
        for s in range(1000):
            n  = 100 * 2 ** i
            if sp!=0:
                epsilon = 2 * 100**sp/n**sp
            else:
                epsilon = 0
            M, B = Gene_Model(n, epsilon, sed=s)
            EER,CE  = Excess_risk(M, B, sed=s)
            Error.append([EER,CE,sp,i,s])
            print(sp,i,s)

DF = pd.DataFrame(Error)

import matplotlib.pyplot as plt
from seaborn import lineplot
lineplot(data=DF[DF[2]==1/5], x=3, y=1, color='red',label='$\zeta=1/5$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==1/4], x=3, y=1, color='blue',label='$\zeta=1/4$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==1/3], x=3, y=1, color='yellow',label='$\zeta=1/3$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==1/2], x=3, y=1, color='grey',label='$\zeta=1/2$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==1], x=3, y=1, color='green',label='$\zeta=1$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==0], x=3, y=1, color='black',label='$\epsilon=0$',  marker= 'o', markersize=5)
plt.xlabel('The training size on a log scale: $i=log_2(n/100)$')
plt.ylabel('Classification Error: CE($\widetilde{f}_n$)')
plt.grid()
plt.ylim(0,1)


lineplot(data=DF[DF[2]==1/5], x=3, y=0, color='red',label='$\zeta=1/5$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==1/4], x=3, y=0, color='blue',label='$\zeta=1/4$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==1/3], x=3, y=0, color='yellow',label='$\zeta=1/3$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==1/2], x=3, y=0, color='grey',label='$\zeta=1/2$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==1], x=3, y=0, color='green',label='$\zeta=1$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==0], x=3, y=0, color='black',label='$\epsilon=0$',  marker= 'o', markersize=5)
plt.xlabel('The training size on a log scale: $i=log_2(n/100)$')
plt.ylabel('Empirical Excess Risk $\widehat{E}(\widetilde{f}_n)$')
plt.grid()
plt.ylim(0,0.18)

DF.to_csv('Res_Scen2')