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

def Gene_Model(n_train,epsi,p=4,sed=1):
    np.random.seed(sed)
    Beta = np.random.uniform(-1, 1, p)
    X_train = np.random.uniform(-1, 1, [n_train, p])
    theta = np.exp(epsi) / (1 + np.exp(epsi))
    P = 1 / (1 + np.exp(-(X_train * Beta).sum(axis=1)))
    P_tilde = theta * P + (1 - theta) * (1 - P)
    Y_tilde = (P_tilde - np.random.uniform(0, 1, n_train) > 0) * 2 - 1
    Model = svm.LinearSVC(loss='hinge',tol=0.001,max_iter=10000)
    Model.fit(X_train,Y_tilde)
    return(Model,Beta)


Error = []
epset = [1,2,3,4,100]
for sp in epset:
    for i in range(9):
        for s in range(1000):
            n  = 100 * 2 ** i
            M, B = Gene_Model(n, sp, sed=s)
            EER,CE  = Excess_risk(M, B, sed=s)
            Error.append([EER,CE,sp,i,s])
            print(sp,1,s)

DF = pd.DataFrame(Error)

import matplotlib.pyplot as plt
from seaborn import lineplot
lineplot(data=DF[DF[2]==1], x=3, y=1, color='red',label='$\epsilon=1$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==2], x=3, y=1, color='blue',label='$\epsilon=2$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==3], x=3, y=1, color='yellow',label='$\epsilon=3$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==4], x=3, y=1, color='grey',label='$\epsilon=4$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==100], x=3, y=1, color='black',label='$\epsilon=\infty$',  marker= 'o', markersize=5)
plt.xlabel('The training size on a log scale: $i=log_2(n/100)$')
plt.ylabel('Classification Error: CE($\widetilde{f}_n$)')
plt.grid()



lineplot(data=DF[DF[2]==1], x=3, y=0, color='red',label='$\epsilon=1$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==2], x=3, y=0, color='blue',label='$\epsilon=2$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==3], x=3, y=0, color='yellow',label='$\epsilon=3$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==4], x=3, y=0, color='grey',label='$\epsilon=4$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]==100], x=3, y=0, color='black',label='$\epsilon=\infty$',  marker= 'o', markersize=5)
plt.xlabel('The training size on a log scale: $i=log_2(n/100)$')
plt.grid()
plt.ylabel('Empirical Excess Risk $\widehat{E}(\widetilde{f}_n)$')
