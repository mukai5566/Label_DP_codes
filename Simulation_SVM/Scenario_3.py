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
epset = ['A','B','C','D']
for sp in epset:
    for i in range(10):
        for s in range(1000):
            n = 100 * 2 ** i
            if sp=='A':
                epsilon = 2 * 1 / n**(0.5) * np.sqrt(100)
            if sp=='B':
                epsilon = 2 * np.sqrt(100) * (np.log(n)) / n**(0.5) / (np.log(100))
            if sp=='C':
                epsilon = 2 * np.sqrt(100) * (np.log(100)) / n**(0.5) / (np.log(n))
            if sp=='D':
                epsilon = 0.
            M, B = Gene_Model(n, epsilon, sed=s)
            EER, CE = Excess_risk(M, B, sed=s)
            Error.append([EER, CE, sp, i, s])
            print(sp, i, s)
DF = pd.DataFrame(Error)


import matplotlib.pyplot as plt
from seaborn import lineplot
lineplot(data=DF[DF[2]=='A'], x=3, y=1, color='red',label='$\epsilon \\asymp n^{-1/2}$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]=='B'], x=3, y=1, color='blue',label='$\epsilon \\asymp \log(n)n^{-1/2}$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]=='C'], x=3, y=1, color='yellow',label='$\epsilon\\asymp \log^{-1}(n)n^{-1/2}$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]=='D'], x=3, y=1, color='grey',label='$\epsilon=0$',  marker= 'o', markersize=5)
plt.xlabel('The training size on a log scale: $i=log_3(n/100)$')
plt.ylabel('Classification Error: CE($\widetilde{f}_n$)')
plt.grid()
plt.ylim(0,1)



import matplotlib.pyplot as plt
from seaborn import lineplot
lineplot(data=DF[DF[2]=='A'], x=3, y=0, color='red',label='$\epsilon \\asymp n^{-1/2}$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]=='B'], x=3, y=0, color='blue',label='$\epsilon \\asymp \log(n)n^{-1/2}$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]=='C'], x=3, y=0, color='yellow',label='$\epsilon=\\asymp \log^{-1}(n)n^{-1/2}$',  marker= 'o', markersize=5)
lineplot(data=DF[DF[2]=='D'], x=3, y=0, color='grey',label='$\epsilon=0$',  marker= 'o', markersize=5)
plt.xlabel('The training size on a log scale: $i=log_3(n/100)$')
plt.ylabel('Empirical Excess Risk $\widehat{E}(\widetilde{f}_n)$')
plt.grid()
plt.ylim(0,0.16)








from scipy.stats import multivariate_normal
p=3
X_1 = np.random.normal(0,1,[1000,p])
X_2 = np.random.normal(0.5,1,[1000,p])
X = np.concatenate([X_1,X_2],axis=0)
D_1 = multivariate_normal.pdf(X_1, mean=np.ones(p)*0, cov=np.diag(np.ones(p)))
D_2 = multivariate_normal.pdf(X_2, mean=np.ones(p)*0.5, cov=np.diag(np.ones(p)))
TV = np.mean(np.abs(D_1 - D_2)) * 0.5


