import numpy as np
def Gene_DNN(n_train=1000,epsi=1,p=4,sed=1,n_test=50000):
    np.random.seed(sed)
    X_train = np.random.uniform(0, 1, [n_train, p])
    Prob_r = (np.sin(X_train*2*np.pi).mean(axis=1) + 1)/2
    theta = np.exp(epsi) / (1 + np.exp(epsi))
    P_tilde_r = theta * Prob_r + (1 - theta) * (1 - Prob_r)
    Y_tilde = np.array(P_tilde_r - np.random.uniform(0, 1, n_train) > 0,dtype=int)
    X_testi = np.random.uniform(0, 1, [n_test, p])
    Prob_e = (np.sin(X_testi*2*np.pi).mean(axis=1) + 1)/2
    Y_testi = np.array(Prob_e >= 1 / 2,dtype=int)
    return([X_train,Y_tilde],[X_testi,Y_testi,Prob_e])