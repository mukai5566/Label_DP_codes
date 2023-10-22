import tensorflow as tf
import numpy as np
import sys
sys.path.append('C:\\D_Disk\\Label_DP_Final\\')
import Simulation_DNN.Data_Gene_DNN as DGNN
def Model_DNN(X):
    Data_R, Data_E, sed, L, units = X
    model = tf.keras.models.Sequential()
    for i in range(L):
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=10,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=tf.keras.losses.mse)
    model.fit(Data_R[0], Data_R[1], epochs=200, verbose=1,callbacks=[callback])
    Y_hat = np.array(model.predict(Data_E[0]).T[0] > 1 / 2, dtype=int)
    Excess_Risk = np.mean((Data_E[1] != Y_hat) * np.abs(2 * Data_E[2] - 1))
    Error = np.mean(Data_E[1] != Y_hat)
    return(Excess_Risk,Error,sed,units,len(Data_R[1]))



if __name__ == '__main__':
    for n in [16000]:
        Data_Set = []
        epsilon = 4
        for i in range(100):
            Data_R, Data_E = DGNN.Gene_DNN(n, epsilon,sed=i)
            Temp = [Data_R,Data_E,i,2,16]
            Data_Set.append(Temp)
        import multiprocessing
        pool = multiprocessing.Pool(processes=4)
        with multiprocessing.Pool(processes=4) as pool:
            result = pool.map(Model_DNN, Data_Set)
        Name = 'Simu_5_3_fix_'+str(n)
        np.save(Name, np.array(result))
        pool.close()
        pool.join()


if __name__ == '__main__':
    for n in [16000]:
        Data_Set = []
        epsilon = 4 / np.sqrt(n) * np.sqrt(1000) / np.log(1000) * np.log(n)
        for i in range(100):
            Data_R, Data_E = DGNN.Gene_DNN(n, epsilon,sed=i)
            Temp = [Data_R,Data_E,i,2,16]
            Data_Set.append(Temp)
        import multiprocessing
        pool = multiprocessing.Pool(processes=4)
        with multiprocessing.Pool(processes=4) as pool:
            result = pool.map(Model_DNN, Data_Set)
        Name = 'Simu_5_3_nlogn_'+str(n)
        np.save(Name, np.array(result))
        pool.close()
        pool.join()





