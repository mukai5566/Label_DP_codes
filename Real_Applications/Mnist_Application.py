import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
X_train = np.load('/Real_Applications\\Mnist_Data_PreProcessed\\X_train.npy')
Y_train = np.load('/Real_Applications\\Mnist_Data_PreProcessed\\Y_train.npy')
X_test = np.load('/Real_Applications\\Mnist_Data_PreProcessed\\X_test.npy')
Y_test = np.load('/Real_Applications\\Mnist_Data_PreProcessed\\Y_test.npy')
def Data_Transform(DR,n=2000,epsi=1,sed=1):
    theta = np.exp(epsi)/(np.exp(epsi)+1)
    L = len(DR[1])
    np.random.seed(sed)
    Prob =  np.random.uniform(0,1,L)
    Change = Prob<=theta
    New_Label = Change * DR[1] + (1-Change) * (1-DR[1])
    Choice = np.random.choice(np.arange(0,L),n,replace=False)
    Index = np.sort(Choice)
    DR_X = DR[0][Index]
    DR_Y = New_Label[Index]
    return([DR_X,DR_Y])


def model_CNN_Base(Input):
    DR, DE, N, eps, se = Input
    Name = 'C:\D_Disk\Label_DP_Final\Mnist' + '_Base'
    models = tf.keras.models.load_model(Name)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.0005,
        patience=5,
        verbose=0,
        mode='auto',
       baseline=None,
       restore_best_weights=True)
    models.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=tf.keras.losses.mse)
    models.fit(DR[0].astype(np.float32), DR[1].astype(np.float32),epochs=500,callbacks=[callback],verbose=2)
    Y_hat_E = np.array((models.predict(DE[0].astype(np.float32)) > 0.5).T[0], dtype=int)
    print('This is the replication', se)
    return([np.mean(Y_hat_E!=DE[1]),N,se])





if __name__ == '__main__':
    for epsilon in [2]:
        Data_ALL = [Data_Transform([X_train, Y_train], n=10000, epsi=epsilon, sed=r) for r in range(50)]
        for u in [6000]:
            Data_Set = []
            for i in range(50):
                DR_temp = [Data_ALL[i][0][0:u,:,:,:],Data_ALL[i][1][0:u]]
                Temp = [DR_temp,[X_test,Y_test],u,epsilon,i]
                Data_Set.append(Temp)
            import multiprocessing
            pool = multiprocessing.Pool(processes=5)
            with multiprocessing.Pool(processes=5) as pool:
                result = pool.map(model_CNN_Base, Data_Set)
            Name = 'MNIST_' + str(epsilon) + '_' + str(u)
            np.save(Name, np.array(result))
            pool.close()
            pool.join()
