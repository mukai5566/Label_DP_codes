import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
X_train = np.load('/Real_Applications\\Mnist_Data_PreProcessed\\X_train.npy')
Y_train = np.load('/Real_Applications\\Mnist_Data_PreProcessed\\Y_train.npy')
X_test = np.load('/Real_Applications\\Mnist_Data_PreProcessed\\X_test.npy')
Y_test = np.load('/Real_Applications\\Mnist_Data_PreProcessed\\Y_test.npy')
#--------------------------Generate New dataset-----------------------------------------
def Data_Transform(DR,n=2000,epsi=1,sed=10000):
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
def Create_model():
    models = Sequential()
    models.add(Conv2D(4, (4, 4),
                      padding="same",
                      activation="relu",
                      input_shape=(28, 28, 1)))
    models.add(MaxPooling2D(pool_size=(2, 2)))
    models.add(Conv2D(4, (4, 4), padding="same",
                      activation="relu"))
    models.add(MaxPooling2D(pool_size=(2, 2)))
    models.add(Conv2D(4, (4, 4), padding="same",
                      activation="relu"))
    models.add(MaxPooling2D(pool_size=(2, 2)))
    models.add(Flatten())
    models.add(Dense(10, activation="relu"))
    models.add(Dense(1))
    models.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mse)
    return(models)
#-----------------------------------------------------------------------------------------------------------------------------------------------
Error = []
for u in [6000]:
    ep = 2* np.sqrt(2000)/ np.sqrt(u)
    dr = Data_Transform([X_train, Y_train], u, epsi=ep)
    Model = Create_model()
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.0001,
        patience=10,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True)
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mse)
    Model.fit(dr[0].astype(np.float32), dr[1].astype(np.float32), epochs=500, callbacks=[callback], verbose=2)
    Name ='Mnist_n_2_' + str(u)
    Model.save(Name)
    Y_hat_E = np.array((Model.predict(X_test.astype(np.float32)) > 0.5).T[0], dtype=int)
    Error.append(np.mean(Y_hat_E != Y_test))



for u in [2000,4000,6000,8000]:
    ep = 2* np.sqrt(2000)/ np.sqrt(u)/np.log(2000)*np.log(u)
    dr = Data_Transform([X_train, Y_train], u, epsi=ep)
    Model = Create_model()
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.0001,
        patience=10,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True)
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mse)
    Model.fit(dr[0].astype(np.float32), dr[1].astype(np.float32), epochs=500, callbacks=[callback], verbose=2)
    Name ='Mnist_nlogn_2_' + str(u)
    Model.save(Name)
    Y_hat_E = np.array((Model.predict(X_test.astype(np.float32)) > 0.5).T[0], dtype=int)
    Error.append(np.mean(Y_hat_E != Y_test))




