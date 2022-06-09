import keras
import tensorflow as tf
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, LeakyReLU, Dense, MaxPool1D
from keras_preprocessing.sequence import pad_sequences
from keras.losses import mse



with open("plucker_s.txt", "rb") as f:
    sample = eval(f.read())
sample = [(s,i) for s,i in sample if len(s) > 6]

for x,i in sample:
    print(i, len(x[0]))
input()



X_pre = []
for S,i in sample:
    cnt = 0
    for s in S:
        X_pre.append((s, i))
        cnt += 1
        if cnt >= 12:
            break

X = np.array([list(s) for s,_ in X_pre])
X=pad_sequences(X, padding='post',value=0)

Y = np.array([i for _,i in X_pre])



#print (list(X))

trainingindex = random.sample([k for k in range(len(X))],
round(len(X)*0.9) )

validateindex = list(set([k for k in range(len(X))]) -
set(trainingindex));


X_train = np.array([X[a] for a in trainingindex]);
Y_train = np.array([Y[a] for a in trainingindex]);
X_test = np.array([X[a] for a in validateindex]);
Y_test = np.array([Y[a] for a in validateindex]);

trainingX = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
validateX  = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
Y_train_re = np.reshape(Y_train,(Y_train.shape[0],1))
Y_test_re = np.reshape(Y_test,(Y_test.shape[0],1))

trainingY=tf.keras.utils.to_categorical(Y_train_re, num_classes=None, dtype='float32')
validateY=tf.keras.utils.to_categorical(Y_test_re, num_classes=None, dtype='float32')

model = Sequential()
input_shapeX = trainingX[0].shape

print (trainingX.shape, input_shapeX, trainingX[0])
model.add(Conv1D(16, kernel_size=4,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPool1D(2,padding='same'))

model.add(Conv1D(16, kernel_size=4,activation='linear',padding='same',input_shape=input_shapeX))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool1D(2,padding='same'))

model.add(Flatten())
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(14,activation='softmax'))

model.compile(
    optimizer='nadam', loss='huber_loss')

model.summary()

model.fit(trainingX,trainingY,batch_size=24,epochs=500,validation_data=(validateX,validateY))

predictY = model.predict(validateX,verbose = 1)

print (sum(mse(validateY, predictY).numpy())/11)
input()

for i,p in enumerate(predictY):
    q = validateY[i]
    print (list(p).index(max(p)), list(q).index(max(q)))
#print (predictY)