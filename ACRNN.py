# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:35:40 2024

@author: Indrajeet_Kumar
"""

#Import image
from skimage import io
image_X = io.imread('path to image') 

#Split into Train & Test set
X_train = image_X[0:102]#80% train
y_train = image_X[102:]#20% Test
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Reshape, UpSampling2D, Dropout, Permute, BatchNormalization
from attention import Attention
from keras.optimizers import Adam
import time
t = time.time() 
inputs = Input(shape = (128, 128,1))
# Convolutional layers
conv1 = Conv2D(32, kernel_size=(3, 3), strides = (1,1), activation='relu', padding='same')(inputs)
batch1 = BatchNormalization()(conv1)
maxpool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(batch1)
conv2 = Conv2D(64, kernel_size=(3, 3), strides = (1,1), activation='relu',  padding='same')(maxpool1)
batch2 = BatchNormalization()(conv2)
maxpool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(batch2)
conv3 = Conv2D(128, kernel_size=(3, 3), strides = (1,1), activation='relu',  padding='same')(maxpool2)
batch2 = BatchNormalization()(conv3)
maxpool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(batch2)
dropout = Dropout(0.5)(maxpool3)
permuted = Permute((1,3, 2))(dropout)
# Recurrent layer
flatten = Flatten()(permuted)
reshape1 = Reshape(target_shape=(64*4, 128))(flatten)
lstm = LSTM(128, dropout = 0.25, return_sequences=True)(reshape1)
attention = Attention(8192)(lstm)
reshape2 = Reshape((8*2, 8*2, 128))(attention)
# Dense layers
dense = Dense(64, activation='softmax')(reshape2)
# Convolutional layers
con1 = Conv2D(128, kernel_size=(3, 3), strides = (1,1), activation='relu', padding='same')(dense)
upsample1 = UpSampling2D(size=(2, 2))(con1)
con2 = Conv2D(64, kernel_size=(3, 3), strides = (1,1), activation='relu', padding='same')(upsample1)
upsample2 = UpSampling2D(size=(2, 2))(con2)
con3 = Conv2D(32, kernel_size=(3, 3), strides = (1,1), activation='relu', padding='same')(upsample2)
upsample3 = UpSampling2D(size=(2, 2))(con3)
outputs = Conv2D(1, kernel_size=(3, 3),strides = (1,1), activation='relu', padding='same')(upsample3)
# Define model
model = Model(inputs=inputs, outputs=outputs)
opt = Adam(lr=0.001,  beta_1 = 0.5)
# Compile model
model.compile(optimizer=opt, loss='mse')
model.summary()
history = model.fit(X_train, y_train, batch_size = 4, epochs=500)
dt = time.time() - t
print("Model Solved in {} ".format(dt) + " secs \n")
