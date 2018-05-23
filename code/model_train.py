
# coding: utf-8

# In[1]:


import keras
import os
os.environ['KERAS_BACKEND']='tensorflow'
import h5py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Flatten, Convolution2D, Conv2DTranspose, Dropout
from keras.layers import MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.callbacks import TensorBoard

def config():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    set_session(tf.Session(config=config))

config()
    
print('libs loaded')


# In[2]:


def get_list():
    X_filenames = []
    path = '../input/train/'
    for root, dirs, files in os.walk(path):
        for name in files:
            X_filenames.append(os.path.join(root, name).split('/')[-1])
    return X_filenames
X_filenames = get_list()


# In[3]:



def load_batch(batch_size):
    path = '../input/train/'
    X_list = []
    Y_list = []
    for i in range(batch_size):
        name = random.choice(X_filenames)
        X_list.append(name)
        Y_list.append(int(name.split('.')[1]))
        
    #print(X_list)
    #print(Y_list)
    
    size = 128
    data_shape = (batch_size, size, size, 3)
    X = np.zeros(data_shape)
    
    for i in range(batch_size):
        f = X_list[i]
        img = np.array(cv2.imread(path + f))# BGR
        #print(img.shape)
        X[i] = img
        
    X = X.astype('float32')
    X /= 255
    
    Y = np.zeros((batch_size, 14951)) # one hot
    Y[np.arange(batch_size), Y_list] = 1
    
    return X, Y


# In[4]:


# continue training
m_num = 10000

# load
model = load_model('../save/' + str(m_num) + '.model.h5')


# In[5]:



# logs
tensorboard = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_grads=True, write_images=True)


# In[6]:


# training
nb_epoch = 100000
for e in range(nb_epoch):
    print()
    print("* Epoch %d" % e)
    X_batch, Y_batch = load_batch(500)
    model.fit(X_batch, Y_batch, 
              validation_split=0.3, 
              shuffle=True,
              callbacks=[tensorboard])
    
    if (e+1) % 1000 == 0:
        model.save('../save/' + str(m_num + e + 1) + '.model.h5')
        print('/// model saved ///')

del model

