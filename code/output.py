
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
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))

#config()
    
print('libs loaded')


# In[2]:


test_data = pd.read_csv('../input/csv/test.csv')


# In[3]:


test_data.head()


# In[4]:


# load
model = load_model('../save/' + '68000' + '.model.h5')


# In[20]:



def predict(data_id):
    path = '../input/test/'
    data_shape = (1, 128, 128, 3)
    X = np.zeros(data_shape)
    X[0] = np.array(cv2.imread(path + data_id + '.jpg'))# BGR
    X = X.astype('float32')
    X /= 255
    #plt.imshow(cv2.cvtColor(X[0], cv2.COLOR_BGR2RGB))

    p_all = model.predict(X)
    p = np.argmax(p_all)
    p_val = p_all[0][p]

    #print('Predict: ', predict)
    #print(p_val)
    
    return p, p_val
    
data_id = test_data['id'][0]
p, p_val = predict(data_id)
print(p, p_val)


# In[6]:


#test_data.shape
#test_data.shape[0]
#117703


# In[23]:


idc = []
landmarks = []

for i in range(test_data.shape[0]):
    data_id = test_data['id'][i]
    p, p_val = predict(data_id)
    idc.append(data_id)
    landmarks.append(str(p) + ' ' + str(round(p_val, 2)))
    if i % 1000 == 0:
        print(i)
    
df_dict = {'id': idc, 'landmarks': landmarks}
df = pd.DataFrame(df_dict)
df = df.set_index('id')
df.to_csv('./output.csv', sep=',')

print("== done")

