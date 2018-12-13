
# coding: utf-8

# ## Specs for running

# In[ ]:


machine_to_run_script = 'local' # 'Sherlock', 'local'
spatial_component = 'vgg16' # 'vgg16', 'ANet', 'dummy'
temporal_component = 'lstm' # '1dconv_v1'


# ### Import Packages

# In[ ]:


import numpy as np
import tensorflow as tf
import random as rn
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)
from keras import backend as K
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)
import sys
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from keras.layers import Conv1D,MaxPooling1D,Conv2D,MaxPooling2D
from keras.layers import TimeDistributed
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TerminateOnNaN
import keras.regularizers
import scipy
import math
import sys
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
from scipy import interpolate
from scipy import signal
import collections
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import gc
from time import time


# In[ ]:


gc.collect()


# ### Load Data

# In[ ]:


# Helper functions to load data

def ImportImage(filepath):
    #Check if file path exists
    img = Image.open(filepath)
    return np.array(img)

def windows(data, size, sample_stride): # define time windows to create each training example
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += sample_stride

def hollywood(data_full, input_window_size, sample_stride): # make video files 
    list_of_examples = []
    labels = np.empty((0))
    for (start, end) in windows(data_full['rownum'], input_window_size, sample_stride):   
        if(end < data_full.shape[0] and # we are not at the end of the total frames
            len(data_full['rownum'][start:end]) == input_window_size and  # not sure
            int(imagepaths_wlabels['image_path'][start][21:29])==int(imagepaths_wlabels['image_path'][end-1][21:29])):  # the end patientID = start patiendID
            # Pull ten frames, crop the images, stack them
            ten_frames = np.array([ImportImage(img) for img in data_full['image_path'][start:end].values])
            if spatial_component == 'vgg16': # crop images to fit into vgg16 model architecture
                ten_frames_crops_res = ten_frames[:,16::2, 50:274, :]
                ten_frames_crops_res = ten_frames_crops_res[:,0:224, :, :]
                list_of_examples.append(ten_frames_crops_res)
            else:
                list_of_examples.append(ten_frames)
            labels = np.append(labels,data_full['labels'][start])          
            
    return np.array(list_of_examples), labels


# In[ ]:


# Pull in path file
if machine_to_run_script == 'local':
    imagepaths_wlabels = pd.read_csv('./data/imagepath_gdi_allframes_205.csv') # TO USE REGRESSION
elif machine_to_run_script == 'Sherlock':
    imagepaths_wlabels = pd.read_csv('../SherlockDataFiles/imagepath_gdi_allframes_205.csv') # TO USE REGRESSION

# Create video outputs with labels
frames_per_video = 5
stride_per_video = 50
videos, labels = hollywood(imagepaths_wlabels, frames_per_video, stride_per_video)

print(videos.shape)

# Normalize data
videos_normalized = videos # (videos - videos.mean())/videos.std()

# Split data into train and validation
msk = np.random.rand(len(videos_normalized)) < 0.8
train_videos=videos_normalized[msk]
train_videos_labels=labels[msk]
validation_videos=videos_normalized[~msk]
validation_videos_labels=labels[~msk]


# In[ ]:


train_videos=videos_normalized[msk]
train_videos_labels=labels[msk]
validation_videos=videos_normalized[~msk]
validation_videos_labels=labels[~msk]


# In[ ]:


# videos_normalized = videos # (videos - videos.mean())/videos.std()

# # Split data into train and validation
# msk = np.random.rand(len(videos_normalized)) < 0.8
# train_videos=videos_normalized[msk]
# train_videos_labels=labels[msk]
# validation_videos=videos_normalized[~msk]
# validation_videos_labels=labels[~msk]


# In[ ]:


type(train_videos)


# In[ ]:


# Check expected shapes and whether data is populated with zeros

print('train images shape : ',train_videos.shape)
print('train labels shape : ',train_videos_labels.shape)
print('validation images shape : ',validation_videos.shape)
print('validation labels shape : ',validation_videos_labels.shape)
print(train_videos[0,1,10:15,15:20,1]) # show a random rectangle of one channel of the the image

# Check if any images are zeros
mask_t = np.ones(len(train_videos), dtype=bool)
for i, video in enumerate(train_videos):
    if (np.count_nonzero(video)<1):
        #print('all zeros at : ', i)
        np.delete(train_videos,i)
        np.delete(train_videos_labels,i)
        mask_t[i] = False

train_videos = train_videos[mask_t,...]
train_videos_labels = train_videos_labels[mask_t,...]

mask_v = np.ones(len(train_videos), dtype=bool)
for i, video in enumerate(validation_videos):
    if (np.count_nonzero(video)<1):
        #print('all zeros at : ', i)
        np.delete(validation_videos,i)
        np.delete(validation_videos_labels,i)
        mask_v = np.ones(len(validation_videos), dtype=bool)
        mask_v[i] = False
        
validation_videos = validation_videos[mask_v,...]
validation_videos_labels = validation_videos_labels[mask_v,...]


# In[ ]:


print('train images shape : ',train_videos.shape)
print('train labels shape : ',train_videos_labels.shape)
print('validation images shape : ',validation_videos.shape)
print('validation labels shape : ',validation_videos_labels.shape)


# ## Spatial Component of Model

# In[ ]:


# Model Parameters

checkpoint_folder = "./cnn_checkpoints_gdicnn"
epochs = 50
epochs_drop,drop_factor = (5,0.8)
batch_size  = 16
video_shape = train_videos[0].shape
frame_shape = train_videos[0][0].shape
kernel_size = 8
conv_dim = 16 # number of kernels
initial_lrate = 0.01
dropout_amount = 0.5
l2_lambda = 10**(-3.5)
reg = keras.regularizers.l2(l2_lambda)

def step_decay(initial_lrate,epochs_drop,drop_factor):
    def step_decay_fcn(epoch):
        return initial_lrate * math.pow(drop_factor, math.floor((1+epoch)/epochs_drop))
    return step_decay_fcn


# In[ ]:


if spatial_component == 'vgg16':

    # Configure VGG Model:
    vgg16_model = keras.applications.vgg16.VGG16() # Download Model
    type(vgg16_model) #This is a Keras Functional API need to convert to sequential
    Frame_model = Sequential() #Iterate over the functional layers and add it as a stack
    for layer in vgg16_model.layers:
        Frame_model.add(layer)

    # Remove last layer of VGG:
    Frame_model.layers.pop()

    # Fix the VGG model
    for layer in Frame_model.layers: #Since the model is already trained with certain weights, we dont want to change it. Let it be the same
        layer.trainable = False

    # Add a Dense layer to VGG
    Frame_model.add(Dense(16, activation='relu')) # Add a connected layer
    
    print(Frame_model.summary())
    

elif spatial_component == 'dummy':
    
    Frame_model = Sequential()
    Frame_model.add(Conv2D(conv_dim, kernel_size=kernel_size, input_shape=frame_shape, padding='valid', strides = 5))
    Frame_model.add(Activation('relu'))
    Frame_model.add(BatchNormalization())
    Frame_model.add(Flatten())
    Frame_model.add(Dense(18,activation='relu'))

elif spatial_component == 'ANet':

    model = Sequential()
    model.add(Conv2D(conv_dim, kernel_size=kernel_size, input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_amount))
    model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_amount))
    model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=3))
    model.add(Dropout(dropout_amount))
    model.add(Flatten())
    model.add(Dense(last_layer_dim,activation='relu'))
    model.add(Dense(1, activation='linear'))


# ## Temporal Component

# In[ ]:


if temporal_component == 'lstm':
    
    # Declare video_model and apply frame_model to each frame
    video_input = Input(shape=video_shape) # usually 10 frames that are 224x224x3 each
    encoded_frame_sequence = TimeDistributed(Frame_model)(video_input) # Run frame_model on each frame

    # Add LSTM                  
    encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector

    # Add a linear layer for output of video model
    output = Dense(1, activation='linear')(encoded_video)

    # Configure video_model
    video_model = Model(inputs=video_input, outputs=output)   
    
    video_model.summary()


# ## Compile Model and Train!

# In[ ]:


# Compile model

if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

filepath=checkpoint_folder+"/weights-{epoch:02d}.hdf5"

# Create optimizer
opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Declare optimizer, lsos function, and reporting metrics
video_model.compile(loss='mse',metrics=['mae'],optimizer=opt)

# Create checkpoints to keep track of weights
checkpoint =     ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Define learning rate function
lr = LearningRateScheduler(step_decay(initial_lrate,epochs_drop,drop_factor))

# Add tensorboard logs
#tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard-logs/{}'.format(time()))


# In[ ]:


# Train Model!

history = video_model.fit(train_videos, train_videos_labels,callbacks=[checkpoint,lr,TerminateOnNaN()], #,tensorboard],
          validation_data=(validation_videos,validation_videos_labels),
          batch_size=batch_size, epochs=epochs,shuffle=True)


# ## Check Predictions and Report Performance

# In[ ]:


Ynew = video_model.predict(validation_videos)
# show the inputs and predicted outputs
for i in range(len(validation_videos)):
    print("True=%s, Predicted=%s" % (validation_videos_labels[i], Ynew[i]))


# ## Scratch

# In[ ]:


# model = Sequential()
# model.add(Conv2D(conv_dim, kernel_size=kernel_size, input_shape=input_shape, padding='same'))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same'))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(dropout_amount))
# model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(dropout_amount))
# model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=3))
# model.add(Dropout(dropout_amount))
# model.add(Flatten())
# model.add(Dense(last_layer_dim,activation='relu'))
# model.add(Dense(18, activation='softmax'))


# from skimage import data, color
# from skimage.transform import rescale, resize, downscale_local_mean
# arr = train_imgs[0]
# #arr_resize = scipy.misc.imresize(arr,(224,224,3))
# image_resized = resize(arr, (224,224,3), anti_aliasing=false)

# print('train images shape : ',train_imgs.shape)
# print('validation images shape : ',validation_imgs.shape)

# # RESHAPE FOR VGG16
# train_imgs_crops_res = train_imgs[:,16::2, 50:274, :]
# train_imgs_crops_res = train_imgs_crops_res[:,0:224, :, :]
# validation_imgs_crops_res = validation_imgs[:,16::2, 50:274, :]
# validation_imgs_crops_res = validation_imgs_crops_res[:,0:224, :, :]

# print(train_imgs_crops_res.shape)
# print(validation_imgs_crops_res.shape)

# print(train_videos.shape)
# print(train_videos_labels.shape)
# # validation_videos=videos[~msk]
# validation_videos_labels=labels[~msk]

