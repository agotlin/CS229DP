
# coding: utf-8

# ## Specs for running

# In[34]:


machine_to_run_script = 'Sherlock' # 'Sherlock', 'local'
spatial_component = 'ANet' # 'vgg16', 'ANet', 'dummy'
temporal_component = 'lstm' # '1dconv', 'lstm'
type_of_input = 'frame' # 'frame', 'video'

print('STARTING SCRIPT!!!!!!!!!!!!!!!!!')


# ### Import Packages

# In[35]:


import numpy as np
import tensorflow as tf
import random as rn
import os
import matplotlib.pyplot as plt
#%matplotlib inline
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
#import cv2
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import gc
from time import time
import datetime
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# In[36]:


gc.collect()


# ### Load Data

# In[37]:


# Helper functions to load data

def ImportImage(filepath):
    #Check if file path exists
    img = Image.open(filepath)
    #img = cv2.imread(filepath, 3)
    return np.array(img)

def windows(data, size, sample_stride): # define time windows to create each training example
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += sample_stride

def hollywood(data_full, input_window_size, sample_stride): # make video files 
    list_of_examples = []
    labels = np.empty((0))
    if machine_to_run_script == 'local':
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
                    #list_of_examples.append(ten_frames)
                    list_of_examples.append(ten_frames[:,:, 0:339, :])
                labels = np.append(labels,data_full['labels'][start])          
    elif machine_to_run_script == 'Sherlock':
        for (start, end) in windows(data_full['rownum'], input_window_size, sample_stride):   
            if(end < data_full.shape[0] and # we are not at the end of the total frames
                len(data_full['rownum'][start:end]) == input_window_size and  # not sure
                int(imagepaths_wlabels['patient_ID'][start])==int(imagepaths_wlabels['patient_ID'][end-1])):  # the end patientID = start patiendID
                # Pull ten frames, crop the images, stack them
                # Check that every file in the frame_sequence exists
                file_names = data_full['image_path'][start:end].values
                all_files_exist = True
                for i, file in enumerate(file_names):
                    if os.path.isfile(file) == False:
                        all_files_exist = False
                if all_files_exist == True:
                    ten_frames = np.array([ImportImage(img) for img in file_names])
                    if spatial_component == 'vgg16': # crop images to fit into vgg16 model architecture
                        ten_frames_crops_res = ten_frames[:,16::2, 50:274, :]
                        ten_frames_crops_res = ten_frames_crops_res[:,0:224, :, :]
                        list_of_examples.append(ten_frames_crops_res)
                    else:
                        #list_of_examples.append(ten_frames)
                        list_of_examples.append(ten_frames[:,:, 0:339, :])
                    labels = np.append(labels,data_full['labels'][start])          
            
    return np.array(list_of_examples), labels


# In[38]:


if type_of_input == 'video':

    # Pull in path file
    print('BEGINNING DATA CREATION')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if machine_to_run_script == 'local':
        imagepaths_wlabels = pd.read_csv('./data/imagepath_gdi_allframes_205.csv') # TO USE REGRESSION
    elif machine_to_run_script == 'Sherlock':
        imagepaths_wlabels = pd.read_csv('./imagepath_gdi_allframes_205_SHER.csv') # TO USE REGRESSION #
        #imagepaths_wlabels = pd.read_csv('./data/imagepath_gdi_allframes_205.csv') # TO USE REGRESSION # FOR TESTING

    # Create video outputs with labels
    frames_per_video = 1
    stride_per_video = 2500
    videos, labels = hollywood(imagepaths_wlabels, frames_per_video, stride_per_video)

    print(videos.shape)

    # Normalize data
    videos_normalized = videos # (videos - videos.mean())/videos.std()
    #videos_normalized = (videos - videos.mean())/videos.std()

    print('finished normalizing')

    # Split data into train and validation
    msk = np.random.rand(len(videos_normalized)) < 0.8
    train_videos=videos_normalized[msk]
    train_videos_labels=labels[msk]
    validation_videos=videos_normalized[~msk]
    validation_videos_labels=labels[~msk]


# In[51]:


if type_of_input == 'frame':

    # Pull in path file
    print('BEGINNING DATA CREATION')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if machine_to_run_script == 'local':
        imagepaths_wlabels = pd.read_csv('./data/imagepath_gdi_10_frames.csv') # TO USE REGRESSION
    elif machine_to_run_script == 'Sherlock':
        imagepaths_wlabels = pd.read_csv('./imagepath_gdi_allframes_205_SHER.csv') # TO USE REGRESSION #

    # Randomly kick out 90% of our data to avoid overflow
    msk_filter = np.random.rand(len(imagepaths_wlabels)) < 0.1
    imagepaths_wlabels = imagepaths_wlabels[msk_filter]
        
    msk = np.random.rand(len(imagepaths_wlabels)) < 0.8
    train=imagepaths_wlabels[msk]
    validation=imagepaths_wlabels[~msk]

    train_imgs = np.array([ImportImage(img) for img in train['image_path'].values]) # if os.path.isfile(img) == True
#     train_labels = np.array([label for label in train['labels'].values if os.path.isfile(img) == True])
    train_labels = np.array([label for label in train['labels'].values])
    validation_imgs = np.array([ImportImage(img) for img in validation['image_path'].values]) # if os.path.isfile(img) == True
#     validation_labels = np.array([label for label in validation['labels'].values if os.path.isfile(img) == True])
    validation_labels = np.array([label for label in validation['labels'].values])
    
    print("Checking zeros")
    
    # Check if any images are zeros
    for i, video in enumerate(train_imgs):
        if (np.count_nonzero(video)<1):
            print('all zeros at : ', i)
    for i, video in enumerate(validation_imgs):
        if (np.count_nonzero(video)<1):
            print('all zeros at : ', i)


# In[52]:


# RANDOM ASIDE FOR CALCULATION ZERO-RULE ERROR (DELETE)

#     imagepaths_wlabels = pd.read_csv('./data/imagepath_gdi_allframes_205_SHER.csv') # TO USE REGRESSION
#     msk = np.random.rand(len(imagepaths_wlabels)) < 0.8
#     train=imagepaths_wlabels[msk]
#     validation=imagepaths_wlabels[~msk]
    
#     train_labels = np.array([label for label in train['labels'].values])
#     validation_labels = np.array([label for label in validation['labels'].values])
    
#     train_mean = np.mean(train_labels)
#     y_pred = np.ones(len(validation_labels))*train_mean
#     y_pred_t = np.ones(len(train_labels))*train_mean
    
#     rms = math.sqrt(mean_squared_error(validation_labels, y_pred))
#     rms_train = math.sqrt(mean_squared_error(train_labels, y_pred_t))
    
#     print('train labels shape : ',train_labels.shape)
#     print('validation labels shape : ',validation_labels.shape)
#     print(train_mean)
#     print(rms)
#     print(rms_train)


# In[53]:


if type_of_input == 'frame':
    #Check if any images are zeros
    mask_t = np.ones(len(train_imgs), dtype=bool)
    for i, video in enumerate(train_imgs):
        if (np.count_nonzero(video)<1):
            #print('all zeros at : ', i)
            np.delete(train_imgs,i)
            np.delete(train_labels,i)
            mask_t[i] = False

    train_imgs = train_imgs[mask_t,...]
    train_labels = train_labels[mask_t,...]

    mask_v = np.ones(len(validation_imgs), dtype=bool)
    for i, video in enumerate(validation_imgs):
        if (np.count_nonzero(video)<1):
            #print('all zeros at : ', i)
            np.delete(validation_imgs,i)
            np.delete(validation_videos_labels,i)
            mask_v = np.ones(len(validation_labels), dtype=bool)
            mask_v[i] = False

    validation_imgs = validation_imgs[mask_v,...]
    validation_labels = validation_labels[mask_v,...]


# In[54]:


if type_of_input == 'video':
   # Check expected shapes and whether data is populated with zeros
   print('train images shape : ',train_videos.shape)
   print('train labels shape : ',train_videos_labels.shape)
   print('validation images shape : ',validation_videos.shape)
   print('validation labels shape : ',validation_videos_labels.shape)
   print(train_videos[0,1,10:15,15:20,1]) # show a random rectangle of one channel of the the image
   print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

   #Check if any images are zeros
   mask_t = np.ones(len(train_videos), dtype=bool)
   for i, video in enumerate(train_videos):
       if (np.count_nonzero(video)<1):
           #print('all zeros at : ', i)
           np.delete(train_videos,i)
           np.delete(train_videos_labels,i)
           mask_t[i] = False

   train_videos = train_videos[mask_t,...]
   train_videos_labels = train_videos_labels[mask_t,...]

   mask_v = np.ones(len(validation_videos), dtype=bool)
   for i, video in enumerate(validation_videos):
       if (np.count_nonzero(video)<1):
           #print('all zeros at : ', i)
           np.delete(validation_videos,i)
           np.delete(validation_videos_labels,i)
           mask_v = np.ones(len(validation_videos), dtype=bool)
           mask_v[i] = False

   validation_videos = validation_videos[mask_v,...]
   validation_videos_labels = validation_videos_labels[mask_v,...]

   # print('Adding a little bit')
   # train_videos = train_videos + 0.00001
   # validation_videos = validation_videos + 0.00001
   # print('Added a little bit')


# In[55]:


if type_of_input == 'video':
   print('train images shape : ',train_videos.shape)
   print('train labels shape : ',train_videos_labels.shape)
   print('validation images shape : ',validation_videos.shape)
   print('validation labels shape : ',validation_videos_labels.shape)
   print(train_videos[0,1,10:15,15:20,1]) # show a random rectangle of one channel of the the image

if type_of_input == 'frame':
   print('train images shape : ',train_imgs.shape)
   print('train labels shape : ',train_labels.shape)
   print('validation images shape : ',validation_imgs.shape)
   print('validation labels shape : ',validation_labels.shape)


# ## Spatial Component of Model

# In[45]:


# Model Parameters

checkpoint_folder = "./cnn_checkpoints_gdicnn"
epochs = 100
epochs_drop,drop_factor = (10,0.95)
batch_size  = 8
if type_of_input == 'video':
    video_shape = train_videos[0].shape
    frame_shape = train_videos[0][0].shape
if type_of_input == 'frame':
    frame_shape = train_imgs[0].shape
kernel_size = 8
conv_dim = 8 # number of kernels
initial_lrate = 0.01
dropout_amount = 0.7
l2_lambda = 10**(-3.5)
reg = keras.regularizers.l2(l2_lambda)

def step_decay(initial_lrate,epochs_drop,drop_factor):
    def step_decay_fcn(epoch):
        return initial_lrate * math.pow(drop_factor, math.floor((1+epoch)/epochs_drop))
    return step_decay_fcn


# In[46]:


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
    Frame_model.add(Dropout(0.9))
    Frame_model.add(Flatten())
    Frame_model.add(Dense(18,activation='relu'))

elif spatial_component == 'ANet':

    Frame_model = Sequential()
    Frame_model.add(Conv2D(conv_dim, kernel_size=kernel_size, input_shape=frame_shape, padding='same'))
    Frame_model.add(Activation('relu'))
    Frame_model.add(BatchNormalization())
    Frame_model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same'))
    Frame_model.add(Activation('relu'))
    Frame_model.add(BatchNormalization())
    Frame_model.add(MaxPooling2D(pool_size=2))
    Frame_model.add(Dropout(dropout_amount))
    Frame_model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
    Frame_model.add(Activation('relu'))
    Frame_model.add(BatchNormalization())
    Frame_model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
    Frame_model.add(Activation('relu'))
    Frame_model.add(BatchNormalization())
    Frame_model.add(MaxPooling2D(pool_size=2))
    Frame_model.add(Dropout(dropout_amount))
    Frame_model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
    Frame_model.add(Activation('relu'))
    Frame_model.add(BatchNormalization())
    Frame_model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))
    Frame_model.add(Activation('relu'))
    Frame_model.add(BatchNormalization())
    Frame_model.add(MaxPooling2D(pool_size=3))
    Frame_model.add(Dropout(dropout_amount))
    Frame_model.add(Flatten())
    Frame_model.add(Dense(16,activation='relu'))
    Frame_model.add(Dense(1, activation='linear'))


# ## Temporal Component

# In[47]:


# if temporal_component == 'lstm':
    
#     # Declare video_model and apply frame_model to each frame
#     video_input = Input(shape=video_shape) # usually 10 frames that are 224x224x3 each
#     encoded_frame_sequence = TimeDistributed(Frame_model)(video_input) # Run frame_model on each frame

#     # Add LSTM                  
#     encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector

#     # Add a linear layer for output of video model
#     output = Dense(1, activation='linear')(encoded_video)

#     # Configure video_model
#     video_model = Model(inputs=video_input, outputs=output)   
    
#     video_model.summary()
    
# elif temporal_component == '1dconv':
    
#     # Declare video_model and apply frame_model to each frame
#     video_input = Input(shape=video_shape) # usually 10 frames that are 224x224x3 each
#     encoded_frame_sequence = TimeDistributed(Frame_model)(video_input) # Run frame_model on each frame
    
#     num_filters = 20
#     kernel_size = 3
#     activation_conv_layer = 'relu'
#     max_pool_kernel_size = 2
#     num_hidden_units_fc_layers_CNN = 16
#     activations_fc_layers_CNN = 'relu'
    
#     # Add 1D convolutions to the sequence (along the time dimension)
#     conv1 = Conv1D(num_filters, kernel_size,activation=activation_conv_layer)(encoded_frame_sequence)
#     pool1 = MaxPooling1D(pool_size=max_pool_kernel_size, padding='valid', strides=(1))(conv1)
#     conv2 = Conv1D(num_filters//5, kernel_size, activation=activation_conv_layer)(pool1) # add additional CNN layer
#     flat1 = Flatten()(pool1)
#     fc_layer = Dense(num_hidden_units_fc_layers_CNN, activation=activations_fc_layers_CNN)(flat1) # add first fully connected layer

# #     # Add fully connected layers (optional)
# #     fc_dict={}
# #     fc_dict[1] = Dense(num_hidden_units_fc_layers_CNN[0], activation=activations_fc_layers_CNN[0])(merged) # add first fully connected layer
# #         for L in range(2, num_hidden_fc_layers_CNN+1):
# #             fc_dict[L] = Dense(num_hidden_units_fc_layers_CNN[L-1], activation=activations_fc_layers_CNN[L-1])(fc_dict[L-1])

#     # Add a softmax layer for output of video_model 
#     output = Dense(1, activation='linear')(fc_layer)

#     # Configure video_model
#     video_model = Model(inputs=[video_input], outputs=output)
    
#     video_model.summary()


# ## Compile Model and Train!

# In[48]:


if type_of_input == 'video':

    # Compile model
    print('COMPILING MODEL')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    filepath=checkpoint_folder+"/weights-{epoch:02d}.hdf5"

    # Create optimizer
    opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Declare optimizer, lsos function, and reporting metrics
    video_model.compile(loss='mse',metrics=['mae'],optimizer=opt)

    # Create checkpoints to keep track of weights
    checkpoint =         ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # Define learning rate function
    lr = LearningRateScheduler(step_decay(initial_lrate,epochs_drop,drop_factor))

    # Add tensorboard logs
    #tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard-logs/{}'.format(time()))


# In[49]:


# Train Model!

if type_of_input == 'video':

    history = video_model.fit(train_videos, train_videos_labels, # ,callbacks=[checkpoint,lr,TerminateOnNaN()], #,tensorboard],
              validation_data=(validation_videos,validation_videos_labels),
              batch_size=batch_size, epochs=epochs,shuffle=True)


# In[50]:


if type_of_input == 'frame':
       
   # Compile model
   print('COMPILING MODEL')
   print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

   if not os.path.exists(checkpoint_folder):
       os.makedirs(checkpoint_folder)

   filepath=checkpoint_folder+"/weights-{epoch:02d}.hdf5"

   # Create optimizer
   opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

   # Declare optimizer, lsos function, and reporting metrics
   Frame_model.compile(loss='mse',metrics=['mae'],optimizer=opt)

   # Create checkpoints to keep track of weights
   checkpoint =        ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

   # Define learning rate function
   lr = LearningRateScheduler(step_decay(initial_lrate,epochs_drop,drop_factor))

   # Add tensorboard logs
   #tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard-logs/{}'.format(time()))
   
   history = Frame_model.fit(train_imgs, train_labels, # ,callbacks=[checkpoint,lr,TerminateOnNaN()], #,tensorboard],
         validation_data=(validation_imgs,validation_labels),
         batch_size=batch_size, epochs=epochs,shuffle=True)
   


# ## Check Predictions and Report Performance

# In[ ]:


if type_of_input == 'video':
   Ynew = video_model.predict(validation_videos)
   # show the inputs and predicted outputs
   for i in range(len(validation_videos)):
       print("True=%s, Predicted=%s" % (validation_videos_labels[i], Ynew[i]))


# In[ ]:


if type_of_input == 'frame':
   Ynew = Frame_model.predict(validation_imgs)
   # show the inputs and predicted outputs
   for i in range(len(validation_imgs)):
       print("True=%s, Predicted=%s" % (validation_labels[i], Ynew[i]))


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

