{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Specs for running"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "machine_to_run_script = 'local' # 'Sherlock', 'local'\n",
    "spatial_component = 'vgg16' # 'vgg16', 'ANet', 'dummy'\n",
    "temporal_component = 'lstm' # '1dconv_v1'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Import Packages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import random as rn\n",
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "os.environ['PYTHONHASHSEED'] = '0'\n",
    "np.random.seed(1)\n",
    "rn.seed(1)\n",
    "from keras import backend as K\n",
    "tf.set_random_seed(1)\n",
    "sess = tf.Session(graph=tf.get_default_graph())\n",
    "K.set_session(sess)\n",
    "import sys\n",
    "import keras\n",
    "from keras.models import Sequential\n",
    "from keras.layers import LSTM, Dense, Activation, Dropout, Flatten\n",
    "from keras.layers import Conv1D,MaxPooling1D,Conv2D,MaxPooling2D\n",
    "from keras.layers import TimeDistributed\n",
    "from keras.layers import Input, LSTM, Embedding, Dense\n",
    "from keras.models import Model, Sequential\n",
    "from keras.layers.normalization import BatchNormalization\n",
    "from keras.optimizers import SGD\n",
    "from keras.optimizers import RMSprop\n",
    "from keras.callbacks import LearningRateScheduler\n",
    "from keras.callbacks import ModelCheckpoint\n",
    "from keras.callbacks import TerminateOnNaN\n",
    "import keras.regularizers\n",
    "import scipy\n",
    "import math\n",
    "import sys\n",
    "import pandas as pd\n",
    "from scipy.ndimage.filters import gaussian_filter1d\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from scipy.stats import linregress\n",
    "from scipy import interpolate\n",
    "from scipy import signal\n",
    "import collections\n",
    "from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img\n",
    "from PIL import Image\n",
    "import matplotlib.pylab as plt\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "import gc\n",
    "from time import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "15"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gc.collect()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Helper functions to load data\n",
    "\n",
    "def ImportImage(filepath):\n",
    "    #Check if file path exists\n",
    "    img = Image.open(filepath)\n",
    "    return np.array(img)\n",
    "\n",
    "def windows(data, size, sample_stride): # define time windows to create each training example\n",
    "    start = 0\n",
    "    while start < data.count():\n",
    "        yield int(start), int(start + size)\n",
    "        start += sample_stride\n",
    "\n",
    "def hollywood(data_full, input_window_size, sample_stride): # make video files \n",
    "    list_of_examples = []\n",
    "    labels = np.empty((0))\n",
    "    for (start, end) in windows(data_full['rownum'], input_window_size, sample_stride):   \n",
    "        if(end < data_full.shape[0] and # we are not at the end of the total frames\n",
    "            len(data_full['rownum'][start:end]) == input_window_size and  # not sure\n",
    "            int(imagepaths_wlabels['image_path'][start][21:29])==int(imagepaths_wlabels['image_path'][end-1][21:29])):  # the end patientID = start patiendID\n",
    "            # Pull ten frames, crop the images, stack them\n",
    "            ten_frames = np.array([ImportImage(img) for img in data_full['image_path'][start:end].values])\n",
    "            if spatial_component == 'vgg16': # crop images to fit into vgg16 model architecture\n",
    "                ten_frames_crops_res = ten_frames[:,16::2, 50:274, :]\n",
    "                ten_frames_crops_res = ten_frames_crops_res[:,0:224, :, :]\n",
    "                list_of_examples.append(ten_frames_crops_res)\n",
    "            else:\n",
    "                list_of_examples.append(ten_frames)\n",
    "            labels = np.append(labels,data_full['labels'][start])          \n",
    "            \n",
    "    return np.array(list_of_examples), labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1369, 5, 224, 224, 3)\n"
     ]
    }
   ],
   "source": [
    "# Pull in path file\n",
    "if machine_to_run_script == 'local':\n",
    "    imagepaths_wlabels = pd.read_csv('./data/imagepath_gdi_allframes_205.csv') # TO USE REGRESSION\n",
    "elif machine_to_run_script == 'Sherlock':\n",
    "    imagepaths_wlabels = pd.read_csv('../SherlockDataFiles/imagepath_gdi_allframes_205.csv') # TO USE REGRESSION\n",
    "\n",
    "# Create video outputs with labels\n",
    "frames_per_video = 5\n",
    "stride_per_video = 50\n",
    "videos, labels = hollywood(imagepaths_wlabels, frames_per_video, stride_per_video)\n",
    "\n",
    "print(videos.shape)\n",
    "\n",
    "# Normalize data\n",
    "videos_normalized = videos # (videos - videos.mean())/videos.std()\n",
    "\n",
    "# Split data into train and validation\n",
    "msk = np.random.rand(len(videos_normalized)) < 0.8\n",
    "train_videos=videos_normalized[msk]\n",
    "train_videos_labels=labels[msk]\n",
    "validation_videos=videos_normalized[~msk]\n",
    "validation_videos_labels=labels[~msk]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_videos=videos_normalized[msk]\n",
    "train_videos_labels=labels[msk]\n",
    "validation_videos=videos_normalized[~msk]\n",
    "validation_videos_labels=labels[~msk]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# videos_normalized = videos # (videos - videos.mean())/videos.std()\n",
    "\n",
    "# # Split data into train and validation\n",
    "# msk = np.random.rand(len(videos_normalized)) < 0.8\n",
    "# train_videos=videos_normalized[msk]\n",
    "# train_videos_labels=labels[msk]\n",
    "# validation_videos=videos_normalized[~msk]\n",
    "# validation_videos_labels=labels[~msk]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(train_videos)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train images shape :  (1096, 5, 224, 224, 3)\n",
      "train labels shape :  (1096,)\n",
      "validation images shape :  (273, 5, 224, 224, 3)\n",
      "validation labels shape :  (273,)\n",
      "[[ 0  0  0  0 63]\n",
      " [ 0  0  0  0 61]\n",
      " [ 0  0  0 58 61]\n",
      " [ 0  0  0 57 60]\n",
      " [ 0  0 53 56 60]]\n",
      "all zeros at :  25\n",
      "all zeros at :  108\n",
      "all zeros at :  187\n",
      "all zeros at :  189\n",
      "all zeros at :  190\n",
      "all zeros at :  390\n",
      "all zeros at :  461\n",
      "all zeros at :  611\n",
      "all zeros at :  612\n",
      "all zeros at :  623\n",
      "all zeros at :  658\n",
      "all zeros at :  744\n",
      "all zeros at :  746\n",
      "all zeros at :  752\n",
      "all zeros at :  753\n",
      "all zeros at :  934\n",
      "all zeros at :  1089\n",
      "all zeros at :  30\n"
     ]
    }
   ],
   "source": [
    "# Check expected shapes and whether data is populated with zeros\n",
    "\n",
    "print('train images shape : ',train_videos.shape)\n",
    "print('train labels shape : ',train_videos_labels.shape)\n",
    "print('validation images shape : ',validation_videos.shape)\n",
    "print('validation labels shape : ',validation_videos_labels.shape)\n",
    "print(train_videos[0,1,10:15,15:20,1]) # show a random rectangle of one channel of the the image\n",
    "\n",
    "# Check if any images are zeros\n",
    "mask_t = np.ones(len(train_videos), dtype=bool)\n",
    "for i, video in enumerate(train_videos):\n",
    "    if (np.count_nonzero(video)<1):\n",
    "        #print('all zeros at : ', i)\n",
    "        np.delete(train_videos,i)\n",
    "        np.delete(train_videos_labels,i)\n",
    "        mask_t[i] = False\n",
    "\n",
    "train_videos = train_videos[mask_t,...]\n",
    "train_videos_labels = train_videos_labels[mask_t,...]\n",
    "\n",
    "mask_v = np.ones(len(train_videos), dtype=bool)\n",
    "for i, video in enumerate(validation_videos):\n",
    "    if (np.count_nonzero(video)<1):\n",
    "        #print('all zeros at : ', i)\n",
    "        np.delete(validation_videos,i)\n",
    "        np.delete(validation_videos_labels,i)\n",
    "        mask_v = np.ones(len(validation_videos), dtype=bool)\n",
    "        mask_v[i] = False\n",
    "        \n",
    "validation_videos = validation_videos[mask_v,...]\n",
    "validation_videos_labels = validation_videos_labels[mask_v,...]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train images shape :  (1079, 5, 224, 224, 3)\n",
      "train labels shape :  (1079,)\n",
      "validation images shape :  (272, 5, 224, 224, 3)\n",
      "validation labels shape :  (272,)\n"
     ]
    }
   ],
   "source": [
    "print('train images shape : ',train_videos.shape)\n",
    "print('train labels shape : ',train_videos_labels.shape)\n",
    "print('validation images shape : ',validation_videos.shape)\n",
    "print('validation labels shape : ',validation_videos_labels.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Spatial Component of Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Model Parameters\n",
    "\n",
    "checkpoint_folder = \"./cnn_checkpoints_gdicnn\"\n",
    "epochs = 50\n",
    "epochs_drop,drop_factor = (5,0.8)\n",
    "batch_size  = 16\n",
    "video_shape = train_videos[0].shape\n",
    "frame_shape = train_videos[0][0].shape\n",
    "kernel_size = 8\n",
    "conv_dim = 16 # number of kernels\n",
    "initial_lrate = 0.01\n",
    "dropout_amount = 0.5\n",
    "l2_lambda = 10**(-3.5)\n",
    "reg = keras.regularizers.l2(l2_lambda)\n",
    "\n",
    "def step_decay(initial_lrate,epochs_drop,drop_factor):\n",
    "    def step_decay_fcn(epoch):\n",
    "        return initial_lrate * math.pow(drop_factor, math.floor((1+epoch)/epochs_drop))\n",
    "    return step_decay_fcn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      \n",
      "_________________________________________________________________\n",
      "block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     \n",
      "_________________________________________________________________\n",
      "block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         \n",
      "_________________________________________________________________\n",
      "block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     \n",
      "_________________________________________________________________\n",
      "block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    \n",
      "_________________________________________________________________\n",
      "block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         \n",
      "_________________________________________________________________\n",
      "block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    \n",
      "_________________________________________________________________\n",
      "block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    \n",
      "_________________________________________________________________\n",
      "block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    \n",
      "_________________________________________________________________\n",
      "block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         \n",
      "_________________________________________________________________\n",
      "block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   \n",
      "_________________________________________________________________\n",
      "block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         \n",
      "_________________________________________________________________\n",
      "block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         \n",
      "_________________________________________________________________\n",
      "flatten (Flatten)            (None, 25088)             0         \n",
      "_________________________________________________________________\n",
      "fc1 (Dense)                  (None, 4096)              102764544 \n",
      "_________________________________________________________________\n",
      "fc2 (Dense)                  (None, 4096)              16781312  \n",
      "_________________________________________________________________\n",
      "predictions (Dense)          (None, 1000)              4097000   \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 16)                16016     \n",
      "=================================================================\n",
      "Total params: 138,373,560\n",
      "Trainable params: 16,016\n",
      "Non-trainable params: 138,357,544\n",
      "_________________________________________________________________\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "if spatial_component == 'vgg16':\n",
    "\n",
    "    # Configure VGG Model:\n",
    "    vgg16_model = keras.applications.vgg16.VGG16() # Download Model\n",
    "    type(vgg16_model) #This is a Keras Functional API need to convert to sequential\n",
    "    Frame_model = Sequential() #Iterate over the functional layers and add it as a stack\n",
    "    for layer in vgg16_model.layers:\n",
    "        Frame_model.add(layer)\n",
    "\n",
    "    # Remove last layer of VGG:\n",
    "    Frame_model.layers.pop()\n",
    "\n",
    "    # Fix the VGG model\n",
    "    for layer in Frame_model.layers: #Since the model is already trained with certain weights, we dont want to change it. Let it be the same\n",
    "        layer.trainable = False\n",
    "\n",
    "    # Add a Dense layer to VGG\n",
    "    Frame_model.add(Dense(16, activation='relu')) # Add a connected layer\n",
    "    \n",
    "    print(Frame_model.summary())\n",
    "    \n",
    "\n",
    "elif spatial_component == 'dummy':\n",
    "    \n",
    "    Frame_model = Sequential()\n",
    "    Frame_model.add(Conv2D(conv_dim, kernel_size=kernel_size, input_shape=frame_shape, padding='valid', strides = 5))\n",
    "    Frame_model.add(Activation('relu'))\n",
    "    Frame_model.add(BatchNormalization())\n",
    "    Frame_model.add(Flatten())\n",
    "    Frame_model.add(Dense(18,activation='relu'))\n",
    "\n",
    "elif spatial_component == 'ANet':\n",
    "\n",
    "    model = Sequential()\n",
    "    model.add(Conv2D(conv_dim, kernel_size=kernel_size, input_shape=input_shape, padding='same'))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same'))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(MaxPooling2D(pool_size=2))\n",
    "    model.add(Dropout(dropout_amount))\n",
    "    model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(MaxPooling2D(pool_size=2))\n",
    "    model.add(Dropout(dropout_amount))\n",
    "    model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(MaxPooling2D(pool_size=3))\n",
    "    model.add(Dropout(dropout_amount))\n",
    "    model.add(Flatten())\n",
    "    model.add(Dense(last_layer_dim,activation='relu'))\n",
    "    model.add(Dense(1, activation='linear'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Temporal Component"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "input_2 (InputLayer)         (None, 5, 224, 224, 3)    0         \n",
      "_________________________________________________________________\n",
      "time_distributed_1 (TimeDist (None, 5, 16)             138373560 \n",
      "_________________________________________________________________\n",
      "lstm_1 (LSTM)                (None, 256)               279552    \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 1)                 257       \n",
      "=================================================================\n",
      "Total params: 138,653,369\n",
      "Trainable params: 295,825\n",
      "Non-trainable params: 138,357,544\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "if temporal_component == 'lstm':\n",
    "    \n",
    "    # Declare video_model and apply frame_model to each frame\n",
    "    video_input = Input(shape=video_shape) # usually 10 frames that are 224x224x3 each\n",
    "    encoded_frame_sequence = TimeDistributed(Frame_model)(video_input) # Run frame_model on each frame\n",
    "\n",
    "    # Add LSTM                  \n",
    "    encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector\n",
    "\n",
    "    # Add a linear layer for output of video model\n",
    "    output = Dense(1, activation='linear')(encoded_video)\n",
    "\n",
    "    # Configure video_model\n",
    "    video_model = Model(inputs=video_input, outputs=output)   \n",
    "    \n",
    "    video_model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Compile Model and Train!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compile model\n",
    "\n",
    "if not os.path.exists(checkpoint_folder):\n",
    "    os.makedirs(checkpoint_folder)\n",
    "\n",
    "filepath=checkpoint_folder+\"/weights-{epoch:02d}.hdf5\"\n",
    "\n",
    "# Create optimizer\n",
    "opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)\n",
    "\n",
    "# Declare optimizer, lsos function, and reporting metrics\n",
    "video_model.compile(loss='mse',metrics=['mae'],optimizer=opt)\n",
    "\n",
    "# Create checkpoints to keep track of weights\n",
    "checkpoint = \\\n",
    "    ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)\n",
    "\n",
    "# Define learning rate function\n",
    "lr = LearningRateScheduler(step_decay(initial_lrate,epochs_drop,drop_factor))\n",
    "\n",
    "# Add tensorboard logs\n",
    "#tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard-logs/{}'.format(time()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 1079 samples, validate on 272 samples\n",
      "Epoch 1/50\n",
      "  32/1079 [..............................] - ETA: 3:17:03 - loss: 5389.3218 - mean_absolute_error: 72.8077"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-34-7f111e38259b>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      3\u001b[0m history = video_model.fit(train_videos, train_videos_labels,callbacks=[checkpoint,lr,TerminateOnNaN()], #,tensorboard],\n\u001b[0;32m      4\u001b[0m           \u001b[0mvalidation_data\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mvalidation_videos\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mvalidation_videos_labels\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m           batch_size=batch_size, epochs=epochs,shuffle=True)\n\u001b[0m",
      "\u001b[1;32m~\\Anaconda3\\envs\\tensorflow_env\\lib\\site-packages\\keras\\engine\\training.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)\u001b[0m\n\u001b[0;32m   1037\u001b[0m                                         \u001b[0minitial_epoch\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0minitial_epoch\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1038\u001b[0m                                         \u001b[0msteps_per_epoch\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0msteps_per_epoch\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1039\u001b[1;33m                                         validation_steps=validation_steps)\n\u001b[0m\u001b[0;32m   1040\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1041\u001b[0m     def evaluate(self, x=None, y=None,\n",
      "\u001b[1;32m~\\Anaconda3\\envs\\tensorflow_env\\lib\\site-packages\\keras\\engine\\training_arrays.py\u001b[0m in \u001b[0;36mfit_loop\u001b[1;34m(model, f, ins, out_labels, batch_size, epochs, verbose, callbacks, val_f, val_ins, shuffle, callback_metrics, initial_epoch, steps_per_epoch, validation_steps)\u001b[0m\n\u001b[0;32m    197\u001b[0m                     \u001b[0mins_batch\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mins_batch\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtoarray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    198\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 199\u001b[1;33m                 \u001b[0mouts\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mf\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mins_batch\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    200\u001b[0m                 \u001b[0mouts\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mto_list\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mouts\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    201\u001b[0m                 \u001b[1;32mfor\u001b[0m \u001b[0ml\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mo\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mout_labels\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mouts\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\envs\\tensorflow_env\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, inputs)\u001b[0m\n\u001b[0;32m   2713\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_legacy_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0minputs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2714\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2715\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0minputs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2716\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2717\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mpy_any\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mis_tensor\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mx\u001b[0m \u001b[1;32min\u001b[0m \u001b[0minputs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\envs\\tensorflow_env\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py\u001b[0m in \u001b[0;36m_call\u001b[1;34m(self, inputs)\u001b[0m\n\u001b[0;32m   2673\u001b[0m             \u001b[0mfetched\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_callable_fn\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0marray_vals\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrun_metadata\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrun_metadata\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2674\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2675\u001b[1;33m             \u001b[0mfetched\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_callable_fn\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0marray_vals\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2676\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mfetched\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moutputs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2677\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\envs\\tensorflow_env\\lib\\site-packages\\tensorflow\\python\\client\\session.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1380\u001b[0m           ret = tf_session.TF_SessionRunCallable(\n\u001b[0;32m   1381\u001b[0m               \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_session\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_session\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_handle\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mstatus\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1382\u001b[1;33m               run_metadata_ptr)\n\u001b[0m\u001b[0;32m   1383\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mrun_metadata\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1384\u001b[0m           \u001b[0mproto_data\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtf_session\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mTF_GetBuffer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrun_metadata_ptr\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "# Train Model!\n",
    "\n",
    "history = video_model.fit(train_videos, train_videos_labels,callbacks=[checkpoint,lr,TerminateOnNaN()], #,tensorboard],\n",
    "          validation_data=(validation_videos,validation_videos_labels),\n",
    "          batch_size=batch_size, epochs=epochs,shuffle=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Check Predictions and Report Performance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True=73.25523767440694, Predicted=[71.599556]\n",
      "True=73.25523767440694, Predicted=[69.644]\n",
      "True=73.25523767440694, Predicted=[72.07249]\n",
      "True=80.44922366806699, Predicted=[82.363655]\n",
      "True=80.44922366806699, Predicted=[83.06998]\n",
      "True=80.44922366806699, Predicted=[80.53925]\n",
      "True=80.44922366806699, Predicted=[73.81894]\n",
      "True=84.74102379560351, Predicted=[84.90053]\n",
      "True=84.74102379560351, Predicted=[80.85953]\n",
      "True=71.6790492593985, Predicted=[71.89114]\n",
      "True=77.1938400448124, Predicted=[73.6715]\n",
      "True=77.1938400448124, Predicted=[77.74444]\n",
      "True=77.1938400448124, Predicted=[80.328606]\n",
      "True=77.1938400448124, Predicted=[74.739876]\n",
      "True=75.07288766535675, Predicted=[76.482605]\n",
      "True=75.07288766535675, Predicted=[71.59193]\n",
      "True=75.07288766535675, Predicted=[84.38763]\n",
      "True=62.133284639290245, Predicted=[63.93198]\n",
      "True=62.133284639290245, Predicted=[62.06683]\n",
      "True=61.39248667646541, Predicted=[60.929028]\n",
      "True=61.39248667646541, Predicted=[63.087093]\n",
      "True=61.39248667646541, Predicted=[56.62547]\n",
      "True=61.39248667646541, Predicted=[58.935467]\n",
      "True=102.7519164554348, Predicted=[102.662674]\n",
      "True=102.7519164554348, Predicted=[102.581764]\n",
      "True=88.17286702598886, Predicted=[85.45616]\n",
      "True=88.17286702598886, Predicted=[86.8857]\n",
      "True=88.17286702598886, Predicted=[79.82423]\n",
      "True=82.21418663880205, Predicted=[80.39486]\n",
      "True=82.21418663880205, Predicted=[83.69624]\n",
      "True=82.21418663880205, Predicted=[83.87299]\n",
      "True=74.70742870540795, Predicted=[75.03192]\n",
      "True=70.95870147644602, Predicted=[67.52883]\n",
      "True=70.95870147644602, Predicted=[71.15849]\n",
      "True=70.95870147644602, Predicted=[74.82227]\n",
      "True=85.32308898899659, Predicted=[84.25356]\n",
      "True=74.10904526666386, Predicted=[74.01431]\n",
      "True=76.2588645555962, Predicted=[68.571175]\n",
      "True=76.2588645555962, Predicted=[76.321686]\n",
      "True=76.2588645555962, Predicted=[75.887215]\n",
      "True=86.9765613362632, Predicted=[86.28792]\n",
      "True=86.9765613362632, Predicted=[87.389244]\n",
      "True=86.9765613362632, Predicted=[82.85294]\n",
      "True=86.9765613362632, Predicted=[80.926544]\n",
      "True=62.046538044625215, Predicted=[60.059086]\n",
      "True=62.046538044625215, Predicted=[63.92905]\n",
      "True=62.046538044625215, Predicted=[60.24662]\n",
      "True=62.046538044625215, Predicted=[62.9279]\n",
      "True=76.6035011342479, Predicted=[78.07855]\n",
      "True=76.6035011342479, Predicted=[75.40444]\n",
      "True=76.6035011342479, Predicted=[74.64684]\n",
      "True=76.6035011342479, Predicted=[74.68578]\n",
      "True=76.6035011342479, Predicted=[73.698]\n",
      "True=95.52373198100672, Predicted=[92.95082]\n",
      "True=95.52373198100672, Predicted=[81.57611]\n",
      "True=77.15815201522035, Predicted=[77.464195]\n",
      "True=77.15815201522035, Predicted=[77.837265]\n",
      "True=76.34763844275375, Predicted=[82.290375]\n",
      "True=76.34763844275375, Predicted=[80.86292]\n",
      "True=76.34763844275375, Predicted=[75.91852]\n",
      "True=73.75502091473605, Predicted=[74.13637]\n",
      "True=73.75502091473605, Predicted=[73.98532]\n",
      "True=69.94689248043511, Predicted=[65.58662]\n",
      "True=69.94689248043511, Predicted=[68.978485]\n",
      "True=69.94689248043511, Predicted=[66.96786]\n",
      "True=75.84350312985775, Predicted=[74.48864]\n",
      "True=68.6030908380578, Predicted=[69.20269]\n",
      "True=68.6030908380578, Predicted=[68.62252]\n",
      "True=68.6030908380578, Predicted=[68.490814]\n",
      "True=62.19011215005239, Predicted=[76.968575]\n",
      "True=62.19011215005239, Predicted=[65.02785]\n",
      "True=63.683676891635045, Predicted=[62.552128]\n",
      "True=81.6561088686743, Predicted=[83.06336]\n",
      "True=81.6561088686743, Predicted=[79.90105]\n",
      "True=81.6561088686743, Predicted=[80.48552]\n",
      "True=80.18111269532929, Predicted=[75.5878]\n",
      "True=80.18111269532929, Predicted=[76.70742]\n",
      "True=80.18111269532929, Predicted=[80.55589]\n",
      "True=97.64540233497351, Predicted=[97.140625]\n",
      "True=97.64540233497351, Predicted=[63.654297]\n",
      "True=61.400407771763895, Predicted=[59.573795]\n",
      "True=61.400407771763895, Predicted=[58.790924]\n",
      "True=73.57620273986909, Predicted=[72.96612]\n",
      "True=73.57620273986909, Predicted=[73.83185]\n",
      "True=70.71924384116333, Predicted=[81.00641]\n"
     ]
    }
   ],
   "source": [
    "Ynew = video_model.predict(validation_videos)\n",
    "# show the inputs and predicted outputs\n",
    "for i in range(len(validation_videos)):\n",
    "    print(\"True=%s, Predicted=%s\" % (validation_videos_labels[i], Ynew[i]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Scratch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# model = Sequential()\n",
    "# model.add(Conv2D(conv_dim, kernel_size=kernel_size, input_shape=input_shape, padding='same'))\n",
    "# model.add(Activation('relu'))\n",
    "# model.add(BatchNormalization())\n",
    "# model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same'))\n",
    "# model.add(Activation('relu'))\n",
    "# model.add(BatchNormalization())\n",
    "# model.add(MaxPooling2D(pool_size=2))\n",
    "# model.add(Dropout(dropout_amount))\n",
    "# model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))\n",
    "# model.add(Activation('relu'))\n",
    "# model.add(BatchNormalization())\n",
    "# model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))\n",
    "# model.add(Activation('relu'))\n",
    "# model.add(BatchNormalization())\n",
    "# model.add(MaxPooling2D(pool_size=2))\n",
    "# model.add(Dropout(dropout_amount))\n",
    "# model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))\n",
    "# model.add(Activation('relu'))\n",
    "# model.add(BatchNormalization())\n",
    "# model.add(Conv2D(conv_dim,kernel_size=kernel_size,padding='same',kernel_regularizer=reg))\n",
    "# model.add(Activation('relu'))\n",
    "# model.add(BatchNormalization())\n",
    "# model.add(MaxPooling2D(pool_size=3))\n",
    "# model.add(Dropout(dropout_amount))\n",
    "# model.add(Flatten())\n",
    "# model.add(Dense(last_layer_dim,activation='relu'))\n",
    "# model.add(Dense(18, activation='softmax'))\n",
    "\n",
    "\n",
    "# from skimage import data, color\n",
    "# from skimage.transform import rescale, resize, downscale_local_mean\n",
    "# arr = train_imgs[0]\n",
    "# #arr_resize = scipy.misc.imresize(arr,(224,224,3))\n",
    "# image_resized = resize(arr, (224,224,3), anti_aliasing=false)\n",
    "\n",
    "# print('train images shape : ',train_imgs.shape)\n",
    "# print('validation images shape : ',validation_imgs.shape)\n",
    "\n",
    "# # RESHAPE FOR VGG16\n",
    "# train_imgs_crops_res = train_imgs[:,16::2, 50:274, :]\n",
    "# train_imgs_crops_res = train_imgs_crops_res[:,0:224, :, :]\n",
    "# validation_imgs_crops_res = validation_imgs[:,16::2, 50:274, :]\n",
    "# validation_imgs_crops_res = validation_imgs_crops_res[:,0:224, :, :]\n",
    "\n",
    "# print(train_imgs_crops_res.shape)\n",
    "# print(validation_imgs_crops_res.shape)\n",
    "\n",
    "# print(train_videos.shape)\n",
    "# print(train_videos_labels.shape)\n",
    "# # validation_videos=videos[~msk]\n",
    "# validation_videos_labels=labels[~msk]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
