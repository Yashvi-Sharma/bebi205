#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import glob2, subprocess
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skimage import exposure
import multiprocessing as mp
import yaml
import warnings

import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from keras import regularizers, optimizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.layers import (Conv2D,Dense,Dropout,Flatten,MaxPooling2D,Input)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')


# ## Exploring the dataset

# In[2]:


def norm_image(img):
    norm = (img - np.min(img))/(np.max(img) - np.min(img))
#     norm = exposure.equalize_adapthist(norm)
    return norm

class CellData:
    def __init__(self):
        return
    
    def load_xy_from_np(self, npfile):
            
        def load_np(npfile):
            return np.load(npfile, allow_pickle=True)
            
        data = load_np(npfile)
        self.X = norm_image(data['X'][0])
        self.y = data['y'][0]
        self.celltypes = data['cell_types'].item()

    def load_xy(self, X, y):
        if len(X)==1:
            self.X = norm_image(X[0])
            self.y = y[0]
        else:
            self.X = norm_image(X)
            self.y = y
        self.celltypes = None

    def crop_cells(self):
        global proc_image, find_cell
        global X, y, celltypes 
        X = self.X
        y = self.y
        celltypes = self.celltypes
        
        def find_cell(y_index):
            if celltypes is not None:
                label = celltypes[y_index]
            else:
                label = None
            inds_0, inds_1 = np.where(y[:,:,1]==y_index)
            cx = X[min(inds_0):max(inds_0),min(inds_1):max(inds_1),:]
#             cx = np.pad(tmp_x,((0,100-tmp_x.shape[0]), (0,100-tmp_x.shape[1]), (0,0)), mode='constant')
            cy = y_index
            return (cx, cy, label)
        
        with mp.Manager() as manager:
            with manager.Pool(10) as pool:
                unq_indices = np.unique(y[:,:,1])
                unq_indices = np.delete(unq_indices, np.where(unq_indices==0))
                result = pool.map(find_cell, [j for j in unq_indices])
        self.cell_x = [res[0] for res in result]
        self.cell_mask = [res[1] for res in result]
        self.cell_y = [res[2] for res in result]

    def create_marker_expression(self):
        ## loop over all cells to get mean marker expression
        marker_exp = np.zeros((len(self.cell_x),51,17))
        print(marker_exp.shape)
        for i,cell in enumerate(self.cell_x):
            for j in range(51):
                marker_exp[i,j,self.cell_y[i]-1] = np.mean(cell[:,:,j])
        mep = np.mean(marker_exp,axis=0)
        return mep

class Model:
    def __init__(self,inp_shape, num_classes):
        self.inp_shape = inp_shape
        self.num_classes = num_classes
        
    def build_model(self):
        if self.num_classes == 1:
            act = 'sigmoid'
        else:
            act = 'softmax'
        print(act)
        inp = Input(shape=self.inp_shape)
        x = Conv2D(filters=32,kernel_size=(3,3),activation='relu')(inp)
#         x = Dropout(0.1)(x)
        x = MaxPooling2D((2,2))(x)
        x = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(x)
#         x = Dropout(0.1)(x)
        x = MaxPooling2D((2,2))(x)
#         x = Conv2D(filters=32,kernel_size=(3,3),activation='relu')(x)
# #         x = Dropout(0.1)(x)
#         x = MaxPooling2D((2,2))(x)
        x = Flatten()(x)
        x = Dense(256,activation='relu')(x)
#         x = Dropout(0.1)(x)
        x = Dense(64,activation='relu')(x)
        out = Dense(self.num_classes,activation=act)(x)
        self.model = keras.Model(inputs=inp,outputs=out)
        
    def compile_model(self,opt='adam',lr=1e-3,momentum=0.9):
        if opt=='adam':
            opt_choice = optimizers.Adam(lr=lr)
        elif opt=='sgd':
            opt_choice = optimizers.SGD(lr=lr,momentum=momentum)
        elif opt=='adagrad':
            opt_choice = optimizers.Adagrad(lr=lr)
        else:
            opt_choice = 'adam'
        
        if self.num_classes == 1:
            loss_choice = 'binary_crossentropy'
        else:
            loss_choice = 'categorical_crossentropy'
        print(loss_choice)
        self.model.compile(optimizer=opt_choice,loss=loss_choice,
                           metrics=['accuracy'])
        
def train_model(model, xtrain, ytrain, epochs=30, batch_size=32):        
    earlystop_callback = keras.callbacks.EarlyStopping(monitor='loss',patience=10)
    model.fit(xtrain, ytrain, validation_split=0.3, 
                epochs=epochs, batch_size=batch_size,callbacks = [earlystop_callback], verbose=2)
    return model
        

def prep_samples(cell_x, cell_y, sel_channels, cell_height=50, cell_width=50):
    ## Using resize with pad instead of padding to a bounding box as it nicely samples and enhances the images
    padded_x = []
    for cell in cell_x:
        padded_x.append(tf.image.resize_with_pad(cell[:,:,sel_channels],target_height=cell_height,
                                                 target_width=cell_width))
#     xtrain, xtest, ytrain, ytest = train_test_split(padded_x, cell_y, shuffle=True, test_size=0.2,
#                                                    random_state=1)
    xtrain = tf.convert_to_tensor(padded_x)
#     xtest = tf.convert_to_tensor(xtest)
    ytrain = to_categorical(np.array(cell_y).astype(int)-1,num_classes=17)
#   ytest = to_categorical(np.array(ytest).astype(int)-1,num_classes=17)
    return xtrain, ytrain
    
##############################################################################################################
##############################################################################################################

# Function to process test data and run it through model

def test_model(X,y,cell_height=40,cell_width=40):
    cd = CellData()
    print('Loading data')
    cd.load_xy(X,y)
    print('Cropping cells')
    cd.crop_cells()
    meta = yaml.safe_load(open('keren/meta.yaml','r'))
    sel_channels = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
                1,1,0,1,0,0,0,1,0]).astype(bool)
    print('Processing data')
    padded_x = []
    for cell in cd.cell_x:
        padded_x.append(tf.image.resize_with_pad(cell[:,:,sel_channels],target_height=cell_height,
                                                 target_width=cell_width))
    xtest = tf.convert_to_tensor(padded_x)
    
    print('Loading model')
    subprocess.call(f'wget https://sites.astro.caltech.edu/~yssharma/cellmodel.tar.xz',shell=True)
    subprocess.call(f'tar -xvf cellmodel.tar.xz',shell=True)
    model = keras.models.load_model('cellmodel',compile=True)
    print('Making predictions')
    preds = np.argmax(model.predict(xtest),axis=-1)
    result = {}
    for i in range(len(preds)):
        result[cd.cell_mask[i]] = preds[i]+1
    return result

###############################################################################################################
###############################################################################################################

meta = yaml.safe_load(open('keren/meta.yaml','r'))
global sel_channels
sel_channels = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
                1,1,0,1,0,0,0,1,0]).astype(bool)
print('Selected channels are ', [x for x in np.array(meta['channels'])[sel_channels]])


