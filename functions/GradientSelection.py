import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


import os
import csv
import umap
import sys
from sklearn.manifold import TSNE
from skimage.io import imsave,imread
from skimage.transform import resize
import glob
import numpy as np
import math

import tensorflow as tf
from keras.layers import Input, BatchNormalization, LeakyReLU, Conv2D, Flatten, Dense, Reshape, Lambda, Conv2DTranspose, concatenate, Concatenate
from keras.regularizers import l2
from keras import optimizers
from keras import metrics
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping
from keras import callbacks as cbks

from clr_callback import CyclicLR
from vae_callback import VAEcallback
from numpydatagenerator import NumpyDataGenerator
from coordplot import CoordPlot
from walk_manifold import WalkPrincipalManifold
#import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['HDF5_USE_FILE_LOCKING']='FALSE' 



###Build Class

class ImageVAE():
    """ 2-dimensional variational autoencoder for latent phenotype capture
    """
    
    def __init__(self):
        """ initialize model with argument parameters and build
        """
        self.chIndex        = 1
        self.data_dir       = '../Inputs/'
        self.save_dir       = '../Outputs/'  
        
        self.use_vaecb      = 1
        self.do_vaecb_each  = 0
        self.use_clr        = 1
        self.earlystop 		= 1
        
        self.latent_dim     = 5
        self.nlayers        = 3
        self.inter_dim      = 64
        self.kernel_size    = 3
        self.batch_size     = 50
        self.epochs         = 1
        self.nfilters       = 16
        self.learn_rate     = 0.001
        
        self.epsilon_std    = 1
        
        self.latent_samp    = 10
        self.num_save       = 8
        
        self.do_tsne        = 0
        self.verbose        = 1
        self.phase          = 'load'
        self.steps_per_epoch = 0
        
        self.data_size = len(os.listdir(self.data_dir))
        self.file_names = os.listdir(self.data_dir)
        
        self.image_size     = 32  # infer?
        self.nchannel       = 25
        self.image_res      = 8
        self.show_channels  = [0,1,2]
        
        if self.steps_per_epoch == 0:
            self.steps_per_epoch = self.data_size // self.batch_size
                
        self.build_model()
        
    
    def sampling(self, sample_args):
        """ sample latent layer from normal prior
        """
        
        z_mean, z_log_var = sample_args
        
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0],
                                         self.latent_dim),
                                  mean=0,
                                  stddev=self.epsilon_std)
    
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    
    def build_model(self):
        """ build VAE model
        """
        def conv_block(x, filters, leaky=True, transpose=False, name=''):
            conv = Conv2DTranspose if transpose else Conv2D
            activation = LeakyReLU(0.2) if leaky else Activation('relu')
            layers = [
                conv(filters, 5, strides=2, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name=name + 'conv'),
            BatchNormalization(momentum=0.9, epsilon=1e-6, name=name + 'bn'),
            activation
            ]
            if x is None:
                return layers
            for layer in layers:
                x = layer(x)
            return x
        
        def set_trainable(model, trainable):
            model.trainable = trainable
            for layer in model.layers:
                layer.trainable = trainable
        
        input_shape = (self.image_size, self.image_size, 1)
        ###-------------------   1   ---------------------------------###
        # build encoder1 model
        inputs1 = Input(shape=input_shape, name='encoder_input1')
        self.inputs1=inputs1
        x1 = inputs1
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x1 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,
                       padding='same')(x1)
        # shape info needed to build decoder model
        shape = K.int_shape(x1)
        # generate latent vector Q(z|X)
        x1 = Flatten()(x1)
        x1 = Dense(self.inter_dim, activation='relu')(x1)
        z1_mean = Dense(self.latent_dim, name='z_mean')(x1)
        z1_log_var = Dense(self.latent_dim, name='z_log_var')(x1)
        # use reparameterization trick to push the sampling out as input
        z1 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z1')([z1_mean, z1_log_var])
        
        ###-------------------   2   ---------------------------------### 
        # build encoder2 model
        inputs2 = Input(shape=input_shape, name='encoder_input2')
        self.inputs2=inputs2
        x2 = inputs2
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x2 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x2)
        x2 = Flatten()(x2)
        x2 = Dense(self.inter_dim, activation='relu')(x2)
        z2_mean = Dense(self.latent_dim, name='z_mean')(x2)
        z2_log_var = Dense(self.latent_dim, name='z_log_var')(x2)
        # use reparameterization trick to push the sampling out as input
        z2 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z2')([z2_mean, z2_log_var])
        
        ###-------------------   3   ---------------------------------### 
        # build encoder3 model
        inputs3 = Input(shape=input_shape, name='encoder_input3')
        self.inputs3=inputs3
        x3 = inputs3
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x3 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x3)
        x3 = Flatten()(x3)
        x3 = Dense(self.inter_dim, activation='relu')(x3)
        z3_mean = Dense(self.latent_dim, name='z_mean')(x3)
        z3_log_var = Dense(self.latent_dim, name='z_log_var')(x3)
        # use reparameterization trick to push the sampling out as input
        z3 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z3')([z3_mean, z3_log_var])
        
        ###-------------------   4   ---------------------------------### 
        # build encoder3 model
        inputs4 = Input(shape=input_shape, name='encoder_input4')
        self.inputs4=inputs4
        x4 = inputs4
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x4 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x4)
        x4 = Flatten()(x4)
        x4 = Dense(self.inter_dim, activation='relu')(x4)
        z4_mean = Dense(self.latent_dim, name='z_mean')(x4)
        z4_log_var = Dense(self.latent_dim, name='z_log_var')(x4)
        # use reparameterization trick to push the sampling out as input
        z4 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z4')([z4_mean, z4_log_var])
        
        ###-------------------   5   ---------------------------------### 
        # build encoder3 model
        inputs5 = Input(shape=input_shape, name='encoder_input5')
        self.inputs5=inputs5
        x5 = inputs5
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x5 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x5)
        x5 = Flatten()(x5)
        x5 = Dense(self.inter_dim, activation='relu')(x5)
        z5_mean = Dense(self.latent_dim, name='z_mean')(x5)
        z5_log_var = Dense(self.latent_dim, name='z_log_var')(x5)
        # use reparameterization trick to push the sampling out as input
        z5 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z5')([z5_mean, z5_log_var])
        
        ###-------------------   6   ---------------------------------### 
        # build encoder3 model
        inputs6 = Input(shape=input_shape, name='encoder_input6')
        self.inputs6=inputs6
        x6 = inputs6
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x6 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x6)
        x6 = Flatten()(x6)
        x6 = Dense(self.inter_dim, activation='relu')(x6)
        z6_mean = Dense(self.latent_dim, name='z_mean')(x6)
        z6_log_var = Dense(self.latent_dim, name='z_log_var')(x6)
        # use reparameterization trick to push the sampling out as input
        z6 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z6')([z6_mean, z6_log_var])
        
        ###-------------------   7   ---------------------------------### 
        # build encoder3 model
        inputs7 = Input(shape=input_shape, name='encoder_input7')
        self.inputs7=inputs7
        x7 = inputs7
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x7 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x7)
        x7 = Flatten()(x7)
        x7 = Dense(self.inter_dim, activation='relu')(x7)
        z7_mean = Dense(self.latent_dim, name='z_mean')(x7)
        z7_log_var = Dense(self.latent_dim, name='z_log_var')(x7)
        # use reparameterization trick to push the sampling out as input
        z7 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z7')([z7_mean, z7_log_var])
        
        ###-------------------   8   ---------------------------------### 
        # build encoder3 model
        inputs8 = Input(shape=input_shape, name='encoder_input8')
        self.inputs8=inputs8
        x8 = inputs8
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x8 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x8)
        x8 = Flatten()(x8)
        x8 = Dense(self.inter_dim, activation='relu')(x8)
        z8_mean = Dense(self.latent_dim, name='z_mean')(x8)
        z8_log_var = Dense(self.latent_dim, name='z_log_var')(x8)
        # use reparameterization trick to push the sampling out as input
        z8 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z8')([z8_mean, z8_log_var])
        
        ###-------------------   9   ---------------------------------### 
        # build encoder3 model
        inputs9 = Input(shape=input_shape, name='encoder_input9')
        self.inputs9=inputs9
        x9 = inputs9
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x9 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x9)
        x9 = Flatten()(x9)
        x9 = Dense(self.inter_dim, activation='relu')(x9)
        z9_mean = Dense(self.latent_dim, name='z_mean')(x9)
        z9_log_var = Dense(self.latent_dim, name='z_log_var')(x9)
        # use reparameterization trick to push the sampling out as input
        z9 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z9')([z9_mean, z9_log_var])
        
        ###-------------------   10   ---------------------------------### 
        # build encoder3 model
        inputs10 = Input(shape=input_shape, name='encoder_input10')
        self.inputs10=inputs10
        x10 = inputs10
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x10 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x10)
        x10 = Flatten()(x10)
        x10 = Dense(self.inter_dim, activation='relu')(x10)
        z10_mean = Dense(self.latent_dim, name='z_mean')(x10)
        z10_log_var = Dense(self.latent_dim, name='z_log_var')(x10)
        # use reparameterization trick to push the sampling out as input
        z10 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z10')([z10_mean, z10_log_var])
        
        ###-------------------   11   ---------------------------------### 
        # build encoder3 model
        inputs11 = Input(shape=input_shape, name='encoder_input11')
        self.inputs11=inputs11
        x11 = inputs11
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x11 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x11)
        x11 = Flatten()(x11)
        x11 = Dense(self.inter_dim, activation='relu')(x11)
        z11_mean = Dense(self.latent_dim, name='z_mean')(x11)
        z11_log_var = Dense(self.latent_dim, name='z_log_var')(x11)
        # use reparameterization trick to push the sampling out as input
        z11 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z11')([z11_mean, z11_log_var])
        
        ###-------------------   12   ---------------------------------### 
        # build encoder3 model
        inputs12 = Input(shape=input_shape, name='encoder_input12')
        self.inputs12=inputs12
        x12 = inputs12
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x12 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x12)
        x12 = Flatten()(x12)
        x12 = Dense(self.inter_dim, activation='relu')(x12)
        z12_mean = Dense(self.latent_dim, name='z_mean')(x12)
        z12_log_var = Dense(self.latent_dim, name='z_log_var')(x12)
        # use reparameterization trick to push the sampling out as input
        z12 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z12')([z12_mean, z12_log_var])
        
        ###-------------------   13   ---------------------------------### 
        # build encoder3 model
        inputs13 = Input(shape=input_shape, name='encoder_input13')
        self.inputs13=inputs13
        x13 = inputs13
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x13 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x13)
        x13 = Flatten()(x13)
        x13 = Dense(self.inter_dim, activation='relu')(x13)
        z13_mean = Dense(self.latent_dim, name='z_mean')(x13)
        z13_log_var = Dense(self.latent_dim, name='z_log_var')(x13)
        # use reparameterization trick to push the sampling out as input
        z13 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z13')([z13_mean, z13_log_var])
        
        ###-------------------   14   ---------------------------------### 
        # build encoder3 model
        inputs14 = Input(shape=input_shape, name='encoder_input14')
        self.inputs14=inputs14
        x14 = inputs14
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x14 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x14)
        x14 = Flatten()(x14)
        x14 = Dense(self.inter_dim, activation='relu')(x14)
        z14_mean = Dense(self.latent_dim, name='z_mean')(x14)
        z14_log_var = Dense(self.latent_dim, name='z_log_var')(x14)
        # use reparameterization trick to push the sampling out as input
        z14 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z14')([z14_mean, z14_log_var])
        
        ###-------------------   15   ---------------------------------### 
        # build encoder3 model
        inputs15 = Input(shape=input_shape, name='encoder_input15')
        self.inputs15=inputs15
        x15 = inputs15
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x15 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x15)
        x15 = Flatten()(x15)
        x15 = Dense(self.inter_dim, activation='relu')(x15)
        z15_mean = Dense(self.latent_dim, name='z_mean')(x15)
        z15_log_var = Dense(self.latent_dim, name='z_log_var')(x15)
        # use reparameterization trick to push the sampling out as input
        z15 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z15')([z15_mean, z15_log_var])
        
        ###-------------------   16   ---------------------------------### 
        # build encoder3 model
        inputs16 = Input(shape=input_shape, name='encoder_input16')
        self.inputs16=inputs16
        x16 = inputs16
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x16 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x16)
        x16 = Flatten()(x16)
        x16 = Dense(self.inter_dim, activation='relu')(x16)
        z16_mean = Dense(self.latent_dim, name='z_mean')(x16)
        z16_log_var = Dense(self.latent_dim, name='z_log_var')(x16)
        # use reparameterization trick to push the sampling out as input
        z16 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z16')([z16_mean, z16_log_var])
        
        ###-------------------   17   ---------------------------------### 
        # build encoder3 model
        inputs17 = Input(shape=input_shape, name='encoder_input17')
        self.inputs17=inputs17
        x17 = inputs17
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x17 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x17)
        x17 = Flatten()(x17)
        x17 = Dense(self.inter_dim, activation='relu')(x17)
        z17_mean = Dense(self.latent_dim, name='z_mean')(x17)
        z17_log_var = Dense(self.latent_dim, name='z_log_var')(x17)
        # use reparameterization trick to push the sampling out as input
        z17 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z17')([z17_mean, z17_log_var])
        
        ###-------------------   18   ---------------------------------### 
        # build encoder3 model
        inputs18 = Input(shape=input_shape, name='encoder_input18')
        self.inputs18=inputs18
        x18 = inputs18
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x18 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x18)
        x18 = Flatten()(x18)
        x18 = Dense(self.inter_dim, activation='relu')(x18)
        z18_mean = Dense(self.latent_dim, name='z_mean')(x18)
        z18_log_var = Dense(self.latent_dim, name='z_log_var')(x18)
        # use reparameterization trick to push the sampling out as input
        z18 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z18')([z18_mean, z18_log_var])
        
        ###-------------------   19   ---------------------------------### 
        # build encoder3 model
        inputs19 = Input(shape=input_shape, name='encoder_input19')
        self.inputs19=inputs19
        x19 = inputs19
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x19 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x19)
        x19 = Flatten()(x19)
        x19 = Dense(self.inter_dim, activation='relu')(x19)
        z19_mean = Dense(self.latent_dim, name='z_mean')(x19)
        z19_log_var = Dense(self.latent_dim, name='z_log_var')(x19)
        # use reparameterization trick to push the sampling out as input
        z19 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z19')([z19_mean, z19_log_var])
        
        ###-------------------   20   ---------------------------------### 
        # build encoder3 model
        inputs20 = Input(shape=input_shape, name='encoder_input20')
        self.inputs20=inputs20
        x20 = inputs20
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x20 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x20)
        x20 = Flatten()(x20)
        x20 = Dense(self.inter_dim, activation='relu')(x20)
        z20_mean = Dense(self.latent_dim, name='z_mean')(x20)
        z20_log_var = Dense(self.latent_dim, name='z_log_var')(x20)
        # use reparameterization trick to push the sampling out as input
        z20 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z20')([z20_mean, z20_log_var])
        
        ###-------------------   21   ---------------------------------### 
        # build encoder3 model
        inputs21 = Input(shape=input_shape, name='encoder_input21')
        self.inputs21=inputs21
        x21 = inputs21
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x21 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x21)
        x21 = Flatten()(x21)
        x21 = Dense(self.inter_dim, activation='relu')(x21)
        z21_mean = Dense(self.latent_dim, name='z_mean')(x21)
        z21_log_var = Dense(self.latent_dim, name='z_log_var')(x21)
        # use reparameterization trick to push the sampling out as input
        z21 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z21')([z21_mean, z21_log_var])
        
        ###-------------------   22   ---------------------------------### 
        # build encoder3 model
        inputs22 = Input(shape=input_shape, name='encoder_input22')
        self.inputs22=inputs22
        x22 = inputs22
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x22 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x22)
        x22 = Flatten()(x22)
        x22 = Dense(self.inter_dim, activation='relu')(x22)
        z22_mean = Dense(self.latent_dim, name='z_mean')(x22)
        z22_log_var = Dense(self.latent_dim, name='z_log_var')(x22)
        # use reparameterization trick to push the sampling out as input
        z22 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z22')([z22_mean, z22_log_var])
        
        ###-------------------   23   ---------------------------------### 
        # build encoder3 model
        inputs23 = Input(shape=input_shape, name='encoder_input23')
        self.inputs23=inputs23
        x23 = inputs23
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x23 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x23)
        x23 = Flatten()(x23)
        x23 = Dense(self.inter_dim, activation='relu')(x23)
        z23_mean = Dense(self.latent_dim, name='z_mean')(x23)
        z23_log_var = Dense(self.latent_dim, name='z_log_var')(x23)
        # use reparameterization trick to push the sampling out as input
        z23 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z23')([z23_mean, z23_log_var])
        
        ###-------------------   24   ---------------------------------### 
        # build encoder3 model
        inputs24 = Input(shape=input_shape, name='encoder_input24')
        self.inputs24=inputs24
        x24 = inputs24
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x24 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x24)
        x24 = Flatten()(x24)
        x24 = Dense(self.inter_dim, activation='relu')(x24)
        z24_mean = Dense(self.latent_dim, name='z_mean')(x24)
        z24_log_var = Dense(self.latent_dim, name='z_log_var')(x24)
        # use reparameterization trick to push the sampling out as input
        z24 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z24')([z24_mean, z24_log_var])
        
        ###-------------------   25   ---------------------------------### 
        # build encoder3 model
        inputs25 = Input(shape=input_shape, name='encoder_input25')
        self.inputs25=inputs25
        x25 = inputs25
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            filters *= 2
            x25 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=1,         
                       padding='same')(x25)
        x25 = Flatten()(x25)
        x25 = Dense(self.inter_dim, activation='relu')(x25)
        z25_mean = Dense(self.latent_dim, name='z_mean')(x25)
        z25_log_var = Dense(self.latent_dim, name='z_log_var')(x25)
        # use reparameterization trick to push the sampling out as input
        z25 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z25')([z25_mean, z25_log_var])
        
        
        ###Add Encoders
        
        ###------------------------   Decoder ---------------------------------------------------------------###
        
        # build decoder model
        latent_inputs1 = Input(shape=(self.latent_dim*self.nchannel,), name='z_sampling')
        d1 = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs1)
        d1 = Reshape((shape[1], shape[2], shape[3]))(d1)
        
        for i in range(self.nlayers):
            d1 = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                activation='relu',
                                strides=1,
                                padding='same')(d1)
            filters //= 2
        
        
        outputs = Conv2DTranspose(filters=self.nchannel,
                                  kernel_size=kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(d1)
        
        
        ###------------------------   Discriminator ----------------------------------------------------------###
        
        dis_inputs = Input(shape=(self.image_size, self.image_size, self.nchannel), name='dis_input')
        z_p = Input(shape=(self.latent_dim*self.nchannel,), name='z_p_input')
        
        layers = [
        Conv2D(32, 5, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='dis_blk_1_conv'),
        LeakyReLU(0.2),
            *conv_block(None, 128, leaky=True, name='dis_blk_2_'),
            *conv_block(None, 256, leaky=True, name='dis_blk_3_'),
            *conv_block(None, 256, leaky=True, name='dis_blk_4_'),
            Flatten(),
            Dense(512, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='dis_dense'),
            BatchNormalization(name='dis_bn'),
            LeakyReLU(0.2),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='dis_output')
        ]
        
        disy = dis_inputs
        disy_feat = None
        for i, layer in enumerate(layers, 1):
            disy = layer(disy)
            # Output the features at the specified depth
            if i == 9:#Depth of model
                disy_feat = disy
            
       ###-------------------   Connect Models   ---------------------------------###
        # instantiate encoder1 model
        self.encoder1 = Model(inputs1, [z1_mean, z1_log_var, z1], name='encoder1')
        self.encoder1.summary()
        plot_model(self.encoder1, to_file=os.path.join(self.save_dir, 'encoder1_model.png'), show_shapes=True)
        
        # instantiate encoder2 model
        self.encoder2 = Model(inputs2, [z2_mean, z2_log_var, z2], name='encoder2')
        self.encoder2.summary()
        plot_model(self.encoder2, to_file=os.path.join(self.save_dir, 'encoder2_model.png'), show_shapes=True)
        
        # instantiate encoder3 model
        self.encoder3 = Model(inputs3, [z3_mean, z3_log_var, z3], name='encoder3')
        self.encoder3.summary()
        plot_model(self.encoder3, to_file=os.path.join(self.save_dir, 'encoder3_model.png'), show_shapes=True)
        
        # instantiate encoder4 model
        self.encoder4 = Model(inputs4, [z4_mean, z4_log_var, z4], name='encoder4')
        self.encoder4.summary()
        plot_model(self.encoder4, to_file=os.path.join(self.save_dir, 'encoder4_model.png'), show_shapes=True)
        
        # instantiate encoder5 model
        self.encoder5 = Model(inputs5, [z5_mean, z5_log_var, z5], name='encoder5')
        self.encoder5.summary()
        plot_model(self.encoder5, to_file=os.path.join(self.save_dir, 'encoder5_model.png'), show_shapes=True)
        
        # instantiate encoder6 model
        self.encoder6 = Model(inputs6, [z6_mean, z6_log_var, z6], name='encoder6')
        self.encoder6.summary()
        plot_model(self.encoder6, to_file=os.path.join(self.save_dir, 'encoder6_model.png'), show_shapes=True)
        
        # instantiate encoder7 model
        self.encoder7 = Model(inputs7, [z7_mean, z7_log_var, z7], name='encoder7')
        self.encoder7.summary()
        plot_model(self.encoder7, to_file=os.path.join(self.save_dir, 'encoder7_model.png'), show_shapes=True)
        
        # instantiate encoder8 model
        self.encoder8 = Model(inputs8, [z8_mean, z8_log_var, z8], name='encoder8')
        self.encoder8.summary()
        plot_model(self.encoder8, to_file=os.path.join(self.save_dir, 'encoder8_model.png'), show_shapes=True)
        
         # instantiate encoder9 model
        self.encoder9 = Model(inputs9, [z9_mean, z9_log_var, z9], name='encoder9')
        self.encoder9.summary()
        plot_model(self.encoder9, to_file=os.path.join(self.save_dir, 'encoder9_model.png'), show_shapes=True)
        
        # instantiate encoder10 model
        self.encoder10 = Model(inputs10, [z10_mean, z10_log_var, z10], name='encoder10')
        self.encoder10.summary()
        plot_model(self.encoder10, to_file=os.path.join(self.save_dir, 'encoder10_model.png'), show_shapes=True)
        
        # instantiate encoder11 model
        self.encoder11 = Model(inputs11, [z11_mean, z11_log_var, z11], name='encoder11')
        self.encoder11.summary()
        plot_model(self.encoder11, to_file=os.path.join(self.save_dir, 'encoder11_model.png'), show_shapes=True)
        
        # instantiate encoder12 model
        self.encoder12 = Model(inputs12, [z12_mean, z12_log_var, z12], name='encoder12')
        self.encoder12.summary()
        plot_model(self.encoder12, to_file=os.path.join(self.save_dir, 'encoder12_model.png'), show_shapes=True)
        
        # instantiate encoder13 model
        self.encoder13 = Model(inputs13, [z13_mean, z13_log_var, z13], name='encoder13')
        self.encoder13.summary()
        plot_model(self.encoder13, to_file=os.path.join(self.save_dir, 'encoder13_model.png'), show_shapes=True)
        
        # instantiate encoder14 model
        self.encoder14 = Model(inputs14, [z14_mean, z14_log_var, z14], name='encoder14')
        self.encoder14.summary()
        plot_model(self.encoder14, to_file=os.path.join(self.save_dir, 'encoder14_model.png'), show_shapes=True)
        
        # instantiate encoder15 model
        self.encoder15 = Model(inputs15, [z15_mean, z15_log_var, z15], name='encoder15')
        self.encoder15.summary()
        plot_model(self.encoder15, to_file=os.path.join(self.save_dir, 'encoder15_model.png'), show_shapes=True)
        
        # instantiate encoder16 model
        self.encoder16 = Model(inputs16, [z16_mean, z16_log_var, z16], name='encoder16')
        self.encoder16.summary()
        plot_model(self.encoder16, to_file=os.path.join(self.save_dir, 'encoder16_model.png'), show_shapes=True)
        
        # instantiate encoder17 model
        self.encoder17 = Model(inputs17, [z17_mean, z17_log_var, z17], name='encoder17')
        self.encoder17.summary()
        plot_model(self.encoder17, to_file=os.path.join(self.save_dir, 'encoder17_model.png'), show_shapes=True)
        
        # instantiate encoder18 model
        self.encoder18 = Model(inputs18, [z18_mean, z18_log_var, z18], name='encoder18')
        self.encoder18.summary()
        plot_model(self.encoder18, to_file=os.path.join(self.save_dir, 'encoder18_model.png'), show_shapes=True)
        
        # instantiate encoder19 model
        self.encoder19 = Model(inputs19, [z19_mean, z19_log_var, z19], name='encoder19')
        self.encoder19.summary()
        plot_model(self.encoder19, to_file=os.path.join(self.save_dir, 'encoder19_model.png'), show_shapes=True)
        
        # instantiate encoder20 model
        self.encoder20 = Model(inputs20, [z20_mean, z20_log_var, z20], name='encoder20')
        self.encoder20.summary()
        plot_model(self.encoder20, to_file=os.path.join(self.save_dir, 'encoder20_model.png'), show_shapes=True)
        
        # instantiate encoder21 model
        self.encoder21 = Model(inputs21, [z21_mean, z21_log_var, z21], name='encoder21')
        self.encoder21.summary()
        plot_model(self.encoder21, to_file=os.path.join(self.save_dir, 'encoder21_model.png'), show_shapes=True)
        
        # instantiate encoder22 model
        self.encoder22 = Model(inputs22, [z22_mean, z22_log_var, z22], name='encoder22')
        self.encoder22.summary()
        plot_model(self.encoder22, to_file=os.path.join(self.save_dir, 'encoder22_model.png'), show_shapes=True)
        
        # instantiate encoder23 model
        self.encoder23 = Model(inputs23, [z23_mean, z23_log_var, z23], name='encoder23')
        self.encoder23.summary()
        plot_model(self.encoder23, to_file=os.path.join(self.save_dir, 'encoder23_model.png'), show_shapes=True)
        
        # instantiate encoder24 model
        self.encoder24 = Model(inputs24, [z24_mean, z24_log_var, z24], name='encoder24')
        self.encoder24.summary()
        plot_model(self.encoder24, to_file=os.path.join(self.save_dir, 'encoder24_model.png'), show_shapes=True)
        
        # instantiate encoder25 model
        self.encoder25 = Model(inputs25, [z25_mean, z25_log_var, z25], name='encoder25')
        self.encoder25.summary()
        plot_model(self.encoder25, to_file=os.path.join(self.save_dir, 'encoder25_model.png'), show_shapes=True)
        
        ###Add Encoders Here
        
        # instantiate decoder1 model
        self.decoder1 = Model(latent_inputs1, outputs, name='decoder1')
        self.decoder1.summary()
        plot_model(self.decoder1, to_file=os.path.join(self.save_dir, 'decoder1_model.png'), show_shapes=True)

        # instantiate VAE model #####Add per Encoder
        outputs1 = self.decoder1(concatenate([  self.encoder1(inputs1)[2], self.encoder2(inputs2)[2], self.encoder3(inputs3)[2], self.encoder4(inputs4)[2],  self.encoder5(inputs5)[2], self.encoder6(inputs6)[2], self.encoder7(inputs7)[2], self.encoder8(inputs8)[2], self.encoder9(inputs9)[2], self.encoder10(inputs10)[2], self.encoder11(inputs11)[2], self.encoder12(inputs12)[2], self.encoder13(inputs13)[2], self.encoder14(inputs14)[2], self.encoder15(inputs15)[2], self.encoder16(inputs16)[2], self.encoder17(inputs17)[2], self.encoder18(inputs18)[2], self.encoder19(inputs19)[2], self.encoder20(inputs20)[2],  self.encoder21(inputs21)[2], self.encoder22(inputs22)[2], self.encoder23(inputs23)[2], self.encoder24(inputs24)[2], self.encoder25(inputs25)[2]  ]))
                 
        ## instantiate discriminator model
        self.discriminator1 = Model(dis_inputs, [disy,disy_feat], name='discriminator1')
        self.discriminator1.summary()
        plot_model(self.discriminator1, to_file=os.path.join(self.save_dir, 'discriminator1_model.png'), show_shapes=True)
            
        # instantiate discriminator_train model
        dis_x, dis_feat = self.discriminator1(dis_inputs)
        dis_x_tilde, dis_feat_tilde = self.discriminator1(outputs1)
        dis_x_p = self.discriminator1(self.decoder1(z_p))[0]
        
        self.discriminator1_train = Model([dis_inputs, z_p, inputs1,inputs2,inputs3,inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10, inputs11, inputs12, inputs13, inputs14, inputs15, inputs16, inputs17, inputs18, inputs19, inputs20, inputs21, inputs22, inputs23, inputs24, inputs25], [dis_x,dis_x_tilde,dis_x_p], name='discriminator1_train')  
            
        #instantiate whole vae
        self.vae = Model(inputs=[dis_inputs,z_p,inputs1,inputs2,inputs3,inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10, inputs11, inputs12, inputs13, inputs14, inputs15, inputs16, inputs17, inputs18, inputs19, inputs20, inputs21, inputs22, inputs23, inputs24, inputs25], outputs=[outputs1], name='vae')

        #instantiate whole VAEGan
        
        self.vaegan = Model(inputs=[inputs1,inputs2,inputs3,inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10, inputs11, inputs12, inputs13, inputs14, inputs15, inputs16, inputs17, inputs18, inputs19, inputs20, inputs21, inputs22, inputs23, inputs24, inputs25], outputs=[dis_x_tilde],name='vaegan')
        
        #   VAE loss terms w/ KL divergence            
        def Decoder1Loss(true, pred,zmean,zlog):
            xent_loss = metrics.binary_crossentropy(K.flatten(true), K.flatten(pred))
            xent_loss *= self.image_size * self.image_size
            kl_loss = 1 + zlog * 2 - K.square(zmean) - K.exp(zlog * 2)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5  
            vae_loss = K.mean(xent_loss + kl_loss)
            return vae_loss/2
             
        def mean_gaussian_negative_log_likelihood(y_true, y_pred):
            nll = 0.5 * np.log(2 * np.pi) + 0.5 * K.square(y_pred - y_true)
            axis = tuple(range(1, len(K.int_shape(y_true))))
            return K.abs(K.mean(K.sum(nll, axis=axis), axis=-1))
            
        def ssim_loss(y_true, y_pred):
            return K.abs(1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0)))*100
        
        def LogLikelihoodLoss(true,pred):
            #Comment
            return mean_gaussian_negative_log_likelihood(dis_feat, dis_feat_tilde)
        
        def BinaryLoss(y_true,y_pred):
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  # Cancels out when target is 1 
            term_1 = y_true * K.log(y_pred + K.epsilon()) # Cancels out when target is 0
            return -K.mean(term_0 + term_1, axis=1)
        
        def CombLoss(true,pred): ###########Add Per Encoder########s
            l1=Decoder1Loss(true, pred, z1_mean, z1_log_var)
            l2=Decoder1Loss(true, pred, z2_mean, z2_log_var)
            l3=Decoder1Loss(true, pred, z3_mean, z3_log_var)
            l4=Decoder1Loss(true, pred, z4_mean, z4_log_var)
            l5=Decoder1Loss(true, pred, z5_mean, z5_log_var)
            l6=Decoder1Loss(true, pred, z6_mean, z6_log_var)
            l7=Decoder1Loss(true, pred, z7_mean, z7_log_var)
            l8=Decoder1Loss(true, pred, z8_mean, z8_log_var)
            l9=Decoder1Loss(true, pred, z9_mean, z9_log_var)
            l10=Decoder1Loss(true, pred, z10_mean, z10_log_var)
            l11=Decoder1Loss(true, pred, z11_mean, z11_log_var)
            l12=Decoder1Loss(true, pred, z12_mean, z12_log_var)
            l13=Decoder1Loss(true, pred, z13_mean, z13_log_var)
            l14=Decoder1Loss(true, pred, z14_mean, z14_log_var)
            l15=Decoder1Loss(true, pred, z15_mean, z15_log_var)
            l16=Decoder1Loss(true, pred, z16_mean, z16_log_var)
            l17=Decoder1Loss(true, pred, z17_mean, z17_log_var)
            l18=Decoder1Loss(true, pred, z18_mean, z18_log_var)
            l19=Decoder1Loss(true, pred, z19_mean, z19_log_var)
            l20=Decoder1Loss(true, pred, z20_mean, z20_log_var)
            l21=Decoder1Loss(true, pred, z21_mean, z21_log_var)
            l22=Decoder1Loss(true, pred, z22_mean, z22_log_var)
            l23=Decoder1Loss(true, pred, z23_mean, z23_log_var)
            l24=Decoder1Loss(true, pred, z24_mean, z24_log_var)
            l25=Decoder1Loss(true, pred, z25_mean, z25_log_var)
            LogLoss=LogLikelihoodLoss(true,pred)
            #BinLoss=BinaryLoss(true,pred)
            SSimLoss=ssim_loss(true,pred)
            return (l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15+l16+l17+l18+l19+l20+l21+l22+l23+l24+l25)/25 + LogLoss/1000 + SSimLoss #+BinLoss
            
        #Binary Cross Entropy Loss
        #bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)       

        optimizer = optimizers.RMSprop(lr = self.learn_rate)    

        losses = {"decoder1": CombLoss}
        #lossWeights = {"decoder1": 1.0,"decoder1": 1.0,"decoder1": 1.0}
            
            

        set_trainable(self.discriminator1, True)
        set_trainable(self.encoder1, False)
        set_trainable(self.encoder2, False)
        set_trainable(self.encoder3, False)
        set_trainable(self.encoder4, False)
        set_trainable(self.encoder5, False)
        set_trainable(self.encoder6, False)
        set_trainable(self.encoder7, False)
        set_trainable(self.encoder8, False)
        set_trainable(self.encoder9, False)
        set_trainable(self.encoder10, False)
        set_trainable(self.encoder11, False)
        set_trainable(self.encoder12, False)
        set_trainable(self.encoder13, False)
        set_trainable(self.encoder14, False)
        set_trainable(self.encoder15, False)
        set_trainable(self.encoder16, False)
        set_trainable(self.encoder17, False)
        set_trainable(self.encoder18, False)
        set_trainable(self.encoder19, False)
        set_trainable(self.encoder20, False)
        set_trainable(self.encoder21, False)
        set_trainable(self.encoder22, False)
        set_trainable(self.encoder23, False)
        set_trainable(self.encoder24, False)
        set_trainable(self.encoder25, False)
        #####Add Encoders Here
        set_trainable(self.decoder1, False)
        
        optimizerDisc = optimizers.RMSprop(lr=0.0001)
        self.discriminator1_train.compile(optimizerDisc, ['binary_crossentropy'] * 3, ['acc'] * 3)
        self.discriminator1_train.summary()
        plot_model(self.discriminator1_train, to_file=os.path.join(self.save_dir, 'discriminator1_train_model.png'), show_shapes=True) 
        
        
        set_trainable(self.discriminator1, False)
        set_trainable(self.encoder1, True)
        set_trainable(self.encoder2, True)
        set_trainable(self.encoder3, True)
        set_trainable(self.encoder4,True)
        set_trainable(self.encoder5, True)
        set_trainable(self.encoder6, True)
        set_trainable(self.encoder7, True)
        set_trainable(self.encoder8, True)
        set_trainable(self.encoder9, True)
        set_trainable(self.encoder10, True)
        set_trainable(self.encoder11, True)
        set_trainable(self.encoder12, True)
        set_trainable(self.encoder13, True)
        set_trainable(self.encoder14, True)
        set_trainable(self.encoder15, True)
        set_trainable(self.encoder16, True)
        set_trainable(self.encoder17, True)
        set_trainable(self.encoder18, True)
        set_trainable(self.encoder19, True)
        set_trainable(self.encoder20, True)
        set_trainable(self.encoder21, True)
        set_trainable(self.encoder22, True)
        set_trainable(self.encoder23, True)
        set_trainable(self.encoder24, True)
        set_trainable(self.encoder25, True)
        #Add Encoders Here
        set_trainable(self.decoder1, True)
        
        
        #self.vae.compile(loss=losses,loss_weights=lossWeights, optimizer=optimizer)
        self.vae.compile(loss=losses, optimizer=optimizer)
        self.vae.summary()       
        plot_model(self.vae, to_file=os.path.join(self.save_dir, 'vae_model.png'), show_shapes=True)
        
        set_trainable(self.vaegan, True)
        #set_trainable(vae, True)
        
        
model = ImageVAE()
model.vae.load_weights('../CellOutputsTest2/models/weights_vae.hdf5')



##Load Inputs Images

imageList=sorted(glob.glob( model.data_dir+'*'))

data1 = []
data2 = []
data3 = [] ### Add Per Encoder
data4 = []
data5 = []
data6 = []
data7 = []
data8 = []
data9 = []
data10 = []
data11 = []
data12 = []
data13 = []
data14 = []
data15 = []
data16 = []
data17 = []
data18 = []
data19 = []
data20 = []
data21 = []
data22 = []
data23 = []
data24 = []
data25 = []
datafull = []
counter=0
while counter < len(imageList):

    image = imread(imageList[rand])

    image=resize(image,(model.image_size,model.image_size,25))#(self.image_size, self.image_size, self.nchannel))

    data1.append(resize(image[:,:,0],(model.image_size,model.image_size,1)))
    data2.append(resize(image[:,:,1],(model.image_size,model.image_size,1)))
    data3.append(resize(image[:,:,2],(model.image_size,model.image_size,1)))
    data4.append(resize(image[:,:,3],(model.image_size,model.image_size,1)))
    data5.append(resize(image[:,:,4],(model.image_size,model.image_size,1)))
    data6.append(resize(image[:,:,5],(model.image_size,model.image_size,1)))
    data7.append(resize(image[:,:,6],(model.image_size,model.image_size,1)))
    data8.append(resize(image[:,:,7],(model.image_size,model.image_size,1)))
    data9.append(resize(image[:,:,8],(model.image_size,model.image_size,1)))
    data10.append(resize(image[:,:,9],(model.image_size,model.image_size,1)))
    data11.append(resize(image[:,:,10],(model.image_size,model.image_size,1)))
    data12.append(resize(image[:,:,11],(model.image_size,model.image_size,1)))
    data13.append(resize(image[:,:,12],(model.image_size,model.image_size,1)))
    data14.append(resize(image[:,:,13],(model.image_size,model.image_size,1)))
    data15.append(resize(image[:,:,14],(model.image_size,model.image_size,1)))
    data16.append(resize(image[:,:,15],(model.image_size,model.image_size,1)))
    data17.append(resize(image[:,:,16],(model.image_size,model.image_size,1)))
    data18.append(resize(image[:,:,17],(model.image_size,model.image_size,1)))
    data19.append(resize(image[:,:,18],(model.image_size,model.image_size,1)))
    data20.append(resize(image[:,:,19],(model.image_size,model.image_size,1)))
    data21.append(resize(image[:,:,20],(model.image_size,model.image_size,1)))
    data22.append(resize(image[:,:,21],(model.image_size,model.image_size,1)))
    data23.append(resize(image[:,:,22],(model.image_size,model.image_size,1)))
    data24.append(resize(image[:,:,23],(model.image_size,model.image_size,1)))
    data25.append(resize(image[:,:,24],(model.image_size,model.image_size,1)))

    datafull.append(resize(image[:,:,0:model.nchannel],(model.image_size,model.image_size,model.nchannel)))
    
    if counter%10000==0:
        print(counter)
    counter+=1
    

model.input1_data = np.array(data1, dtype="float")# / 255.0
model.input2_data = np.array(data2, dtype="float")# / 255.0
model.input3_data = np.array(data3, dtype="float")# / 255.0
model.input4_data = np.array(data4, dtype="float")# / 255.0
model.input5_data = np.array(data5, dtype="float")# / 255.0
model.input6_data = np.array(data6, dtype="float")# / 255.0
model.input7_data = np.array(data7, dtype="float")# / 255.0
model.input8_data = np.array(data8, dtype="float")# / 255.0
model.input9_data = np.array(data9, dtype="float")# / 255.0
model.input10_data = np.array(data10, dtype="float")# / 255.0
model.input11_data = np.array(data11, dtype="float")# / 255.0
model.input12_data = np.array(data12, dtype="float")# / 255.0
model.input13_data = np.array(data13, dtype="float")# / 255.0
model.input14_data = np.array(data14, dtype="float")# / 255.0
model.input15_data = np.array(data15, dtype="float")# / 255.0
model.input16_data = np.array(data16, dtype="float")# / 255.0
model.input17_data = np.array(data17, dtype="float")# / 255.0
model.input18_data = np.array(data18, dtype="float")# / 255.0
model.input19_data = np.array(data19, dtype="float")# / 255.0
model.input20_data = np.array(data20, dtype="float")# / 255.0
model.input21_data = np.array(data21, dtype="float")# / 255.0
model.input22_data = np.array(data22, dtype="float")# / 255.0
model.input23_data = np.array(data23, dtype="float")# / 255.0
model.input24_data = np.array(data24, dtype="float")# / 255.0
model.input25_data = np.array(data25, dtype="float")# / 255.0
model.out1_data = np.array(datafull, dtype="float")# / 255.0




## Run Test Prediction

n=-1
rng = np.random.RandomState(0)
model.z_norm = rng.normal(size=(len(imageList), model.latent_dim*25))
preds=model.vae.predict([model.out1_data[:n],model.z_norm[:n],model.input1_data[:n], model.input2_data[:n],model.input3_data[:n],model.input4_data[:n], 
                   model.input5_data[:n], model.input6_data[:n], model.input7_data[:n], model.input8_data[:n], model.input9_data[:n], model.input10_data[:n],
                   model.input11_data[:n], model.input12_data[:n], model.input13_data[:n], model.input14_data[:n], model.input15_data[:n], model.input16_data[:n],
                   model.input17_data[:n], model.input18_data[:n], model.input19_data[:n], model.input20_data[:n], model.input21_data[:n], model.input22_data[:n], 
                   model.input23_data[:n], model.input24_data[:n], model.input25_data[:n]],verbose=1)


## Calculate Gradients

from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    
    grad_model = tf.keras.models.Model(
        [model.vae.inputs, model.decoder1.inputs], [model.vae.get_layer(last_conv_layer_name).output, model.vae.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    last_conv_layer_output = last_conv_layer_output[0]

    return last_conv_layer_output, grads


n=-1
a=0
out,grads = make_gradcam_heatmap([model.out1_data[a:n],model.z_norm[a:n],model.input1_data[a:n], model.input2_data[a:n],model.input3_data[a:n],model.input4_data[a:n], 
                   model.input5_data[a:n], model.input6_data[a:n], model.input7_data[a:n], model.input8_data[a:n], model.input9_data[a:n], model.input10_data[a:n],
                   model.input11_data[a:n], model.input12_data[a:n], model.input13_data[a:n], model.input14_data[a:n], model.input15_data[a:n], model.input16_data[a:n],
                   model.input17_data[a:n], model.input18_data[a:n], model.input19_data[a:n], model.input20_data[a:n], model.input21_data[a:n], model.input22_data[a:n], 
                   model.input23_data[a:n], model.input24_data[a:n], model.input25_data[a:n]], model, 'concatenate')


gradmean=np.mean(abs(grads.numpy()),axis=0)

gradmeanmean=[]
for i in range(0,25):
    arr=gradmean[i*5:i*5+5]
    mean=np.max(arr)
    gradmeanmean.append(mean)


plt.bar(range(1,26),gradmeanmean)
print(gradmeanmean)

            