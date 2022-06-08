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
import pandas as pd

import tensorflow as tf
from keras.layers import Input, BatchNormalization, LeakyReLU, Conv2D, Flatten, Dense, Reshape, Lambda, Conv2DTranspose, concatenate, Concatenate, Layer
from keras.regularizers import l2
from tensorflow.keras import optimizers
from keras import metrics
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping
from keras import callbacks as cbks
from keras.losses import mse



class LinearCoeff(Layer):
    def __init__(self, batches=50,num_enc=25, lat_dim=5):
        super(LinearCoeff, self).__init__()
        self.num_enc=num_enc
        self.lat_dim=lat_dim
        self.batches=batches
        self.diagonal=np.zeros(self.num_enc)
        self.w = self.add_weight("kernel",
            shape=(num_enc, num_enc), initializer="uniform", trainable=True)

    def call(self, inputs):
        self.inshape=tf.shape(inputs)  
        outputs=tf.reshape(inputs, (self.inshape[0],self.num_enc,self.lat_dim))
        outputs=tf.matmul((tf.linalg.set_diag(self.w,self.diagonal,k=0)), outputs)
        outputs=tf.reshape(outputs, (self.inshape[0],self.num_enc*self.lat_dim))
        self.add_loss(tf.reduce_sum(tf.math.pow(self.w,2.0)))
        return outputs
    
    
class NewLinearCoeff(Layer):
    def __init__(self, num_enc=25, lat_dim=5):
        super(NewLinearCoeff, self).__init__()
        tf.random.set_seed(5)
        self.diagonal=np.zeros(num_enc)
        self.w = self.add_weight(
            shape=(num_enc, num_enc), initializer="random_normal", trainable=True
        )
     
    def call(self, inputs):
        
        rand=tf.random.uniform(shape=[],minval=0,maxval=9, dtype=tf.int32)
        out=tf.matmul(tf.math.abs(tf.linalg.set_diag(self.w,self.diagonal,k=0)),inputs)
        self.add_loss(tf.reduce_sum(tf.math.abs(self.w)))   
        return out


class SubspaceModel():
    
        def __init__(self):
            """ initialize model with argument parameters and build
            """


            direc='SubSpace-FullIntensityEvaluation/'



            self.save_dir       = '/SaveDir/
            self.data_dir       = '/IntensityDir/'
            self.latent_dim     = 1
            self.batch_size     = 50
            self.epochs         = 2
            self.learn_rate     = .001
            self.verbose        = 1
            self.nchannel       = 25

            self.build_model()



        def build_model(self):

                inputsC = Input(shape=(self.nchannel,self.latent_dim,), name='Coeff_inputs')
                linear_layer = NewLinearCoeff(self.nchannel,self.latent_dim)
                self.z_c_out = linear_layer(inputsC)



                self.CoeffModel = Model(inputsC,self.z_c_out, name='Coeff1')
                plot_model(self.CoeffModel, to_file=os.path.join(self.save_dir, 'Coeff1_model.png'), show_shapes=True)



                def CMatLoss(y_true,y_pred):
                    SmallCLoss=tf.reduce_sum(tf.pow(self.Coeff,2.0)/25)
                    SelfExpressionLoss= 0.5 * tf.reduce_sum(tf.pow(tf.subtract(y_true,y_pred),2))
                    return SelfExpressionLoss #+ SmallCLoss

                optimizer = optimizers.RMSprop(lr = self.learn_rate)    
                self.CoeffModel.compile(loss=CMatLoss,optimizer=optimizer,metrics=[SelfELoss])
                self.CoeffModel.summary() 

                self.model_dir = os.path.join(self.save_dir, 'models')
                os.makedirs(self.model_dir, exist_ok=True)
                print('saving model architectures to', self.model_dir)

                with open(os.path.join(self.model_dir, 'CoeffModel.json'), 'w') as file:
                    file.write(self.CoeffModel.to_json())

        def train(self):
            data=np.array(pd.read_csv('/DataDir/IntensityFile.csv',index_col=0))[:,4:-1]
            np.random.shuffle(data)
            row,col=np.shape(data)
            self.dataIn=[]
            
            for idx in list(range(0,row)):
                self.dataIn.append(np.reshape( data[idx,:],(self.nchannel,self.latent_dim) ))
              
            self.dataIn=np.reshape(np.array(self.dataIn),(idx+1,self.nchannel,self.latent_dim))
            history=self.CoeffModel.fit(x=self.dataIn,y=self.dataIn,verbose=1,batch_size=self.batch_size,epochs=self.epochs)
            
            self.CoeffModel.save_weights(os.path.join(self.model_dir, 'weights_CoeffModel.hdf5'))

            print('done!')

            

model=SubspaceModel()
model.train()

### Show Subspace Coefficient Matrix

Coeff=model.CoeffModel.get_layer('new_linear_coeff').get_weights()[0]

data=pd.read_csv('../IntensityFile.csv',index_col=0)
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=[15,15]

temp=abs(Coeff)
np.fill_diagonal(temp, 0, wrap=False)
mask=temp==0
g=sns.heatmap(temp,cmap='bwr',mask=mask,vmin=0, vmax=.5,xticklabels=data.columns[4:-1],yticklabels=data.columns[4:-1])
g.set_facecolor('xkcd:black')
g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 30)
g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 30)
cbar = g.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)


#Combinatorially Test and Select panel to optimize subspace interactions

def CalculateScore(MatrixOfCorrelations,selection):
    
    restricted=MatrixOfCorrelations[selection,:]
    restricted=restricted[:,selection]
    includedinteraction=np.mean(restricted)
    
    restricted=MatrixOfCorrelations[np.array(list(set(range(0,25))-set(selection))),:]
    restricted=restricted[:,np.array(list(set(range(0,25))-set(selection)))]
    witheldinteraction=np.mean(restricted)
    
    return witheldinteraction-includedinteraction


from itertools import combinations
import math
MatrixOfCorrelations=temp
names=data.columns[4:-1].to_numpy()
MaxxMean=0
FinalSelection=[]
MeanArr=[]
for idx in list(range(0,25)):
    MaxxMean=0
    comb = combinations(list(range(1,25)),idx)
    comblist=np.array(list(comb))
    for n,selection in enumerate(comblist):
        selection=list(selection)
        selection.append(0)
               
        CurMean=CalculateScore(MatrixOfCorrelations,selection)
        
        if CurMean >MaxxMean:
            MaxxMean=CurMean
            FinalSelection=selection

    MeanArr.append(MaxxMean)
    
    print('Selection for panel of size: '+str(idx))
    print(names[FinalSelection])








