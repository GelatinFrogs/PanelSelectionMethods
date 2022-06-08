import glob
import pandas as pd
from skimage.io import imread
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
import seaborn as sns


## Read and normalize intensity values
MeanIntensities=pd.read_csv('./IntensitiesFile.csv',index_col=0)

qt= QuantileTransformer(n_quantiles=10000,random_state =30)
MeanMatTot=MeanIntensities[MeanIntensities.columns[4:-1]]
MeanMat=MeanMatTot[MeanIntensities['TMA-Section']=='TMA004'].to_numpy()
MeanMat=qt.fit_transform(MeanMat)


# Create correlation matrix
row1,col=np.shape(MeanMat)

corrMat=np.array(list(range(0,col1)))
for i in range(0,col):
    corrVec=[]
    for j in range(0,col):
        corrVec.append(np.corrcoef(MeanMat[:,j],MeanMat[:,i])[0,1])
    corrMat=np.vstack([corrMat,corrVec])
    
corrFrame=pd.DataFrame(corrMat[1:,:])
corrFrame=corrFrame.set_index(pd.Index(MeanIntensities.columns[4:-1]))
corrFrame.columns=MeanIntensities.columns[4:-1]

###Show Correlation matrix
cg2 = sns.clustermap(corrFrame,cmap="coolwarm",cbar_kws={"ticks":[-1,-.5,0,.5,1]},cbar_pos=(.02, .32, .05, .3),vmin = -1.0, vmax=1.0,row_cluster=True,col_cluster=True)
cg2.ax_col_dendrogram.set_visible(False)
cg2.ax_row_dendrogram.set_visible(False)
plt.setp(cg2.ax_cbar.yaxis.get_majorticklabels(),fontsize=20)


###Calulate highest performing set for every panel size

from itertools import combinations
import math

MatrixOfCorrelations=corrMat[1:,:]
names=MeanIntensities.columns[4:-1].to_numpy()
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
        SelectedCorrs=np.max(abs(MatrixOfCorrelations[selection,:]),axis=0)
        CurMean=np.mean(SelectedCorrs[SelectedCorrs<.99])
        if CurMean >MaxxMean:
            MaxxMean=CurMean
            FinalSelection=selection

    
    MeanArr.append(MaxxMean)
    
    print('Selection for panel of size: '+str(idx))
    print(names[FinalSelection])

