"""
This file contains functions that relate to PCA.
"""
import numpy as np
import pandas as pd
import pylab
from sklearn.decomposition import RandomizedPCA
from pseudomonas_gwas import PseudomonasGWAS
#%%
painst = PseudomonasGWAS()
padf = painst.complete_dataframe()
#%%
padft = padf.transpose()
padft_centered = padft.copy()
#%%
for i in range(np.shape(padft)[1]):
    mean = np.mean(padft.iloc[:,i])
    stdev = np.std(padft.iloc[:,i])
    padft_centered.iloc[:,i] = (padft.iloc[:,i] - mean)/stdev

    
#%%
