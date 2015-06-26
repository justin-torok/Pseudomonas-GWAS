"""
This file contains functions that relate to PCA.
"""
import numpy as np
import pandas as pd
import pylab
from sklearn.decomposition import RandomizedPCA
import pseudomonas_gwas as pgwas
reload(pgwas)
#%%
"""
Defining two different types of dataframes, one using all 39 genomes and the
other using the 30 for which phenotype data is available.
"""
pa1 = pgwas.PseudomonasGWAS(phenotype=True)
pa2 = pgwas.PseudomonasGWAS()
phen_df = pa1.complete_dataframe()
all_df = pa2.complete_dataframe()
#%%
"""
This function normalizes the rows of the dataframe to (mu,sigma) = (0,1). It
is necessary to first convert to a numpy array for computational efficiency.
"""
def normalizeddf(df):
    df_array = np.array(df)
    df_new = np.zeros(np.shape(df_array))
    for i in range(np.shape(df_array)[0]):
        mean = np.mean(df_array[i])
        stdev = np.std(df_array[i])
        df_new[i] = (df_array[i]-mean)/stdev
    df_new = pd.DataFrame(df_new, index=df.index, columns=df.columns)
    return df_new
#%%
def pcaanalysis(df,n_comp=20,evr=True,pc1=True):
    randpca = RandomizedPCA(n_components=n_comp)
    pca_fit = randpca.fit(df)
    expvar = pca_fit.explained_variance_ratio_
    if evr==True:
        histplot = pylab.bar(np.arange(1,n_comp+1), expvar)
        pylab.xlabel('PC #')
        pylab.ylabel('Fraction of Explained Variance')
        pylab.show()
        print 'Total explained variance in %d PCs is %f'%(n_comp, sum(expvar))
    princompdict = {}
    for i in range(n_comp):
        key = 'princomp_%d'%(i+1)
        pc = pca_fit.transform(df)[:,i]
        princompdict[key] = pd.Series(pc, index=df.index)
    return princompdict


        
    