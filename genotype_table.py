"""This is a file for prototyping code to create the binary genotype matrix for
P. aeruginosa strains. These will be incorporated into a neater package once
the code is optimized."""

import SpringDb_local as sdb
reload(sdb)
import numpy as np
import pandas as pd
#%%
"""Import database of orfs (cdhits) and the genotypes of the 30 strains"""
db = sdb.SpringDb()
table = db.getAllResultsFromDbQuery('SELECT cdhit_id, genome_id FROM orf \
                                    WHERE genome_id IN %s AND cdhit_id IS NOT \
                                                NULL'%(str(tuple([1,5,6,7,8,9,10,11,12,13,14,24,25,26,27,28,29,30,31,32,
                            33,35,36,37,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53])))) 
#%%
"""Create a binary genotype matrix indicating the presence or absence of a
cdhit in a given genome; the presence of a gene is given a value of '1'."""
                                    
orfgenarray = np.array(list(set(table)))

orfs = orfgenarray[:,0]
uni_orfs = np.unique(orfs)
series_orfs = pd.Series(np.arange(len(uni_orfs)), index=uni_orfs)
ind_orfs = series_orfs[orfs].values

genomes = orfgenarray[:,1]
uni_genomes = np.unique(genomes)
series_genomes = pd.Series(np.arange(len(uni_genomes)), index=uni_genomes)
ind_genomes = series_genomes[genomes].values

presence_absence_table = np.zeros((np.shape(uni_orfs)[0], np.shape(uni_genomes)[0]), dtype=int)
presence_absence_table[ind_orfs,ind_genomes] = 1
#%%
"""Take original genotype matrix and conform it to our convention: the most
prevalent genotype at a given locus is given a value of '0'.  The code below
inverts any row where the cdhit is present in more than 50% of the strains"""

filter_table = []
for i in range(np.size(uni_orfs)):
    entry = sum(presence_absence_table[i,:])>19
    filter_table.append(entry)
filter_table = np.array(filter_table)
presence_absence_table[filter_table,:] = presence_absence_table[filter_table,:]+1
presence_absence_table[presence_absence_table == 2] = 0
#%%
"""Identifying the subset of cdhits that comprises the core genome; that is, 
genes that are shared by all thirty strains."""

core_genome = presence_absence_table.copy()
filter_table_2 = []
for i in range(np.size(uni_orfs)):
    core = sum(core_genome[i,:])==0
    filter_table_2.append(core)
filter_table_2 = np.array(filter_table_2)
core_genome = core_genome[filter_table_2,:]
core_cdhits = uni_orfs[filter_table_2]
presence_absence_table = presence_absence_table[-filter_table_2]
#shape(core_genome) = (4243, 30)
#%%
"""Writing to text files"""
"""
import SpringDb_local as sdb
reload(sdb)
import numpy as np
import pandas as pd
np.savetxt('pan_genome_table_060915.csv', presence_absence_table, fmt='%d', delimiter=',')
np.savetxt('core_orfs_060915.csv', core_cdhits, fmt='%d', delimiter=',')
np.savetxt('genomes_060915.csv', uni_genomes, fmt='%d', delimiter=',')
"""

#%%
    