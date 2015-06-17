"""
This file accesses and uses the subset of orfs (1921) that are complete and 
aligned already, allowing for easy annotation of all SNPs between genomes. The
goal is to create a function(orthid_list, genomeid_list) that returns a binary matrix
encoding the presence/absence of all SNPs in the queried list of genome ids.
"""
#%%
import SpringDb_local as sdb
reload(sdb)
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
#%%
def get_mutation_dataframe(orthlist, genomelist):
    """Takes a valid orthid list and genome list and queries the database, returning
    a pandas dataframe containing a binary representation of the presence/absence
    of all SNPs for the queried orfs and genomes."""
    
    #Respository for mutation ids and presence/absence matrix, respectively
    mutationlistindex = []
    mutationbinary = []
    
    #Initialize database
    db = sdb.SpringDb()    
    
    #Query database, obtain infoarray.
    for orth in orthlist:
        genometuples = tuple(genomelist)
        dbquery = 'SELECT orth_id, genome_id, seq_inframe_aligned FROM orf WHERE \
        genome_id IN %s AND orth_id = %s'%(str(genometuples), str(orth))
        infotable = db.getAllResultsFromDbQuery(dbquery)
        for i in range(np.shape(infotable)[0]):
            infotable[i] = list(infotable[i])
        infoarray = np.array(sorted(infotable, key=lambda x: x[1]))
 
    #Create aligned array of amino acid sequences for each orf/genome
        dnas = [len(list(x)) for x in infoarray[:,2]]
        if dnas.count(dnas[0]) == len(dnas) and dnas[0]%3 == 0:
            protlen = dnas[0]/3
            aminoacidarray = np.zeros((np.shape(infoarray)[0],protlen), dtype=str)
            for i in range(np.shape(aminoacidarray)[0]):
                dnaseq = Seq(infoarray[i][2], IUPAC.unambiguous_dna)
                protseq = dnaseq.translate()
                aminoacidarray[i,:] = np.array(list(protseq))
            print 'Success'
        else:
            print 'Orth %s seqs are of different lengths, cannot align.'%(str(orth))
            continue
        
    #Create mutation table
        for i in range(protlen):
            resarray = aminoacidarray[:,i]
            if np.size(np.unique(resarray)) != 1:
                rescountdict = {}
                for res in np.unique(resarray):
                    ct = resarray.tolist().count(res)
                    rescountdict[ct] = res
                aawt = rescountdict[max(rescountdict.keys())]
                for res in rescountdict.values():
                    if res != aawt:
                        binaryrow = np.zeros(np.size(resarray), dtype=int)
                        rowindex = (resarray == res)
                        binaryrow[rowindex] = 1
                        mutationbinary.append(binaryrow)
                        mutation = str(orth)+'_'+aawt+str(i)+res
                        mutationlistindex.append(mutation)
    
    #Generate final dataframe
    if len(mutationlistindex) == 0:
        print 'No mutations detected.'
    else:
        mutationarray = np.array(mutationbinary)
        mutationdf = pd.DataFrame(mutationarray, index=mutationlistindex, columns=genomelist)    
        return mutationdf
        
#%%
"""
Querying the database. There are two possibilities: 1) specify the queried 
orth_id in the pqsl query, or 2) make a generic function that searches the 
table of 1940 orfs to determine if the queried orf is in this subset, returning
an error if it is not. Below is the code prototyping a necessary piece for option 2).
"""
orfquery = 969 # input orth_id
genomequery = [11,12,13,14,24,25,26,27,28,29,30,31,32,33,35,36,37,39,
               40,41,42,43,44,45,46,47,48,49,50,51] #input genome_id list
genometuplequery = tuple(genomequery)

db = sdb.SpringDb()
query = 'SELECT orth_id, genome_id FROM orth_fam'
table = db.getAllResultsFromDbQuery(query)
tablearray = np.array(table)
queryarray = (tablearray[:,0]==orfquery) & (np.in1d(tablearray[:,1],np.array(genomequery)))
tablearray = tablearray[queryarray]
#%%
"""
Taking the shortcut option 1) described above.  Creating a function that will 
detect SNPs and output them to a binary matrix.
"""
query2 = 'SELECT orth_id, genome_id, seq_inframe_aligned FROM view_orth_fam WHERE genome_id IN %s AND \
        orth_id = %s'%(str(genometuplequery), str(orfquery))
table1 = db.getAllResultsFromDbQuery(query2)
for i in range(len(genometuplequery)):
    table1[i] = list(table1[i])
table2 = np.array(sorted(table1, key=lambda x: x[1]))

indexlist = list(zip(table2[:,0],table2[:,1]))

dnas = [len(list(x)) for x in table2[:,2]]
if dnas.count(dnas[0]) == len(dnas) and len(dnas)%3 == 0:
    protlen = dnas[0]/3
    aacidarray = np.zeros((len(genomequery),protlen), dtype=str)
    for i in range(np.shape(aacidarray)[0]):
        dnaseq = Seq(table2[i][2], IUPAC.unambiguous_dna)
        protseq = dnaseq.translate()
        aacidarray[i,:] = np.array(list(protseq))
    print 'Success!'
else: #change to try/except
    print 'These DNA sequences are of different lengths.'
#%%
"""
Create the mutation array (without indices) and a mutation dataframe that is
properly indexed, each row indicating another mutation and each column
indicating a orth_id, genome_id pair.
"""
mutationlistindex = []
mutationbinary = []
for i in range(protlen):
    resarray = aacidarray[:,i]
    if np.size(np.unique(resarray)) != 1:
        rescountdict = {}
        for res in np.unique(resarray):
            ct = resarray.tolist().count(res)
            rescountdict[ct] = res
        aawt = rescountdict[max(rescountdict.keys())]
        for res in rescountdict.values():
            if res != aawt:
                binaryrow = np.zeros(np.size(resarray), dtype=int)
                rowindex = (resarray == res)
                binaryrow[rowindex] = 1
                mutationbinary.append(binaryrow)
                mutation = aawt+str(i)+res
                mutationlistindex.append(mutation)
if len(mutationlistindex) == 0:
    print 'All sequences are identical'
else:    
    mutationarray = np.array(mutationbinary)
    mutationdf = pd.DataFrame(mutationarray, index=mutationlistindex, columns=indexlist)
