"""
This file creates a class containing all of the necessary functions to generate
a binary matrix (pandas dataframe) of genotype differences between a set of
queried strains with respect to: 1) presence/absence of non-core genes and
2) presence/absence of SNPs in the core genome. Additional functionality may be
added later on.
"""
import SpringDb_local as sdb
reload(sdb)
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
import pylab
from sklearn.decomposition import RandomizedPCA
#%%
class PseudomonasGWAS:
    def __init__(self, phenotype=False):
        """
        Default is to query using all 39 strains for which genotype information is
        known.  If phenotype is set to something other than False, then analysis
        is restricted to the 30 strains for which there is phenotype information.
        """
        if phenotype == False:
            self.strains = [1,5,6,7,8,9,10,11,12,13,14,24,25,26,27,28,29,30,31,32,
                            33,35,36,37,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]
        else:                    
            self.strains = [11,12,13,14,24,25,26,27,28,29,30,31,32,33,35,36,37,
                                39,40,41,42,43,44,45,46,47,48,49,50,51]
        self.db = sdb.SpringDb()
    def orf_presence_absence_dataframe(self):
        """
        Import database of orfs (orthids and the genotypes) of selected strains.
        """
        #Build initial table
        table = self.db.getAllResultsFromDbQuery('SELECT cdhit_id, genome_id FROM orf \
                                                WHERE genome_id IN %s AND cdhit_id IS NOT \
                                                NULL'%(str(tuple(self.strains))))                                                
        orfgenarray = np.array(list(set(table)))
        
        orfs = orfgenarray[:,0]
        uni_orfs = np.unique(orfs)
        series_orfs = pd.Series(np.arange(len(uni_orfs)), index=uni_orfs)
        ind_orfs = series_orfs[orfs].values
        
        genomes = orfgenarray[:,1]
        uni_genomes = np.unique(genomes)
        series_genomes = pd.Series(np.arange(len(uni_genomes)), index=uni_genomes)
        ind_genomes = series_genomes[genomes].values
        
        presence_absence_table = np.zeros((np.shape(uni_orfs)[0], 
                                           np.shape(uni_genomes)[0]), dtype=int)
        presence_absence_table[ind_orfs,ind_genomes] = 1
        
        #Convert to the convention that the predominant genotype gets a value
        #of 0
        filter_table = []
        for i in range(np.size(uni_orfs)):
            entry = sum(presence_absence_table[i,:])>15
            filter_table.append(entry)
        filter_table = np.array(filter_table)
        presence_absence_table[filter_table,:] = presence_absence_table[filter_table,:]+1
        presence_absence_table[presence_absence_table == 2] = 0
        presence_absence_df = pd.DataFrame(presence_absence_table,
                                           index=uni_orfs,
                                           columns=uni_genomes)
        return presence_absence_df
    def core_genes(self):
        """
        Returns an array containing the ids of the core genes for all strains.
        """
        genome = self.orf_presence_absence_table(phenotype)
        genomearr = genome.values
        filter_table_2 = []
        for i in range(np.size(genome.index)):
            core = sum(genomearr[i,:])==0
            filter_table_2.append(core)
        filter_table_2 = np.array(filter_table_2)
        core_cdhits = np.array(genome.index[filter_table_2])
        return core_cdhits
    def pan_genome(self):
        """
        Returns a dataframe containing an binary matrix indicating the presence
        or absence of non-core genes.
        """
        genome = self.orf_presence_absence_dataframe()
        genomearr = genome.values
        filter_table_2 = []
        for i in range(np.size(genome.index)):
            core = sum(genomearr[i,:])==0
            filter_table_2.append(core)
        filter_table_2 = np.array(filter_table_2)
        pan_genome = genome[-filter_table_2]
        return pan_genome
    
    def get_orth_list(self):
        """
        Obtains the (stable) cdhit_ids from the 1921 orfs for which aligned 
        sequences are available.
        """
        query = 'SELECT DISTINCT(cdhit_id), orth_id FROM view_orth_fam WHERE \
                orth_id >= 1 AND orth_id <= 1921'
        table = self.db.getAllResultsFromDbQuery(query)
        table = np.array(sorted(table, key = lambda x: x[0]))
        return table[:,0].tolist()
        
    def get_mutation_dataframe(self, orthlist=None, genomelist=None):
        """
        Takes a valid orthid list and genome list and queries the database, 
        returning a dataframe containing a binary representation of the 
        presence/absence of all SNPs for the queried orfs and genomes. The
        predominant residue is given a 0. The default is to compile a table
        using all genes (1921 pre-aligned, in-frame sequences) and genomes, but
        these can be specified.
        """
        #Default is all 30 genomes, all orfs
        if genomelist == None:
            genomelist = self.strains
        if orthlist == None:
            orthlist = self.get_orth_list()
                    
        #Respository for mutation ids and presence/absence matrix, respectively
        mutationlistindex = []
        mutationbinary = []
        
        #Query database, obtain infoarray.
        for orth in orthlist:
            genometuples = tuple(genomelist)
            dbquery = 'SELECT cdhit_id, genome_id, seq_inframe_aligned FROM \
            view_orth_fam WHERE genome_id IN %s AND cdhit_id = %s'%(str(genometuples), 
                                                                   str(orth))
            infotable = self.db.getAllResultsFromDbQuery(dbquery)
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
                print 'Orth %s parsed successfully.'%(str(orth))
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
    def complete_dataframe(self, write=False):
        """
        Concatenates the presence/absence dataframe for non-core genes and the 
        complete mutation dataframe for the core genes. Optionally creates a
        .csv file with this information.
        """
        pan_genes = self.pan_genome()
        mutations = self.get_mutation_dataframe()
        completedf = pd.concat([pan_genes, mutations], keys=['pan_gene', 'mutation'])
        if write == True:
            from datetime import date
            filename = 'complete_genotype_df_'+str(date.today())+'.csv'
            completedf.to_csv(filename)
        return completedf
    
        