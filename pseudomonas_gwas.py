"""
This file creates several classes to facilitate linear regression using several
scientific computing packages within Python. The datasets used are 

This module is meant to be used in the following way: 1) create a complete_dataframe
object using the PseudomonasDataframes class (OR build a custom binary genotype df); 
2) use this df as an argument for instances of the other classes. The reason for
doing it this way is that the most computationally-intensive step is the creation 
of the genotype dataframe, so only building it once and using it repeatedly 
facilitates faster analysis than if it is built de novo each time an analysis 
class is instantiated.

Author: Justin Torok, jlt46@cornell.edu
Last updated: 9/29/2015

Note: If SpringDb_local can't find the server, it may not be running; use the following command
in the terminal: 'pg_ctl -D /usr/local/var/postgres -l /usr/local/var/postgres/server.log start'
to manually start the server.
"""
import SpringDb_local as sdb #Accession to database; see SpringDb_local for details
reload(sdb)
import numpy as np
import pandas as pd
from Bio.Seq import Seq #mutation analysis
from Bio.Alphabet import IUPAC #mutation analysis
import pylab #graphical output
from sklearn.decomposition import RandomizedPCA #PCA
from patsy import dmatrices #Linear regression
import statsmodels.api as sm #Linear regression
from scipy.stats import chi2 #Linear regression
from scipy.stats import f
#%%
class MiscFunctions:
    """
    All methods that would otherwise be called as static methods will be placed in this class, so that all of the
    classes below can access them.  These methods are required for multiple classes and this is more concise than
    redefining the functions each time.
    """
    def zscore(self, vector):
        mean, std = vector.mean(), vector.std()
        zscores = (vector-mean)/std
        return zscores

    def arrays_df(self, df):
        """
        This function converts the binary arrays in the input dataframe into their decimal representations (words),
        determines the unique set, and then outputs a df of the unique binary arrays reindexed. It is used by
        the PCA_Regression and GLS_Regression classes below.
        """
        def bin_array(value):
            binary = bin(value)
            binarylist = [int(x) for x in binary[2:]]  # List of ints instead of binary string; '0b' is omitted
            binarylist = binarylist[::-1]  # Reverse order is necessary so that it agrees with genome_id order
            while len(binarylist) < np.shape(df)[1]:
                binarylist.append(0)  # Extend all binary lists to 30 digits (or less/more, as appropriate)
            return np.array(binarylist)

        bin2dec_array = np.dot(df.values, 2**np.arange(0, np.shape(df)[1]))  # Word creation
        uni_words = np.unique(bin2dec_array)
        uni_arrs = np.zeros((np.size(uni_words), np.shape(df)[1]), dtype=int)  # Becomes the smaller binary array
        for i in range(np.size(uni_words)):
            arr = bin_array(uni_words[i])
            uni_arrs[i] = arr
        uni_arrs = uni_arrs.T
        wordids = np.arange(1, np.size(uni_words)+1)
        wordids = np.array(['word%d'%(x) for x in wordids])
        uni_arrs_df = pd.DataFrame(uni_arrs, index=np.arange(1, np.shape(uni_arrs)[0]+1), columns=wordids)
        return uni_arrs_df

#%%
class PseudomonasDataframes:
    """
    Methods build several useful pandas dataframes containing genotype and 
    phenotype information that is imported from a postgres database.  These
    dataframes are used in downstream PCA and regression analysis, but may also
    be used for other purposes; in particular, the maximum entropy approach.
    """
    def __init__(self, phenotype=True, query=None):
        """
        Default is to query using the 30 strains for which genotype and phenotype 
        information is known.  If phenotype is set to something other than True, 
        then analysis handles all 39 strains in the database or a manual query.
        """
        self.phenotype = phenotype
        self.query = query
        if query != None:
            self.strains = list(query)
        elif self.phenotype == True:
            self.strains = [11,12,13,14,24,25,26,27,28,29,30,31,32,33,35,36,37,
                            39,40,41,42,43,44,45,46,47,48,49,50,51]
        else:
            self.strains = [1,5,6,7,8,9,10,11,12,13,14,24,25,26,27,28,29,30,31,32,
                            33,35,36,37,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]
        self.db = sdb.SpringDb()
        self.misc = MiscFunctions()
        self.strain_legend = self.db.getAllResultsFromDbQuery('SELECT genome_id, strain_name FROM \
                                                              genome WHERE genome_id IN %s'%(str(tuple(self.strains))))
        self.orf_legend = self.db.getAllResultsFromDbQuery('SELECT DISTINCT(cdhit_id), start, std_locus_name FROM orf \
                                                           WHERE cdhit_id IS NOT NULL')

    def orf_presence_absence_dataframe(self):
        """
        Import database of orfs (orthids and the genotypes) of selected strains.
        """
        #Build initial table
        table = self.db.getAllResultsFromDbQuery('SELECT cdhit_id, genome_id FROM orf WHERE genome_id IN %s AND cdhit_id \
                                                IS NOT NULL'%(str(tuple(self.strains))))
        orfgenarray = np.array(list(set(table))) #Removes redundant pairs and converts to an array
        
        orfs = orfgenarray[:,0]
        uni_orfs = np.unique(orfs)
        series_orfs = pd.Series(np.arange(len(uni_orfs)), index=uni_orfs)
        ind_orfs = series_orfs[orfs].values #Establishes an arbitrary index for each unique orf
        
        genomes = orfgenarray[:,1]
        uni_genomes = np.unique(genomes)
        series_genomes = pd.Series(np.arange(len(uni_genomes)), index=uni_genomes)
        ind_genomes = series_genomes[genomes].values #Establishes an arbitrary index for each unique genome (strain)
        
        presence_absence_table = np.zeros((np.shape(uni_orfs)[0], 
                                           np.shape(uni_genomes)[0]), dtype=int)
        presence_absence_table[ind_orfs,ind_genomes] = 1 #Each unique (orf, genome) pair from the query substitutes a
                                                         #'1' for a '0' using the preestablished indices.
        
        #Convert to the convention that the predominant genotype gets a value of 0
        filter_table = []
        for i in range(np.size(uni_orfs)):
            entry = sum(presence_absence_table[i,:])>len(self.strains)/2
            filter_table.append(entry)
        filter_table = np.array(filter_table) #Boolean array used as an index
        presence_absence_table[filter_table,:] = presence_absence_table[filter_table,:]+1
        presence_absence_table[presence_absence_table == 2] = 0
        presence_absence_df = pd.DataFrame(presence_absence_table,
                                           index=uni_orfs,
                                           columns=uni_genomes)
        return presence_absence_df

    def core_genes(self):
        """
        Returns an array containing the ids of the core genes for all strains (i.e. orfs corresponding to each row of
        all zeroes in the presence_absence_dataframe)
        """
        genome = self.orf_presence_absence_dataframe()
        genomearr = genome.values
        filter_table_2 = []
        for i in range(np.size(genome.index)):
            core = sum(genomearr[i,:])==0
            filter_table_2.append(core)
        filter_table_2 = np.array(filter_table_2)
        core_genes = np.array(genome.index[filter_table_2])
        return core_genes

    def pan_genome(self):
        """
        Returns a dataframe containing an binary matrix indicating the presence or absence of non-core genes.
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
        Obtains the (stable) cdhit_ids, orth_ids for which in-frame, aligned
        sequences are available. Currently this includes batches 1 (1921) and 3 (1059).
        """
        query = 'SELECT DISTINCT ON(cdhit_id, orth_id) cdhit_id, orth_id FROM orf WHERE \
                orth_batch_id = 1 or orth_batch_id = 3'
        table = self.db.getAllResultsFromDbQuery(query)
        table = np.array(sorted(table, key = lambda x: x[0]))
        return table[:,0].tolist()

    def get_protein_mutation_dataframe(self, orthlist=None, genomelist=None, overrule=True):
        """
        Takes a valid orthid list and genome list and queries the database, 
        returning a dataframe containing a binary representation of the 
        presence/absence of all nonsynonymous SNPs for the queried orfs and genomes.
        The predominant residue is given a 0. The default is to compile a table
        using all orthologs (from batches 1 and 3) and genomes, but
        these can be specified. Both of these batches have fully aligned, high-
        quality sequences, so there is no need at this time to do an alignment.
        
        
        In the case that there is only partial sequence information available 
        for a given orth, the exception is handled and a dataframe containing 
        the orths which have missing sequences is optionally returned instead of 
        the binary dataframe if overrule is set to False. Otherwise the binary 
        matrix containing all the successfully parsed sequences is returned (default).
        """
        #Default is all 30 genomes, all orfs
        if genomelist == None:
            genomelist = self.strains
        if orthlist == None:
            orthlist = self.get_orth_list()
                    
        #Respository for mutation ids and presence/absence matrix, respectively
        mutationlistindex = []
        mutationbinary = []
        missinglist = []
        
        #Query database, obtain infoarray. An exception for missing sequences is handled
        for orth in orthlist:
            try:
                genometuples = tuple(genomelist)
                dbquery = 'SELECT cdhit_id, genome_id, seq_inframe_aligned FROM \
                view_orth WHERE genome_id IN %s AND cdhit_id = %s'%(str(genometuples),
                                                                       str(orth))
                infotable = self.db.getAllResultsFromDbQuery(dbquery)
                for i in range(np.shape(infotable)[0]):
                    infotable[i] = list(infotable[i])
                infoarray = np.array(sorted(infotable, key=lambda x: x[1]))
            
                #Create aligned array of amino acid sequences for each orf/genome
                dnas = [len(list(x)) for x in infoarray[:,2]]
                if dnas.count(dnas[0]) == len(dnas):
                    if dnas[0]%3 == 0:
                        protlen = dnas[0]/3
                        aminoacidarray = np.zeros((np.shape(infoarray)[0],protlen), dtype=str)
                        for i in range(np.shape(aminoacidarray)[0]):
                            dnaseq = Seq(infoarray[i][2], IUPAC.unambiguous_dna)
                            protseq = dnaseq.translate()
                            aminoacidarray[i,:] = np.array(list(protseq))
                        print 'Orth %s parsed successfully.'%(str(orth))
                    else:
                        print 'Orth %s seqs cannot be converted to protein seqs'%(str(orth))
                        continue
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
            except TypeError:
                print '%d did not parse because of missing seqs.'%(orth)
                for row in infoarray:
                    if row[2] is None:
                        missinglist.append(np.array([row[0],row[1]]))
                continue
            
        #Generate final dataframe
        if len(mutationlistindex) == 0:
            print 'No mutations detected.'
        elif len(missinglist) != 0 and overrule == False:
            print 'Missing sequences detected. Returning dataframe of misses.'
            missingarray = np.array(missinglist)
            cols = {'cdhit_id', 'genome_id'}
            missingdf = pd.DataFrame(missingarray, columns=cols)
            return missingdf
        else:
            print 'Orths parsed successfully. Returning binary dataframe.'
            mutationarray = np.array(mutationbinary)
            mutationdf = pd.DataFrame(mutationarray, index=mutationlistindex, 
                                      columns=genomelist)    
            return mutationdf

    def get_genetic_mutation_dataframe(self, orthlist=None, genomelist=None,
                                       overrule=True):
        """
        Essentially runs identically to get_protein_mutation_dataframe, except synonymous mutations are
        included here.  For identifying protein mutations, the sequences are first translated using
        BioPython methods and then compared, which excludes all synonymous SNPs, whereas the comparisons
        made here are at the genetic sequence level.
        """
        #Default is all 30 genomes, all orfs
        if genomelist == None:
            genomelist = self.strains
        if orthlist == None:
            orthlist = self.get_orth_list()

        #Respository for mutation ids and presence/absence matrix, respectively
        mutationlistindex = []
        mutationbinary = []
        missinglist = []

        #Query database, obtain infoarray. An exception for missing sequences is handled
        for orth in orthlist:
            try:
                genometuples = tuple(genomelist)
                dbquery = 'SELECT cdhit_id, genome_id, seq_inframe_aligned FROM \
                view_orth WHERE genome_id IN %s AND cdhit_id = %s'%(str(genometuples),
                                                                       str(orth))
                infotable = self.db.getAllResultsFromDbQuery(dbquery)
                for i in range(np.shape(infotable)[0]):
                    infotable[i] = list(infotable[i])
                infoarray = np.array(sorted(infotable, key=lambda x: x[1]))

                #Create aligned array of amino acid sequences for each orf/genome
                dnas = [len(list(x)) for x in infoarray[:,2]]
                if dnas.count(dnas[0]) == len(dnas):
                    seqlen = dnas[0]
                    nucleotidearray = np.zeros((np.shape(infoarray)[0],seqlen), dtype=str)
                    for i in range(np.shape(nucleotidearray)[0]):
                        dnaseq = Seq(infoarray[i][2], IUPAC.unambiguous_dna)
                        nucleotidearray[i,:] = np.array(list(dnaseq))
                    print 'Orth %s parsed successfully.'%(str(orth))
                else:
                    print 'Orth %s seqs are of different lengths, cannot align.'%(str(orth))
                    continue

                #Create mutation table
                for i in range(seqlen):
                    basearray = nucleotidearray[:,i]
                    if np.size(np.unique(basearray)) != 1:
                        basecountdict = {}
                        for base in np.unique(basearray):
                            ct = basearray.tolist().count(base)
                            basecountdict[ct] = base
                        basewt = basecountdict[max(basecountdict.keys())]
                        for base in basecountdict.values():
                            if base != basewt:
                                binaryrow = np.zeros(np.size(basearray), dtype=int)
                                rowindex = (basearray == base)
                                binaryrow[rowindex] = 1
                                mutationbinary.append(binaryrow)
                                mutation = str(orth)+'_'+basewt+str(i)+base
                                mutationlistindex.append(mutation)
            except TypeError:
                print '%d did not parse because of missing seqs.'%(orth)
                for row in infoarray:
                    if row[2] is None:
                        missinglist.append(np.array([row[0],row[1]]))
                continue

        #Generate final dataframe
        if len(mutationlistindex) == 0:
            print 'No mutations detected.'
        elif len(missinglist) != 0 and overrule == False:
            print 'Missing sequences detected. Returning dataframe of misses.'
            missingarray = np.array(missinglist)
            cols = {'cdhit_id', 'genome_id'}
            missingdf = pd.DataFrame(missingarray, columns=cols)
            return missingdf
        else:
            print 'Orths parsed successfully. Returning binary dataframe.'
            mutationarray = np.array(mutationbinary)
            mutationdf = pd.DataFrame(mutationarray, index=mutationlistindex,
                                      columns=genomelist)
            return mutationdf
            
    def complete_dataframe(self, synonymous=True, pangenes=True, write=False):
        """
        Concatenates the presence/absence dataframe for non-core genes and the 
        complete mutation dataframe for the core genes. Optionally creates a
        .csv file with this information. If synonymous is set to True, then the
        get genetic mutations method is used; else the protein mutations method is
        used. *Key method for analysis classes*.
        """
        pan_genes = self.pan_genome()
        if synonymous == True:
            mutations = self.get_genetic_mutation_dataframe(overrule=True)
        else:
            mutations = self.get_protein_mutation_dataframe(overrule=True)
        if pangenes == True:
            completedf = pd.concat([pan_genes, mutations], keys=['pan_gene', 'mutation'])
        else:
            completedf = mutations
        if write == True:
            from datetime import date
            filename = 'complete_genotype_df_'+str(date.today())+'.csv'
            completedf.to_csv(filename)
        return completedf

    def phenotype_dataframe(self, phenlist=['swarm_diameter', 'biofilm'], continuous=True):
        """
        Generates a phenotype dataframe for subsequent analysis. The option to 
        return either the binned (binary) representation of biofilm formation 
        and swarming or the normalized analog representation is given, with the
        default set to binary. Additional phenotypes and custom queries may be 
        added in the future as the need arises.  As of 9/28/15, there are many
        more phenotypes to examine - waiting for them to be uploaded into the
        database.
        """
        cols = ['genome_id'] + phenlist
        phen_dat = self.db.getAllResultsFromDbQuery('SELECT %s FROM phenotype WHERE genome_id \
                                                    IN %s'%(', '.join(cols), str(tuple(self.strains))))
        phen_arr = np.array(phen_dat)
        phen_df_cont = pd.DataFrame(phen_arr, columns=cols)
        phen_df_cont[['genome_id']] = phen_df_cont[['genome_id']].astype(int)
        phen_df_cont = phen_df_cont.sort(columns=['genome_id'])
        for phen in phenlist:
            phen_df_cont[phen] = self.misc.zscore(phen_df_cont[phen])
        strainslen = np.shape(phen_df_cont.values)[0]
        phen_df_cont.index = np.arange(1, strainslen+1)

        # THIS HAS TO BE PUT INTO A FUNCTION SOMEHOW TO ACCOMMODATE NEW PHENOTYPES; HOW TO CLUSTER?
        if continuous is False:
            swarm_cutoff = 25  # Divides swarmers into two classes based on a priori k-means clustering
            swarm_filter = []
            for i in range(strainslen):
                entry = phen_arr[i, 1] > swarm_cutoff
                swarm_filter.append(entry)
            biofilm_cutoff = 1.2  # Divides biofilm into two classes based on k-means clustering
            biofilm_filter = []
            for i in range(strainslen):
                entry = phen_arr[i,2]>biofilm_cutoff
                biofilm_filter.append(entry)
            swarm_filter = np.array(swarm_filter)
            biofilm_filter = np.array(biofilm_filter)
            phen_arr[swarm_filter,1], phen_arr[~swarm_filter,1] = 1, 0
            phen_arr[biofilm_filter,2], phen_arr[~biofilm_filter,2] = 1, 0
            phen_arr = phen_arr.astype(int)
            phen_df_bin = pd.DataFrame(phen_arr, columns=['genome_id', 'swarm_diameter',
                                                          'biofilm']).sort('genome_id')
            phen_df_bin.index = np.arange(1, strainslen+1)
            return phen_df_bin
        else:
            return phen_df_cont
        # Note: The indices of phen_df are arbitrary at this stage, but they become convenient for downstream operations
            
    def swarm_biofilm_plot(self, write=False):
        """
        This function creates a scatterplot that is divided into four regions as 
        determined by k-means clustering (performed elsewhere; the red lines are 
        drawn at the appropriate boundaries). It may not be useful in its current 
        state if more phenotypes and strains are added, but it can serve as a 
        template if nothing else.
        """
        phen_dat = self.db.getAllResultsFromDbQuery('select genome_id, swarm_diameter, \
                                                biofilm from phenotype')
        phen_dat = np.array(phen_dat)
        swarm = phen_dat[:,1]
        biofilm = phen_dat[:,2]
        nonsnonb = phen_dat[(swarm<25)&(biofilm<1.2)]
        snonb = phen_dat[(swarm>25)&(biofilm<1.2)]
        nonsb = phen_dat[(swarm<25)&(biofilm>1.2)]
        sb = phen_dat[(swarm>25)&(biofilm>1.2)]
        plot1 = pylab.plot(nonsnonb[:,1], nonsnonb[:,2], 'go')
        plot2 = pylab.plot(snonb[:,1], snonb[:,2], 'bo')
        plot3 = pylab.plot(nonsb[:,1], nonsb[:,2], 'mo')
        plot4 = pylab.plot(sb[:,1], sb[:,2], 'co')
        plot5 = pylab.plot([0,80],[1.2,1.2],'r-')
        plot6 = pylab.plot([25,25],[-0.5,3],'r-')
        pylab.xlabel('Swarm Diameter (mm)')
        pylab.ylabel('Biofilm Index')
        pylab.title('P. aeruginosa Phenotypes')
        if write == True:
            from datetime import date
            filename = 'phenotypes_scatterplot_'+str(date.today())+'.png'
            pylab.savefig(filename)
        pylab.show()
        
#%%
class PCA_Tools:
    """
    This class has a number of methods associated with related to outputting 
    data from PCA decomposition. Instantiating this class creates a fit object
    using the RandomizedPCA function from the Scikit Learn package (see
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.RandomizedPCA.html).
    This fit object is then passed to the methods defined below.
    """
    def __init__(self, df, ncomps=20):
        """
        The df to be passed is meant to be a complete_dataframe object, but it 
        will work on any dataframe assuming that it is arrayed as
        (features x samples). The number of components to be returned is by
        default 20 but can be set to any number < # samples. The df is first normalized.
        """
        self.ncomps = ncomps
        df_array = np.array(df)
        dfmeans = np.mean(df_array, axis=1)
        dfstd = np.std(df_array, axis=1)
        df_array_norm = (df_array-dfmeans[:, np.newaxis])/dfstd[:, np.newaxis]
        df_norm = pd.DataFrame(df_array_norm, index=df.index, columns=df.columns)
        randpca = RandomizedPCA(n_components=ncomps)
        self.pcafit = randpca.fit(df_norm)
    
    def pca_dataframe(self):
        """
        Builds a dataframe of principal components.
        """
        pca_arr = self.pcafit.components_.T  #default is to have PCs along the 0 axis
        pcacols = ['PC%d'%(x) for x in np.arange(1, np.shape(pca_arr)[1]+1)]
        pca_df = pd.DataFrame(pca_arr, index = np.arange(1,np.shape(pca_arr)[0]+1), 
                              columns = pcacols)
        return pca_df
    
    def variance_histogram(self, write=False):
        """
        Outputs a histogram showing the distribution of explained variance by
        each principal component. Option to write to .png file.
        """
        expvar = self.pcafit.explained_variance_ratio_
        histplot = pylab.bar(np.arange(1,self.ncomps+1), expvar)
        pylab.xlabel('PC #')
        pylab.ylabel('Density')
        percentv = 100*sum(expvar)
        pylab.title('Total explained variance is %f percent'%(percentv))
        if write == True:
            from datetime import date
            filename = 'pca_variance_histogram'+str(date.today())+'.png'
            pylab.savefig(filename)
        pylab.show()
    
    def pc_2dplot(self, princomp1='PC1', princomp2='PC2', write=False):
        """
        Outputs a 2-D scatterplot of two principal components against each other;
        by default these are set up to be the first two principal components. The 
        arguments should be strings in the form 'PC#' if user-specified. Option 
        to write to .png file given. An option for a 3-D plot may be added in the 
        future.
        """
        pca_df = self.pca_dataframe()
        scatterplot = pylab.plot(pca_df[princomp1], pca_df[princomp2], 'ro')
        pylab.xlabel(princomp1)
        pylab.ylabel(princomp2)
        pylab.title(princomp1+' vs '+princomp2)
        if write == True:
            from datetime import date
            filename = 'pc_2d_scatterplot'+str(date.today())+'.png'
            pylab.savefig(filename)
        pylab.show()
        # More to come as desired
#%%
class PCA_Regression:
    """
    This class is designed to contain all relevant functions to conduct a linear
    regression using principal components as covariates (or not). 
    
    Currently these methods are set up to generate the outputs shown in the
    'Xavier Lab Meeting 7-6-15' presentation, but the code should be a useful 
    template should other types of regression analysis become worthwhile to 
    pursue.
    
    A conceptual overview of the approach:
    1) Convert the 30-digit (at this time) binary arrays corresponding to each  
    genotypic feature into their decimal representations (herein referred to as 
    'words').
    2) Determine the unique set of words (some features will have the same
    genotypic pattern, especially for small sample sizes) to reduce the
    computational burden of performing the regression. 
    3) Assign these words simple indices ('word ids'). Convert the words back 
    into their binary representation, now indexed by the word ids.
    4) Perform the regression and extract the top hits.
    5) Determine which genotypic features correspond to the hits by reindexing.
    """
    def __init__(self, gen_df, ncomps=20, continuous_phen=False):
        """
        A full (strains x phenotypes/PCs/genotypes) df is constructed upon class
        instantiation, as it will be used repeatedly in the other class methods.
        The option to leave the phenotypes continuous is given, but the more
        relevant encoding for regression analysis (for biofilm/swarming) is binary.
        """
        self.df = gen_df
        self.ncomps = ncomps
        self.cont = continuous_phen
        psdfs = PseudomonasDataframes()
        self.phen_df = psdfs.phenotype_dataframe(continuous=self.cont)
        
        pctools = PCA_Tools(self.df, ncomps=self.ncomps)
        self.pca_df = pctools.pca_dataframe()

        self.misc = MiscFunctions()
        self.unique_arrays_df = self.misc.arrays_df(self.df)
        self.word_count = np.shape(self.unique_arrays_df)[1]      
        self.complete_df = pd.concat((self.phen_df, self.pca_df, self.unique_arrays_df), axis=1)

        # The self.unique_words_df created below is important for reindexing in significant_hits_summary
        arrstranspose = self.unique_arrays_df.transpose()
        uniquewords = np.dot(arrstranspose.values, 2**np.arange(0,np.shape(self.df)[1]))
        self.unique_words_df = pd.DataFrame(self.unique_arrays_df.columns, index=uniquewords)
                                     
    def simple_regression(self, intercept=True, bio=True, swarm=True): 
        """
        This function performs a linear regression without incorporating any 
        covariates (and therefore not correcting for population structure). It 
        returns an array of p-values obtained using the StatsModels OLS package. 
        It can take the convenient R-like syntax for its lm() function using the 
        package patsy, which is used here. Swarming and biofilm formation are 
        separate phenotypes and are fitted individually. Default is to perform 
        both regressions, but this is not necessary.
        
        At the time of this version, no analysis has been done without including 
        a parameter in the models for the intercept. This can be fixed at 0 with
        the intercept parameter set to False, which may be desired in future tests.

        Additionally, this code can probably be written more succinctly. To that point,
        since new phenotypes may be added in the future, the input should be a list of
        strings corresponding to the phenotypes of interest.  The code could then define
        a function and loop through this list so that the p-value generation code would be
        stated only once. For now (as of 9/25/15) this will be left alone.
        """
        compdf = self.complete_df
        
        if bio is True:
            biofilmpvals = []
            for i in range(1, self.word_count+1):
                if intercept is True:
                    y1, X1 = dmatrices('biofilm ~ word%d'%(i), data=compdf, return_type='dataframe')
                else:
                    y1, X1 = dmatrices('biofilm ~ word%d - 1'%(i), data=compdf, return_type='dataframe')
                mod1 = sm.OLS(y1, X1)
                res1 = mod1.fit()
                pval1 = res1.f_pvalue
                biofilmpvals.append(pval1)
            logbio = -np.log10(biofilmpvals)
            biopvaldf = pd.DataFrame(np.array(logbio), index=np.arange(1, self.word_count+1),
                                     columns=['biofilm_pvals'])
            itemb = 1
        else:
            itemb = 0
        
        if swarm is True:
            swarmpvals = []
            for i in range(1, self.word_count+1):
                if intercept is True:
                    y2, X2 = dmatrices('swarm_diameter ~ word%d'%(i), data=compdf, return_type='dataframe')
                else:
                    y2, X2 = dmatrices('swarm_diameter ~ word%d - 1'%(i), data=compdf, return_type='dataframe')
                mod2 = sm.OLS(y1, X1)
                res2 = mod2.fit()
                pval2 = res2.f_pvalue
                swarmpvals.append(pval2)
            logswarm = -np.log10(swarmpvals)
            swarmpvaldf = pd.DataFrame(np.array(logswarm), index=np.arange(1, self.word_count+1),
                                       columns=['swarm_pvals'])
            items = 1
        else:
            items = 0
        
        if itemb == 1 and items == 1:
            pvaldf = pd.concat((biopvaldf, swarmpvaldf), axis=1)
            return pvaldf
        elif itemb == 1:
            return biopvaldf
        elif items == 1:
            return swarmpvaldf
        else:
            print 'Specify bio, swarm, or both.'
        
    def pca_regression(self, princomps=3, intercept=True, bio=True, swarm=True):
        """
        This function performs a linear regression using the first X (default 3) 
        principal components as covariates. A likelihood ratio test is employed 
        to test the null and alternative models (an F-statistic could be found 
        as well, but this seemed easier to implement in StatsModels and there is
        virtually no difference between the results of both). Again, the intercept 
        as a variable has not been explored yet but the option is included here.
        
        bio and swarm are specified as in simple_regression. This can probably 
        be written more succinctly (See docstring of simple_regression).
        """
        compdf = self.complete_df
        # Creates the input string argument to dmatrices corresponding to the PC columns of compdf
        pclist = range(1, princomps+1)
        pcstring = ''
        i = 0
        while i < princomps-1:
            pcstring = pcstring + 'PC%d + '%(pclist[i])
            i += 1
        pcstring = pcstring + 'PC%d'%(pclist[-1])
        
        if bio is True:
            biofilmpvals = []
            if intercept is True:
                yn1, Xn1 = dmatrices('biofilm ~ %s'%(pcstring), data=compdf, return_type='dataframe')
            else:
                yn1, Xn1 = dmatrices('biofilm ~ %s - 1'%(pcstring), data=compdf, return_type='dataframe')
            modn1 = sm.OLS(yn1, Xn1)
            resn1 = modn1.fit()
            llhn1 = resn1.llf
            for i in range(1,self.word_count+1):
                if intercept is True:
                    y1, X1 = dmatrices('biofilm ~ %s + word%d'%(pcstring, i), data=compdf, return_type='dataframe')
                else:
                    y1, X1 = dmatrices('biofilm ~ %s + word%d - 1'%(pcstring, i), data=compdf, return_type='dataframe')
                mod1 = sm.OLS(y1, X1)
                res1 = mod1.fit()
                llh1 = res1.llf
                pval1 = 1-chi2.cdf(2*(llh1-llhn1), 1)
                biofilmpvals.append(pval1)
            logbio = -np.log10(biofilmpvals)
            biopvaldf = pd.DataFrame(np.array(logbio), index=np.arange(1,self.word_count+1), columns=['biofilm_pvals'])
            itemb = 1
        else:
            itemb = 0
        
        if swarm is True:
            swarmpvals = []
            if intercept is True:
                yn2, Xn2 = dmatrices('swarm_diameter ~ %s'%(pcstring), data=compdf, return_type='dataframe')
            else:
                yn2, Xn2 = dmatrices('swarm_diameter ~ %s - 1'%(pcstring), data=compdf, return_type='dataframe')
            modn2 = sm.OLS(yn2, Xn2)
            resn2 = modn2.fit()
            llhn2 = resn2.llf
            for i in range(1,self.word_count+1):
                if intercept == True:
                    y2, X2 = dmatrices('swarm_diameter ~ %s + word%d'%(pcstring, i), data=compdf,
                                       return_type='dataframe')
                else:
                    y2, X2 = dmatrices('swarm_diameter ~ %s + word%d - 1'%(pcstring, i), data=compdf, return_type='dataframe')
                mod2 = sm.OLS(y2, X2)
                res2 = mod2.fit()
                llh2 = res2.llf
                pval2 = 1-chi2.cdf(2*(llh2-llhn2), 1)
                swarmpvals.append(pval2)
            logswarm = -np.log10(swarmpvals)
            swarmpvaldf = pd.DataFrame(np.array(logswarm), index=np.arange(1,self.word_count+1),
                                       columns=['swarm_pvals'])
            items = 1
        else:
            items = 0
            
        if itemb == 1 and items == 1:
            pvaldf = pd.concat((biopvaldf, swarmpvaldf), axis=1)
            return pvaldf
        elif itemb == 1:
            return biopvaldf
        elif items == 1:
            return swarmpvaldf
        else:
            print 'Specify bio, swarm, or both.'

    def significant_hit_arrays(self, simple=False, princomps=3, intercept=True, phenotype='bio', signif=0.05):
        """
        This function returns a dataframe that contains the relevant genome_ids, 
        phenotypes, and all regression hits given the parameters specified. Default 
        is to do a PCA regression with 3 PCs (this is how all figures were generated 
        previously).
        
        For the moment, phenotype can either be set to bio or swarm, but both are 
        not handled together. That flexibility should be added in a future version.
        The unimplemented idea suggested in the simple_regression docstring would
        more easily accommodate this.
        """
        if phenotype is 'bio':
            if simple is False:
                pvaldf = self.pca_regression(princomps=princomps, intercept=intercept, swarm=False)
            else:
                pvaldf = self.simple_regression(intercept=intercept, swarm=False)
        
        elif phenotype is 'swarm':
            if simple is False:
                pvaldf = self.pca_regression(princomps=princomps, intercept=intercept, bio=False)
            else:
                pvaldf = self.simple_regression(intercept=intercept, bio=False)
        
        bonferroni = -np.log10(signif/self.word_count)
        compdf = self.complete_df
        sighits = pvaldf.values[pvaldf.values > bonferroni]
        sighits.index = ['word%d' % x for x in sighits.index]
        hitsdf = compdf[sighits.index]
        if phenotype is 'bio':
            fulldf = pd.concat((compdf['genome_id'], compdf['biofilm'], hitsdf),
                               axis=1) 
        elif phenotype is 'swarm':
            fulldf = pd.concat((compdf['genome_id'], compdf['swarm_diameter'], hitsdf),
                               axis=1)         
        return fulldf
    
    def significant_hits_summary(self, simple=False, princomps=3,
                                 intercept=True, phenotype='bio', signif=0.05, write=False):
        """
        This function is similar to the previous one and shares much of the same 
        code, but returns a dataframe (can write to file) that contains all the 
        important information about the significant hits.
        
        For the moment, phenotype can either be set to bio or swarm, but both are 
        not handled together. That flexibility should be added in a future version
        (see docstrings for significant_hits_array and simple_regression).

        Additionally, currently cdhit_id is used to index, but in the future this should
        be converted to the standard locus names (or they should just be added)
        """
        if phenotype == 'bio':
            if simple == False:
                pvaldf = self.pca_regression(princomps=princomps, intercept=intercept, swarm=False)
            else:
                pvaldf = self.simple_regression(intercept=intercept, swarm=False)
        
        elif phenotype == 'swarm':
            if simple == False:
                pvaldf = self.pca_regression(princomps=princomps, intercept=intercept, bio=False)
            else:
                pvaldf = self.simple_regression(intercept=intercept, bio=False)
        
        bonferroni = -np.log10(signif/self.word_count)
        compdf = self.complete_df
        sighits = pvaldf.values[pvaldf.values > bonferroni]
        sighits.index = ['word%d'%(x) for x in sighits.index]
        hitsdf = compdf[sighits.index]
        if phenotype == 'bio':
            fulldf = pd.concat((compdf['genome_id'], compdf['biofilm'], hitsdf), axis=1)
        elif phenotype == 'swarm':
            fulldf = pd.concat((compdf['genome_id'], compdf['swarm_diameter'], hitsdf), axis=1)

        wordlen = np.shape(self.df)[1]
        wordsarray = np.dot(self.df, 2**np.arange(0, wordlen))
        # Recall that self.unique_words_df contains the word id strings indexed by word (decimal) values
        wordidsarray = np.array([self.unique_words_df.ix[word] for word in wordsarray])
        # Associate both words and word ids with the appropriate cdhit_id.  Probably could be accomplished more
        # stylistically in a single df using pandas hierarchical indexing, but it's unnecessary here.
        wordsdf = pd.DataFrame(wordsarray, index=self.df.index, columns=['words'])
        wordidsdf = pd.DataFrame(wordidsarray, index=self.df.index, columns=['wordids'])
        
        justhits = fulldf.ix[:, 2:]
        hitwords = np.dot(justhits.values.T, 2**np.arange(0, wordlen))
        sumwordidsdf = wordidsdf[wordsdf.values == hitwords]
        # np.squeeze is necessary below because the listcomp won't work with a column vector, which is 2D
        pvalarray = np.array([sighits.ix[wordid] for wordid in np.squeeze(sumwordidsdf.values)])
        sumdf = pd.DataFrame(pvalarray, index=sumwordidsdf.index, columns=['p_values'])
        if write is True:
            from datetime import date
            filename = 'hits_summary_'+phenotype+'_'+str(date.today())+'.csv'
            sumdf.to_csv(filename)         
        return sumdf
#%%
class PCA_Plotter:
    """
    This class contains methods to make Manhattan and Q-Q plots, and maybe more
    in the future. It requires specific arguments to pass to the various methods
    of the Regression class.

    Currently it plots using word ids, but in later versions it should plot by
    genomic position instead.
    """
    def __init__(self, gen_df, simple=False, princomps=3, intercept=True,
                 phenotype='bio', signif=0.05):
        self.df = gen_df
        reg = PCA_Regression(self.df)
        if phenotype == 'bio':
            if simple == False:
                pvaldf = reg.pca_regression(princomps=princomps, intercept=intercept,
                                             swarm=False)
            else:
                pvaldf = reg.simple_regression(intercept=intercept, swarm=False)

        elif phenotype == 'swarm':
            if simple == False:
                pvaldf = reg.pca_regression(princomps=princomps, intercept=intercept,
                                             bio=False)
            else:
                pvaldf = reg.simple_regression(intercept=intercept, bio=False)
        self.pvals = np.squeeze(pvaldf.values)
        self.wordids = np.array(pvaldf.index)
        self.bonferroni = -np.log10(signif/np.size(self.wordids))
        self.phenotype = phenotype

    def manhattan(self, write=False):
        graph1 = pylab.plot(self.wordids, self.pvals, 'bo')
        graph2 = pylab.plot([self.wordids[0],self.wordids[-1]],
                            [self.bonferroni,self.bonferroni],'r-')
        pylab.xlabel('word id')
        pylab.ylabel('-log10(p-val)')
        if self.phenotype == 'bio':
            pylab.title('Biofilm Manhattan Plot')
        elif self.phenotype == 'swarm':
            pylab.title('Swarming Manhattan Plot')
        if write == True:
            from datetime import date
            filename = 'manhattan_'+self.phenotype+'_'+str(date.today())+'.png'
            pylab.savefig(filename)
        pylab.show()

    def qq(self, write=False):
        sortedpvals = np.sort(self.pvals)
        sortednull = np.sort(-np.log10(np.linspace(1./self.wordids[-1],1,self.wordids[-1])))
        graph3 = pylab.plot(sortednull, sortedpvals, 'bo')
        graph4 = pylab.plot([0,sortednull[-1]],[0,sortednull[-1]],'r-')
        pylab.xlabel('Expected p-val')
        pylab.ylabel('Actual p-val')
        if self.phenotype == 'bio':
            pylab.title('Biofilm Q-Q Plot')
        elif self.phenotype == 'swarm':
            pylab.title('swarm_diameter Q-Q Plot')
        if write == True:
            from datetime import date
            filename = 'qq_'+self.phenotype+'_'+str(date.today())+'.png'
            pylab.savefig(filename)
        pylab.show()
#%%
class GLS_Regression:
    """
    This class contains all the necessary components to perform a GLS regression using the covariance matrix generated
    from the genotype dataframe to account for relatedness between strains. See the docstrings for individual methods
    for further information.
    """
    def __init__(self, gen_df, phens=['biofilm', 'swarm_diameter'], binary_phen=True):
        """
        The __init__ method contains everything necessary to perform any kind of GLS regression, making it easier and
        faster to perform the subsequent analyses.  Included is the arrays_df function introduced in PCA_Regression, as
        it will be necessary test word by word here as well.  It also tries to calculate the covariance matrix from the
        genotype dataframe, raising an error if the covariance matrix is not positive definite (a second error will
        automatically be raised as it tries to execute 'self.covariance = covmat').

        phens should (must) be a list of strings that correspond to the names of columns within the full phenotype
        dataframe generated by PseudomonasDataframes()
        """
        self.gen_df = gen_df
        try:
            covmat = np.cov(gen_df.transpose())
        except np.linalg.LinAlgError:
            print 'Fatal error: Covariance matrix is not positive-definite'
        self.covariance = covmat
        self.phenlist = ['genome_id']+phens
        self.psdfs = PseudomonasDataframes()
        self.misc = MiscFunctions()
        self.binary = binary_phen
        if self.binary is False:
            phenfull = self.psdfs.phenotype_dataframe(continuous=True)
        else:
            phenfull = self.psdfs.phenotype_dataframe(continuous=False)
        self.phen_df = phenfull[self.phenlist]
        self.strains = self.phen_df['genome_id']

        self.misc = MiscFunctions()
        self.unique_arrays_df = self.misc.arrays_df(self.gen_df)
        self.word_count = np.shape(self.unique_arrays_df)[1]
        self.complete_df = pd.concat((self.phen_df, self.unique_arrays_df), axis=1)

        # The self.unique_words_df created below is important for reindexing in significant_hits_summary
        arrstranspose = self.unique_arrays_df.transpose()
        uniquewords = np.dot(arrstranspose.values, 2**np.arange(0, np.shape(self.gen_df)[1]))
        self.unique_words_df = pd.DataFrame(self.unique_arrays_df.columns, index=uniquewords)

    def phenotype_regression(self, phen_ind='swarm_diameter', phen_dep='biofilm', intercept=True, plot=False):
        """
        Performs the GLS regression between two phenotypes using the binary genotype matrix to construct the covariance
        matrix used in the regression, returning a vector of coefficients (beta).  The analytical solution to the
        least-squares problem is beta = (S_c'S_c)^-1 S_c'B_c, where S_c and B_c represent the independent
        and dependent variables, respectively, which have been transformed using the Cholesky decomposition
        of the covariance matrix R = TT', X_c = T'X, where T is the lower triangular Cholesky factor.
        """
        gendf = self.gen_df
        phen_cont = self.psdfs.phenotype_dataframe(continuous=True)     # Cont. required for phen/phen regression

        indphen = phen_cont[phen_ind].reshape(len(self.strains), 1)
        depphen = phen_cont[phen_dep].reshape(len(self.strains), 1)
        strainlen = len(self.strains)
        covmat = np.cov(gendf.transpose())

        T = np.linalg.cholesky(covmat)
        Y_c = np.dot(np.linalg.inv(T), depphen)
        if intercept is False:
            X_c = np.dot(np.linalg.inv(T), indphen)
        else:
            X_c = np.dot(np.linalg.inv(T), indphen)
            X_c = np.concatenate((np.ones((strainlen, 1)), X_c), axis=1)
        beta = np.dot(np.linalg.inv(np.dot(X_c.T, X_c)), np.dot(X_c.T, Y_c))
        y_exp = np.dot(X_c, beta)
        sst = np.sum((Y_c-np.mean(Y_c))**2)
        ssr = np.sum((y_exp-np.mean(Y_c))**2)
        sse = np.sum((Y_c-y_exp)**2)
        rsq = 1-sse/sst
        fstat = ssr/(sse/28)
        pval = 1-f.cdf(fstat, 1, 28) # Check to make sure that this is the way the p-value was calculated in paper.
        summary = pd.Series([float(beta[0]), float(beta[1]), rsq, pval], index=['Intercept', 'Slope', 'R^2', 'P_value'])
        print summary
        if plot is True:    # To be expanded upon later...
            plot1, = pylab.plot(X_c[:, -1], Y_c, 'bx')
            x = np.linspace(int(np.min(X_c))-50, int(np.max(X_c))+50, 50)
            y = beta[0]+x*beta[1]
            plot2, = pylab.plot(x, y, 'r-')
            pylab.title('Phenotype Regression')
            pylab.show()
        return summary

    def genotype_regression1(self, phenotypes=['biofilm'], intercept=True):
        """
        Goal: Repeat the PCA_Regression analysis, independently testing each gene against a particular phenotype,
        except using the (Cholesky-decomposed) covariance matrix to transform the data instead of incorporating PCs
        as covariates.  Inputs are gen_df and phenotype_df as constructed in the PseudomonasDataframes class above.
        This allows for multiple phenotypes to be tested at once if df is set to True; otherwise a single phenotype can
        be examined independently.  binary_phen specifies whether continuous or binary phenotypes should be tested as
        it applies to the different traits). Intercept is as above in PCA_Regression.

        Ideas/Notes:
        - Binary or continuous phenotype?  Try both, but binary seems like a good starting point.
        - Normalization? Necessary for phenotype (if continuous), but for genotype? Probably not.
        - Testing the transformed phenotypes against the transformed genotypes should be no different from testing them
          straight up: T_inv(y) = alpha + beta*T_inv(x) is no different from y = alpha' + beta*x, barring a trivial
          readjustment of the intercept (not an interesting parameter).  Here, only genotypes should be transformed to
          reflect their phylogeny: y = alpha + beta*T_inv(x).
        - For y = alpha + beta*T_inv(x), there is only one degree of freedom, so the simple OLS regression, which is
          similarly integrated in the simple_regression method of PCA_Regression, should be adequate.
        - An alternative treatment is to use untransformed genotypes and include the covariance matrix as a covariate:
          y = alpha + beta*x + T*Gamma. *These should yield the same result; test this!* There is still only one degree
          of freedom, though Gamma will be 30x1 and T*Gamma is really np.multiply(T, Gamma).
        """
        if intercept is False:
            pass
        covmat = self.covariance
        T = np.linalg.cholesky(covmat)
        Tinv = np.linalg.inv(T)
        compdf = self.complete_df.copy()
        for i in range(1, self.word_count+1):
            word = 'word%d'%(i)
            compdf[word] = np.dot(Tinv, compdf[word].reshape((len(self.strains), 1)))
        phenpvaldf = pd.DataFrame(np.zeros((self.word_count, len(phenotypes))),
                                  index=np.arange(1, self.word_count+1), columns=phenotypes)
        for phen in phenotypes:
            phenpvals = []
            for i in range(1, self.word_count+1):
                if intercept is True:
                    y1, X1 = dmatrices('%(phenotype)s ~ word%(index)d'%{'phenotype':phen, 'index':i}, data=compdf,
                                       return_type='dataframe')
                else:
                    y1, X1 = dmatrices('%(phenotype)s ~ word%(index)d - 1'%{'phenotype':phen, 'index':i}, data=compdf,
                                       return_type='dataframe')
                mod1 = sm.OLS(y1, X1)
                res1 = mod1.fit()
                pval1 = res1.f_pvalue
                phenpvals.append(pval1)
            logphen = -np.log10(phenpvals)
            phenpvaldf[phen] = np.array(logphen)
        print 'Finished'
        return phenpvaldf

    def genotype_regression2(self, phenotypes=['biofilm'], intercept=True):
        """
        Goal: Repeat the PCA_Regression analysis, independently testing each gene against a particular phenotype,
        except using the (Cholesky-decomposed) covariance matrix to transform the data instead of incorporating PCs
        as covariates.  Inputs are gen_df and phenotype_df as constructed in the PseudomonasDataframes class above.
        This allows for multiple phenotypes to be tested at once if df is set to True; otherwise a single phenotype can
        be examined independently.  binary_phen specifies whether continuous or binary phenotypes should be tested as
        it applies to the different traits). Intercept is as above in PCA_Regression.

        Ideas/Notes:
        - Binary or continuous phenotype?  Try both, but binary seems like a good starting point.
        - Normalization? Necessary for phenotype (if continuous), but for genotype? Probably not.
        - Testing the transformed phenotypes against the transformed genotypes should be no different from testing them
          straight up: T_inv(y) = alpha + beta*T_inv(x) is no different from y = alpha' + beta*x, barring a trivial
          readjustment of the intercept (not an interesting parameter).  Here, only genotypes should be transformed to
          reflect their phylogeny: y = alpha + beta*T_inv(x).
        - For y = alpha + beta*T_inv(x), there is only one degree of freedom, so the simple OLS regression, which is
          similarly integrated in the simple_regression method of PCA_Regression, should be adequate.
        - An alternative treatment is to use untransformed genotypes and include the covariance matrix as a covariate:
          y = alpha + beta*x + T*Gamma. *These should yield the same result; test this!* There is still only one degree
          of freedom, though Gamma will be 30x1 and T*Gamma is really np.multiply(T, Gamma).
        """
        if intercept is False:
            pass
        covmat = self.covariance
        T = np.linalg.cholesky(covmat)
        Tinv = np.linalg.inv(T)
        compdf = self.complete_df.copy()
        for i in range(1, self.word_count+1):
            word = 'word%d'%(i)
            compdf[word] = np.dot(Tinv, compdf[word].reshape((len(self.strains), 1)))
        phenpvaldf = pd.DataFrame(np.zeros((self.word_count, len(phenotypes))),
                                  index=np.arange(1, self.word_count+1), columns=phenotypes)
        for phen in phenotypes:
            phenpvals = []
            Y = compdf[phen].reshape((len(self.strains), 1))
            for i in range(1, self.word_count+1):
                if intercept is False:
                    X_c = compdf['word%d'%i].reshape((len(self.strains), 1))
                else:
                    X_c = compdf['word%d'%i].reshape((len(self.strains), 1))
                    X_c = np.concatenate((np.ones((len(self.strains), 1)), X_c), axis=1)
                beta = np.dot(np.linalg.inv(np.dot(X_c.T, X_c)), np.dot(X_c.T, Y))
                y_exp = np.dot(X_c, beta)
                ssr = np.sum((y_exp-np.mean(Y))**2)
                sse = np.sum((Y-y_exp)**2)
                fstat = ssr/(sse/28)
                pval = 1-f.cdf(fstat, 1, 28)
                phenpvals.append(pval)
            logphen = -np.log10(phenpvals)
            phenpvaldf[phen] = np.array(logphen)
        print 'Finished'
        return phenpvaldf
#%%
class GLS_Plotter:
    pass