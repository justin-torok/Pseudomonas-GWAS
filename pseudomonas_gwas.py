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
Last updated: 7/8/2015
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
        The query should be in the form of a tuple.
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
            entry = sum(presence_absence_table[i,:])>len(self.strains)/2
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
        genome = self.orf_presence_absence_dataframe()
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
        Obtains the (stable) cdhit_ids, orth_ids for which in-frame, aligned
        sequences are available. Currently this includes batches 1 and 3.
        """
        query = 'SELECT DISTINCT(cdhit_id), orth_id FROM view_orth WHERE \
                orth_id IN (SELECT DISTINCT(orth_id) FROM orf WHERE orth_batch_id \
                = 1 or orth_batch_id = 3)'
        table = self.db.getAllResultsFromDbQuery(query)
        table = np.array(sorted(table, key = lambda x: x[0]))
        return table[:,0].tolist()
        
    def get_mutation_dataframe(self, orthlist=None, genomelist=None, 
                               overrule=False):
        """
        Takes a valid orthid list and genome list and queries the database, 
        returning a dataframe containing a binary representation of the 
        presence/absence of all SNPs for the queried orfs and genomes. The
        predominant residue is given a 0. The default is to compile a table
        using all orthologs (from batches 1 and 3) and genomes, but
        these can be specified. 
        
        In the case that there is only partial sequence information available 
        for a given orth, the exception is handled and a dataframe containing 
        the orths which have missing sequences is returned instead of the
        binary dataframe. The binary matrix containing all the successfully
        parsed sequences is returned instead if overrule is set to True.
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
            
    def complete_dataframe(self, write=False):
        """
        Concatenates the presence/absence dataframe for non-core genes and the 
        complete mutation dataframe for the core genes. Optionally creates a
        .csv file with this information. *Key method for analysis classes*.
        """
        pan_genes = self.pan_genome()
        mutations = self.get_mutation_dataframe(overrule=True)
        completedf = pd.concat([pan_genes, mutations], keys=['pan_gene', 'mutation'])
        if write == True:
            from datetime import date
            filename = 'complete_genotype_df_'+str(date.today())+'.csv'
            completedf.to_csv(filename)
        return completedf
    
    def phenotype_dataframe(self, continuous=False):
        """
        Generates a phenotype dataframe for subsequent analysis. The option to 
        return either the binned (binary) representation of biofilm formation 
        and swarming or the normalized analog representation is given, with the
        default set to binary. Additional phenotypes and custom queries may be 
        added in the future as the need arises.
        """
        phen_dat = self.db.getAllResultsFromDbQuery('SELECT genome_id, \
                                        swarm_diameter, biofilm FROM phenotype \
                                        WHERE genome_id IN %s'%(str(tuple(self.strains))))
        phen_arr = np.array(phen_dat)
        phen_df_cont = pd.DataFrame(phen_arr, columns=['genome_id', 'swarming', 
                                                       'biofilm']).sort('genome_id')
        phen_df_cont[['genome_id']] = phen_df_cont[['genome_id']].astype(int)
        
        swarm_mean = np.mean(phen_df_cont.values[:,1])
        swarm_std = np.std(phen_df_cont.values[:,1])
        swarm_norm = (phen_df_cont.values[:,1] - swarm_mean)/swarm_std
        phen_df_cont[['swarming']] = swarm_norm
        
        bio_mean = np.mean(phen_df_cont.values[:,2])
        bio_std = np.std(phen_df_cont.values[:,2])
        bio_norm = (phen_df_cont.values[:,2] - bio_mean)/bio_std
        phen_df_cont[['biofilm']] = bio_norm
        
        strainslen = np.shape(phen_df_cont.values)[0]
        phen_df_cont.index = np.arange(1, strainslen+1)
        if continuous == False:
            swarm_cutoff = 25 #Divides swarmers into two classes based on k-means clustering
            swarm_filter = []
            for i in range(strainslen):
                entry = phen_arr[i,1]>swarm_cutoff
                swarm_filter.append(entry)
            biofilm_cutoff = 1.2 #Divides biofilm into two classes based on k-means clustering
            biofilm_filter = []
            for i in range(strainslen):
                entry = phen_arr[i,2]>biofilm_cutoff
                biofilm_filter.append(entry)
            swarm_filter = np.array(swarm_filter)
            biofilm_filter = np.array(biofilm_filter)
            phen_arr[swarm_filter,1], phen_arr[~swarm_filter,1] = 1, 0
            phen_arr[biofilm_filter,2], phen_arr[~biofilm_filter,2] = 1, 0
            phen_arr = phen_arr.astype(int)
            phen_df_bin = pd.DataFrame(phen_arr, columns=['genome_id', 'swarming', 
                                                          'biofilm']).sort('genome_id')
            phen_df_bin.index = np.arange(1, strainslen+1)
            return phen_df_bin
        else:
            return phen_df_cont
        #Note: The indices of phen_df are arbitrary at this stage, but they
        #become convenient for downstream operations
            
    def phenotype_plot(self, write=False):
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
class PCATools:
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
        default 20 but can be set to any number < # samples.
        
        The df is first normalized. For efficiency reasons, the conversion to 
        and from an array is necessary.
        """
        self.ncomps = ncomps
        df_array = np.array(df)
        df_norm = np.zeros(np.shape(df_array))
        for i in range(np.shape(df_array)[0]):
            mean = np.mean(df_array[i])
            stdev = np.std(df_array[i])
            df_norm[i] = (df_array[i]-mean)/stdev
        df_norm = pd.DataFrame(df_norm, index=df.index, columns=df.columns)
        randpca = RandomizedPCA(n_components=ncomps)
        pcafit = randpca.fit(df_norm)
        self.pcafit = pcafit
    
    def pca_dataframe(self):
        """
        Builds a dataframe of principal components.
        """
        pca_arr = self.pcafit.components_.T #default is to have PCs along the 0 axis
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
        
    #More to come as desired
#%%
class Regression:
    """
    This class is designed to contain all relevant functions to conduct a linear
    regression using principal components as covariates (or not). 
    
    Currently these methods are set up to generate the outputs shown in the
    'Xavier Lab Meeting 7-6-15' presentation, but the code should be a useful 
    template should other types of regression analysis become worthwhile to 
    pursue.
    
    A conceptual overview of the following approach:
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
        
        The functions words_df and bin_array are necessary for the creation of 
        regression objects but are not needed anywhere else. The stylistic choice 
        made here was to nest these within the __init__ method so that all of 
        these objects are automatically created for convenient use in later methods.
        """
        self.df = gen_df
        self.ncomps = ncomps
        self.cont = continuous_phen
        psdfs = PseudomonasDataframes()
        self.phen_df = psdfs.phenotype_dataframe(continuous=self.cont)
        
        pctools = PCATools(self.df, ncomps=self.ncomps)
        self.pca_df = pctools.pca_dataframe()
        
        def arrays_df(df):
            """
            This function converts the binary arrays in the input dataframe into 
            their decimal representations (words), determines the unique set, and 
            then outputs a df of the unique binary arrays reindexed. 
            """
            def bin_array(value):
                binary = bin(value)
                binarylist = [int(x) for x in binary[2:]] #list of ints instead of binary string
                binarylist = binarylist[::-1] #reverse order is necessary
                while len(binarylist) < np.shape(df)[1]:
                    binarylist.append(0) #extend all binary lists to 30 digits
                return np.array(binarylist)
            bin2dec_array = np.dot(df.values, 2**np.arange(0,np.shape(df)[1]))      
            uni_words = np.unique(bin2dec_array)
            uni_arrs = np.zeros((np.size(uni_words),np.shape(df)[1]), dtype=int)
            for i in range(np.size(uni_words)):
                arr = bin_array(uni_words[i])
                uni_arrs[i] = arr
            uni_arrs = uni_arrs.T
            wordids = np.arange(1,np.size(uni_words)+1)
            wordids = np.array(['word%d'%(x) for x in wordids])   
            uni_arrs_df = pd.DataFrame(uni_arrs, index =
                            np.arange(1,np.shape(uni_arrs)[0]+1), columns=wordids)    
            return uni_arrs_df
        self.unique_arrays_df = arrays_df(self.df)
        self.word_count = np.shape(self.unique_arrays_df)[1]      
        self.complete_df = pd.concat((self.phen_df, self.pca_df, self.unique_arrays_df), 
                                     axis=1)
                                     
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
        a parameter in the models for the intercept. This can be fixed at 0, which 
        may be desired in future tests. Additionally, this code can probably be 
        written more succinctly.
        """
        compdf = self.complete_df
        
        if bio == True:        
            biofilmpvals = []
            for i in range(1,self.word_count+1):
                if intercept == True:
                    y1, X1 = dmatrices('biofilm ~ word%d'%(i), data=compdf, return_type='dataframe')
                    mod1 = sm.OLS(y1, X1)
                    res1 = mod1.fit()
                    pval1 = res1.f_pvalue
                    biofilmpvals.append(pval1)
                else:
                    y1, X1 = dmatrices('biofilm ~ word%d - 1'%(i), data=compdf, return_type='dataframe')
                    mod1 = sm.OLS(y1, X1)
                    res1 = mod1.fit()
                    pval1 = res1.f_pvalue
                    biofilmpvals.append(pval1)                
            logbio = -np.log10(biofilmpvals)
            biopvaldf = pd.DataFrame(np.array(logbio), index=np.arange(1,self.word_count+1), 
                                     columns=['biofilm_pvals'])
            itemb = 1
        else:
            itemb = 0
        
        if swarm == True:
            swarmpvals = []
            for i in range(1,self.word_count+1):
                if intercept == True:
                    y2, X2 = dmatrices('swarming ~ word%d'%(i), data=compdf, return_type='dataframe')
                    mod2 = sm.OLS(y1, X1)
                    res2 = mod2.fit()
                    pval2 = res2.f_pvalue
                    swarmpvals.append(pval2)
                else:
                    y2, X2 = dmatrices('swarming ~ word%d - 1'%(i), data=compdf, return_type='dataframe')
                    mod2 = sm.OLS(y1, X1)
                    res2 = mod2.fit()
                    pval2 = res2.f_pvalue
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
        
    def pca_regression(self, princomps=3, intercept=True, bio=True, swarm=True):
        """
        This function performs a linear regression using the first X (default 3) 
        principal components as covariates. A likelihood ratio test is employed 
        to test the null and alternative models (an F-statistic could be found 
        as well, but this seemed easier to implement in StatsModels and there is
        virtually no difference between the results of both). Again, the intercept 
        as a variable has not been explored yet but the option is included here.
        
        bio and swarm are specified as in simple_regression. This can probably 
        be written more succinctly.
        """
        compdf = self.complete_df
        pclist = range(1, princomps+1)
        pcstring = ''
        i = 0
        while i<princomps-1:
            pcstring = pcstring + 'PC%d + '%(pclist[i])
            i += 1
        pcstring = pcstring + 'PC%d'%(pclist[-1])
        
        if bio == True:
            biofilmpvals = []
            if intercept == True:
                yn1, Xn1 = dmatrices('biofilm ~ %s'%(pcstring), data=compdf, return_type='dataframe')
                modn1 = sm.OLS(yn1, Xn1)
                resn1 = modn1.fit()
                llhn1 = resn1.llf
            else:
                yn1, Xn1 = dmatrices('biofilm ~ %s - 1'%(pcstring), data=compdf, return_type='dataframe')
                modn1 = sm.OLS(yn1, Xn1)
                resn1 = modn1.fit()
                llhn1 = resn1.llf
            for i in range(1,self.word_count+1):
                if intercept == True:
                    y1, X1 = dmatrices('biofilm ~ %s + word%d'%(pcstring, i), data=compdf, 
                                       return_type='dataframe')
                    mod1 = sm.OLS(y1, X1)
                    res1 = mod1.fit()
                    llh1 = res1.llf
                    pval1 = 1-chi2.cdf(2*(llh1-llhn1), 1)
                    biofilmpvals.append(pval1)
                else:
                    y1, X1 = dmatrices('biofilm ~ %s + word%d - 1'%(pcstring, i), data=compdf, 
                                       return_type='dataframe')
                    mod1 = sm.OLS(y1, X1)
                    res1 = mod1.fit()
                    llh1 = res1.llf
                    pval1 = 1-chi2.cdf(2*(llh1-llhn1), 1)
                    biofilmpvals.append(pval1)          
            logbio = -np.log10(biofilmpvals)
            biopvaldf = pd.DataFrame(np.array(logbio), index=np.arange(1,self.word_count+1), 
                                  columns=['biofilm_pvals'])
            itemb = 1
        else:
            itemb = 0
        
        if swarm == True:
            swarmpvals = []
            if intercept == True:
                yn2, Xn2 = dmatrices('swarming ~ %s'%(pcstring), data=compdf, 
                                     return_type='dataframe')
                modn2 = sm.OLS(yn2, Xn2)
                resn2 = modn2.fit()
                llhn2 = resn2.llf
            else:
                yn2, Xn2 = dmatrices('swarming ~ %s - 1'%(pcstring), data=compdf, 
                                     return_type='dataframe')
                modn2 = sm.OLS(yn2, Xn2)
                resn2 = modn2.fit()
                llhn2 = resn2.llf
            for i in range(1,self.word_count+1):
                if intercept == True:
                    y2, X2 = dmatrices('swarming ~ %s + word%d'%(pcstring, i), data=compdf, 
                                       return_type='dataframe')
                    mod2 = sm.OLS(y2, X2)
                    res2 = mod2.fit()
                    llh2 = res2.llf
                    pval2 = 1-chi2.cdf(2*(llh2-llhn2), 1)
                    swarmpvals.append(pval2)
                else:
                    y2, X2 = dmatrices('swarming ~ %s + word%d - 1'%(pcstring, i), data=compdf, 
                                       return_type='dataframe')
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

    def significant_hit_arrays(self, simple=False, princomps=3, 
                         intercept=True, phenotype='bio', signif=0.05):
        """
        This function returns a dataframe that contains the relevant genome_ids, 
        phenotypes, and all regression hits given the parameters specified. Default 
        is to do a PCA regression with 3 PCs (this is how all figures were generated 
        previously).
        
        For the moment, phenotype can either be set to bio or swarm, but both are 
        not handled together. That flexibility should be added in a future version.
        """
        if phenotype == 'bio':
            if simple == False:
                pvaldf = self.pca_regression(princomps=princomps, intercept=intercept, 
                                             swarm=False)
            else:
                pvaldf = self.simple_regression(intercept=intercept, swarm=False)
        
        elif phenotype == 'swarm':
            if simple == False:
                pvaldf = self.pca_regression(princomps=princomps, intercept=intercept, 
                                             bio=False)
            else:
                pvaldf = self.simple_regression(intercept=intercept, bio=False)
        
        bonferroni = -np.log10(signif/self.word_count)
        compdf = self.complete_df
        sighits = pvaldf[pvaldf.values>bonferroni]
        sighits.index = ['word%d'%(x) for x in sighits.index]
        hitsdf = compdf[sighits.index]
        if phenotype == 'bio':
            fulldf = pd.concat((compdf['genome_id'], compdf['biofilm'], hitsdf),
                               axis=1) 
        elif phenotype == 'swarm':
            fulldf = pd.concat((compdf['genome_id'], compdf['swarming'], hitsdf),
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
        """
        if phenotype == 'bio':
            if simple == False:
                pvaldf = self.pca_regression(princomps=princomps, intercept=intercept, 
                                             swarm=False)
            else:
                pvaldf = self.simple_regression(intercept=intercept, swarm=False)
        
        elif phenotype == 'swarm':
            if simple == False:
                pvaldf = self.pca_regression(princomps=princomps, intercept=intercept, 
                                             bio=False)
            else:
                pvaldf = self.simple_regression(intercept=intercept, bio=False)
        
        bonferroni = -np.log10(signif/self.word_count)
        compdf = self.complete_df
        sighits = pvaldf[pvaldf.values>bonferroni]
        sighits.index = ['word%d'%(x) for x in sighits.index]
        hitsdf = compdf[sighits.index]
        if phenotype == 'bio':
            fulldf = pd.concat((compdf['genome_id'], compdf['biofilm'], hitsdf),
                               axis=1) 
        elif phenotype == 'swarm':
            fulldf = pd.concat((compdf['genome_id'], compdf['swarming'], hitsdf),
                               axis=1)         
        
        wordlen = np.shape(self.df)[1]
        wordsarray = np.dot(self.df, 2**np.arange(0,wordlen))
        wordsdf = pd.DataFrame(wordsarray, index = self.df.index, columns=['words'])          
        
        justhits = fulldf.iloc[:,2:]
        hitwords = np.dot(justhits.values.T, 2**np.arange(0,wordlen))
        sumwordsdf = wordsdf[wordsdf.values==hitwords]
        sumdf = pd.DataFrame(sighits.values, index=sumwordsdf.index, 
                             columns=['p_values'])
        if write == True:
            from datetime import date
            filename = 'hits_summary_'+phenotype+'_'+str(date.today())+'.csv'
            sumdf.to_csv(filename)         
        return sumdf

class Plotter:
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
        reg = Regression(self.df)
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
            pylab.title('Swarming Q-Q Plot')
        if write == True:
            from datetime import date
            filename = 'qq_'+self.phenotype+'_'+str(date.today())+'.png'
            pylab.savefig(filename) 
        pylab.show()
            
        
        
        
        
            
            
        