"""
This file creates several classes to facilitate linear regression using several scientific computing packages within
Python.

This module is meant to be used in the following way: 1) create a complete_dataframe object using the
PseudomonasDataframes class (OR build a custom binary genotype df); 2) use this df as an argument for instances of the
other classes. The reason for doing it this way is that the most computationally-intensive step is the creation of the
genotype dataframe, so only building it once and using it repeatedly facilitates faster analysis than if it is built de
novo each time an analysis class is instantiated.  Check the list of imports for the necessary packages to run this
file to make sure that there are no missing dependencies.

Author: Justin Torok, jlt46@cornell.edu
Last updated: 2/16/2015

Note: If SpringDb_local can't find the server, it may not be running; use the following command in the terminal:
'pg_ctl -D /usr/local/var/postgres -l /usr/local/var/postgres/server.log start' (or the equivalent paths in PostgreSQL)
to manually start the server.
"""
import SpringDb_local as sdb
reload(sdb)
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
import pylab
from patsy import dmatrices
import statsmodels.api as sm
from sklearn.decomposition import RandomizedPCA
from scipy.stats import chi2
from scipy.stats import f

#%%

class MiscFunctions:
    """
    All methods that would otherwise be called as static methods will be placed in this class, so that all of the
    classes below can access them.  These methods are required for multiple classes and this is more concise than
    redefining the functions each time.
    """
    def __init__(self):
        pass

    def zscore(self, vector):
        mean, std = vector.mean(), vector.std()
        zscores = (vector-mean)/std
        return zscores

    def normalize(self, vector):
        mean = vector.mean()
        cent = vector-mean
        norm = np.sqrt(np.dot(cent, cent))
        normed = cent/norm
        return normed

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
    dataframes are used in downstream PCA and regression analysis.
    """
    def __init__(self, phenotype=True, query=None):
        """
        Default is to query using the 30 strains for which genotype and phenotype 
        information is known.  If phenotype is set to something other than True, 
        then analysis handles all 39 strains in the database or a manual query.
        """
        self.phenotype = phenotype
        self.query = query
        self.db = sdb.SpringDb()
        if query is not None:
            self.strains = list(query)
        elif self.phenotype is True:
            strains = self.db.getAllResultsFromDbQuery('SELECT DISTINCT(genome_id) FROM phenotype')
            self.strains = sorted(np.squeeze(np.array(strains)))
        else:
            strains = self.db.getAllResultsFromDbQuery('SELECT DISTINCT(genome_id) FROM orf')
            self.strains = sorted(np.squeeze(np.array(strains)))
        self.misc = MiscFunctions()
        self.strain_legend_raw = self.db.getAllResultsFromDbQuery('SELECT genome_id, strain_name FROM \
                                                              genome WHERE genome_id IN %s'%(str(tuple(self.strains))))
        strain_array = np.array(self.strain_legend_raw)
        number_array = strain_array[:, 0].astype(int)
        strain_df = pd.DataFrame(number_array, columns=['ID'])
        strain_df['Name'] = strain_array[:, 1]
        self.strain_legend = strain_df.sort(columns=['ID'])
        self.orf_legend = self.db.getAllResultsFromDbQuery('SELECT DISTINCT(cdhit_id), start, std_locus_name FROM orf \
                                                           WHERE cdhit_id IS NOT NULL')

    def raw_presence_absence_dataframe(self):
        """
        Import database of orfs (orthids and the genotypes) of selected strains.
        """
        # Build initial table
        table = self.db.getAllResultsFromDbQuery('SELECT cdhit_id, genome_id FROM orf WHERE genome_id IN %s AND \
                                                cdhit_id IS NOT NULL'%(str(tuple(self.strains))))
        orfgenarray = np.array(list(set(table)))    # Removes redundant pairs and converts to an array

        orfs = orfgenarray[:, 0]
        uni_orfs = np.unique(orfs)
        series_orfs = pd.Series(np.arange(len(uni_orfs)), index=uni_orfs)
        ind_orfs = series_orfs[orfs].values     # Establishes an arbitrary index for each unique orf

        genomes = orfgenarray[:, 1]
        uni_genomes = np.unique(genomes)
        series_genomes = pd.Series(np.arange(len(uni_genomes)), index=uni_genomes)
        ind_genomes = series_genomes[genomes].values    # Establishes an arbitrary index for each unique genome (strain)

        presence_absence_table = np.zeros((np.shape(uni_orfs)[0],
                                           np.shape(uni_genomes)[0]), dtype=int)
        presence_absence_table[ind_orfs, ind_genomes] = 1    # Each unique (orf, genome) pair from the query substitutes
                                                             # a '1' for a '0' using the preestablished indices.
        presence_absence_df = pd.DataFrame(presence_absence_table, index=uni_orfs, columns=uni_genomes)
        return presence_absence_df

    # Originally the following method was the default; however, at this point it is important for the presence of a
    # specific gene be unambiguously given a value of '1'. It is left here for completeness but it is not used by any
    # other methods
    def orf_presence_absence_dataframe(self):
        """
        Same as the previous method, except the ancestral state is set to '0' for the presence/absence of a gene.
        """
        table = self.db.getAllResultsFromDbQuery('SELECT cdhit_id, genome_id FROM orf WHERE genome_id IN %s AND \
                                                 cdhit_id IS NOT NULL'%(str(tuple(self.strains))))
        orfgenarray = np.array(list(set(table)))

        orfs = orfgenarray[:, 0]
        uni_orfs = np.unique(orfs)
        series_orfs = pd.Series(np.arange(len(uni_orfs)), index=uni_orfs)
        ind_orfs = series_orfs[orfs].values

        genomes = orfgenarray[:, 1]
        uni_genomes = np.unique(genomes)
        series_genomes = pd.Series(np.arange(len(uni_genomes)), index=uni_genomes)
        ind_genomes = series_genomes[genomes].values

        presence_absence_table = np.zeros((np.shape(uni_orfs)[0], np.shape(uni_genomes)[0]), dtype=int)
        presence_absence_table[ind_orfs, ind_genomes] = 1

        # Convert to the convention that the predominant genotype gets a value of 0
        filter_table = []
        for i in range(np.size(uni_orfs)):
            entry = sum(presence_absence_table[i, :]) > len(self.strains)/2
            filter_table.append(entry)
        filter_table = np.array(filter_table)   # Boolean array used as an index
        presence_absence_table[filter_table, :] = presence_absence_table[filter_table, :]+1
        presence_absence_table[presence_absence_table == 2] = 0
        presence_absence_df = pd.DataFrame(presence_absence_table, index=uni_orfs, columns=uni_genomes)
        return presence_absence_df

    def core_genome(self):
        """
        Returns an array containing the ids of the core genes for all strains (i.e. orfs present in all strains)
        """
        genome = self.raw_presence_absence_dataframe()
        genomearr = genome.values
        filter_table_2 = []
        for i in range(np.size(genome.index)):
            core = sum(genomearr[i, :]) % len(self.strains) == 0
            filter_table_2.append(core)
        filter_table_2 = np.array(filter_table_2)
        core_genome = genome[filter_table_2]
        return core_genome

    def accessory_genome(self):
        """
        Returns an array containing the ids of the accessory genes for all strains (i.e. orfs absent in one or more
        strains)
        """
        genome = self.raw_presence_absence_dataframe()
        genomearr = genome.values
        filter_table_2 = []
        for i in range(np.size(genome.index)):
            accessory = sum(genomearr[i, :]) % len(self.strains) != 0
            filter_table_2.append(accessory)
        filter_table_2 = np.array(filter_table_2)
        accessory_genome = genome[filter_table_2]
        return accessory_genome

    def get_orth_list(self):
        """
        Obtains the (stable) cdhit_ids, orth_ids for which in-frame, aligned sequences are available. Currently this
        includes batches 1 (1921) and 3 (1059) in orf.
        """
        query = 'SELECT DISTINCT ON(cdhit_id, orth_id) cdhit_id, orth_id FROM orf WHERE orth_batch_id = 1 or \
                orth_batch_id = 3'
        table = self.db.getAllResultsFromDbQuery(query)
        table = np.array(sorted(table, key=lambda x: x[0]))
        return table[:, 0].tolist()

    def get_protein_mutation_dataframe(self, orthlist=None, genomelist=None, overrule=True):
        """
        Takes a valid orthid list and genome list and queries the database, returning a dataframe containing a binary
        representation of the presence/absence of all nonsynonymous SNPs for the queried orfs and genomes.
        The predominant residue is given a 0. The default is to compile a table using all orthologs (from batches 1 and
        3) and genomes, but these can be specified. Both of these batches have fully aligned, high-quality sequences, so
        there is no need at this time to do an alignment.
        
        
        In the case that there is only partial sequence information available for a given orth, the exception is handled
        and a dataframe containing the orths which have missing sequences is optionally returned instead of the binary
        dataframe if overrule is set to False. Otherwise the binary matrix containing all the successfully parsed
        sequences is returned (default).
        """
        # Default is all 30 genomes, all orfs
        if genomelist is None:
            genomelist = self.strains
        if orthlist is None:
            orthlist = self.get_orth_list()
                    
        # Respository for mutation ids and presence/absence matrix, respectively
        mutationlistindex = []
        mutationbinary = []
        missinglist = []
        
        # Query database, obtain infoarray. An exception for missing sequences is handled
        for orth in orthlist:
            try:
                genometuples = tuple(genomelist)
                dbquery = 'SELECT cdhit_id, genome_id, seq_inframe_aligned FROM \
                view_orth WHERE genome_id IN %s AND cdhit_id = %s'%(str(genometuples), str(orth))
                infotable = self.db.getAllResultsFromDbQuery(dbquery)
                for i in range(np.shape(infotable)[0]):
                    infotable[i] = list(infotable[i])
                infoarray = np.array(sorted(infotable, key=lambda x: x[1]))
            
                # Create aligned array of amino acid sequences for each orf/genome
                dnas = [len(list(x)) for x in infoarray[:, 2]]
                if dnas.count(dnas[0]) == len(dnas):
                    if dnas[0] % 3 == 0:
                        protlen = dnas[0]/3
                        aminoacidarray = np.zeros((np.shape(infoarray)[0], protlen), dtype=str)
                        for i in range(np.shape(aminoacidarray)[0]):
                            dnaseq = Seq(infoarray[i][2], IUPAC.unambiguous_dna)
                            protseq = dnaseq.translate()
                            aminoacidarray[i, :] = np.array(list(protseq))
                        print 'Orth %s parsed successfully.'%(str(orth))
                    else:
                        print 'Orth %s seqs cannot be converted to protein seqs'%(str(orth))
                        continue
                else:
                    print 'Orth %s seqs are of different lengths, cannot align.'%(str(orth))
                    continue
                
                # Create mutation table
                for i in range(protlen):
                    resarray = aminoacidarray[:, i]
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
                        missinglist.append(np.array([row[0], row[1]]))
                continue
            
        # Generate final dataframe
        if len(mutationlistindex) == 0:
            print 'No mutations detected.'
        elif len(missinglist) != 0 and overrule is False:
            print 'Missing sequences detected. Returning dataframe of misses.'
            missingarray = np.array(missinglist)
            cols = {'cdhit_id', 'genome_id'}
            missingdf = pd.DataFrame(missingarray, columns=cols)
            return missingdf
        else:
            print 'Orths parsed successfully. Returning binary dataframe.'
            mutationarray = np.array(mutationbinary)
            mutationdf = pd.DataFrame(mutationarray, index=mutationlistindex, columns=genomelist)
            return mutationdf

    def get_genetic_mutation_dataframe(self, orthlist=None, genomelist=None, overrule=True):
        """
        Essentially runs identically to get_protein_mutation_dataframe, except synonymous mutations are
        included here.  For identifying protein mutations, the sequences are first translated using
        BioPython methods and then compared, which excludes all synonymous SNPs, whereas the comparisons
        made here are at the genetic sequence level.
        """
        # Default is all 30 genomes, all orfs
        if genomelist is None:
            genomelist = self.strains
        if orthlist is None:
            orthlist = self.get_orth_list()

        # Respository for mutation ids and presence/absence matrix, respectively
        mutationlistindex = []
        mutationbinary = []
        missinglist = []

        # Query database, obtain infoarray. An exception for missing sequences is handled
        for orth in orthlist:
            try:
                genometuples = tuple(genomelist)
                dbquery = 'SELECT cdhit_id, genome_id, seq_inframe_aligned FROM \
                view_orth WHERE genome_id IN %s AND cdhit_id = %s'%(str(genometuples), str(orth))
                infotable = self.db.getAllResultsFromDbQuery(dbquery)
                for i in range(np.shape(infotable)[0]):
                    infotable[i] = list(infotable[i])
                infoarray = np.array(sorted(infotable, key=lambda k: k[1]))

                # Create aligned array of amino acid sequences for each orf/genome
                dnas = [len(list(x)) for x in infoarray[:, 2]]
                if dnas.count(dnas[0]) == len(dnas):
                    seqlen = dnas[0]
                    nucleotidearray = np.zeros((np.shape(infoarray)[0], seqlen), dtype=str)
                    for i in range(np.shape(nucleotidearray)[0]):
                        dnaseq = Seq(infoarray[i][2], IUPAC.unambiguous_dna)
                        nucleotidearray[i, :] = np.array(list(dnaseq))
                    print 'Orth %s parsed successfully.'%(str(orth))
                else:
                    print 'Orth %s seqs are of different lengths, cannot align.'%(str(orth))
                    continue

                # Create mutation table
                for i in range(seqlen):
                    basearray = nucleotidearray[:, i]
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
                        missinglist.append(np.array([row[0], row[1]]))
                continue

        # Generate final dataframe
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
            mutationdf = pd.DataFrame(mutationarray, index=mutationlistindex, columns=genomelist)
            return mutationdf
            
    def complete_dataframe(self, synonymous=False, write=False):
        """
        Concatenates the presence/absence dataframe for non-core genes and the 
        complete mutation dataframe for the core genes. Optionally creates a
        .csv file with this information. If synonymous is set to True, then the
        get genetic mutations method is used; else the protein mutations method is
        used. *Key method for analysis classes*.
        """
        acc_genes = self.accessory_genome()
        if synonymous is True:
            mutations = self.get_genetic_mutation_dataframe(overrule=True)
        else:
            mutations = self.get_protein_mutation_dataframe(overrule=True)
        completedf = pd.concat([acc_genes, mutations], keys=['acc_gene', 'mutation'])
        if write is True:
            from datetime import date
            filename = 'complete_genotype_df_'+str(date.today())+'.csv'
            completedf.to_csv(filename)
        return completedf

    def phenotype_dataframe(self, phenlist=['swarm_diameter', 'biofilm'], cutoffs=[25, 1.2], continuous=False):
        """
        Generates a phenotype dataframe for subsequent analysis. The option to 
        return either the binned (binary) representation of biofilm formation 
        and swarming or the normalized analog representation is given, with the
        default set to binary. Additional phenotypes and custom queries may be 
        added in the future as the need arises.

        The default values for 'cutoffs' are those previously used to cluster
        the biofilm and swarming phenotypes. Since phenotypes will need to be
        converted to binary, determining an appropriate cutoff and then creating
        the dataframe with that value seems like the most economical option to
        accommodate additional phenotypes.
        """
        cols = ['genome_id'] + phenlist
        phen_dat = self.db.getAllResultsFromDbQuery('SELECT %s FROM phenotype WHERE genome_id IN %s'%(', '.join(cols),
                                                    str(tuple(self.strains))))
        phen_arr = np.array(phen_dat)

        phen_df_cont = pd.DataFrame(phen_arr, columns=cols)
        phen_df_cont[['genome_id']] = phen_df_cont[['genome_id']].astype(int)
        phen_df_cont = phen_df_cont.sort('genome_id')
        for phen in phenlist:
            phen_df_cont[phen] = self.misc.zscore(phen_df_cont[phen])
        strainslen = np.shape(phen_df_cont.values)[0]
        phen_df_cont.index = np.arange(1, strainslen+1)

        if continuous is False:
            for j in range(len(phenlist)):
                cutoff = cutoffs[j]
                filterer = []
                for i in range(strainslen):
                    entry = phen_arr[i, j+1] > cutoff
                    filterer.append(entry)
                filtererarr = np.array(filterer)
                phen_arr[filtererarr, j+1], phen_arr[~filtererarr, j+1] = 1, 0
            phen_arr = phen_arr.astype(int)
            phen_df_bin = pd.DataFrame(phen_arr, columns=cols).sort('genome_id')
            phen_df_bin.index = np.arange(1, strainslen+1)
            return phen_df_bin
        else:
            return phen_df_cont
        # Note: The index of phen_df is arbitrary at this stage, but it becomes convenient for downstream operations
            
    def swarm_biofilm_plot(self, save=False):
        """
        This function creates a scatterplot that is divided into four regions as 
        determined by k-means clustering (performed elsewhere; the red lines are 
        drawn at the appropriate boundaries). It may not be useful in its current 
        state if more phenotypes and strains are added, but it can serve as a 
        template if nothing else.
        """
        phen_dat = self.db.getAllResultsFromDbQuery('select genome_id, swarm_diameter, biofilm from phenotype')
        phen_dat = np.array(phen_dat)
        swarm = phen_dat[:, 1]
        biofilm = phen_dat[:, 2]
        nonsnonb = phen_dat[(swarm < 25) & (biofilm < 1.2)]
        snonb = phen_dat[(swarm > 25) & (biofilm < 1.2)]
        nonsb = phen_dat[(swarm < 25) & (biofilm > 1.2)]
        sb = phen_dat[(swarm > 25) & (biofilm > 1.2)]
        plot1 = pylab.plot(nonsnonb[:, 1], nonsnonb[:, 2], 'go')
        plot2 = pylab.plot(snonb[:, 1], snonb[:, 2], 'bo')
        plot3 = pylab.plot(nonsb[:, 1], nonsb[:, 2], 'mo')
        plot4 = pylab.plot(sb[:, 1], sb[:, 2], 'co')
        plot5 = pylab.plot([0, 80], [1.2, 1.2], 'r-')
        plot6 = pylab.plot([25, 25], [-0.5, 3], 'r-')
        pylab.xlabel('Swarm Diameter (mm)')
        pylab.ylabel('Biofilm Index')
        pylab.title('P. aeruginosa Phenotypes')
        if save is True:
            from datetime import date
            filename = 'phenotypes_scatterplot_'+str(date.today())+'.svg'
            pylab.savefig(filename, format='svg')
        pylab.show()

    def core_genome_bar(self, flip=False, save=False):
        """
        Note: One strain name is cut off if save is set to True; better to expand the pop-out figure and save it after.
        """
        strleg = self.strain_legend
        coregen = np.array(self.core_genome())
        totalgen = np.array(self.raw_presence_absence_dataframe())
        coresum = coregen.sum(axis=0)
        totalsum = totalgen.sum(axis=0).astype(float)
        fraction = np.divide(coresum, totalsum)
        strleg['Fraction'] = fraction
        strleg = strleg.sort(['Fraction'], ascending=False)

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ind = np.arange(len(self.strains))
        width = 1.0
        if flip is False:
            bars = ax.bar(ind, strleg['Fraction'], width)
            ax.set_xlim(0, len(self.strains))
            ax.set_ylim([np.min(fraction)-0.05, np.max(fraction)+0.05])
            ax.set_ylabel('Fraction')
            ax.set_title('Core Genome Fraction')
            ax.set_xticks(ind+width/2)
            xticks = strleg['Name']
            xticklabels = ax.set_xticklabels(xticks)
            pylab.setp(xticklabels, rotation=90, fontsize=10)
            ax.xaxis.set_ticks_position('none')
            pylab.show()
            if save is True:
                from datetime import date
                filename = 'core_fraction_bar_'+str(date.today())+'.svg'
                pylab.savefig(filename, format='svg')
        else:
            height = 1.0
            bars = ax.barh(ind, strleg['Fraction'], height=height)
            ax.set_ylim(0, len(self.strains)+3)
            ax.set_xlim(np.min(fraction)-0.05, np.max(fraction)+0.05)
            ax.set_xlabel('Fraction')
            ax.set_title('Core Genome Fraction')
            ax.set_yticks(ind+width/2)
            yticks = strleg['Name']
            yticklabels = ax.set_yticklabels(yticks)
            pylab.setp(yticklabels, fontsize=10)
            ax.yaxis.set_ticks_position('none')
            pylab.show()
            if save is True:
                from datetime import date
                filename = 'core_fraction_bar_flip_'+str(date.today())+'.svg'
                pylab.savefig(filename, format='svg')

    def rarefaction_curves(self, iterations=30, save=False):
        """
        Generates the rarefaction curves for the core genome, accessory genome, and pan genome, using a specified number
        of iterations.  The error bars are one standard deviation from the mean.  Creates a .svg file if save is True.
        """
        totalgen = self.raw_presence_absence_dataframe()
        strnum = len(self.strains)
        coregenomesizes = np.zeros((iterations, strnum))
        pangenomesizes = np.zeros((iterations, strnum))
        accgenomesizes = np.zeros((iterations, strnum))
        j = 0
        while j < iterations:
            corelist = []
            panlist = []
            acclist = []
            index = np.arange(strnum)
            np.random.shuffle(index)
            indexlist = list(index)
            i0 = indexlist.pop(0)
            strain0 = totalgen.iloc[:, i0]
            inter0 = np.array(strain0[strain0 == 1].index)
            union0 = inter0
            diff0 = []
            corelist.append(np.size(inter0))
            panlist.append(np.size(union0))
            acclist.append(np.size(diff0))
            while len(indexlist) > 0:
                i = indexlist.pop(0)
                strain = totalgen.iloc[:, i]
                genes = np.array(strain[strain == 1].index)
                inter = np.intersect1d(inter0, genes)
                corelist.append(np.size(inter))
                union = np.union1d(union0, genes)
                panlist.append(np.size(union))
                diff = np.setdiff1d(union, inter)
                acclist.append(np.size(diff))
                inter0 = inter
                union0 = union
            coregenomesizes[j, :] = corelist
            pangenomesizes[j, :] = panlist
            accgenomesizes[j, :] = acclist
            j += 1
        coremeans = np.mean(coregenomesizes, axis=0)
        corestds = np.std(coregenomesizes, axis=0)
        panmeans = np.mean(pangenomesizes, axis=0)
        panstds = np.std(pangenomesizes, axis=0)
        accmeans = np.mean(accgenomesizes, axis=0)
        accstds = np.std(accgenomesizes, axis=0)

        pylab.subplot(121)
        cores = pylab.errorbar(np.arange(1, strnum+1), coremeans, yerr=corestds, fmt='ro', label='Core')
        pans = pylab.errorbar(np.arange(1, strnum+1), panmeans, yerr=panstds, fmt='bo', label='Pan')
        pylab.legend(handles=[cores, pans])
        pylab.xlabel('Number of Included Genomes')
        pylab.ylabel('Number of Genes')
        pylab.title('Pan/Core Genome Rarefaction Curves')

        pylab.subplot(122)
        diffs = pylab.errorbar(np.arange(1, strnum+1), accmeans, yerr=accstds, fmt='go', label='Accessory')
        pylab.xlabel('Number of Included Genomes')
        pylab.ylabel('Number of Genes')
        pylab.title('Accessory Genome Rarefaction Curve')
        pylab.show()

        if save is True:
            from datetime import date
            filename = 'rarefaction_curves_'+str(date.today())+'.svg'
            pylab.savefig(filename, format='svg')

    def snp_vs_presence(self, mut=None, save=False):
        """
        This creates a plot of normalized number of SNPs vs normalized number of accessory genes for each genome.
        """
        accgen = self.accessory_genome()
        accsize = np.shape(accgen)[0]
        accsum = np.sum(np.array(accgen), axis=0)
        accnorm = accsum/float(accsize)
        if mut is None:
            mutgen = self.get_protein_mutation_dataframe()
        else:
            mutgen = mut
        mutsize = np.shape(mutgen)[0]
        mutsum = np.sum(np.array(mutgen), axis=0)
        mutnorm = mutsum/float(mutsize)
        strleg = self.strain_legend

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        plot1 = ax.plot(accnorm, mutnorm, 'bo')
        plot2 = ax.plot([np.min(accnorm)-0.025, np.max(accnorm)+0.025],
                        [np.min(mutnorm)-0.025, np.max(mutnorm)+0.025], 'r-')
        labels = strleg['Name']
        for i in range(len(labels)):
            pylab.annotate(labels[i], xy=(accnorm[i], mutnorm[i]), xytext=(10, 10), textcoords='offset points')
        ax.set_xlim(np.min(accnorm)-0.025, np.max(accnorm)+0.025)
        ax.set_ylim(np.min(mutnorm)-0.025, np.max(mutnorm)+0.025)
        ax.set_xlabel('Fraction of Accessory Genome')
        ax.set_ylabel('Fraction of Nonsynonyous SNPs')
        ax.set_title('Strain Genome Content')
        pylab.show()

        if save is True:
            from datetime import date
            filename = 'snp_presence_'+str(date.today())+'.svg'
            pylab.savefig(filename, format='svg')

    def pairwise_diff_plot(self, mut=None, save=False):
        """
        This plots the differences between each pair of strains in terms of presence/absence of genes and nonsyn. SNPs.
        """
        if mut is None:
            mutgen = self.get_protein_mutation_dataframe()
        else:
            mutgen = mut
        mutarr = np.array(mutgen)
        mutsize = float(np.shape(mutarr)[0])
        accgen = self.accessory_genome()
        accarr = np.array(accgen)
        accsize = float(np.shape(accarr)[0])

        i = 0
        mutfraclist = []
        accfraclist = []
        while i < len(self.strains):
            for j in range(i+1, len(self.strains)):
                accdiff = np.abs(accarr[:, i] - accarr[:, j])
                accsum = np.sum(np.squeeze(accdiff))
                accfrac = accsum/accsize
                accfraclist.append(accfrac)
                mutdiff = np.abs(mutarr[:, i] - mutarr[:, j])
                mutsum = np.sum(np.squeeze(mutdiff))
                mutfrac = mutsum/mutsize
                mutfraclist.append(mutfrac)
            i += 1
        relatedacc = 1-np.array(accfraclist)
        relatedmut = 1-np.array(mutfraclist)

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        plot1 = ax.plot(relatedacc, relatedmut, 'go')
        plot2 = ax.plot([0.5, 1], [0.5, 1], 'r-')
        ax.set_xlim(0.5, 1)
        ax.set_ylim(0.5, 1)
        ax.set_xlabel('Accessory Genome Similarity')
        ax.set_ylabel('Nonsynonyous SNP Similarity')
        ax.set_title('Pairwise Relatedness Between Strains')
        pylab.show()

        if save is True:
            from datetime import date
            filename = 'strain_diffs_'+str(date.today())+'.svg'
            pylab.savefig(filename, format='svg')

#%%

class PCA_Tools:
    """
    This class has a number of methods associated with related to outputting 
    data from PCA decomposition. Instantiating this class creates a fit object
    using the RandomizedPCA function from the Scikit Learn package (see
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.RandomizedPCA.html).
    This fit object is then passed to the methods defined below.
    """
    def __init__(self, df, ncomps=10):
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
        pca_arr = self.pcafit.components_.T  # default is to have PCs along the 0 axis
        pcacols = ['PC%d'%(x) for x in np.arange(1, np.shape(pca_arr)[1]+1)]
        pca_df = pd.DataFrame(pca_arr, index=np.arange(1, np.shape(pca_arr)[0]+1), columns=pcacols)
        return pca_df
    
    def variance_histogram(self, write=False):
        """
        Outputs a histogram showing the distribution of explained variance by each principal component. Option to write
        to a .png file.
        """
        expvar = self.pcafit.explained_variance_ratio_
        histplot = pylab.bar(np.arange(1, self.ncomps+1), expvar)
        pylab.xlabel('PC #')
        pylab.ylabel('Density')
        percentv = 100*sum(expvar)
        pylab.title('Total explained variance is %f percent'%(percentv))
        if write is True:
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
        if write is True:
            from datetime import date
            filename = 'pc_2d_scatterplot'+str(date.today())+'.png'
            pylab.savefig(filename)
        pylab.show()
    # More to come as desired

#%%
class PCA_Regression:
    """
    This class is designed to contain all relevant functions to conduct a linear
    regression using principal components as covariates (or not). A conceptual
    overview of the approach:
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
        self.gen_df = gen_df
        self.ncomps = ncomps
        self.cont = continuous_phen
        psdfs = PseudomonasDataframes()
        self.phen_df = psdfs.phenotype_dataframe(continuous=self.cont)
        self.strains = self.phen_df['genome_id']

        pctools = PCA_Tools(self.gen_df, ncomps=self.ncomps)
        self.pca_df = pctools.pca_dataframe()

        self.misc = MiscFunctions()
        self.unique_arrays_df = self.misc.arrays_df(self.gen_df)
        self.word_count = np.shape(self.unique_arrays_df)[1]      
        self.complete_df = pd.concat((self.phen_df, self.pca_df, self.unique_arrays_df), axis=1)

        # The self.unique_words_df created below is important for reindexing in significant_hits_summary
        arrstranspose = self.unique_arrays_df.transpose()
        uniquewords = np.dot(arrstranspose.values, 2**np.arange(0,np.shape(self.gen_df)[1]))
        self.unique_words_df = pd.DataFrame(self.unique_arrays_df.columns, index=uniquewords)
                                     
    def simple_regression(self, intercept=True, phenotypes=['biofilm', 'swarm_diameter']):
        """
        This function performs a linear regression without incorporating any 
        covariates (and therefore not correcting for population structure). It 
        returns an array of p-values obtained using the StatsModels OLS package. 
        It can take the convenient R-like syntax for its lm() function using the 
        package patsy, which is used here.
        
        At the time of this version, no analysis has been done without including 
        a parameter in the models for the intercept. This can be fixed at 0 with
        the intercept parameter set to False, which may be desired in future tests.
        """
        compdf = self.complete_df
        phenpvaldf = pd.DataFrame(np.zeros((self.word_count, len(phenotypes))), index=np.arange(1, self.word_count+1),
                                  columns=phenotypes)
        
        for phen in phenotypes:
            pvals = []
            for i in range(1, self.word_count+1):
                if intercept is True:
                    y1, X1 = dmatrices('%(phenotype)s ~ word%(index)d'%{'phenotype': phen, 'index': i}, data=compdf,
                                       return_type='dataframe')
                else:
                    y1, X1 = dmatrices('%(phenotype)s ~ word%(index)d - 1'%{'phenotype': phen, 'index': i}, data=compdf,
                                       return_type='dataframe')
                mod1 = sm.OLS(y1, X1)
                res1 = mod1.fit()
                pval1 = res1.f_pvalue
                pvals.append(pval1)
            logp = -np.log10(pvals)
            phenpvaldf[phen] = np.array(logp)
        print 'Simple regression finished'
        return phenpvaldf
        
    def pca_regression(self, princomps=3, intercept=True, phenotypes=['biofilm', 'swarm_diameter']):
        """
        This function performs a linear regression using the first X (default 3) 
        principal components as covariates. A likelihood ratio test is employed 
        to test the null and alternative models (an F-statistic could be found 
        as well, but this seemed easier to implement in StatsModels and there is
        virtually no difference between the results of both when I tested them).
        Again, the intercept as a variable has not been explored yet but the option
        is included here.
        """
        compdf = self.complete_df
        phenpvaldf = pd.DataFrame(np.zeros((self.word_count, len(phenotypes))), index=np.arange(1, self.word_count+1),
                                  columns=phenotypes)
        # Creates the input string argument to dmatrices corresponding to the PC columns of compdf
        pclist = range(1, princomps+1)
        pcstring = ''
        i = 0
        while i < princomps-1:
            pcstring = pcstring + 'PC%d + '%(pclist[i])
            i += 1
        pcstring = pcstring + 'PC%d'%(pclist[-1])
        
        for phenotype in phenotypes:
            pvals = []
            if intercept is True:
                yn1, Xn1 = dmatrices('%s ~ %s'%(phenotype, pcstring), data=compdf, return_type='dataframe')
            else:
                yn1, Xn1 = dmatrices('%s ~ %s - 1'%(phenotype, pcstring), data=compdf, return_type='dataframe')
            modn1 = sm.OLS(yn1, Xn1)
            resn1 = modn1.fit()
            llhn1 = resn1.llf
            for i in range(1,self.word_count+1):
                if intercept is True:
                    y1, X1 = dmatrices('%s ~ %s + word%d'%(phenotype, pcstring, i), data=compdf,
                                       return_type='dataframe')
                else:
                    y1, X1 = dmatrices('%s ~ %s + word%d - 1'%(phenotype, pcstring, i), data=compdf,
                                       return_type='dataframe')
                mod1 = sm.OLS(y1, X1)
                res1 = mod1.fit()
                llh1 = res1.llf
                pval1 = 1-chi2.cdf(2*(llh1-llhn1), 1)
                pvals.append(pval1)
            logp = -np.log10(pvals)
            phenpvaldf[phenotype] = np.array(logp)
        print 'PC regression finished'
        return phenpvaldf

    def significant_hit_arrays(self, simple=False, princomps=3, intercept=True, phenotype='biofilm', signif=0.05):
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
        if simple is True:
            pvaldf = self.simple_regression(intercept=intercept, phenotypes=[phenotype])
        else:
            pvaldf = self.pca_regression(princomps=princomps, intercept=intercept, phenotypes=[phenotype])
        phenlist = ['genome_id']+phenotype
        bonferroni = -np.log10(signif/self.word_count)
        compdf = self.complete_df.copy()
        sighits = pvaldf.values[pvaldf.values > bonferroni]
        sighits.index = ['word%d' % x for x in sighits.index]
        hitsdf = compdf[sighits.index]
        fulldf = pd.concat((compdf[phenlist], hitsdf), axis=1)
        return fulldf
    
    def significant_hits_summary(self, simple=False, princomps=3, intercept=True, phenotype='biofilm', signif=0.05,
                                 write=False):
        """
        This function is similar to the previous one and shares much of the same 
        code, but returns a dataframe (can write to file) that contains all the 
        important information about the significant hits.
        
        For the moment, phenotype can either be set to biofilm or swarm diameter, but both are
        not handled together. That flexibility should be added in a future version
        (see docstrings for significant_hits_array and simple_regression).

        Additionally, currently cdhit_id is used to index, but in the future this should
        be converted to the standard locus names (or they should just be added).
        """
        if simple is True:
            pvaldf = self.simple_regression(intercept=intercept, phenotypes=[phenotype])
        else:
            pvaldf = self.pca_regression(princomps=princomps, intercept=intercept, phenotypes=[phenotype])

        pshape = np.shape(pvaldf)[0]
        pvalarray = np.concatenate((np.array(pvaldf.index).reshape(pshape, 1),
                                    pvaldf[phenotype].values.reshape(pshape, 1)), axis=1)  # clumsy but bypasses pd bug
        bonferroni = -np.log10(signif/self.word_count)
        compdf = self.complete_df.copy()
        sighitsarray = pvalarray[pvalarray[:, 1] > bonferroni]
        sighits = pd.DataFrame(sighitsarray[:, 1], index=sighitsarray[:, 0])
        sighits.index = ['word%d' % x for x in sighits.index]
        hitsdf = compdf[sighits.index]
        wordsarray = np.dot(self.gen_df, 2**np.arange(0, len(self.strains))).reshape(np.shape(self.gen_df)[0], 1)

        # Recall that self.unique_words_df contains the word id strings indexed by word (decimal) values
        wordidsarray = np.array([self.unique_words_df.ix[word]
                                 for word in wordsarray]).reshape(np.shape(self.gen_df)[0], 1)
        wordsdf = pd.DataFrame(np.concatenate((wordsarray, wordidsarray), axis=1), index=self.gen_df.index,
                               columns=['words', 'wordids'])
        hitwords = np.dot(hitsdf.values.T, 2**np.arange(0, len(self.strains)))
        filterlist = [x in hitwords for x in wordsdf['words']]
        sumwordidsdf = wordsdf[filterlist]

        # np.squeeze is necessary below because the listcomp won't work with a column vector, which is 2D
        pvalarray = np.array([sighits.ix[wordid] for wordid in np.squeeze(sumwordidsdf['wordids'])])
        sumdf = pd.DataFrame(pvalarray, index=sumwordidsdf.index, columns=['-log(p_values)'])
        if write is True:
            from datetime import date
            filename = 'hits_summary_'+phenotype+'_'+str(date.today())+'.csv'
            sumdf.to_csv(filename)
        print 'Returning significant hits:'
        return sumdf
#%%
class PCA_Plotter:
    """
    This class contains methods to make Manhattan and Q-Q plots, and maybe more
    in the future. It requires specific arguments to pass to the various methods
    of the Regression class.

    Currently it plots using word ids, but in later versions it should plot by
    genomic position (in PA14, for example) instead.
    """
    def __init__(self, gen_df, simple=False, princomps=3, intercept=True, signif=0.05):
        self.gen_df = gen_df
        reg = PCA_Regression(self.gen_df)
        self.phens = ['biofilm', 'swarm_diameter']
        if simple is True:
            self.pvaldf = reg.simple_regression(intercept=intercept, phenotypes=self.phens)
        else:
            self.pvaldf = reg.pca_regression(princomps=princomps, intercept=intercept, phenotypes=self.phens)
        self.bonferroni = -np.log10(signif/np.size(self.pvaldf))
        self.phens = ['biofilm', 'swarm_diameter']

    def qq(self, phenotype=None,  write=False):
        if phenotype is None:
            phenotype = self.phens[0]
        pvaldf = self.pvaldf[phenotype]
        sortedpvals = np.sort(np.squeeze(pvaldf))
        sortednull = np.sort(-np.log10(np.linspace(1./self.pvaldf.index[-1], 1, self.pvaldf.index[-1])))
        plot1 = pylab.plot(sortednull, sortedpvals, 'bo')
        plot2 = pylab.plot([0, sortednull[-1]], [0, sortednull[-1]], 'r-')
        pylab.xlabel('Expected p-val')
        pylab.ylabel('Actual p-val')
        pylab.title('PCA Q-Q Plot: %s'%(phenotype))
        if write is True:
            from datetime import date
            filename = 'qq_'+phenotype+'_'+str(date.today())+'.svg'
            pylab.savefig(filename, format='svg')
        pylab.show()

    def manhattan(self, phenotype=None, write=False):
        if phenotype is None:
            phenotype = self.phens[0]
        pvaldf = self.pvaldf[phenotype]
        graph1 = pylab.plot(pvaldf.index, pvaldf.values, 'bo')
        graph2 = pylab.plot([pvaldf.index[0], pvaldf.index[-1]], [self.bonferroni, self.bonferroni], 'r-')
        pylab.xlabel('word id')
        pylab.ylabel('-log10(p-val)')
        pylab.title('PCA Manhattan Plot: %s'%(phenotype))
        if write is True:
            from datetime import date
            filename = 'manhattan_'+phenotype+'_'+str(date.today())+'.svg'
            pylab.savefig(filename, format='svg')
        pylab.show()

class GLS_Regression:
    """
    This class contains all the necessary components to perform a GLS regression using the covariance matrix generated
    from the genotype dataframe to account for relatedness between strains. See the docstrings of individual methods
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
        self.strain_legend = np.array(list(self.psdfs.strain_legend))
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

    def phenotype_regression(self, phen_ind='swarm_diameter', phen_dep='biofilm', intercept=True):
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
        pval = 1-f.cdf(fstat, 1, 28)
        summary = pd.Series([float(beta[0]), float(beta[1]), rsq, pval], index=['Intercept', 'Slope', 'R^2', 'P_value'])
        return summary

    def genotype_regression1(self, phenotypes=['biofilm'], intercept=True):
        """
        The next few methods are designed to perform the same analysis in different ways.

        Goal: Repeat the PCA_Regression analysis, independently testing each gene against a particular phenotype,
        except using the (Cholesky-decomposed) covariance matrix to transform the data instead of incorporating PCs
        as covariates.  Inputs are gen_df and phenotype_df as constructed in the PseudomonasDataframes class above.
        This allows for multiple phenotypes to be tested at once if df is set to True; otherwise a single phenotype can
        be examined independently.  binary_phen specifies whether continuous or binary phenotypes should be tested as
        it applies to the different traits). Intercept is as above in PCA_Regression.

        Notes on this method:
        - Binary phenotype (at first)
        - Testing the transformed phenotypes against the transformed genotypes should be no different from testing them
          straight up: T_inv(y) = alpha + beta*T_inv(x) is no different from y = alpha' + beta*x, barring a trivial
          readjustment of the intercept (not an interesting parameter).  Here, only genotypes should be transformed to
          reflect their phylogeny: y = alpha + beta*T_inv(x).
        - Simple OLS regression, no covariates, one degree of freedom (of the mean):
        - Uses statsmodels and patsy (slow)
        """
        if intercept is False:
            pass
        covmat = self.covariance
        T = np.linalg.cholesky(covmat)
        Tinv = np.linalg.inv(T)
        compdf = self.complete_df.copy()    # Regression modifies original df if a copy isn't used
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
        print 'GLS regression finished'
        return phenpvaldf

    def genotype_regression2(self, phenotypes=['biofilm'], intercept=True):
        """
        Notes on this method:
        - Binary phenotype (at first)
        - Testing the transformed phenotypes against the transformed genotypes should be no different from testing them
          straight up: T_inv(y) = alpha + beta*T_inv(x) is no different from y = alpha' + beta*x, barring a trivial
          readjustment of the intercept (not an interesting parameter).  Here, only genotypes should be transformed to
          reflect their phylogeny: y = alpha + beta*T_inv(x).
        - Simple OLS regression, no covariates, one degree of freedom (of the mean):
        - Does regression from scratch - exact analytical solution (fast)
        - As of 10/2/15, the results from the previous method and this one are identical
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
        print 'GLS regression finished'
        return phenpvaldf

    def significant_hits_df(self, phenotype=['biofilm'], intercept=True, signif=0.05):
        """
        Performs a GLS regression for the given phenotypes and choice of inclusion of intercept, returning a subset of
        the complete dataframe (all genotypes) including the phenotypes and the significant genotypes (words).  Can only
        look at one phenotype at a time.
        """
        pvaldf = self.genotype_regression2(phenotypes=phenotype, intercept=intercept)
        pvalarray = np.concatenate((np.array(pvaldf.index).reshape(np.shape(pvaldf)[0], 1),
                                    pvaldf[phenotype].values), axis=1)  # This is clumsy but bypasses a bug in pandas
        phenlist = ['genome_id']+phenotype
        bonferroni = -np.log10(signif/self.word_count)
        compdf = self.complete_df.copy()
        sighitsarray = pvalarray[pvalarray[:, 1] > bonferroni]
        sighits = pd.DataFrame(sighitsarray[:, 1], index=sighitsarray[:, 0])
        sighits.index = ['word%d' % x for x in sighits.index]
        hitsdf = compdf[sighits.index]
        fulldf = pd.concat((compdf[phenlist], hitsdf), axis=1)
        return fulldf

    def significant_hits_summary(self, phenotype='biofilm', intercept=True, signif=0.05, write=False):
        """
        This function is similar to the previous one and shares much of the same code, but returns a dataframe (can
        write to file) that contains all the important information about the significant hits.  Can only look at one
        phenotype at a time.

        Additionally, currently cdhit_id is used to index, but in the future this should
        be converted to the standard locus names (or they should just be added)
        """
        pvaldf = self.genotype_regression2(phenotypes=[phenotype], intercept=intercept)
        pshape = np.shape(pvaldf)[0]
        pvalarray = np.concatenate((np.array(pvaldf.index).reshape(pshape, 1),
                                    pvaldf[phenotype].values.reshape(pshape, 1)), axis=1)  # clumsy but bypasses pd bug
        bonferroni = -np.log10(signif/self.word_count)
        compdf = self.complete_df.copy()
        sighitsarray = pvalarray[pvalarray[:, 1] > bonferroni]
        sighits = pd.DataFrame(sighitsarray[:, 1], index=sighitsarray[:, 0])
        sighits.index = ['word%d' % x for x in sighits.index]
        hitsdf = compdf[sighits.index]
        wordsarray = np.dot(self.gen_df, 2**np.arange(0, len(self.strains))).reshape(np.shape(self.gen_df)[0], 1)
        # Recall that self.unique_words_df contains the word id strings indexed by word (decimal) values
        wordidsarray = np.array([self.unique_words_df.ix[word]
                                 for word in wordsarray]).reshape(np.shape(self.gen_df)[0], 1)
        wordsdf = pd.DataFrame(np.concatenate((wordsarray, wordidsarray), axis=1), index=self.gen_df.index,
                               columns=['words', 'wordids'])
        hitwords = np.dot(hitsdf.values.T, 2**np.arange(0, len(self.strains)))
        filterlist = [x in hitwords for x in wordsdf['words']]
        sumwordidsdf = wordsdf[filterlist]
        # np.squeeze is necessary below because the listcomp won't work with a column vector, which is 2D
        pvalarray = np.array([sighits.ix[wordid] for wordid in np.squeeze(sumwordidsdf['wordids'])])
        sumdf = pd.DataFrame(pvalarray, index=sumwordidsdf.index, columns=['-log(p_values)'])
        if write is True:
            from datetime import date
            filename = 'hits_summary_'+phenotype+'_'+str(date.today())+'.csv'
            sumdf.to_csv(filename)
        return sumdf


#%%
class GLS_Plotter:
    """
    This class contains methods to make Phenotype Regression, Manhattan, Q-Q plots, and maybe more
    in the future. It requires specific arguments to pass to the various methods of the Regression class.

    Note: instantiating this class will yield the following output:
    'Connecting to SPRING-DB...
    Connected
    Connecting to SPRING-DB...
    Connected
    Finished'

    The reason for the duplicate 'Connecting...Connected' output is that the PseudomonasDataframes class is called
    twice (once in GLS_Regression, once here).  'Finished' refers to performing the genotype2_regression, which will
    be preceded by a delay - wait until it appears before executing further commands.
    """
    def __init__(self, gen_df, phens=['swarm_diameter', 'biofilm'], binary_phen=True, intercept=True, signif=0.05):
        """
        Relatively straightforward; GLS_Regression.genotype_regression2 is performed here so that it can be passed
        to the multiple methods below that require its output.
        """
        self.gen_df = gen_df
        self.binary = binary_phen
        self.phens = phens
        self.intercept = intercept
        self.psdfs = PseudomonasDataframes()
        self.misc = MiscFunctions()
        self.gls = GLS_Regression(self.gen_df, phens=self.phens, binary_phen=self.binary)
        self.bonferroni = -np.log10(signif/self.gls.word_count)
        self.phen_df = self.gls.phen_df
        self.strains = self.phen_df['genome_id']
        self.pvaldf = self.gls.genotype_regression2(phenotypes=self.phens, intercept=self.intercept)

    def phenotype_regression_plot(self, phen_ind=None, phen_dep=None):
        """
        Recreates the phenotype regression plot in the manuscript.  Unfortunately the phenotype regression must be
        calculated from scratch (code pasted from GLS_Regression.phenotype_regression), but it's a quick calculation.
        """
        if phen_ind is None:
            phen_ind = self.phens[0]
        if phen_dep is None:
            phen_dep = self.phens[1]
        gendf = self.gen_df
        if self.binary is True:
            phen_cont = self.psdfs.phenotype_dataframe(continuous=True)    # Cont. required for phen/phen regression
        else:
            phen_cont = self.phen_df
        indphen = phen_cont[phen_ind].reshape(len(self.strains), 1)
        depphen = phen_cont[phen_dep].reshape(len(self.strains), 1)
        strainlen = len(self.strains)
        covmat = np.cov(gendf.transpose())

        T = np.linalg.cholesky(covmat)
        Y_c = np.dot(np.linalg.inv(T), depphen)
        X_c = np.dot(np.linalg.inv(T), indphen)
        X_c = np.concatenate((np.ones((strainlen, 1)), X_c), axis=1)
        beta = np.dot(np.linalg.inv(np.dot(X_c.T, X_c)), np.dot(X_c.T, Y_c))
        y_exp = np.dot(X_c, beta)
        sst = np.sum((Y_c-np.mean(Y_c))**2)
        ssr = np.sum((y_exp-np.mean(Y_c))**2)
        sse = np.sum((Y_c-y_exp)**2)
        rsq = 1-sse/sst
        fstat = ssr/(sse/28)
        pval = 1-f.cdf(fstat, 1, 28)

        x = X_c[:, -1]
        y = Y_c
        plot1, = pylab.plot(x, y, 'bo')
        labels = [np.array(self.psdfs.strain_legend)[i, 1] for i in range(len(self.strains))]
        for i in range(len(labels)):
            pylab.annotate(labels[i], xy=(x[i], y[i]), xytext=(10, 10), textcoords='offset points')
        x2 = np.linspace(int(np.min(X_c))-50, int(np.max(X_c))+50, 50)
        y2 = beta[0]+x2*beta[1]
        plot2, = pylab.plot(x2, y2, 'r-')
        pylab.text(np.min(X_c)-50, min(Y_c)-25, 'y = {0} + {1}(x) \nR-Squared = {2} \nP-value = {3}'.format(float(beta[0]), float(beta[1]), rsq,
                                                                              pval), horizontalalignment='left')
        pylab.xlabel('%s (transformed)'%(phen_ind))
        pylab.ylabel('%s (transformed)'%(phen_dep))
        pylab.title('Phenotype Regression')
        pylab.show()

    def qq(self, phenotype=None, write=False):
        """
        Creates a quantile-quantile plot.
        """
        if phenotype is None:
            phenotype = self.phens[0]
        pvaldf = self.pvaldf[phenotype]
        sortedpvals = np.sort(np.squeeze(pvaldf))
        sortednull = np.sort(-np.log10(np.linspace(1./self.pvaldf.index[-1], 1, self.pvaldf.index[-1])))
        plot1 = pylab.plot(sortednull, sortedpvals, 'bo')
        plot2 = pylab.plot([0, sortednull[-1]], [0, sortednull[-1]], 'r-')
        pylab.xlabel('Expected p-val')
        pylab.ylabel('Actual p-val')
        pylab.title('Phyl. Q-Q Plot: %s'%(phenotype))
        if write is True:
            from datetime import date
            filename = 'qq_'+phenotype+'_'+str(date.today())+'.png'
            pylab.savefig(filename)
        pylab.show()

    def manhattan(self, phenotype=None, write=False):
        """
        Need to queue this by start site, for example PA14.

        orf_legend = np.array(list(self.psdfs.orf_legend))
        orflegdf = pd.DataFrame(orf_legend, columns=['cdhit_id', 'start', 'std_locus_name'])
        orflegdf = orflegdf[orflegdf['std_locus_name'].notnull()]
        """
        if phenotype is None:
            phenotype = self.phens[0]
        pvaldf = self.pvaldf[phenotype]
        graph1 = pylab.plot(pvaldf.index, pvaldf.values, 'bo')
        graph2 = pylab.plot([pvaldf.index[0], pvaldf.index[-1]],
                            [self.bonferroni, self.bonferroni], 'r-')
        pylab.xlabel('word id')
        pylab.ylabel('-log10(p-val)')
        pylab.title('Phyl. Manhattan Plot: %s'%(phenotype))
        if write is True:
            from datetime import date
            filename = 'manhattan_'+phenotype+'_'+str(date.today())+'.png'
            pylab.savefig(filename)
        pylab.show()

class GenomeExporter:
    """
    This class contains the necessary methods to write the genome sequences and their gene annotations of strains
    in the borreliabase.org pa2 database to a fasta file.  The 'genomes' argument should be in the form of a list, even
    if there is only one genome to be passed to the function.  fasta_genome() exports the raw sequence (separated into
    contigs if necessary) with a header of: >gi|strain_name|genome_id.contig_id|contig_start|contig_end.
    """
    def __init__(self):
        """
        The several dictionaries created here are necessary for various aspects of creating the fasta headers. Any gene
        annotations originate from loci in either PA01 or PA14, as reflected by the boolean indexing of the raw df
        gene_code_df.  The genecode.csv file derives from the 'paerug' database in borreliabase.org, which was created
        directly from postgres.  Writing the csv file was less arduous than trying to use psycopg2 to query both pa2
        and paerug, but the databases have yet to be condensed.  Be aware that the path will be different on different
        machines.

        Worth noting: The formation of cdhit_locus_dict from locus_cdhit_dict as opposed to reversing the order of the
        locus/cdhit query is necessary because multiple a single cdhit id can map to multiple locus names and keys must
        be unique. The problematic scenario is that the reversed query could map a cdhit id to a locus name of a
        strain other than PA01 or PA14, which would then not map to a gene in locus_gene_dict (which only contains loci
        from PA01 and PA14). As written there should be minimal loss of information.
        """
        import SpringDb_local as sdb
        reload(sdb)
        self.db = sdb.SpringDb()
        self.strain_legend = dict(self.db.getAllResultsFromDbQuery('SELECT genome_id, strain_name FROM genome'))
        self.assembly_legend = dict(self.db.getAllResultsFromDbQuery('SELECT genome_id, assembly_status FROM genome'))
        # The following file is located within the github directory - change the path as needed.
        self.gene_code_df = pd.read_csv('/Users/torokj/Documents/Pseudomonas-GWAS/genecode.csv', header=None)
        gene_code_arr = np.array(self.gene_code_df)
        loci_number = np.shape(gene_code_arr)[0]
        boolean_index = []
        for i in range(loci_number):
            val = len(list(gene_code_arr[i, 0])) == 6 or 'PA14' in gene_code_arr[i, 0]
            boolean_index.append(val)
        gene_code_arr2 = gene_code_arr[np.array(boolean_index)]
        self.locus_gene_dict = dict(gene_code_arr2)
        self.locus_cdhit_dict = dict(self.db.getAllResultsFromDbQuery('SELECT std_locus_name, cdhit_id FROM orf '
                                                                      'WHERE std_locus_name IS NOT NULL'))
        self.cdhit_gene_dict = dict()
        self.cdhit_locus_dict = dict()
        for key in self.locus_gene_dict.keys():
            if key in self.locus_cdhit_dict.keys():
                self.cdhit_gene_dict[self.locus_cdhit_dict[key]] = self.locus_gene_dict[key]
                self.cdhit_locus_dict[self.locus_cdhit_dict[key]] = key

    @staticmethod
    def spacer_finder(seq):
        seqlist = list(seq)
        spacer = 'NNNNNCACACACTTAATTAATTAAGTGTGTGNNNNN'
        spacerlist = list(spacer)
        seqlen = len(seqlist)
        spacerlen = len(spacerlist)
        spalocs = []
        for i in range(1, 1+seqlen-spacerlen):
            if seqlist[i:i+spacerlen] == spacerlist:
                spalocs.append(i)
        return spalocs

    def fasta_genome(self, genomes=None):
        """
        The genomes in pa2 are either closed or consist of several long contigs that have been concatenated together and
        separated by a characteristic 36-bp spacer.  This is for getting raw whole-genome sequence files.
        """
        from Bio.Seq import Seq
        from Bio.Alphabet import IUPAC
        from Bio.SeqRecord import SeqRecord
        from Bio import SeqIO
        gen = genomes
        if gen is None:
            query = self.db.getAllResultsFromDbQuery('SELECT genome_id FROM phenotype')
        elif len(gen) == 1:
            query_text = 'SELECT DISTINCT(genome_id) FROM orf WHERE genome_id = {}'.format(str(gen[0]))
            query = self.db.getAllResultsFromDbQuery(query_text)
        else:
            strains = sorted(np.squeeze(np.array(gen)))
            query_text = 'SELECT DISTINCT(genome_id) FROM orf WHERE genome_id IN {}'.format(str(tuple(strains)))
            query = self.db.getAllResultsFromDbQuery(query_text)
        query = [x[0] for x in query]
        gen_dict = dict(enumerate(query))
        for i in range(len(gen_dict.keys())):
            seq_query = 'SELECT seq FROM genome_seq WHERE genome_id = {}'.format(gen_dict[i])
            raw_data = self.db.getAllResultsFromDbQuery(seq_query)
            seq = Seq(str(raw_data)[3:-3], alphabet=IUPAC.unambiguous_dna)
            strain_note = 'strain_name={}'.format(self.strain_legend[gen_dict[i]])
            gen_note = 'genome_id={}'.format(gen_dict[i])
            if self.assembly_legend[gen_dict[i]] == 'closed':
                contig_note = 'contig_id=0'
                start_note = 'start=1'
                end_note = 'end={}'.format(len(seq))
                seq_id = 'gi|' + strain_note + '|' + gen_note + '|' + contig_note + '|' + start_note + '|' + end_note
                seq_rec = SeqRecord(seq, id=seq_id, description='')
                SeqIO.write(seq_rec, 'genome_{}_raw_sequence'.format(self.strain_legend[gen_dict[i]])+'.fasta',
                            'fasta')
                print 'strain_{}'.format(self.strain_legend[gen_dict[i]])+' is complete. #{}/{}'.format(i+1, len(query))
            elif self.assembly_legend[gen_dict[i]] == 'supercontig':
                contig_locs = list([0])
                contig_locs.extend(self.spacer_finder(seq))
                contig_locs.append(len(list(seq)))
                buff = 36
                rec_list = []
                for j in range(len(contig_locs)-1):
                    if j == 0:
                        start = contig_locs[j]   # This is the Pythonic start of the sequence in 'seq'
                    else:
                        start = contig_locs[j]+buff
                    end = contig_locs[j+1]   # This is 1 + the Pythonic end in 'seq'; i.e. the start of the next spacer
                    contig = seq[start: end]
                    contig_note = 'contig_id={}'.format(j+1)
                    start_note = 'start={}'.format(1+start-buff*j)
                    end_note = 'end={}'.format(end-buff*j)
                    seq_id = 'gi|' + strain_note + '|' + gen_note + '|' + contig_note + '|' + \
                             start_note + '|' + end_note
                    seq_rec = SeqRecord(contig, id=seq_id, description='')
                    rec_list.append(seq_rec)
                SeqIO.write(rec_list, 'genome_{}_raw_sequence'.format(self.strain_legend[gen_dict[i]])+'.fasta',
                            'fasta')
                print 'strain_{}'.format(self.strain_legend[gen_dict[i]])+' is complete. #{}/{}'.format(i+1, len(query))

    def fasta_annotation(self, genomes=None):
        """
        Outputs the ORFs and their annotations (if they're documented).
        """
        from Bio.Seq import Seq
        from Bio.Alphabet import IUPAC
        from Bio.SeqRecord import SeqRecord
        from Bio import SeqIO
        gen = genomes
        if gen is None:
            query = self.db.getAllResultsFromDbQuery('SELECT genome_id FROM phenotype')
        elif len(gen) == 1:
            query_text = 'SELECT DISTINCT(genome_id) FROM orf WHERE genome_id = {}'.format(str(gen[0]))
            query = self.db.getAllResultsFromDbQuery(query_text)
        else:
            strains = sorted(np.squeeze(np.array(gen)))
            query_text = 'SELECT DISTINCT(genome_id) FROM orf WHERE genome_id IN {}'.format(str(tuple(strains)))
            query = self.db.getAllResultsFromDbQuery(query_text)
        query = [x[0] for x in query]
        gen_dict = dict(enumerate(query))
        for i in range(len(gen_dict.keys())):
            seq_query = 'SELECT cdhit_id, seq, start, stop, strand FROM orf WHERE cdhit_id IS NOT NULL AND ' \
                        'genome_id = {}'.format(gen_dict[i])
            raw_data = self.db.getAllResultsFromDbQuery(seq_query)
            rec_list = []
            if self.assembly_legend[gen_dict[i]] == 'closed':
                for j in range(len(raw_data)):
                    seq = Seq(raw_data[j][1], alphabet=IUPAC.unambiguous_dna)
                    cdhit = raw_data[j][0]
                    seq_id_list = ['strain_name=' + self.strain_legend[gen_dict[i]], 'genome_id={}'.format(gen_dict[i]),
                                   'contig_id=0', 'orf_id={}'.format(cdhit)]
                    if cdhit in self.cdhit_locus_dict.keys():
                        locus = self.cdhit_locus_dict[cdhit]
                        seq_id_list.append('locus_name={}'.format(locus))
                    else:
                        seq_id_list.append('locus_name= ')
                    if cdhit in self.cdhit_gene_dict.keys():
                        gene = self.cdhit_gene_dict[cdhit]
                        seq_id_list.append('gene_name={}'.format(gene))
                    else:
                        seq_id_list.append('gene_name= ')
                    start = raw_data[j][2]
                    seq_id_list.append('start={}'.format(start))
                    stop = raw_data[j][3]
                    seq_id_list.append('end={}'.format(stop))
                    if raw_data[j][4] is True:
                        orient = '+1'
                    elif raw_data[j][4] is False:
                        orient = '-1'
                    else:
                        orient = '0'
                    seq_id_list.append('orientation={}'.format(orient))
                    seq_id = ''
                    x = 0
                    while x < len(seq_id_list)-1:
                        seq_id = seq_id + '{}|'.format(seq_id_list[x])
                        x += 1
                    seq_id = 'gi|' + seq_id + '{}'.format(seq_id_list[-1])
                    seq_rec = SeqRecord(seq, id=seq_id, description='')
                    rec_list.append(seq_rec)
                SeqIO.write(rec_list, 'genome_{}_annotations'.format(self.strain_legend[gen_dict[i]])+'.fasta', 'fasta')
                print 'strain_{}'.format(self.strain_legend[gen_dict[i]])+' is complete. #{}/{}'.format(i+1, len(query))

            elif self.assembly_legend[gen_dict[i]] == 'supercontig':
                seq_query = 'SELECT seq FROM genome_seq WHERE genome_id = {}'.format(gen_dict[i])
                raw_seq = self.db.getAllResultsFromDbQuery(seq_query)
                gen_seq = Seq(str(raw_seq)[3:-3], alphabet=IUPAC.unambiguous_dna)
                contig_locs = list([0])
                contig_locs.extend(self.spacer_finder(gen_seq))
                contig_locs.append(len(list(gen_seq)))
                for j in range(len(raw_data)):
                    seq = Seq(raw_data[j][1], alphabet=IUPAC.unambiguous_dna)
                    cdhit = raw_data[j][0]
                    raw_start = raw_data[j][2]
                    raw_stop = raw_data[j][3]
                    for n in range(len(contig_locs)-1):
                        if raw_start < contig_locs[n+1]:
                            contig_id = n+1
                            start = raw_start-36*n
                            stop = raw_stop-36*n
                            break
                    seq_id_list = ['strain_name=' + self.strain_legend[gen_dict[i]], 'genome_id={}'.format(gen_dict[i]),
                                   'contig_id={}'.format(contig_id), 'orf_id={}'.format(cdhit)]
                    if cdhit in self.cdhit_locus_dict.keys():
                        locus = self.cdhit_locus_dict[cdhit]
                        seq_id_list.append('locus_name={}'.format(locus))
                    else:
                        seq_id_list.append('locus_name= ')
                    if cdhit in self.cdhit_gene_dict.keys():
                        gene = self.cdhit_gene_dict[cdhit]
                        seq_id_list.append('gene_name={}'.format(gene))
                    else:
                        seq_id_list.append('gene_name= ')
                    seq_id_list.append('start={}'.format(start))
                    seq_id_list.append('end={}'.format(stop))
                    if raw_data[j][4] is True:
                        orient = '+1'
                    elif raw_data[j][4] is False:
                        orient = '-1'
                    else:
                        orient = '0'
                    seq_id_list.append('orientation={}'.format(orient))
                    seq_id = ''
                    x = 0
                    while x < len(seq_id_list)-1:
                        seq_id = seq_id + '{}|'.format(seq_id_list[x])
                        x += 1
                    seq_id = 'gi|' + seq_id + '{}'.format(seq_id_list[-1])
                    seq_rec = SeqRecord(seq, id=seq_id, description='')
                    rec_list.append(seq_rec)
                SeqIO.write(rec_list, 'genome_{}_annotations'.format(self.strain_legend[gen_dict[i]])+'.fasta', 'fasta')
                print 'strain_{}'.format(self.strain_legend[gen_dict[i]])+' is complete. #{}/{}'.format(i+1, len(query))
