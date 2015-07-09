#!/usr/bin/python
import psycopg2
import sys
import pprint
import os
import shutil

from Bio import AlignIO
from Bio import Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_protein
from Bio import SeqIO
from Bio.Align.Applications import ClustalwCommandline
import numpy as np

 
class SpringDb:
	"""Interfaces with cloud SPRING-DB"""
   	empCount = 0
   	connection = []
   	# change with path to database config file springDbConfig.txt
   	CONFIGFILE = "/home/justin/Documents/Xavier_Rotation/springDbConfig_local.txt"

   	def __init__(self):
		"""Contructor"""

		print "Connecting to SPRING-DB..."
		self.connect()
		print "Connected"
   

	def getGenomeIdOfStrain(self, strain):
		"""return the genome_id of a strain (string)"""
		
		return self.getFirstResultFromDbQuery("select genome_id from genome where strain_name = '" + strain + "'")
	
	
	def getLocusNameOfGeneInStrain(self, gene, strain):
		"""search for the locus_name of a gene containing strin gene in its name. Search restricted to strain"""
		
		return self.getFirstResultFromDbQuery("select locus_name from orf where genome_id="+ str(strain) + " and gene_name like '%" + gene +"%'")

	
	def getOrfIdForLocusName(self, locusName):
		"""return the OrfId for the orf with locusName"""
		
		return self.getFirstResultFromDbQuery("select orth_orf_id from orth_orf where locus_name='" + locusName + "'")

	
	def getFastaFormatedSequencesFromOrthId(self, orfId):
		"""return the OrfId for the orf with locusName"""

		cursor = self.getCursor()
		cursor.execute("SELECT locus_name from orth_orf where orth_orf_id = " + str(orfId) + "  and genome_id in (select genome_id from view_genome)")
		sequenceArray = []
		for row in cursor:
			c = self.getFirstRowOfResultsFromDbQuery("select genome_id, seq from orf where locus_name = '" + row[0] + "'")
			try:
				gid = c[0]
				seq = Seq(c[1])
				genome = self.getFirstResultFromDbQuery("select strain_name from genome where genome_id = '" + str(gid) + "'")
				# print everything except for last element, which is stop codon
				sequenceArray.append(SeqRecord(seq.translate()[0:-1], id=genome, description=""))
			except TypeError:
				print "The locus name " + row[0] + " did not return any reslts"
 	
 		#SeqIO.write(sequenceArray, gene + ".faa", "fasta")
		return sequenceArray


	
	def getSeqrecordOfORFInStrain(self, orfId, strain):
		"""return a SeqReord for the sequence with orfId in a strain (genome_id)"""

		locusName = self.getFirstResultFromDbQuery("SELECT locus_name from orth_orf where orth_orf_id = " + str(orfId) + "  and genome_id='" + str(strain)+ "'")
		c = self.getFirstRowOfResultsFromDbQuery("select genome_id, seq from orf where locus_name = '" + locusName + "'")
		gid = c[0]
		seq = Seq(c[1])
		genome = self.getFirstResultFromDbQuery("select strain_name from genome where genome_id = '" + str(gid) + "'")
		return SeqRecord(seq, id=locusName, description="")


	def saveFastaOfORFsInStrain(self, orfIdArray, strain, fn):
		"""save a fasta file (filename fn) with the sequences for the orfs in the array OrfIdArray in strain"""

		strOrfs = SpringDb.convertNumPyArrayToSqlArray(orfIdArray)
		cursor = self.getCursor()
		cursor.execute("select locus_name from orth_orf where orth_orf_id in " + strOrfs + " and genome_id=" + str(strain))
		locusNameArray = cursor.fetchall()
		locusNameArray = np.array(locusNameArray)
		locusNameArrayFormatted = SpringDb.convertNumPyArrayToSqlStrArray(locusNameArray.reshape(locusNameArray.size))
		#for locusName in a:
		cursor.execute("select seq, locus_name, gene_name, product_name from orf where locus_name in " + locusNameArrayFormatted)
		f = open(fn, 'w')
		i = 0
		for locus in cursor:
			f.write(">" + locus[1] + "; " + locus[2] + "; " + locus[3] + "\n")
			f.write(locus[0] + "\n")
			i += 1
		f.close()
		# read and re-write to reformat nicely using the SeIO.write
		s = SeqIO.parse(fn, "fasta")
		SeqIO.write(s, fn + "_tmp", "fasta")
		shutil.move(fn + "_tmp", fn)

	##############################
	## CORE/PAN GENOME ANALYSIS ##
	##############################

	def getOrfIdOfCoreGenes(self):
		"""returns an array with the  Ids of the core genes"""

		cursor = self.getCursor()
		cursor.execute("select orth_orf_id,genome_id from orth_orf where genome_id in (select genome_id from view_genome) and (not exclude or exclude is NULL);")
		a = cursor.fetchall()
		x = np.array(a)
		orfs    = x[:, 0]
		genomes = x[:, 1]
		# unique ORFs
		uOrfs   = np.unique(orfs)
		uGenomes = np.unique(genomes) 

		nGenomes = uGenomes.size

		nGenomesWithOrfI = np.zeros(uOrfs.size)
		i = 0
		for orfI in uOrfs:
			nGenomesWithOrfI[i] = (orfs[orfs == orfI]).size
			i += 1
		coreGenomeOrfs = uOrfs[nGenomesWithOrfI == nGenomes]
		return coreGenomeOrfs

	##########################
	## PAIRWISE COMPARISONS ##
	##########################

	def getMatrixOfOrfsInCommonAndGenomicCoordinates(self, strain1, strain2):

		# get orfs in first strain
		query = "select orth_orf_id from orf where genome_id=" + str(strain1)
		a = self.getAllResultsFromDbQuery(query)
		orfs1 = np.array(a)

		# get orfs in second strain
		query = "select orth_orf_id from orf where genome_id=" + str(strain2)
		a = self.getAllResultsFromDbQuery(query)
		orfs2 = np.array(a)

		# find orfs in common
		orfs = np.intersect1d(orfs1, orfs2)
		orfs = filter(lambda x: x != None, orfs)

		query = "select orth_orf_id, start, stop, strand from orf where orth_orf_id IN" + self.convertNumPyArrayToSqlArray(orfs) + " and genome_id=" + str(strain1) + " order by orth_orf_id"
		a = self.getAllResultsFromDbQuery(query)
		x1 = np.array(a)

		query = "select orth_orf_id, start, stop, strand from orf where orth_orf_id IN" + self.convertNumPyArrayToSqlArray(orfs) + " and genome_id=" + str(strain2) + " order by orth_orf_id"
		a = self.getAllResultsFromDbQuery(query)
		x2 = np.array(a)

		return np.hstack((x1, x2))


	##################################
	## DATABASE AUXILIARY FUNCTIONS ##
	##################################

	@staticmethod
	def convertNumPyArrayToSqlArray(npArray):
		"""to be used in sql queries with IN"""
		sqlArray = "("
		for e in npArray:
			sqlArray += str(e) + ","
		sqlArray = sqlArray[0:-1] + ")"
		return sqlArray		

	@staticmethod
	def convertNumPyArrayToSqlStrArray(npArray):
		"""same as convertNumPyArrayToSqlArray but for strings"""
		sqlArray = "('"
		for e in npArray:
			sqlArray += str(e) + "','"
		sqlArray = sqlArray[0:-2] + ")"
		return sqlArray		

	#########################
	## DATABASE CONNECTION ##
	#########################

	def getFirstResultFromDbQuery(self, query):
		"""make a query and return the very first result"""
		cursor = self.getCursor()
		cursor.execute(query)
		return cursor.fetchone()[0]

	def getFirstRowOfResultsFromDbQuery(self, query):
		"""make a query and return the first row of results"""
		cursor = self.getCursor()
		cursor.execute(query)
		return cursor.fetchone()

	def getAllResultsFromDbQuery(self, query):
		"""make a query and returnall results"""
		cursor = self.getCursor()
		cursor.execute(query)
		return cursor.fetchall()

	def getCursor(self):
		"""get the cursor to the data base. If connection is lost then reconnect"""
		if (self.connection.closed == 1):
			self.connect()
		return self.connection.cursor()

	def connect(self):
		""" connect to SPRING-DB using information for text file configFile"""
		with open(self.CONFIGFILE, 'r') as f:
  			configLine = f.readline()	
		self.connection = psycopg2.connect(configLine)

	def __repr__(self):
		"""output for pretty print"""
		return 'Interfaces with cloud SPRING-DB'

if __name__ == "__main__":
	# """some example code"""	
	# strain    = "pa_UCBPP_PA14"
	# gene      = "fleN"
	# db        = SpringDb()
	# strainId  = db.getGenomeIdOfStrain(strain)
	# locusName = db.getLocusNameOfGeneInStrain(gene, strainId)
	# orfId     = db.getOrfIdForLocusName(locusName)
	# sequenceArray = db.getFastaFormatedSequencesFromOrthId(orfId)

	# SeqIO.write(sequenceArray, gene + ".faa_test", "fasta")

	# r = db.getSeqrecordOfORFInStrain(607, 11)

	# a = db.getOrfIdOfCoreGenes()
	# print "SPRING-DB\n"

	# initiate connection to database
	db        = SpringDb()
	qr = db.getAllResultsFromDbQuery('SELECT cdhit_id, genome_id FROM orf WHERE genome_id IN (SELECT genome_id FROM phenotype) AND cdhit_id IS NOT NULL ')
	resultTable = np.array(qr)



