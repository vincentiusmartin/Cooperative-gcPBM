import pandas as pd
from coopgcpbm.sitespredict import basepred, basemodel
from coopgcpbm.util import bio as bio
import itertools
import math
import matplotlib.patches as patches

# imports for Kompas
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import reverse_complement, Seq
import os
from itertools import groupby
from operator import itemgetter
from pybedtools import BedTool

import sys

"""
if self.protein == 'ets1':
    core = (11,15)
    centerPos = 12
if self.protein == 'runx1':
    core = (12, 17)
    centerPos = 14
"""

class Kompas(basemodel.BaseModel):
    def __init__(self, kmer_align_path, core_start, core_end, core_center, escore_thres = 0.38, clean=False):
        """
        Args
            kmer_align_path: path of the kmer alignment for kompas
            escore_thres: bound escore threshold
            clean: ???
        """
        self.threshold = escore_thres
        self.kmer_df = pd.read_csv(kmer_align_path, sep="\t")
        self.clean = clean
        self.core = (core_start, core_end)
        self.centerPos = core_center
        self.k = len(self.kmer_df['kmer'][0])

    def kmerMatch(self, seq, k, kDict):
        """
        Returns matched positions in the sequence and their kpositions
        Input: sequence and kmer length (k)
        Output: consecutive positions, kpositions, and scores above threshold
        """
        # Get the kposition and kscore for the peak, save a numpy array
        kpos,kscore = [], []
        for i in range(len(seq) - self.k + 1):
            window = seq[i:i+k]
            if window in kDict:
                kpos.append(kDict[window][0])
                kscore.append(kDict[window][1])
            else:
                kpos.append(0)
                kscore.append(-0.5)
        kpos = np.array(kpos)
        kscore = np.array(kscore)
        coreLen = self.core[1] - self.core[0]
        # Get consecutive positions, kpositions, and score via numpy operations
        if self.k >= coreLen:
            position = list(filter(lambda x: len(x) != 1,np.split(np.r_[:len(kpos)], np.where(np.diff(kpos) != 1)[0]+1)))
            kpos = list(filter(lambda x: len(x) != 1,np.split(kpos, np.where(np.diff(kpos) != 1)[0]+1)))
        elif self.k < coreLen:
            reqLen = len(ReqKpos)
            position = list(filter(lambda x: len(x) == reqLen,np.split(np.r_[:len(kpos)], np.where(np.diff(kpos) != 1)[0]+1)))
            kpos = list(filter(lambda x: len(x) == reqLen,np.split(kpos, np.where(np.diff(kpos) != 1)[0]+1)))
        kScore = []
        for pos in position:
            kScore.append(kscore[pos])
        return(zip(position, kpos, kScore))

    def findCenter(self, zippedMatch, orient, seqLen):
        """
        Given a zip of match position, kposition, and kscore
        Returns the center sites and threshold kscore
        """
        centerSites = []
        siteScores = []
        coreLen = self.core[1] - self.core[0]
        for pos, kpos, kScore in zippedMatch:
            centerSite = (self.centerPos - kpos[0]) + pos[0]
            if orient == 'rc':
                centerSite = (seqLen - centerSite) -1
            centerSites.append(centerSite)
            if self.k >= coreLen:
                score = self.threshold
                for score1, score2 in zip(kScore, kScore[1:]):
                    caniScore = sorted([score1, score2])[0]
                    if caniScore > score:
                        score = caniScore
                siteScores.append(score)
            elif self.k < coreLen:
                siteScores.append(min(kScore))
        return(pd.Series([centerSites, siteScores]))

    def convertToBed(self, df, isPalindrome):
        if isPalindrome == False:
            chrom, start, orient,scores = [],[],[],[]
            for row in zip(df['Chromosome'],df['Start'],df['centerPlus'],df['scorePlus'],df['centerMinus'],df['scoreMinus']):
                if row[2]:
                    for centerP, score in zip(row[2], row[3]): # + sites
                        chrom.append(row[0])
                        start.append(row[1] + centerP)
                        orient.append('+')
                        scores.append(score)
                if row[4]:
                    for centerN, score in zip(row[4],row[5]): # - sites
                        chrom.append(row[0])
                        start.append(row[1] + centerN)
                        orient.append('-')
                        scores.append(score)
        else:
            for row in zip(df['Chromosome'],df['Start'],df['centerPlus'],df['scorePlus']):
                if row[2]:
                    for centerP, score in zip(row[2], row[3]): # + sites
                        chrom.append(row[0])
                        start.append(row[1] + centerP)
                        orient.append('+')
                        scores.append(score)
        bedDF = pd.DataFrame({'chrom':chrom,'start':start, 'end':start,'orient':orient, 'score':scores})
        bedDF['end'] = bedDF['end'] + 1 # exclusive end position
        bedDF = bedDF.drop_duplicates().sort_values(by=['chrom','start']) # some sites overlap, will call the same centers
        return bedDF

    def predict_sequence(self, sequence):
        '''This function uses kompas to predict the position of a
           tf binding site on a given sequence'''
        ##### Read in kmer data and process ####
        coreLen = self.core[1] - self.core[0]
        # Find the kPositions required, any would be sufficient to call
        if self.k > coreLen:
            searchEnd = self.core[1]
            checkK = 0
            ReqKpos = set() #
            while checkK != self.core[0]:
                checkK = searchEnd - self.k
                if checkK <= self.core[0]:
                    ReqKpos.add(checkK)
                    searchEnd = searchEnd + 1
        # Or find the group of all kPositions that are needed, all or none
        else:
            searchStart = core[0]
            checkK = 0
            ReqKpos = set()
            while searchStart + self.k <= core[1]:
                ReqKpos.add(searchStart)
                searchStart = searchStart + 1
        # Determine flanks of ReqKPos for threshold score reporting
        ScoredKpos = ReqKpos.copy()
        if self.k >= coreLen:
            ScoredKpos.add(min(ReqKpos) - 1)
            ScoredKpos.add(max(ReqKpos) + 1)

        # Generate dictionary for quick retreaval of relevant kmers
        thrKmers = self.kmer_df[(self.kmer_df['Escore'] > self.threshold) & (self.kmer_df['kposition'].isin(ScoredKpos))]
        kDict = dict(zip(thrKmers['kmer'],zip(thrKmers['kposition'],thrKmers['Escore'])))

        #-----------------------start prediction-------------------------------
        isPalindrome = False
        chromosome = ""
        start = 0
        fwd = sequence
        seq = Seq(fwd)
        seq_len = len(fwd)
        rev_comp = str(seq.reverse_complement())
        col_names = ["Chromosome", "Start", "fwd", "seq_len", "rev_comp"]
        lst = [[chromosome, start, fwd, seq_len, rev_comp]]
        df = pd.DataFrame.from_records(lst, columns = col_names)
        if isPalindrome == True:
            df[["centerPlus","scorePlus"]] = df.apply(lambda df: self.findCenter(self.kmerMatch(df['fwd'], self.k, kDict), 'fwd', len(df['fwd'])), axis = 1)
            df = df[~df['centerPlus'].isna()]
        else:
            # search forward strand
            df[["centerPlus","scorePlus"]] = df.apply(lambda df: self.findCenter(self.kmerMatch(df['fwd'], self.k, kDict), 'fwd',len(df['fwd'])), axis = 1)
            # search reverse compliment
            df[["centerMinus","scoreMinus"]] = df.apply(lambda df: self.findCenter(self.kmerMatch(df['rev_comp'], self.k, kDict), 'rc',len(df['rev_comp'])), axis = 1)
            df = df[~df['centerPlus'].isna() | ~df['centerMinus'].isna()]

        # convert dataframe to bed file format
        finalBed = self.convertToBed(df, isPalindrome)

        # get all binding site for this protein in this sequence
        prediction = []
        for index, row in finalBed.iterrows():
            if coreLen % 2 == 0:
                if row[3] == '+':
                    to_start = -1
                else:
                    to_start = -2
            else:
                to_start = -2
            position = int(row[1]) + to_start

            prediction.append({"site_start": position - 7, "site_width": 2 * 7 + coreLen,
                               "core_start": position, "core_width": coreLen,
                               "core_mid": int(row[1]),
                               "score": 0
                               })
        return prediction

    def predict_sequences(self, sequence_df, key_colname="",
                          sequence_colname="sequence",
                          flank_colname="flank", predict_flanks=False,
                          flank_len=10):
        '''This is a temporary function that makes predictions dict
           using the dataframe'''

        seqdict = bio.get_seqdict(sequence_df, sequence_col=sequence_colname, keycol=key_colname)
        if predict_flanks:
            flank_left = bio.get_seqdict(sequence_df,"%s_left" % flank_colname,
                                         ignore_missing_colname=True,
                                         keycolname=key_colname)
            flank_right = bio.get_seqdict(sequence_df,"%s_right" % flank_colname,
                                          ignore_missing_colname=True,
                                          keycolname=key_colname)
        predictions = {}
        # for each sequence we want to predict
        for key in seqdict:
            sequence = seqdict[key]
            if predict_flanks:
                sequence = flank_left[key][-10:] + seqdict[key] + flank_right[key][:10]
            prediction = self.predict_sequence(sequence)
            if predict_flanks:
                for result in prediction:
                    result['site_start'] = result['site_start'] - flank_len
                    result['core_start'] = result['core_start'] - flank_len
                    # if a prediction is in the flanks
                    if result['core_start'] < 0 or \
                       result['core_start'] + result['core_width'] > len(seqdict[key]) - 1:
                       # remove the prediction
                       prediction.remove(result)
            predictions[key] = basepred.BasePrediction(sequence, prediction)
        return predictions

    def make_plot_data(self, predictions_dict, color = "mediumspringgreen",
                       show_model_flanks=False):
        func_dict = {}
        for key in predictions_dict:
            sequence = predictions_dict[key].sequence
            sites_prediction = predictions_dict[key].predictions
            # if clean is true, and sequence has no imads object, indicate set this sequence to None
            if self.clean and len(sites_prediction) == 0:
                func_dict[key] = None
                continue
            # otherwise, set the appropriate dictionary as value for this key
            func_pred = []
            #i = 0
            #cs = ["#F7A091","red"]
            for pred in sites_prediction:
                # first argument is x,y and y is basically just starts at 0
                # first plot the binding site
                # Note that it is important to do -1 on rectangle width so we only cover the site
                site_rect = patches.Rectangle((pred["core_start"],0),\
                            pred["core_width"] - 1,height = 1, facecolor=color,\
                            alpha=0.6,edgecolor='black')
                func_pred.append({"func":"add_patch","args":[site_rect],"kwargs":{}})
            func_dict[key] = {"sequence":sequence,
                              "plt":func_pred}
        return func_dict
