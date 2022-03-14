"""
This file contains a class for PBMEscore object.

Created on Jul 22, 2019

Authors: Vincentius Martin, Farica Zhuang
"""
from chip2probe.sitespredict import basepred, basemodel
from chip2probe.util import bio as bio
import pandas as pd

class PBMEscore(basemodel.BaseModel):
    """PBMEscore sitespredict class

    Make E-score predictions for all k-mers (currently only accept 8) in the
    input sequences.

    Example:
    >>> escore = PBMEscore("<insert path here>")
    >>> # predict one sequence
    >>> single_sequence = "TTACGGCAAGCGGGCCGGAAGCCACTCCTCGAGTCT"
    >>> singlepred = escore.predict_sequence(single_sequence)
    >>> # predict many sequence
    >>> many_sequences = ["ACTGGCAGGAAGGGCAGTTTTGGCAGGAAAAGCCAT", "CAGCTGGCCGGAACCTGCGTCCCCTTCCCCCGCCGC"]
    >>> manypredlist = escore.predict_sequences(many_sequences)
    >>> # or as a data frame input
    >>> df = pd.DataFrame(list(zip(many_sequences, ['seq1','seq2'])), columns=['sequence', 'key'])
    >>> manypredlist = escore.predict_sequences(df)

    TODO:
        Make this available for gap E-score and different k-mer length
    """

    def __init__(self, escore_path, escore_map_path="", kmer=8):
        """
        Args:
            escore_path: path to the escore file, can be in the long format (i.e. a file
                with k-mer sequence, its reverse complement, and E-score) or just the short
                format with the just the E-score in sorted k-mer order.
            escore_map_path: when the short format is used, the map of the index
                short to long needs to be provided.
            kmer: the value of k, default 8
        """
        self.escore = self.read_escore(escore_path, escore_map_path)
        self.kmer = kmer


    def read_escore(self, filepath, map_path=""):
        """
        Read an escore file to produce PBMEscore object. If map_path is provided
        then the function assumes that the condensed version of the input file is used.

        Args:
            filepath: The path to the escore file
            map_path: Path to the short to log map
        Return:
            dictionary of sequence to E-score
        """

        # if map_path is provided, assume short escore
        if map_path:
            with open(map_path) as f:
                next(f)
                emap = [int(line.split(",")[1])-1 for line in f]
            with open(filepath) as f:
                eshort = [float(line) for line in f]
            # kmer is always 8 in escore
            elong = {bio.itoseq(idx,self.kmer):eshort[idx] for idx in emap}
            return elong
        else: # full escore file
            df = pd.read_csv(filepath, sep="\t")
            d1 = pd.Series(df["E-score"].values,index=df["8-mer"]).to_dict()
            d2 = pd.Series(df["E-score"].values,index=df["8-mer.1"]).to_dict()
            dboth = {**d1, **d2}
            return dboth

    def predict_sequence(self, sequence):
        """
        Make E-score predictions of an input sequence

        Args:
            sequence (str): input sequence
        Return:
            list of dictionary of sequence to E-score
        """
        prediction = []
        for i in range(0,len(sequence)-self.kmer+1):
            score = self.escore[sequence[i:i+self.kmer]]
            # list index is the position from the first e-scoreo (i.e. i-th position)
            prediction.append({"position":i+(self.kmer+1)//2,"escore_seq":sequence[i:i+self.kmer],"score":score,"start_idx":i})
        return prediction

    def predict_sequences(self, sequences, sequence_colname="sequence",
                          key_colname="", predict_flanks=False, flank_colname="flank",
                          flank_len=10,  only_pred = False):
        """
        Get a dictionary of escore predictions for each sequence.

        Args:
            sequences: list / data frame / dictionary of sequences (see bio.get_seqdict)
            sequence_colname: when input is a data frame, this is the column name of
                the sequence (default: sequence)
            key_colname: when input is data frame, this is the column with the key
                that denotes distict row (default: "")
            predict_flanks: default False, when True check flank column--input needs
                to be a data frame
            flank_colname: the column name of the flank sequence
            flank_len: length of the flanking sequences
            only_pred: by default we return result as `BasePred` object for plotting
        Return:
            list of dictionary of the predicted sequences as a BasePred object
            if `only_pred` is False, else just return the list
        """
        seqdict = bio.get_seqdict(sequences, sequence_col=sequence_colname, keycol=key_colname)
        # get the flanks if we are including flank predictions
        if predict_flanks:
            flank_left = bio.get_seqdict(sequence_df, "%s_left" % flank_colname,
                                         keycol=key_colname,
                                         ignore_missing_colname=True)
            flank_right = bio.get_seqdict(sequence_df, "%s_right" % flank_colname,
                                          keycol=key_colname,
                                          ignore_missing_colname=True)
        # get prediction of each sequence
        predictions = {}
        for key, sequence in seqdict.items():
            # if we are including flanks in the prediction
            if predict_flanks:
                # make sure there are enough flanks to take
                if len(flank_left[key]) < flank_len or len(flank_right[key]) < flank_len:
                    raise Exception("flank_len is greater than the length of flanks available")
                # update the sequence to be predicted
                sequence = flank_left[key][-flank_len:] + sequence + flank_right[key][:flank_len]
            # get the prediction for this sequence
            prediction = self.predict_sequence(sequence)
            if only_pred:
                predictions[key] = prediction
            else:
                predictions[key] = basepred.BasePrediction(sequence, prediction)
        # return the dictionary of predictions for each sequence
        return predictions

    # TODO: modify sequence to use this function instead
    def get_escores_specific(self, sequence, escore_cutoff = 0.4, escore_gap = 0):
        """
        Get the list of specific escore regions in the given sequence.

        Using this, we get only speicifc escore list based on the cutoff and gap
        threshold.

        Args:
            sequence: sequence string
        """
        escores = self.predict_sequence(sequence)
        signifcount = 0
        startidx = -1
        gapcount = 0
        escore_signifsites = []
        for i in range(0, len(escores)):
            escoresite = escores[i]
            if escoresite["score"] > escore_cutoff :
                if signifcount == 0:
                    startidx = i
                signifcount += 1
                gapcount = 0
            # we can ignore else if here since we need i == len(esores)-1
            if escoresite["score"] <= escore_cutoff and i != len(escores)-1 and gapcount < escore_gap:
                # check if the sequence is still within
                gapcount += 1
            elif escoresite["score"] <= escore_cutoff or i == len(escores)-1:
                if signifcount > 0:
                    # if we have found sufficient e-scores above the cutoff then get the binding sites
                    if signifcount >= 2:
                        # startpos: the start of binding
                        escore_bind = {"startpos":escores[startidx]['position'],  "escorelength":signifcount + gapcount,
                                "escore_startidx":escores[startidx]['start_idx']}
                        escore_signifsites.append(escore_bind)
                    startidx = -1
                    signifcount = 0
        # list of significant regions are in sorted order
        return escore_signifsites

    def make_plot_data(self, predictions_dict, scale=1, escore_cutoff=0.4,
                        additional_functions={}, color="orange",
                        line_color="darkorange"):
        """
        Make plot data from `predict_sequences` result

        Args:
            predictions_dict: result from `predict_sequences`
            scale: scale the E-score, useful when plotting E-score with another
                model with (usually) larger score such as PWM
            escore_cutoff: where to plot the horizontal threshold line
            additional_function: additional plotting functions
            color: color of the E-score line
            line_color: color for the horizontal threshold line
        """
        func_dict = {}
        for key in predictions_dict:
            sequence = predictions_dict[key].sequence
            escores = predictions_dict[key].predictions
            func_pred =[]
            y_escore = [x["score"] * scale for x in escores]
            x_escore = [x["position"] for x in escores]
            func_pred.append({"func":"plot","args":[x_escore, y_escore],"kwargs":{"color":color, "linewidth" : 2.5}})
            func_pred.append({"func":"axhline", "args":[escore_cutoff * scale], "kwargs":{"color":line_color, "linestyle" : "dashed", "linewidth":1}})
            if key in additional_functions and additional_functions[key]:
                func_pred.extend(additional_functions[key])
            func_dict[key] = {"sequence":sequence,
                                  "plt":func_pred}
        return func_dict
