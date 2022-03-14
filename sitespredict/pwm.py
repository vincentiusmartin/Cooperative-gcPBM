'''
Created on Sep 24, 2020
Authors: Vincentius Martin
'''
import numpy as np
import matplotlib.patches as patches

from chip2probe.sitespredict import basepred, basemodel
from chip2probe.util import bio as bio

"""
Some start end data:
    -Ets1: (6,16) -> 3 + 4 + 3
    -Runx1: (7,18) -> 3 + 5 + 3
"""

# TODO: handle palindrome
class PWM(basemodel.BaseModel):
    def __init__(self, pwm_path, startidx=0, endidx=-1, log=True, reverse=False):
        """
        if reverse is True then use the reverse complement as the forward
        """
        self.pwm_fwd, self.pwm_rev = self.read_pwm(pwm_path, startidx, endidx, log, reverse)
        self.length = len(self.pwm_fwd['A'])
        self.log = log

    def read_pwm(self, pwmfile, startidx=0, endidx=-1, log=True, reverse=False):
        with open(pwmfile,'r') as f:
            end = len(f.readline().split(":")[1].split()) if endidx == -1 else endidx
        pwm_fwd = {}
        bases = []
        with open(pwmfile,'r') as f:
            fclean = f.read().strip().split("\n")
        for line in fclean:
            base,scores = line.strip().split(":")
            base = base.strip()
            bases.append(base)
            scores = scores.strip().split()[startidx:end]
            if log:
                with np.errstate(divide='ignore'): # ignore divide by zero warning
                    pwm_fwd[base] = [np.log2(float(score)/0.25) for score in scores]

            else:
                pwm_fwd[base] = [float(score) for score in scores]
        bases_rev = bases[::-1]
        if not reverse: # just take the rc as the reverse
            pwm_rev = {bases[i] : pwm_fwd[bases_rev[i]][::-1] for i in range(len(bases))}
        else:
            pwm_rev = dict(pwm_fwd)
            pwm_fwd = {bases[i] : pwm_rev[bases_rev[i]][::-1] for i in range(len(bases))}
        # for a in pwm_rev:
        #     print(a,"\t".join(map(str,pwm_rev[a])))
        return pwm_fwd, pwm_rev

    def generate_pwm_file(self, path):
        bases = ['A','C','G','T']
        text = ""
        for b in bases:
            text += "%s:\t%s\n" % (b,"\t".join([str(n) for n in self.pwm_fwd[b]]))
        with open(path,'w') as f:
            f.write(text[:-2])

    def predict_sequence(self, sequence, zero_thres=True):
        """
        Args:
            -sequence: input sequence
            -zero_thres: if true then only return of prediction is above zero
        """
        prediction = []
        for i in range(0, len(sequence)-self.length+1):
            if self.log:
                score_fwd = sum([self.pwm_fwd[sequence[j]][j-i] for j in range(i,i+self.length)])
                score_rev = sum([self.pwm_rev[sequence[j]][j-i] for j in range(i,i+self.length)])
            else:
                score_fwd = np.prod([self.pwm_fwd[sequence[j]][j-i] for j in range(i,i+self.length)])
                score_rev = np.prod([self.pwm_rev[sequence[j]][j-i] for j in range(i,i+self.length)])
            score, ori = (score_fwd, 1) if score_fwd > score_rev else (score_rev, -1)
            mid = (i + self.length) // 2
            if not zero_thres or score > 0: # TODO: what threshold for non log?
                prediction.append({"site_start": i,
                                 "site_width": self.length,
                                 "score": score,
                                 "mid": mid if self.length % 2 == 0 else mid + 1,
                                 "seq": sequence[i:i+self.length],
                                 "orientation": ori
                                 })
        return prediction

    def predict_sequences(self, sequences, sequence_colname="sequence",
                    key_colname="", only_pred = False):
        """
        """
        seqdict = bio.get_seqdict(sequences, sequence_col=sequence_colname, keycol=key_colname)
        predictions = {}
        for key in seqdict:
            prediction = self.predict_sequence(seqdict[key])
            if only_pred:
                predictions[key] = prediction
            else:
                predictions[key] = basepred.BasePrediction(seqdict[key], prediction)
        return predictions

    def make_plot_data(self, predictions_dict, color = "cyan"):
        func_dict = {}
        for key in predictions_dict:
            sequence = predictions_dict[key].sequence
            sites_prediction = predictions_dict[key].predictions
            func_pred = []
            max_y = 1
            for pred in sites_prediction:
                core_rect = patches.Rectangle((pred["site_start"],0),pred["site_width"] - 1,pred["score"],
                                 facecolor=color,alpha=0.9,edgecolor='black')
                func_pred.append({"func": "add_patch",
                                  "args": [core_rect],
                                  "kwargs": {}})
                if pred["score"] > max_y:
                    max_y = pred["score"]
            func_pred.append({"func": "set_ylim",
                              "args": [],
                              "kwargs": {"top":max_y+1}})
            func_dict[key] = {"sequence": sequence,
                              "plt": func_pred}
        return func_dict
