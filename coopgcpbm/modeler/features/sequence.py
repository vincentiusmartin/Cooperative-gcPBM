from coopgcpbm.modeler.features import basefeature
import coopgcpbm.util.bio as bio
import itertools

import numpy as np

class Sequence(basefeature.BaseFeature):
    def __init__(self, traindf, params):
        """
        DNA sequence feature prediction class

        Args:
            traindf: dataframe containing the "name", "sequence" column
            params:


         Returns:
            NA
        """
        default_args = {
            "seqin": 0,
            "smode": "positional",
            "poscols": [],
            "namecol":"name",
            "seqcol":"Sequence",
            "maxk":2,
        }
        self.df = traindf
        self.set_attrs(params, default_args)
        if self.smode != "relative" and self.smode != "positional" and self.smode != "linker":
            raise TypeError("Smode can only be 'relative', 'positional', or 'linker'")

    def get_feature(self,seqcolname="Sequence"):
        rfeature = []
        # s1 is always on the left
        for idx, row in self.df.iterrows():
            pc1, pc2 = int(row[self.poscols[0]]), int(row[self.poscols[1]])
            if self.smode == "positional" or self.smode == "relative":
                if self.smode == "positional":
                    s1, s2 = pc1, pc2
                    s1type, s2type = "s1",  "s2"
                elif self.smode == "relative" or self.smode == "linker":
                    if pc1 < pc2:
                        s1, s2 = pc1, pc2
                        s1type, s2type = "s1","s2"
                    else:
                        s1, s2 = pc2, pc1
                        s1type, s2type = "s2", "s1"
                flank1 = row[self.seqcol][s1:s1 + self.seqin] if self.seqin > 0 else row[self.seqcol][s1 + self.seqin:s1][::-1]
                flank2 = row[self.seqcol][s2 - self.seqin:s2][::-1] if self.seqin > 0 else row[self.seqcol][s2:s2 - self.seqin]
                label = "inner" if self.seqin > 0 else "outer"
            else: #linker
                center = (pc1 + pc2) // 2
                s1type, s2type = "toright", "toleft"
                flank1 =  row[self.seqcol][center:center+self.seqin]
                flank2 =  row[self.seqcol][center-self.seqin:center]
                label = "linker"
            d1 = self.extract_positional(flank1, stype=s1type, label=label, maxk=self.maxk)
            d2 = self.extract_positional(flank2, stype=s2type, label=label, maxk=self.maxk)
            rfeature.append({**d1, **d2})
        return rfeature


    def extract_positional(self, seq, maxk=2, stype="site", label="seq", minseqlen=-float("inf")):
        '''
        orientation: if right, then start from 0 to the right, else start from
        len(seq)-1 to the left
        minseqlen: will fill -1 if not enough bases
        '''
        iterseq = str(seq)
        nucleotides = ['A','C','G','T']
        features = {}
        for k in range(1,maxk+1):
            perm = ["".join(p) for p in itertools.product(nucleotides, repeat=k)]
            i = 0
            while i < len(iterseq)+1-k:
                for kmer in perm:
                    seqcmp = iterseq[i:i+k]
                    if seqcmp == kmer:
                        features["%s_%s_pos%d_%s" % (stype,label,i,kmer)] = 1
                    else:
                        features["%s_%s_pos%d_%s" % (stype,label,i,kmer)] = 0
                i += 1
            # append the rest with -1
            if minseqlen > 0:
                while i < minseqlen + 1 - k:
                    for kmer in perm:
                        features["%s_%s_pos%d_%s" % (stype,label,i,kmer)] = -1
                    i += 1
        return features
