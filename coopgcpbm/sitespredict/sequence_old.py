"""
This file contains BindingSite class and Sequence class.

Created on Jul 25, 2019

Authors: Vincentius Martin, Farica ZHuang
"""
import collections
import copy


class BindingSite(object):
    """Class for BindingSite object."""

    def __init__(self, site_pos, site_start, site_end, core_start, core_end, core_mid,
                 sequence_in_site, protein, score=1, barrier=1):
        """
        Initialize class variables for BindingSite object.

        For kompas, site_start=core_start and site_end=core_end.
        For imads, they will be different since site_start and site_end are
        20bp to the left and right of core
        """
        self.site_pos = site_pos
        self.site_start = site_start
        self.site_end = site_end
        self.core_start = core_start
        self.core_end = core_end
        self.core_mid = core_mid
        self.site_sequence = sequence_in_site
        self.score = score
        # bp from the core that shouldn't be mutated
        self.barrier = barrier
        self.protein = protein

    def __str__(self):
        """Return the string representation of the BindingSite object."""
        return "site_pos: {}, score: {}, site_start: {}, site_end: {}, \
               core_start: {}, core_end: {}, site_sequence {}, protein: {}"\
               .format(self.site_pos, self.score, self.site_start,
                       self.site_end, self.core_start, self.core_end,
                       self.site_sequence, self.protein)

MutatedSequence = collections.namedtuple('MutatedSequence',
                                         'sequence, \
                                          escore_preds, \
                                          model_preds, \
                                          proteins, \
                                          escore_cutoff, \
                                          escore_gap, \
                                          mutpos, \
                                          plot_functions')


class Sequence(object):
    """Class for Sequence object."""

    def __init__(self, escore_preds, model_preds, proteins, pbmescores, escore_cutoff=0.4, escore_gap=0):
        """
        Initialize a sequence object.

        :param escore_preds: prediction from the Escore class, the sequence string will
               be obtained from this object.
        :param model_preds: give the position of the core predicted by a model (kompas or imads).
                            The quality of the prediction will be assessed by using 'escore_preds'.
        :param proteins: list of proteins with length 1 if homotypic and 2 if heterotypic
        :param escore_cutoff: we determine that the site is specific when there are 2
               consecutive positions with escore above this cutoff. The escore range from
               -0.5 to 0.5, default cutoff is 0.4.
        :param escore_gap: how many gaps (i.e. value below cutoff) are permitted between specific
               escore position, the value is 0 by default
        """
        self.escore_preds = escore_preds
        self.model_preds = model_preds
        self.bsites = {}
        self.proteins = proteins
        self.escore_cutoff = escore_cutoff
        self.pbmescores = pbmescores
        self.escore_gap = escore_gap

        seq = ""
        # initialize class variables
        for protein in proteins:
            # check that the sequences are the same for both proteins
            if seq == "":
                seq = escore_preds[protein].sequence
            else:
                assert seq == escore_preds[protein].sequence

            self.bsites[protein] = self.get_bsite(self.escore_preds[protein],
                                                  self.model_preds[protein],
                                                  protein, escore_cutoff,
                                                  escore_gap)

        self.sequence = seq

    def __str__(self):
        """Get string representation of Sequence object."""
        return "Sequence object: {}\nsites {}".format(self.sequence, str(self.bsites))

    def mutate_escore_seq_at_pos(self, seq, pos, threshold):
        """Return a mutated sequence that is nonspecific for all proteins."""
        if len(seq) != 8:
            raise Exception("sequence must be at length of 8")

        nucleotides = "ACGT"
        all_muts = {}

        # get every mutation for the sequence at the given position
        for n in nucleotides:
            seqlist = list(seq)
            # mutate the sequence
            seqlist[pos] = n
            mutseq = "".join(seqlist)
            all_muts[mutseq] = {}
            # get esocre of this mutation for each protein
            for protein in self.proteins:
                # get the escore for the mutated sequence
                all_muts[mutseq][protein] = self.pbmescores[protein].predict_sequence(mutseq).predictions[0]['score']

        # find a mutated sequence that is non-specific for all proteins
        min_sum = float("inf")
        min_seq = ""

        # iterate through mutated sequences
        for seq in all_muts:
            # get escores
            if all(i < threshold for i in all_muts[seq].values()):
                if sum(all_muts[seq].values()) < min_sum:
                    min_sum = sum(all_muts[seq].values())
                    min_seq = seq

        return min_seq

    def get_max_non_intersecting_escore(self, protein, full_seq, site_index):
        """
        Get maxescore from given core that doesn't intersect a different core.

        This is to prevent mutating other cores.
        """
        # get the sequence we want to get the escore of
        s_start = self.bsites[protein][site_index].site_start
        s_end = self.bsites[protein][site_index].site_end
        site_seq = full_seq[s_start:s_end]
        # get the escore prediction for the sequence
        epreds = self.pbmescores[protein].predict_sequence(site_seq).predictions
        # initialize non intersecting max escore
        max_escore = {"score": float("inf")}
        # find non intersecting max escore
        while max_escore["score"] == float("inf") and epreds:
            # get 8mer in the given core with the highest escore
            max_val = max(epreds, key=lambda x: x['score'])
            # get position of this max escore
            max_val_seqpos = self.bsites[protein][site_index].site_start + max_val['position']
            # loop through each protein
            for curr_protein in self.bsites:
                # loop through each binding site of this protein
                for i in range(len(self.bsites[curr_protein])):
                    # skip if checking against itself
                    if i == site_index and curr_protein == protein:
                        continue
                    # get region of this other core that the
                    unmutated_start = self.bsites[curr_protein][i].core_start - self.bsites[curr_protein][i].barrier
                    unmutated_end = self.bsites[curr_protein][i].core_end + self.bsites[curr_protein][i].barrier
                    # if this max escore intersects with another core, remove
                    if max_val_seqpos >= unmutated_start and max_val_seqpos < unmutated_end and max_val in epreds:
                        epreds.remove(max_val)
                    # otherwise, initialize this as max non intersecting escore
                    else:
                        max_escore = copy.copy(max_val)
        return max_escore

    def eliminate_site(self, protein, sequence, site_index,
                       escore_threshold=0.3):
        """
        Eliminate the given site from the sequence.

        This assumes that input is a sequence of a site
        site_index: which site to mutate
        """
        maxescore = 1  # just to initialize with a large value
        prevmax = -1  # just to avoid infinite loop
        # list of mutated sites
        site_mutpos = []
        # sequence
        full_sequence = str(sequence)

        flag = True
        while flag and prevmax != maxescore:
            prevmax = float(maxescore)
            maxepreds = self.get_max_non_intersecting_escore(protein=protein,
                                                             full_seq=full_sequence,
                                                             site_index=site_index)
            maxescore = maxepreds['score']
            # if the max non intersecting escore is below the threshold, nothing to mutate
            if maxescore < escore_threshold:
                flag = False
            else:
                # return immediately if the site can't be mutated
                if maxescore == float("inf"):  # no e-score that can be chosen
                    # since there is no site to be mutated, then just use empty list
                    return full_sequence, []
                seq_tomutate = maxepreds["escore_seq"]
                midpos = len(seq_tomutate) // 2
                # get new mutated sequence
                mutated_escore_seq = self.mutate_escore_seq_at_pos(seq_tomutate, midpos, escore_threshold)
                if mutated_escore_seq != "":
                    # mutate the sequence
                    mut_start = self.bsites[protein][site_index].site_start + maxepreds["start_idx"]
                    mut_end = mut_start + len(mutated_escore_seq)
                    full_sequence = full_sequence[:mut_start] + mutated_escore_seq + full_sequence[mut_end:]
                    site_mutpos.append(mut_start + midpos)
                else:
                    full_sequence = ""
                    site_mutpos = []
        # return the new mutated sequence and the positions mutated
        return full_sequence, site_mutpos

    def abolish_sites(self, sites, mode="to_eliminate",
                      escore_threshold=0.3):
        """
        proteins: list of proteins whose core to be abolished.

        type can either be to_eliminate or to_keep
        """
        # if we have multiple pbmescore and proteins to abolish
        mutated = str(self.sequence)
        for j in range(len(self.proteins)):
            protein = self.proteins[j]
            sites_to_mutate = []
            if mode == "to_eliminate":
                sites_to_mutate = sites[protein]
            elif mode == "to_keep":
                sites_to_mutate = [i for i in range(len(self.bsites[protein])) if i not in sites[protein]]

            else:
                raise Exception("type should either be to_eliminate or to_keep")

            mutpos = []
            for i in sites_to_mutate:
                mutated, site_mutpos = self.eliminate_site(protein=protein, sequence=mutated,
                                                           site_index=i,
                                                           escore_threshold=escore_threshold)
                mutpos.extend(site_mutpos)

            functions = []
            for pos in mutpos:
                functions.append({"func": "axvline", "args": [pos],
                                  "kwargs": {"color": "purple",
                                             "linestyle": "dashed",
                                             "linewidth": 1}})
        return MutatedSequence(mutated, self.escore_preds, self.model_preds,
                               self.proteins, self.escore_cutoff,
                               self.escore_gap, mutpos, functions)

    # TODO: change to pbmescore.get_escores_specific
    def get_bsite(self, escore_preds, model_preds, protein,
                  escore_cutoff=0.4, escore_gap=0):
        """
        Get binding site objects.

        escore_gap : value below threshold allowed to still say that an 8-mer is still within specific window
        """
        # initialize a list for the model predictions we want to keep
        model_pred_keep = []
        sequence = escore_preds.sequence
        escores = escore_preds.predictions
        signifcount = 0
        startidx = -1
        gapcount = 0
        bindingsites = []
        for i in range(0, len(escores)):
            escoresite = escores[i]
            if escoresite["score"] > escore_cutoff:
                if signifcount == 0:
                    startidx = i
                signifcount += 1
                gapcount = 0
            # we can ignore else if here since we need i == len(esores)-1
            if escoresite["score"] <= escore_cutoff and i != len(escores) - 1 and gapcount < escore_gap:
                # check if the sequence is still within
                gapcount += 1
            elif escoresite["score"] <= escore_cutoff or i == len(escores) - 1:
                if signifcount > 0:
                    # if we have found sufficient e-scores above the cutoff then get the binding sites
                    if signifcount >= 2:
                        # startpos: the start of binding
                        escore_bind = {"startpos": escores[startidx]['position'],
                                       "escorelength": signifcount + gapcount,
                                       "escore_startidx": escores[startidx]['start_idx']}
                        # we get e-scores now we look for its imads binding site
                        for model_pred in model_preds.predictions:
                            escore_start = escore_bind['startpos']
                            escore_end = escore_start + escore_bind["escorelength"]
                            core_start = model_pred["core_start"]
                            core_end = core_start + model_pred["core_width"]
                            if escore_start <= core_end and core_start <= escore_end: #overlap
                                # a site is found,
                                # also sequence can be negative or have length above sequence length since we appended
                                # flanking regions, so we need to define site start and site end
                                site_start = max(0, model_pred["site_start"])
                                site_end = min(model_pred["site_start"] + model_pred["site_width"], len(sequence))
                                bsite = BindingSite(site_pos=(core_start + core_end) // 2,
                                                    score=model_pred["score"],
                                                    site_start=site_start,
                                                    site_end=site_end,
                                                    core_start=model_pred["core_start"],
                                                    core_mid=model_pred["core_mid"],
                                                    core_end=model_pred["core_start"] + model_pred["core_width"],
                                                    sequence_in_site=sequence[site_start:site_end],
                                                    protein=protein)
                                bindingsites.append(bsite)
                                # add this imads pred to the list of imads preds we
                                # want to keep
                                model_pred_keep.append(model_pred)
                    startidx = -1
                    signifcount = 0
                    gapcount = 0
        self.model_preds[protein].predictions = model_pred_keep
        # return the list of binding sites for this protein
        return bindingsites

    def sites_to_dict(self, bindingsites):
        """Put binding site objects into a dictionary of attributes."""
        out_dict = {}
        # loop through each list of binding site objects
        for protein in bindingsites:
            bs_list = bindingsites[protein]
            # loop through each binding site object
            for i in range(0, len(bs_list)):
                attrs = [attr for attr in dir(bs_list[i])
                         if not callable(getattr(bs_list[i], attr)) \
                         and not attr.startswith("__")]
                for attr in attrs:
                    out_dict["%s_%d" % (attr, i+1)] = getattr(bs_list[i], attr)
        return out_dict

    def get_sites_dist(self, site1=0, site2=1):
        """Get distance between two binding sites."""
        if len(self.proteins) == 1:
            protein = self.proteins[0]
            return abs(self.bsites[protein][site2].core_mid - self.bsites[protein][site1].core_mid)
        protein1 = self.proteins[0]
        protein2 = self.proteins[1]
        return abs(self.bsites[protein1][site1].core_mid - self.bsites[protein2][site1].core_mid)

    def get_sites_dict(self):
        """Get dictionary of sites in this sequence."""
        return self.sites_to_dict(self.bsites)

    def site_exist(self):
        """Return true if there is at least 1 site in the sequence."""
        return self.site_count != 0

    def site_count_all(self):
        """Return the number of binding sites."""
        tot = 0
        for protein in self.proteins:
            tot += self.site_count(protein)
        return tot

    def site_count(self, protein):
        """Get number of binding sites in the sequence for the given protein."""
        return len(self.bsites[protein])

    def get_center_bsites(self):
        """Find the indices of non-center binding sites."""
        # get the lists of midpoints of binding sites for all proteins
        midpoints = {}
        for protein in self.proteins:
            preds = self.model_preds[protein].predictions
            midpoints[protein] = [(int(d['core_start']) + int(d['core_width'])/2) for d in preds]

        min_dist = float('inf')
        mid_lst1 = midpoints[self.proteins[0]]
        if len(self.proteins) > 1:
            mid_lst2 = midpoints[self.proteins[1]]
        else:
            mid_lst2 = midpoints[self.proteins[0]]

        # find the pair centered binding sites
        for i in range(len(mid_lst1)):
            for j in range(len(mid_lst2)):
                mid1 = mid_lst1[i]
                mid2 = mid_lst2[j]
                dist = (mid1 + mid2) / 2 - 18
                if abs(dist) < min_dist:
                    min_dist = abs(dist)
                    keep_mid1 = i
                    keep_mid2 = j

        # returns a dictionary of index of binding sites to keep for each protein
        if len(self.proteins) > 1:
            return {self.proteins[0]: [keep_mid1], self.proteins[1]: [keep_mid2]}
        return {self.proteins[0]: [keep_mid1, keep_mid2]}

    def are_centered(self):
        """Check if the pair of binding sites are centered."""
        # if more than 2 binding sites, raise exception
        if self.site_count_all() != 2:
            return False
            # raise Exception("Number of binding sites found is {}. Should be 2."
            #                 .format(self.site_count_all()))

        # get the midpoints of binding sites for all proteins
        midpoints = []
        for protein in self.proteins:
            preds = self.model_preds[protein].predictions
            midpoints += [(d['core_start'] + d['core_width'] / 2) for d in preds]

        dist = ((midpoints[0] + midpoints[1]) / 2) - 18
        if abs(dist) > 3:
            return False
        return True

    def remove_pos(self, pos):
        """
        Return the indices of each protein after removing binding sites.

        Binding site(s) removed are specified by pos.
        pos: list of indices of binding sites to be removed from the sequence
        Return: a dictionary of indices to mutate for each protein
        """
        # make sure there are 2 binding sites left
        if not self.is_valid():
            raise Exception("Not a valid wild type")

        if len(self.proteins) == 1:
            return {self.proteins[0]: pos}

        if pos == [0, 1]:
            return {self.proteins[0]: [0], self.proteins[1]:[0]}

        # find left hand binding site
        if pos == [0]:
            if self.bsites[self.proteins[0]][0]['core_start']\
               < self.bsites[self.proteins[1]][0]['core_start']:
                return {self.proteins[0]: [0], self.proteins[1]: []}
            return {self.proteins[0]: [], self.proteins[1]: [0]}
        else:
            if self.bsites[self.proteins[0]][0]['core_start'] \
               < self.bsites[self.proteins[1]][0]['core_start']:
                return {self.proteins[0]: [], self.proteins[1]: [0]}
            return {self.proteins[0]: [0], self.proteins[1]: []}

    def is_valid(self):
        """
        Mutate a sequence to make it valid.

        A valid sequence has exactly two centered binding site.
        Return: True if resulting sequence is valid, False otherwise
        """
        # get number of proteins we are dealing with
        num_prot = len(self.proteins)
        # check base cases for homotypic cluster
        if num_prot == 1:
            if self.site_count_all() == 2 and self.are_centered():
                return True
            elif self.site_count_all() < 2:
                return False
        # check base cases for heterotypic cluster
        elif num_prot == 2:
            if self.site_count(self.proteins[0]) == self.site_count(self.proteins[1])==1 \
               and self.are_centered():
                return True
            elif self.site_count(self.proteins[0]) == 0 \
                    or self.site_count(self.proteins[1]) == 0:
                return False

        # if there are more than 2 significant binding sites, mutate
        else:
            to_keep = self.get_center_bsites()
            mut_seq = self.abolish_sites(to_keep, mode="to_keep",
                                         escore_threshold=self.escore_cutoff)
            # Update the current Sequence object with the valid, mutated sequence
            self.__init__(mut_seq.escore_preds, mut_seq.model_preds,
                          proteins=mut_seq.proteins,
                          escore_cutoff=self.escore_cutoff,
                          escore_gap=self.escore_gap, pbmescores=self.pbmescores)
            # check if mutation was successful
            if ((len(self.proteins) == 1 and self.site_count_all() == 2) \
               or (len(self.proteins) == 2 and self.site_count(self.proteins[0])==self.site_count(self.proteins[1])==1)) \
               and self.are_centered():
                return True
            else:
                return False
        return False
