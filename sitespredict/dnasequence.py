"""
Sequence class

Created on ?

Authors: Vincentius Martin, Farica Zhuang
"""
import copy


class DNASequence(object):
    """DNASequence class

    Generate sequence based on a sites predictor model (imads/pwm/kompas) and E-score.
    The sites are predictor sites that overlap E-score coordinate.

    Example:
    >>>
    """

    def __init__(self, sequence, predictor, escore,
                 escore_cutoff=0.4, escore_gap = 0):
        """
        Args:
            sequence: sequence string
            predictor: imads/pwm/kompas
            escore: escore object, if none, we just use predictor
        """
        self.sequence = sequence
        self.predictor = predictor
        self.escore = escore
        self.sites, self.sites_specific = self.predict_sites(escore_cutoff, escore_gap)

    def get_sites(self):
        """
        getter for sites
        """
        return self.sites, self.sites_specific

    def predict_sites(self, escore_cutoff=0.4, escore_gap = 0):
        """
        Predict binding sites using predictor and escore models.

        We only consider a site as specific if any escores overlap with
        the imads_prediction.
        The complexity here is O(mn) based on the number of sites in predictor
        and escore, another solution will be to take sequence from escore list
        if we found overlap, but this might not be a safe assumption.

        Args:
            predictor:
            escore:
        Returns:
            unfiltered, filtered sequences
        """
        escore_sites = self.escore.get_escores_specific(self.sequence,
                        escore_cutoff=escore_cutoff, escore_gap=escore_gap)
        predictor_sites = self.predictor.predict_sequence(self.sequence)
        specific_sites = []
        for psite in predictor_sites:
            c_start = psite["core_start"]
            c_end = psite["core_start"] + psite["core_width"]
            for esite in escore_sites:
                e_start = esite["startpos"]
                e_end = esite["startpos"] + esite["escorelength"]
                if c_end >= e_start and e_end >= c_start:
                    specific_sites.append(psite)
                    continue
        return predictor_sites, specific_sites

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
            # get escore of this mutation,
            # since we input 8-mer there should be only 1 site
            all_muts[mutseq] = self.escore.predict_sequence(mutseq)[0]['score']

        # find a mutated sequence that is non-specific for all proteins
        min_sum = float("inf")
        min_seq = ""

        # iterate through mutated sequences
        min_seq = min(all_muts.keys(), key=(lambda k: all_muts[k]))

        return min_seq

    def get_max_non_intersecting_escore(self, mutantseq, site_index, barrier=1, ignorelist=[], nomutrange=[]):
        """
        Get maxescore from given core that doesn't intersect a different core.

        Args:
            mutantseq: a different full sequence from what the sequece has originally
            ignorelist: if maxepred equal sequence in ignore list, then go to the next one

        This is to prevent mutating other cores.
        """
        # get the sequence we want to get the escore of
        s_start = self.sites[site_index]["site_start"]
        s_end = self.sites[site_index]["site_start"] + self.sites[site_index]["site_width"]
        site_seq = mutantseq[s_start:s_end]
        # get the escore prediction for the sequence
        epreds = self.escore.predict_sequence(site_seq)
        # initialize non intersecting max escore
        max_escore = {"score": float("inf")}
        # find non intersecting max escore
        while max_escore["score"] == float("inf") and epreds:
            # get 8mer in the given core with the highest escore
            max_val = max(epreds, key=lambda x: x['score'])
            if max_val in ignorelist:
                epreds.remove(max_val)
                continue
            # get position of this max escore
            max_val_seqpos = self.sites[site_index]["site_start"] + max_val['position']
            flag = True
            # loop through each protein
            #for curr_protein in self.sites_specific:
            # loop through each binding site of this protein
            if len(self.sites) != 1:
                # then we check other sites
                for i in range(len(self.sites)):
                    # skip if checking against itself
                    if i == site_index:
                        continue
                    # get region of this other core that the
                    unmutated_start = self.sites[i]["core_start"] - barrier
                    unmutated_end = self.sites[i]["core_start"] + self.sites[i]["core_width"] + barrier
                    # if this max escore intersects with another core, remove
                    # or if we found maxval previously, we got this from ignore list
                    if (max_val_seqpos >= unmutated_start and max_val_seqpos < unmutated_end and max_val in epreds):
                        epreds.remove(max_val)
                        flag = False
                        break
            if flag and max_val in epreds and ( \
                (len(nomutrange) > 0 and max_val_seqpos >= nomutrange[0] and max_val_seqpos < nomutrange[1])):
                epreds.remove(max_val)
            else:
                max_escore = copy.copy(max_val)
        return max_escore

    def _eliminate_site(self, sequence, site_index,
                       escore_threshold=0.3, barrier=1, nomutrange=[]):
        """
        Eliminate the given site from the sequence.

        This is a local function, please use abolish site, the wrapper for this.

        This assumes that input is a sequence of a site
        site_index: which site to mutate
        """
        maxescore = 1  # just to initialize with a large value
        prevmax = -1  # just to avoid infinite loop
        # list of mutated sites
        site_mutpos = []

        # sequence
        # we use self.sites here because the site to eliminate is based on the original
        # wt. E.g. when making m3, after we remove 1 site, we want to mutate the second
        # site, but the first site is already gone. That's why we need to use self.sites.
        target_cstart = self.sites[site_index]["core_start"]
        full_sequence = str(sequence)
        prevseq = str(sequence)

        init_sites = self.predictor.predict_sequence(sequence)
        ignlist = []
        site_exist = True
        while prevmax != maxescore and maxescore != float("inf"):
            maxepreds = self.get_max_non_intersecting_escore(full_sequence,site_index,barrier,ignorelist=ignlist,nomutrange=nomutrange)
            if maxepreds not in ignlist:
                ignlist.append(maxepreds)
            else:
                return "",[] # not sure about this, think again later
            maxescore = maxepreds['score']

            # EXIT CONDITION, we successfully mutate the sites
            if maxescore < escore_threshold and not site_exist:
                break

            # return immediately if the site can't be mutated
            if maxescore == float("inf"):  # no e-score that can be chosen
                print("Couldn't choose mutated spot")
                # since there is no site to be mutated, then just use empty list
                return full_sequence, []
            seq_tomutate = maxepreds["escore_seq"]
            midpos = len(seq_tomutate) // 2
            # get new mutated sequence
            mutated_escore_seq = self.mutate_escore_seq_at_pos(seq_tomutate, midpos, escore_threshold)
            if mutated_escore_seq != "":
                # mutate the sequence
                mut_start = self.sites[site_index]["site_start"] + maxepreds["start_idx"]
                mut_end = mut_start + len(mutated_escore_seq)
                newseq = full_sequence[:mut_start] + mutated_escore_seq + full_sequence[mut_end:]
                # if the max non intersecting escore is below the threshold
                # and imads prediction below threshold, then nothing to mutate
                # Note that we use imads here because there are cases where escore is
                # below threshold but imads is still above
                newpred  = self.predictor.predict_sequence(newseq)
                # check if the site we want to abolish still exist and be sure we are not creating new site
                site_exist = any([npred['core_start'] == target_cstart for npred in newpred])
                if not site_exist and len(newpred) != len(init_sites)-1:
                    # if not site exist means we are successful in deleting the sequence but,
                    # if we still have >= n sequences, this means we create new sequence
                    # or if we have < (n-1) this means we destroy existing site
                    # Therefore, keep going and choose different spot to mutate
                    continue
                full_sequence = full_sequence[:mut_start] + mutated_escore_seq + full_sequence[mut_end:]
                site_mutpos.append(mut_start + midpos)
            else:
                full_sequence = ""
                site_mutpos = []
        # return the new mutated sequence and the positions mutated
        return full_sequence, site_mutpos

    def abolish_sites(self, sites, mode="to_eliminate",
                      escore_threshold=0.3 , barrier=1, nomutrange=[],
                      seqonly=True):
        """
        Abolish sites from a sequence

        Args:

        Returns:

        """
        # if we have multiple pbmescore and proteins to abolish
        mutated = str(self.sequence)

        sites_to_mutate = []
        if mode == "to_eliminate":
            sites_to_mutate = sites
        elif mode == "to_keep":
            sites_to_mutate = list(set(range(len(self.sites))) - set(sites))
        else:
            raise Exception("type should either be to_eliminate or to_keep")

        mutpos = []
        for i in sites_to_mutate:
            mutated, site_mutpos = self._eliminate_site(sequence=mutated,
                                                       site_index=i,
                                                       escore_threshold=escore_threshold,
                                                       barrier=barrier,
                                                       nomutrange=nomutrange)
            mutpos.extend(site_mutpos)

        if seqonly:
            return mutated
        else:
            functions = []
            for pos in mutpos:
                functions.append({"func": "axvline", "args": [pos],
                                  "kwargs": {"color": "purple",
                                             "linestyle": "dashed",
                                             "linewidth": 1}})
            return {"sequence":mutated, "plt":functions}
