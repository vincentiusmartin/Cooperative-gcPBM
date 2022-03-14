'''
Created on May, 2020

@author: vincentiusmartin
@description: Make mutations for custom probes. All columns on the mutants are
    repredicted so we can validate that mutations don't change features that
    should not be changed.
'''

# 1. minimum distance??

from chip2probe.util import bio
import pandas as pd
import math
import chip2probe.util.coopgeneral as cg
import chip2probe.training_gen.traingen as tg

def makeseqdict(seq, id, preds, type, comment):
    return {"id":id,
            "Sequence":seq,
            "s1_score": preds[0]["score"],
            "s1_ori": preds[0]["ori"],
            "s1_tf":preds[0]["tf"],
            "s2_score": preds[1]["score"],
            "s2_ori": preds[1]["ori"],
            "s2_tf":preds[1]["tf"],
            "distance":preds[1]["pos"] - preds[0]["pos"],
            "muttype":type,
            "comment":comment}

# =========================

def mutate_core(seq, id, kompdict, pwmdict, fwdcore, revcore):
    initpreds = tg.pred_all(seq, kompdict, pwmdict)
    scr1, scr2 = initpreds[0]['score'], initpreds[1]['score']
    res = []
    for pred in initpreds:
        c_st = pred["core_start"]
        c_end = pred["core_start"]+pred["core_width"]
        curcore = seq[c_st:c_end]
        if curcore in fwdcore[pred["tf"]]:
            corelist = set(fwdcore[pred["tf"]]) - {curcore}
        elif curcore in revcore[pred["tf"]]:
            corelist = set(revcore[pred["tf"]]) - {curcore}
        else:
            print("Could not find core in the corelist")
            corelist = [] # can just break
        for c in corelist:
            curmut = seq[:c_st] + c + seq[c_end:]
            mutpreds = tg.pred_all(curmut, kompdict, pwmdict)
            if len(mutpreds) != 2 or \
                initpreds[0]["core_start"] != mutpreds[0]["core_start"] or \
                initpreds[1]["core_start"] != mutpreds[1]["core_start"] or \
                initpreds[0]["tf"] != mutpreds[0]["tf"] or \
                initpreds[1]["tf"] != mutpreds[1]["tf"]:
                continue
            suffix = "s2" if math.isclose(scr1,mutpreds[0]['score'],rel_tol=0.01) else "s1"
            res.append(makeseqdict(curmut, id, mutpreds, type="strength", comment="core_%s" % suffix))
    return res

def mutate_flank(seq, id, kompdict, pwmdict, flanklen=2, max_score_delta=0.01):
    """

    """
    base = {'A', 'T', 'G', 'C'}
    initpreds = tg.pred_all(seq, kompdict, pwmdict)
    scr1, scr2 = initpreds[0]['score'], initpreds[1]['score']
    res = []
    mutpos = []
    corepos = []
    # get the position to mutate
    for pred in initpreds:
        c_st = pred["core_start"]
        c_end = pred["core_start"]+pred["core_width"]
        curcrpos = list(range(c_st,c_end))
        curmutpos = list(range(c_st-flanklen,c_st)) + list(range(c_end,c_end+flanklen))
        corepos.extend(curcrpos)
        mutpos.extend(curmutpos)
    # get unique position to mutate
    mutpos = sorted(set(mutpos) - set(corepos))
    for mp in mutpos:
        btargets = base - {seq[mp]}
        for bt in btargets:
            curmut = seq[:mp] + bt + seq[mp+1:]
            mutpreds = tg.pred_all(curmut, kompdict, pwmdict)
            # make sure we have 2 sites and mutation only change 1 site
            if len(mutpreds) != 2 or \
                initpreds[0]["core_start"] != mutpreds[0]["core_start"] or \
                initpreds[1]["core_start"] != mutpreds[1]["core_start"] or \
                initpreds[0]["tf"] != mutpreds[0]["tf"] or \
                initpreds[1]["tf"] != mutpreds[1]["tf"] or \
                (not math.isclose(scr1,mutpreds[0]['score'],rel_tol=max_score_delta) and \
                 not math.isclose(scr2,mutpreds[1]['score'],rel_tol=max_score_delta)):
                continue
            suffix = "s2" if math.isclose(scr1,mutpreds[0]['score'],rel_tol=max_score_delta) else "s1"
            res.append(makeseqdict(curmut, id, mutpreds, type="strength", comment="flank_%s" % suffix))
    return res

# ========================

def mutate_affinity(seqdf, kompdict, pwmdict, coredict, pwmthres=0):
    """
    Make mutation to change the affinity (i.e. strength) prediction.
    But not eliminating the sites (pwm score > 0)

    1. change the core to other version
    2. change flank
    """
    # we need the index
    seqs = seqdf.to_dict('dict')['Sequence']
    fwd_cores = {}
    rev_cores = {}
    for tf in kompdict.keys():
        fwd_cores[tf] = coredict[tf]
        rev_cores[tf] = [bio.revcompstr(c) for c in fwd_cores[tf]]

    mutres = []
    for key in seqs:
        seq = seqs[key]
        curmutres = []

        # 1. Mutate the cores
        coremts = mutate_core(seq, key, kompdict, pwmdict, fwd_cores, rev_cores)
        curmutres.extend(coremts)

        # 2. Mutate the flanks
        flankmts = mutate_flank(seq, key, kompdict, pwmdict)
        curmutres.extend(flankmts)

        if len(curmutres) > 0:
            mutres.extend(curmutres)
    return mutres

# =========================

def mutate_distance(seqdf, kompdict, pwmdict, mindist=5):
    """
    """
    seqs = seqdf.to_dict('dict')['Sequence']
    mutres = []
    for key in seqs:
        seq = seqs[key]
        initpreds = tg.pred_all(seq, kompdict, pwmdict)
        s1_end = initpreds[0]["core_start"] + initpreds[0]["core_width"]
        s2_start = initpreds[1]["core_start"]
        initdist = initpreds[1]["pos"] - initpreds[0]["pos"]
        move = 0
        while initdist - move > mindist:
            move += 1
            curmut = cg.move_single_site(seq, s1_end, s2_start, -move)
            mutpreds = tg.pred_all(curmut, kompdict, pwmdict)
            if len(mutpreds) != 2 or \
                not math.isclose(initpreds[0]["score"],mutpreds[0]['score'],rel_tol=0.1) or \
                not math.isclose(initpreds[1]["score"],mutpreds[1]['score'],rel_tol=0.1) or \
                initpreds[0]["tf"] != mutpreds[0]["tf"] or \
                initpreds[1]["tf"] != mutpreds[1]["tf"]:
                continue
            mutres.append(makeseqdict(curmut, key, mutpreds, type="distance", comment="fixed_s1"))
    return mutres

def mutate_orientation(seqdf, kompdict, pwmdict, spread=3):
    seqs = seqdf.to_dict('dict')['Sequence']
    mutres = []
    for key in seqs:
        seq = seqs[key]
        initpreds = tg.pred_all(seq, kompdict, pwmdict)
        s1_end = initpreds[0]["core_start"]+initpreds[0]["core_width"]
        s2_start = initpreds[1]["core_start"]
        if spread >= (s2_start-s1_end): # can't mutate without affecting the other core
            continue
        sitestarget = [[0],[1],[0,1]]
        for sites in sitestarget:
            curmut = str(seq)
            # mutate sites
            for s in sites:
                pred = initpreds[s]
                start, end = pred["core_start"]-spread, pred["core_start"]+pred["core_width"]+spread
                curmut = curmut[:start] + bio.revcompstr(curmut[start:end]) + curmut[end:]
            mutpreds = tg.pred_all(curmut, kompdict, pwmdict)
            if len(mutpreds) != 2 or \
              initpreds[0]["core_start"] != mutpreds[0]["core_start"] or \
              initpreds[1]["core_start"] != mutpreds[1]["core_start"] or \
              initpreds[0]["tf"] != mutpreds[0]["tf"] or \
              initpreds[1]["tf"] != mutpreds[1]["tf"]:
                continue
            mutres.append(makeseqdict(curmut, key, mutpreds, type="orientation", comment="site_%s"%"_".join([str(s+1) for s in sites])))
    return mutres

def initiate_wt(seqdf, kompdict, pwmdict):
    seqs = seqdf.to_dict('dict')['Sequence']
    mutres = []
    for key in seqs:
        seq = seqs[key]
        preds = tg.pred_all(seq, kompdict, pwmdict)
        if len(preds) != 2:
            print("predictions error for %s" % seq)
            continue
        # append wt to the result
        mutres.append(makeseqdict(seq, key, preds, "distance", "wt"))
        mutres.append(makeseqdict(seq, key, preds, "strength", "wt"))
        mutres.append(makeseqdict(seq, key, preds, "orientation", "wt"))
    return mutres
