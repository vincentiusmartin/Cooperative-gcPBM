import pandas as pd
from operator import itemgetter

from coopgcpbm.util import bio

def pred_komp_pwm(seq, kompas, pwm):
    pred = kompas.predict_sequence(seq)
    for k in pred:
        flanklen = (pwm.length - k["core_width"])//2
        start, end = k["core_start"]-flanklen, k["core_start"]+k["core_width"]+flanklen
        pwmpred =  pwm.predict_sequence(seq[start:end],zero_thres=False)
        if len(pwmpred) == 0:
            return []
        pwmpred = pwmpred[0]
        k["score"] = pwmpred["score"]
        k["pos"] = k["core_mid"]
        k["ori"] = pwmpred["orientation"]
    return pred

def pred_all(seq, kompdict, pwmdict):
    tfs = kompdict.keys()
    preds = []
    for tf in tfs:
        curp = pred_komp_pwm(seq, kompdict[tf], pwmdict[tf])
        for c in curp:
            c["tf"] = tf
        preds.extend(curp)
    preds = sorted(preds, key=itemgetter('pos'))
    return preds

def pwm_score(dfpos, pwm, startcol, corelen, flanklen, seqcol="Sequence"):
    """
    e.g. pwm_score(dft, pwm_ets, "ets_start", 4, 3)
    """
    res = []
    ori = []
    core = []
    for idx, row in dfpos.iterrows():
        seq = row[seqcol][row[startcol]-flanklen:row[startcol]+corelen+flanklen]
        core.append(row[seqcol][row[startcol]:row[startcol]+corelen])
        if len(seq) < (corelen + 2*flanklen): # error
            res.append(-999)
            ori.append(-999)
        else:
            pred = pwm.predict_sequence(seq,zero_thres=False)[0]
            res.append(pred["score"])
            cur_ori = 0 if pred["orientation"] == -1 else 1
            ori.append(cur_ori)
    return res, ori, core

def get_sites_pos(df, kompas, pwm, seqcol="Sequence"):
    """
    Get site position for each sequence

    Args:
        df: input data frame

    """
    if  df.empty:
        return df
    seqlist = df[seqcol].unique().tolist()
    poslist = []
    misscount = 0
    for seq in seqlist:
        x = kompas.predict_sequence(seq)
        if len(x) != 2:
            continue
        # WE LET "SITE STR" BE THE FIRST SITE IN THE BEGINNING
        poslist.append({seqcol:seq, "site_str_pos":x[0]['core_start'] + 2, 'site_str_start':x[0]['core_start'], 'site_wk_pos':x[1]['core_start'] + 2, 'site_wk_start':x[1]['core_start']})
    posdf = pd.DataFrame(poslist)
    posdf['site_str_score'], posdf['site_str_ori'], posdf['site_str_core'] =  pwm_score(posdf, pwm, "site_str_start", 4, 3, seqcol=seqcol)
    posdf['site_wk_score'], posdf['site_wk_ori'],  posdf['site_wk_core'] =  pwm_score(posdf, pwm, "site_wk_start", 4, 3, seqcol=seqcol)
    posdf = posdf[(posdf["site_str_score"] != -999) & (posdf["site_wk_score"] != -999)]

    orimap = {0:"-",1:"+"}
    posdf["orientation"] = posdf.apply(lambda x: "%s/%s" % (orimap[int(x["site_str_ori"])], orimap[int(x["site_wk_ori"])]),axis=1)
    posdf["distance"] = posdf["site_wk_pos"] - posdf["site_str_pos"]

    # now we flip the left and right, we flip all but orientation
    flip_target = []
    for i,r in posdf.iterrows():
        if r["site_str_score"] < r["site_wk_score"]:
            flip_target.append(i)
    posdf.loc[flip_target,['site_str_score','site_wk_score']] = posdf.loc[flip_target,['site_wk_score','site_str_score']].values
    posdf.loc[flip_target,['site_str_pos','site_wk_pos']] = posdf.loc[flip_target,['site_wk_pos','site_str_pos']].values
    posdf.loc[flip_target,['site_str_ori','site_wk_ori']] = posdf.loc[flip_target,['site_wk_ori','site_str_ori']].values
    posdf.loc[flip_target,['site_str_core','site_wk_core']] = posdf.loc[flip_target,['site_wk_core','site_str_core']].values

    posdf = posdf[[seqcol,"site_str_pos","site_str_score","site_wk_pos","site_wk_score" ,"distance","site_str_ori","site_str_core", "site_wk_ori","site_wk_core","orientation"]]
    posdf = df.merge(posdf,on=seqcol)
    return posdf

def gen_training(seqlist, pwm, kompas):
    df = pd.DataFrame({'Sequence':seqlist})
    train = get_sites_pos(df, kompas, pwm)
    # reverse -- to ++
    train00 = train[train["orientation"] == "-/-"]
    train00["Sequence"] = train00["Sequence"].apply(lambda x: bio.revcompstr(x))
    train00 = get_sites_pos(train00, kompas, pwm)
    train = pd.concat([train[train["orientation"] != "-/-"], train00])
    return train.drop_duplicates()
