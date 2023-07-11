import os

import pandas as pd

import coopgcpbm.modeler.plotlib as plot
import coopgcpbm.traingen as tg
from coopgcpbm.sitespredict.pwm import PWM


def predict_strength(train, pred, tfname, flanklen=0):
    afflist = []
    for idx, row in train.iterrows():
        pos = row["%s_pos"%tfname]
        sites = pred.predict_sequence(row["fullseq"])
        aff = None
        for predsite in sites:
            predcpos = predsite["site_start"] - flanklen
            if pos > predcpos and pos < predcpos + predsite["site_width"]:
                aff = predsite['score']
                break
        afflist.append(aff)
    return afflist

if __name__ == "__main__":
    pd.set_option("display.max_rows",None)

    # ========= INPUT FIELDS =========

    # outdir = "data/analysis_files/ETS1_RUNX1/training"
    # lbled_path = "data/analysis_files/ETS1_RUNX1/labeled/ets1_runx1_seqlbled.tsv"
    # maintf = "ets1"
    # cooptf = "runx1"
    # color = ["#b22222","#FFA07A"]

    outdir = "data/analysis_files/RUNX1_ETS1/training"
    lbled_path = "data/analysis_files/RUNX1_ETS1/labeled/runx1_ets1_seqlbled.tsv"
    maintf = "runx1"
    cooptf = "ets1"
    color = ["#0343df","#75bbfd"]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if maintf == "ets1":
        pwm_main = PWM("data/sitemodels/%s.txt" % maintf, log=True, reverse=False)
        pwm_coop = PWM("data/sitemodels/%s.txt" % cooptf, 8, 17, log=True, reverse=True)
    else:
        pwm_main = PWM("data/sitemodels/%s.txt" % maintf, 8, 17, log=True, reverse=True)
        pwm_coop = PWM("data/sitemodels/%s.txt" % cooptf, log=True, reverse=False)

    # ======================================================

    train = pd.read_csv(lbled_path, sep="\t").drop_duplicates(subset=["Name"], keep="first")
    train = train[(train["label"] == "cooperative") | (train["label"] == "independent")]

    corelendict = {"ets1":4,"runx1":5}
    flanklendict = {"ets1":3,"runx1":2}

    train["%s_score" % maintf], train["%s_ori" % maintf], train['%s_core' % maintf] =  tg.pwm_score(train, pwm_main, "%s_start" % maintf, corelendict[maintf], flanklendict[maintf])
    train["%s_score" % cooptf], train["%s_ori" % cooptf], train['%s_core' % cooptf] = tg.pwm_score(train, pwm_coop, "%s_start" % cooptf, corelendict[cooptf], flanklendict[cooptf])
    train = train[(train["%s_score" % maintf] != -999) & (train["%s_score" % cooptf] != -999)]

    orimap = {0:"-",1:"+"}
    train["orientation"] = train.apply(lambda x: "%s/%s" % (orimap[int(x["%s_ori" % maintf])], orimap[int(x["%s_ori" % cooptf])]),axis=1)
    train = train[(train["ets1_score"] != - 999) & (train["runx1_score"] != - 999)]

    # print(train.groupby(["distance","orientation"])["label"].value_counts())
    train.to_csv(os.path.join(outdir, "train_%s_%s.tsv" % (maintf,cooptf)),sep="\t", index=False, float_format='%.3f')

    train.rename(columns={'%s_score' % maintf: '%s strength\n(main TF)' % maintf.capitalize(), '%s_score' % cooptf: '%s strength\n(cooperator TF)' % cooptf.capitalize()}, inplace=True)
    plot.plot_stacked_categories(train, "distance", path=os.path.join(outdir, "distance_bar.png"), title="Distance distribution", ratio=True, figsize=(17,4), color=color)
    plot.plot_stacked_categories(train, "orientation", path=os.path.join(outdir, "ori_bar.png"), title="Relative sites orientation\ndistribution", ratio=True, figsize=(9,5), color=color)
    plot.plot_box_categories(train, path=os.path.join(outdir, "boxplot.png"), incols=["%s strength\n(main TF)" % maintf.capitalize(), "%s strength\n(cooperator TF)" % cooptf.capitalize()], alternative="smaller", color=color)
