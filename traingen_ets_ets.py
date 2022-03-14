import pandas as pd
import argparse

from coopgcpbm.modeler.cooptrain import CoopTrain
import coopgcpbm.traingen as tg
import coopgcpbm.modeler.plotlib as pl

from sitespredict.imads import iMADS
from sitespredict.imadsmodel import iMADSModel
from sitespredict.pwm import PWM
from sitespredict.kompas import Kompas

from util import bio

from chip2probe.modeler.cooptrain import CoopTrain

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
    posdf['site_str_score'], posdf['site_str_ori'], posdf['site_str_core'] =  tg.pwm_score(posdf, pwm, "site_str_start", 4, 3, seqcol=seqcol)
    posdf['site_wk_score'], posdf['site_wk_ori'],  posdf['site_wk_core'] =  tg.pwm_score(posdf, pwm, "site_wk_start", 4, 3, seqcol=seqcol)
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

def gen_training(df, pwm, kompas):
    train = get_sites_pos(df, kompas, pwm)
    # reverse -- to ++
    train00 = train[train["orientation"] == "-/-"][["Name","Sequence","label"]]
    train00["Sequence"] = train00["Sequence"].apply(lambda x: bio.revcompstr(x))
    train00 = get_sites_pos(train00, kompas, pwm)
    train = pd.concat([train[train["orientation"] != "-/-"], train00])
    return train.drop_duplicates()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Label sequences as cooperative/ambiguous/independent')
    parser.add_argument(action="store", dest="path", type=str, help='Labeled files input path')
    parser.add_argument('-p', '--pwmpath', action="store", dest="pwmpath", type=str,  help='PWM path')
    parser.add_argument('-k', '--kmeralign', action="store", dest="kmeralign", type=str,  help='K-mer Ets1_kmer_alignment path')
    args = parser.parse_args()

    pwm_ets = PWM(args.pwmpath, log=True)
    kompas_ets = Kompas(args.kmeralign,
                    core_start = 11, core_end = 15, core_center = 12)
    df = pd.read_csv(args.path).drop_duplicates()


    df = df[(df["label"] == "cooperative") | (df["label"] == "independent")]
    print(df[["label"]].value_counts())
    train = gen_training(df, pwm_ets, kompas_ets).drop_duplicates(["Sequence"])

    print(train["label"].value_counts())
    train.to_csv("train_ets1_ets1.tsv", index=False, sep="\t")

    train.rename(columns={'site_str_score': 'Binding strength of the stronger site', 'site_wk_score': 'Binding strength of the weaker site'}, inplace=True)
    pl.plot_stacked_categories(train, "distance", path="distance_bar.png", title="Distance distribution", ratio=True, figsize=(17,4), color = ["#b22222","#FFA07A"])
    pl.plot_stacked_categories(train, "orientation", path="ori_bar.png", title="Relative sites orientation\ndistribution", ratio=True, figsize=(9,5), color = ["#b22222","#FFA07A"])
    pl.plot_box_categories(train, path="boxplot.png", incols=["Binding strength of the stronger site", "Binding strength of the weaker site"], alternative="smaller", color = ["#b22222","#FFA07A"])
