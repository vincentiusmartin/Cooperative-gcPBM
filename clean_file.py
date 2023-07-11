import argparse
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
import os

import coopgcpbm.arranalysis as arr
import coopgcpbm.util.bio as bio

def fix_naming(df):
    df = df.drop(columns=["ori"])
    df_ori = df[["Name","Sequence","type"]] \
        .drop_duplicates()
    df_ori["ori"] = df_ori \
        .groupby(["Name","type"],as_index=False) \
        .cumcount() + 1
    df_ori["ori"] = "o" + df_ori["ori"].astype(str)
    df = df.merge(df_ori, on=["Name","Sequence","type"])
    return df

def fixdup_single_name(nm, df):
    dseq1 = df[(df["type"] == "wt")][["Sequence"]].rename({"Sequence":"Sequence_x"}).drop_duplicates()
    dseq2 = df[["Sequence"]].rename({"Sequence":"Sequence_y"}).drop_duplicates()

    dseq1_o1_seqs = df[(df["type"] == "wt") & (df["ori"] == "o1")]["Sequence"].drop_duplicates().tolist()
    rcgrp = {}
    for i in range(len(dseq1_o1_seqs)):
        rcgrp[dseq1_o1_seqs[i]] = i
        rcgrp[bio.revcompstr(dseq1_o1_seqs[i])] = i

    dseq1["key"], dseq2["key"] = 1,1
    comb = dseq1.merge(dseq2, on="key").drop("key", 1)  # get cartesian product between the two

    comb["simscore"] = comb.apply(lambda row: SequenceMatcher(None, row["Sequence_x"], row["Sequence_y"]).ratio(), axis=1)
    comb = comb.sort_values(["Sequence_x","simscore"], ascending=[True,False])
    selectedcmb = comb.groupby("Sequence_x").head(4).rename(columns={"Sequence_y":"Sequence"})
    selectedcmb["group"] = selectedcmb["Sequence_x"].apply(lambda x: "%s_%s" % (nm, rcgrp[x]))
    selectedcmb = selectedcmb[["Sequence", "group"]].drop_duplicates()

    named = df.merge(selectedcmb, on="Sequence")
    return named

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Generate clean probe file with information needed for the pipeline')
    parser.add_argument(action="store", dest="path", type=str, help='File input path')
    parser.add_argument('-k', '--key', action="store", dest="key", type=str,  help='include key string')
    parser.add_argument('-l', '--keycol', action="store", dest="keycol", type=str, default="Name",  help='column for the main probes key')
    parser.add_argument('-e', '--exkey', action="store", dest="exkey", type=str,  help='exclude key string')
    parser.add_argument('-n', '--negkey', action="store", dest="negkey", default="NegativeCtrl", type=str, help='include negctrl key string')
    parser.add_argument('-c', '--seqcols', action="store", dest="sc", default="Name,type,rep,ori", type=str, help='sequence cols order, str separated by comma')
    parser.add_argument('-d', '--negcols', action="store", dest="nc", default="Name,rep,ori", type=str, help='negctrl cols order, str separated by comma')
    parser.add_argument('-f', '--fixnaming', action="store_true", dest="fixnaming", help='fix naming error in the orientatiion files')
    parser.add_argument('-g', '--fixdup', action="store_true", dest="fixdup", help='fix naming duplicates')
    parser.add_argument('-i', '--probeid', action="store_true", dest="prbid", help='include probe id in the output')
    parser.add_argument('-o', '--outdir', action="store", dest="outdir", default=".", help='output directory to store output files')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    pd.set_option("display.max_columns",None)
    filename = Path(args.path).stem
    df, neg = arr.read_chamber_file(args.path, includekey=args.key, keycol=args.keycol, negkey=args.negkey, excludekey=args.exkey,
                seqcols=args.sc.split(","), negcols=args.nc.split(","), include_id=args.prbid)

    df.rename(columns={"Alexa488Adjusted": "intensity"}, inplace=True)
    neg.rename(columns={"Alexa488Adjusted": "intensity"}, inplace=True)

    if args.fixnaming:
        df = fix_naming(df)
    if args.fixdup:
        pd.set_option("display.max_columns",None)
        g = df.groupby("Name")
        dflist = []
        for nm, gdf in g:
            if gdf.shape[0] != 24:
                gdf = fixdup_single_name(nm, gdf)
                gdf["Name"] = gdf["group"]
                gdf = gdf.drop(columns=["group"])
            dflist.extend(gdf.to_dict('records'))
        df = pd.DataFrame(dflist)

    df.to_csv(os.path.join(args.outdir, "%s_pr_clean.csv" % filename), index=False)
    neg.to_csv(os.path.join(args.outdir, "%s_neg_clean.csv" % filename), index=False)
