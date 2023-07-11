import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import coopgcpbm.util.stats as st
import coopgcpbm.arranalysis as arr

def assign_label(l1, l2):
    # We restrict same label in both orientations
    if l1 == "fail_cutoff" or l2 == "fail_cutoff":
        return "fail_cutoff"
    elif l1 == "cooperative" and l2 == "cooperative":
        return "cooperative"
    elif l1 == "anticooperative" and l2 == "anticooperative":
        return "anticooperative"
    elif (l1 == "independent" and l2 == "independent") or \
         (l1 == "independent" and l2 == "ambiguous") or \
         (l1 == "ambiguous" and l2 == "independent"):
        return "independent"
    else:
        return "ambiguous"

def plot_pval(input, path, ori):
    df = input.copy()
    dfin = pd.DataFrame(df[(df["label_%s" % ori] != "below_cutoff") & (df["label_%s" % ori] != "anticooperative")])[["p_%s"%ori, "label"]]
    dfin['p_%s' % ori] = dfin['p_%s' % ori].apply(lambda x: "%.4f" % x)
    dfin_c = dfin.groupby('p_%s' % ori)['p_%s' % ori].count().reset_index(name="count")
    dfin_c = dfin_c.set_index('p_%s' % ori)
    plt.rcParams["figure.figsize"] = (5,10)
    dfin_c.plot.barh(rot=0)
    plt.savefig(path)
    plt.clf()

def create_cooplbl(indivsum, twosites, pcutoff, pambig):
    p_coop = st.wilcox(twosites, indivsum, "greater")
    p_anti = st.wilcox(twosites, indivsum, "less")
    if p_coop < pcutoff:
        return "cooperative", p_coop
    elif p_coop < pambig:
        return "ambiguous", p_coop
    elif p_anti < pcutoff:
        return "anticooperative", p_anti
    else:
        return 'independent', p_coop

# label the probes for ets1 ets1
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Label sequences as cooperative/ambiguous/independent')
    parser.add_argument(action="store", dest="path", type=str, help='File input path')
    parser.add_argument('-n', '--negpath', action="store", dest="negpath", type=str,  help='Negative ctrl file path')
    parser.add_argument('-e', '--negpercent', action="store", dest="negpercent", type=float, default=0.95,  help='Negative ctrl percentile to use as the negative control intensity cutoff')
    parser.add_argument('-p', '--pcut', action="store", dest="pcut", type=float, default=0.0003,  help='P-value cutoff for sequences labeled as cooperative')
    parser.add_argument('-q', '--pambig', action="store", dest="pambig", type=float, default=0.105,  help='P-value cutoff for sequences labeled as cooperative')
    parser.add_argument('-f', '--fdrcorr',  action="store_true", dest="fdrcorr", help='Perform fdr correction')
    parser.add_argument('-o', '--outdir', action="store", dest="outdir", default=".", help='output directory to store output files')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # read input clean probe and negative control files
    df, negdf = pd.read_csv(args.path), pd.read_csv(args.negpath)
    df = df[df["Name"] != "ets1_GM23338_seq16_0"] # err

    # assign args to variables
    neg_percentile = args.negpercent
    pcut = args.pcut
    p_ambig = args.pambig
    fdrcorr = args.fdrcorr

    nmsqmap = df[(df["ori"] == "o1") & (df["type"] == "wt")][["Name","Sequence"]].drop_duplicates()
    dftype = df[["Name","intensity","type"]].groupby(["Name","type"]).mean().reset_index()

    dftype_med = dftype.groupby(["Name","type"]).median().reset_index().pivot(index='Name', columns='type')
    dftype_med.columns = dftype_med.columns.droplevel(0)
    dftype_med = dftype_med.reset_index().merge(nmsqmap, on="Name")

    print("Number of distinct names %d" % df["Name"].nunique())

    # Get the negative control cutoff
    seqnmneg = negdf[["Name","Sequence"]].drop_duplicates(subset="Name")
    negdf = negdf[["Name","intensity"]].groupby("Name") \
            .median().reset_index() \
            .merge(seqnmneg, on="Name")[["Sequence","intensity"]]
    cutoff = float(negdf[["intensity"]].quantile(neg_percentile))
    print("Negative control cutoff", cutoff)

    # Make permutation of m1+m2-m3 (i.e. indiv) and wt (i.e. two)
    indiv,two = arr.make_replicas_permutation(df, affcol="intensity")
    # indiv, two = pickle.load( open( "indiv.p", "rb" ) ), pickle.load( open( "two.p", "rb" ) )

    arr.permutdict2df(indiv).drop_duplicates().sort_values(["Name","ori"]).to_csv(os.path.join(args.outdir, "ets1_ets1_indiv.csv"), index=False)
    arr.permutdict2df(two).drop_duplicates().sort_values(["Name","ori"]).to_csv(os.path.join(args.outdir, "ets1_ets1_two.csv"), index=False)

    labeled_dict = {}
    median_dict = df.groupby(["Name", "ori", "type"])["intensity"].median().to_dict()
    for ori in list(indiv.keys()):
        orilbls = []
        for k in indiv[ori]:
            rowdict = {}
            if median_dict[(k,ori,'wt')] < cutoff or median_dict[(k,ori,'m3')] > cutoff:
                rowdict['label'], rowdict['p'] = "fail_cutoff", 1
            else:
                rowdict['label'], rowdict['p'] = create_cooplbl(indiv[ori][k], two[ori][k], pcut, p_ambig)
            rowdict['indiv_median'] = np.median(indiv[ori][k])
            rowdict['two_median'] = np.median(two[ori][k])
            rowdict['Name'] = k
            orilbls.append(rowdict)
        labeled_dict[ori] = pd.DataFrame(orilbls)

    df["intensity"] = np.log(df["intensity"])
    lbled_both = labeled_dict["o1"].merge(labeled_dict["o2"], on="Name", suffixes=("_o1", "_o2"))
    lbled_both["label"] = lbled_both.apply(lambda x: assign_label(x["label_o1"], x["label_o2"]), axis=1)

    seqdf = df[(df["type"] == "wt") & (df["ori"] == "o1")][["Name","Sequence"]]
    seqlbled = lbled_both.merge(seqdf, on="Name")[["Name","Sequence","label","p_o1","label_o1","p_o2","label_o2"]].drop_duplicates()
    seqlbled.to_csv(os.path.join(args.outdir, "ets_ets_seqlabeled.csv"), index=False)
    print("Label count", seqlbled["label"].value_counts())

    # plot both result in one orientation only; only take independent and cooperative since we have little to no anticooperative
    filt = lbled_both[(lbled_both['label'] != "fail_cutoff") & (lbled_both['label'] != "anticooperative")]
    lbled_one_ori = labeled_dict['o1'].merge(filt[["Name"]],on="Name")
    arr.plot_classified_labels(lbled_one_ori[lbled_one_ori["label"] != "anticooperative"], col1="indiv_median", col2="two_median", plotnonsignif=False,
                       xlab="M1-M3+M2-M3", ylab="WT-M3", path=os.path.join(args.outdir, "labeled_ets_ets_scatter.pdf"), title="Cooperative vs independent binding of Ets1-Ets1",
                       labelnames=["cooperative","independent","anticooperative"], showlog=True)
