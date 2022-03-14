import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import logomaker
import argparse
import os

import coopgcpbm.modeler.dnashape as ds
from util import bio

def align(seq, move):
    """
    move, if negative, move to the left (append right)
    """
    mv = abs(move)
    if move > 0:
        return bio.gen_randseq(mv) + seq[:-mv]
    else:
        return seq[mv:] + bio.gen_randseq(mv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Shape feature analysis')
    parser.add_argument(action="store", dest="path", type=str, help='Path to training data')
    parser.add_argument('-p', '--poscols', action="store", dest="poscols", type=str,  help='Column name for the binding site positions, separated by comma')
    parser.add_argument('-o', '--outdir', action="store", dest="outdir", type=str, default="shape_out", help='Output directory')
    args = parser.parse_args()

    params = {'axes.labelsize': 22,
          'axes.titlesize': 18,
          "xtick.labelsize" : 15, "ytick.labelsize" : 15 , "axes.labelsize" : 14}
    plt.rcParams.update(params)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)


    df = pd.read_csv(args.path, sep="\t")
    poscol1, poscol2 = args.poscols.split(",")

    df["s1pos"] = df.apply(lambda x: x[poscol1] if x[poscol1] < x[poscol2] else x[poscol2], axis=1)
    df["s2pos"] = df.apply(lambda x: x[poscol1] if x[poscol1] > x[poscol2] else x[poscol2], axis=1)


    dist = sorted(df["distance"].unique().tolist())[1:]
    oris = df["orientation"].unique().tolist()

    fastadict = dict(zip(df["Name"], df["Sequence"]))
    ds = ds.DNAShape(fastadict)
    labels = ["independent", "cooperative"]

    labels = df["label"].unique()
    with PdfPages("%s/motifs.pdf" % args.outdir) as pdf:
        for d in dist:
            curdf_dist = df[df['distance'] == d]
            fig = plt.figure(figsize=(12,12))
            n = 0
            for ori in oris:
                curdf_ori = curdf_dist[curdf_dist['orientation'] == ori]
                for label in labels:
                    if curdf_ori.shape[0] <= 1:
                        continue
                    curdf = curdf_ori[curdf_ori["label"] == label]
                    if curdf.shape[0] <= 1:
                        continue
                    n = n + 1
                    ax = fig.add_subplot(7,2,n)
                    s1 = int(curdf['s1pos'].mode().iloc[0])
                    s2 = curdf[curdf['s1pos'] == s1].iloc[0]['s2pos']
                    seqalign = curdf.apply(lambda x: align(x['Sequence'], s1-x['s1pos']), axis=1).tolist()
                    m = logomaker.alignment_to_matrix(seqalign, to_type="weight")
                    m[m < 0] = 0
                    # min_m = m.min(axis=1)
                    # m = m.sub(min_m, axis=0)
                    mlogo = logomaker.Logo(m,shade_below=.5, fade_below=.5, ax=ax)
                    ax.set_title("%s, distance %d, orientation %s" % (label, d,ori))
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close()

    with PdfPages("%s/shape_plots.pdf" % args.outdir) as pdf:
        for d in dist:
            curdf_dist = pd.DataFrame(df[df['distance'] == d])
            for ori in oris: # oris
                curdf = pd.DataFrame(curdf_dist[curdf_dist['orientation'] == ori])
                if curdf.shape[0] <= 1:
                    continue
                s1 = int(curdf['s1pos'].mode().iloc[0])
                s2 = curdf[curdf['s1pos'] == s1].iloc[0]['s2pos']
                curdf["seqalign"] = curdf.apply(lambda x: align(x['Sequence'], s1-x['s1pos']), axis=1)
                fig = plt.figure(figsize=(12,12))
                ds.plot_average(curdf, linemarks=[s1, s2], base_count=False, in_fig=fig, lblnames=["cooperative","independent"], pltlabel="dist %d, ori %s"%(d,ori), label_title=True,pthres=0.05)
                pdf.savefig(fig)
                plt.close()
                oristr = ori.replace("/","")
                for l in labels:
                    curdflbl = curdf[curdf["label"] == l]
                    with open("%s/seqs_dist%d_%s_%s.txt"%(args.outdir,d,oristr,l),'w') as f:
                        f.write("\n".join(curdflbl["seqalign"].tolist()))
