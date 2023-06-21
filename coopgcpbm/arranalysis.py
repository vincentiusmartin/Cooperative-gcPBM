import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import itertools
import statsmodels.stats.multitest as sm
import os

import coopgcpbm.util.stats_r as st

def read_chamber_file(path, includekey, excludekey=None, keycol="Name", seqcols=["Name","type","ori","rep"], negcols=["Name","ori","rep"],
                     negkey="NegativeCtrl", include_id=False):
    """
    the ori, rep order might be different
    """
    cols = ["Name", "Sequence", "Alexa488Adjusted"] if not include_id else ["ID_REF","ID", "Name", "Sequence", "Alexa488Adjusted"]

    df = pd.read_csv(path, sep="\t").rename(columns={"Column":"ID_REF"})
    df["Sequence"] = df["Sequence"].str[:36]

    # get negctrl
    negdf = df[df["Name"].str.contains(negkey, na=False)][cols]
    negdf[negcols] =  negdf["Name"].str.rsplit("_", n = 2, expand = True)
    negdf["type"] = "negctrl"

    df = df[~df["Name"].str.contains("NegativeCtrl", na=False) & df[keycol].str.contains(includekey, na=False)][cols]
    if excludekey != None:
        df = df[~df['Name'].str.contains(excludekey, na=False)]

    df[seqcols] = df["Name"].str.rsplit("_", n = 3, expand = True)
    return df.sort_values(["Name","ori","type","rep"]), negdf

def assign_fdrcor_class(p, prevlbl, pcut=0.05):
    if prevlbl == "fail_cutoff":
        return 'fail_cutoff'
    elif prevlbl == 'anticooperative' and p < pcut:
        return 'anticooperative'
    elif prevlbl == 'cooperative' and p < pcut:
        return 'cooperative'
    else:
        return 'independent'

def create_cooplbl(indivsum, twosites, pcutoff = 0.05):
    """
    """
    p_coop = st.wilcox(twosites, indivsum, "greater")
    p_anti = st.wilcox(twosites, indivsum, "less")
    if p_coop < pcutoff:
        return "cooperative", p_coop
    elif p_anti < pcutoff:
        return "anticooperative", p_anti
    else:
        return 'independent', p_coop

def label_replicas_permutation(indiv, two, arrdf, cutoff=0, oricol="ori", namecol="Name", affcol="affinity", typecol="type", pcut = 0.05, fdrcor=True):
    median_dict = arrdf.groupby([namecol, oricol, typecol])[affcol].median().to_dict()
    labeled_dict = {}
    for ori in list(indiv.keys()):
        orilbls = []
        for k in indiv[ori]:
            rowdict = {}
            if median_dict[(k,ori,'wt')] < cutoff or median_dict[(k,ori,'m3')] > cutoff:
                rowdict['label'], rowdict['p'] = "fail_cutoff", 1
            else:
                if not (len(indiv[ori][k]) == 27 and len(two[ori][k]) == 3):
                    continue
                rowdict['label'], rowdict['p'] = create_cooplbl(indiv[ori][k], two[ori][k], pcut)
            rowdict['indiv_median'] = np.median(indiv[ori][k])
            rowdict['two_median'] = np.median(two[ori][k])
            rowdict['Name'] = k
            orilbls.append(rowdict)
        labeled_dict[ori] = pd.DataFrame(orilbls)
        labeled_dict[ori].to_csv("%s.csv"%ori,index=False)
        if fdrcor:
            newdf = pd.DataFrame()
            newdf["Name"] = labeled_dict[ori]['Name']
            newdf["p_old"] = labeled_dict[ori]['p']
            labeled_dict[ori]['p'] = sm.fdrcorrection(labeled_dict[ori]['p'])[1]
            newdf["p_new"] = labeled_dict[ori]['p']
            newdf.to_csv("p_%s.csv"%ori,index=False)
            labeled_dict[ori]['label'] = labeled_dict[ori].apply(lambda row: assign_fdrcor_class(row['p'],row['label'],pcut),axis=1)
    return labeled_dict

def make_replicas_permutation(indf, oricol="ori", namecol="Name", affcol="affinity", typecol="type"):
    """
    Make permutation across replicas

    Args:
        type: wt,m1,m2,m3
    """
    oris = indf[oricol].unique()
    df = indf[[namecol, affcol, oricol, typecol, "Sequence"]].sort_values([namecol, oricol])
    wt = df[df[typecol] == "wt"][[namecol, affcol, oricol]]
    m1 = df[df[typecol] == "m1"][[namecol, affcol, oricol]]
    m2 = df[df[typecol] == "m2"][[namecol, affcol, oricol]]
    m3 = df[df[typecol] == "m3"][[namecol, affcol, oricol]]
    twodict = {}
    onedict = {}
    for o in oris:
        wt_o = wt[wt[oricol] == o]
        m1_o = m1[m1[oricol] == o]
        m2_o = m2[m2[oricol] == o]
        m3_o = m3[m3[oricol] == o]
        twodict[o] = wt_o.groupby(namecol)[affcol].apply(list).to_dict()
        one = m1_o.merge(m2_o, on=namecol).merge(m3_o, on=namecol)
        one['indiv_aff'] = one["%s_x"%affcol] + one["%s_y"%affcol] - one[affcol]
        onedict[o] = one.groupby(namecol)["indiv_aff"].apply(list).to_dict()
    return onedict, twodict

def permutdict2df(permutdict):
    """
    convert indivsum/twosites to a data frame
    """
    ld = [{"Name":k, "ori": ori, "intensity":aff} for ori in permutdict for k in permutdict[ori] for aff in permutdict[ori][k]]
    return pd.DataFrame(ld)

# --------

def plot_chamber_corr(dfx, dfy, xlab="Chamber 1", ylab="Chamber 2",
                namecol="Name", valcol="Alexa488Adjusted", extrajoincols=[],
                title="", cutoff="", ax_in=False,
                median=False, log=False, showlog=False,
                shownames=False, path=""):
    """
    Get correlation between 2 chambers

    Args:
        dfx: data frame from chamber 1
        dfy: data frame from chamber 2
        namecol: column name for the probe names
        valcol: column name for the array intensity
        extrajoincols: column name for the extras such as orientation
        title: plot title
        median: get median between replicates
        log: return result in log scale
        shownames: show names of the probe on the plot, useful for debugging
        path: if empty then show the plot
    Return:
        R^2
    """
    # if showlog and log:
    #     raise ValueError("showlog and log could not be both True")
    # correlation plot of negative controls in chamber 1 vs chamber 2

    joincols = [namecol] + extrajoincols

    if median:
        dfx = dfx.groupby([namecol] + extrajoincols)[[valcol]].median().reset_index()
        dfy = dfy.groupby([namecol] + extrajoincols)[[valcol]].median().reset_index()

    dfcombined = dfx.merge(dfy, on=joincols)[joincols + ["%s_x"%valcol, "%s_y"%valcol]]

    if log:
        dfcombined["%s_x"%valcol] = np.log(dfcombined["%s_x"%valcol])
        dfcombined["%s_y"%valcol] = np.log(dfcombined["%s_y"%valcol])

    x = dfcombined["%s_x"%valcol].values
    y = dfcombined["%s_y"%valcol].values
    # x = [i/10000 for i in x]
    # y= [i/10000 for i in y]

    ax = ax if ax_in else plt.axes()
    # plot diagonal
    ax.plot([min(min(x),min(y)),max(max(x),max(y))], [min(min(x),min(y)),max(max(x),max(y))], color='black', label='diagonal') # , linewidth=0.5

    if cutoff:
        c = np.log(cutoff) if log else cutoff
        ax.axhline(cutoff, color='black', linewidth=0.5, linestyle=":")
        ax.axvline(cutoff, color='black', linewidth=0.5, linestyle=":")

    slope, intercept = np.polyfit(x, y, 1)
    abline_values = [slope * i + intercept for i in x]

    ax.plot(x, abline_values, color='red', linestyle='dashed', label='best fit')

    if showlog:
        ax.loglog(base=np.e)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())

        # locs, labels = plt.xticks()
        # labels = [np.exp(item) for item in locs]
        # print(labels)
        # plt.xticks(locs, labels)ax.set_xticks([250,1000,3000,10000,25000,60000, 150000])
    #ax.plot([min(x), max(x)], [min(abline_values),max(abline_values)], color='black', label='best fit')
    r_squared = np.corrcoef(x,y)[0,1]**2
    ax.scatter(x, y, s=1) # color="#FFA07A"  color="#DCDCDC"

    if shownames:
        names = dfcombined[namecol].values
        for i in range(len(names)):
            ax.text(x[i], y[i], names[i], fontsize=3)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_title(title)
    ax.text(0.02, 0.94, 'R² = %0.2f' % r_squared, color='red', fontsize=12, transform = ax.transAxes)
    sign = '+' if intercept > 0 else '-'
    prefix = "linear " if showlog else ""
    ax.text(0.02, 0.9, '%sbest fit: y=%0.2fx %s %0.2f' % (prefix,slope,sign,abs(intercept)), color='red', transform = ax.transAxes, fontsize=12)
    ax.legend(loc='lower right', prop={'size': 11})

    if not ax_in:
        if path:
            plt.savefig(path)
        else:
            plt.show()
        plt.clf()
    return r_squared

def plot_classified_labels(df, path="", col1="Alexa488Adjusted_x", col2="Alexa488Adjusted_y",
                           xlab="Chamber1", ylab="Chamber2", log=False, showlog=False , title="", axes=None,
                           shownames=False, namecol="Name", labelcol="label", plotnonsignif=True,
                           labelnames=["cooperative","independent","anticooperative"], colors=["#b22222","#FFA07A"],
                           showlegend=True):
    """
    Desc

    Args:
        df: with "Name", "Intensity_one", "Intensity_two", "label".
            Label -> cooperative, independent, anticooperative, fail_cutoff
            shownames: show names of the probe on the plot, useful for debugging. Names are obtained from namecol.
        showlog: show the plot in log

    Return:

    """
    plt.clf()
    if log and showlog:
        raise ValueError("log and showlog can't be both true")

    permitted_labels = list(labelnames)
    if plotnonsignif:
        permitted_labels.append("fail_cutoff")
    newdf = df[df[labelcol].isin(permitted_labels)]

    if axes:
        ax = axes
    else:
        ax = plt.axes()

    # red/firebrick, mistyrose, blue, skyblue  blue ["#0343df","#75bbfd"] red ["#b22222","#FFA07A"]
    lblclr = [("fail_cutoff","yellow", 0.7), (labelnames[0],colors[0],1), (labelnames[1],colors[1],1), (labelnames[2],"gray",1)]
    if log:
        newdf.loc[:,col1] = np.log(newdf[col1].tolist())
        newdf.loc[:,col2] = np.log(newdf[col2].tolist())

    for lc in lblclr:
        if not newdf[newdf[labelcol]==lc[0]].empty:
            x = newdf[newdf[labelcol]==lc[0]][col1].values
            y = newdf[newdf[labelcol]==lc[0]][col2].values
            label = lc[0] if lc[0] != "fail_cutoff" else None
            ax.scatter(x, y, color=lc[1], s=3, alpha=lc[2], label=label)
            if shownames:
                names = newdf[newdf['label']==lc[0]][namecol].values
                for i in range(len(names)):
                    ax.text(x[i], y[i], names[i], fontsize=5)


    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    x,y = newdf[col1].values, newdf[col2].values
    ax.plot([min(min(x),min(y)),max(max(x),max(y))], [min(min(x),min(y)),max(max(x),max(y))], color='black')
    ax.set_title(title, fontsize=15)

    if showlegend:
        ax.legend(loc="lower right", markerscale=5, fontsize=10)

    if showlog:
        ax.set_xlim(left=0.8*min(x),right=1.2*max(x))
        ax.loglog(base=np.e)
        ax.set_xticks([250,1000,3000,10000,25000,60000, 150000])
        ax.set_yticks([250,1000,3000,10000,25000,60000, 150000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if not axes:
        if path == "":
            plt.show()
        else:
            plt.savefig(path)
    #     plt.clf()

def scatter_boxplot(input_df, cols=[], log=False,
                    ax=None, path = "box.png", repcol="rep",
                    title=""):
    """
    Scatter boxplot of intensity among replicates

    Args:
        input_df: df input
        path: where to save the plot
        cols: columns with intensity
    Return:

    """
    df = input_df[cols] if cols else input_df.copy()

    if not ax:
        df.boxplot(grid=False)
        p = plt
    else:
        p = df.boxplot(grid=False, ax=ax)

    for i in range(len(cols)):
        y = df[cols[i]]
        x = np.random.normal(i+1, 0.04, size=len(y))
        p.plot(x, y, 'r.', alpha=0.2)

    if ax:
        p.set_title(title)
    else:
        p.title(title)
        p.savefig(path)
        p.clf() # clear canvas

def scatter_boxplot_col(input_df, coltype="type", colval="affinity", plotorder=[], colororder=[], ax=None, path = "", title=""):
    """
    """
    plotlist = []
    for po in plotorder:
        plotlist.append(input_df[input_df[coltype] == po][colval].tolist())

    idxs = list(range(1,len(plotorder)+1))
    if ax:
        ax.boxplot(plotlist)
        ax.set_xticklabels(plotorder)
        ax.set_title(title)
    else:
        plt.ylim(6,11) ###
        plt.boxplot(plotlist)
        plt.xticks(idxs,plotorder)
        plt.title(title)
    ax = plt.axes()
    # ax.set_yscale("log")
    # ax.set_yticks([1000,3000,10000,25000,60000])
    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    p = plt if not ax else ax
    for i in range(len(plotlist)):
        y = plotlist[i]
        x = np.random.normal(i+1, 0.04, size=len(y))
        curc = colororder[i] if len(colororder) > i else None
        # red/firebrick, mistyrose, blue, skyblue  blue ["#0343df","#75bbfd"] red ["#b22222","#FFA07A"]
        p.plot(x, y, 'r.', markersize=5, alpha=1, c="#b22222")

    if path != "":
        plt.savefig(path)
        plt.clf() # clear canvas

def plot_multi_scatterbox(path, indf, po, co=[], namecol="Name", oricol="ori", affcol="affinity", allplot=True,
                          numcol=4, numrow=4, typecol="type", thresline=False, fixed_ax=False, pline=True):
    """
    po: plot order based on the typecol
    co: color order
    allplot: plot the summary of all as the first plot
    """
    fig, ax = plt.subplots(numrow, numcol, figsize=(25, 5))
    plt.subplots_adjust(hspace = 0.4, wspace=0.2)
    i = 0
    with PdfPages(path) as pdf:
        if allplot:
            fig = plt.figure(figsize=(25,14))
            fig.subplots_adjust(hspace=0.4,wspace=0.2)
            i += 1 # start from 1 for the all plot
            cur_ax = fig.add_subplot(numcol,numrow,i)
            df_median = indf.groupby([namecol,typecol,oricol]).median().reset_index()
            main_title = os.path.splitext(path)[0]
            scatter_boxplot_col(df_median, typecol, affcol, plotorder=po, colororder=co, title=main_title, ax=cur_ax)
        namelist = indf[namecol].drop_duplicates().tolist()
        len_namelist = len(namelist)
        for n_idx in range(len_namelist):
            print("Progress %d/%d" % (n_idx, len_namelist))
            n = namelist[n_idx]
            if i == 0:
                fig = plt.figure(figsize=(25,14))
                fig.subplots_adjust(hspace=0.4,wspace=0.2)
            i += 1
            cur_ax = fig.add_subplot(numcol,numrow,i)
            indf_col = indf[indf[namecol] == n]
            scatter_boxplot_col(indf_col, typecol, affcol, plotorder=po, colororder=co, title=n, ax=cur_ax)
            # statistical annotation
            if pline:
                for pos in [1, 3]:
                    p = st.wilcox(indf_col[indf_col[typecol] == po[pos-1]][affcol].tolist(),
                                  indf_col[indf_col[typecol] == po[pos]][affcol].tolist(), 'less')
                    x1, x2 = pos, pos+1
                    y, h, col = max(indf_col[affcol]) + 0.2, 0.2, 'k'
                    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], c=col)
                    plt.text((x1+x2)*.5, y+h, "%.4f" % p, ha='center', va='bottom', color=col)
            if thresline:
                cur_ax.axhline(thresline, linestyle='--', color="orange", lw=0.5)
            if fixed_ax:
                min_y = min(indf[affcol].tolist())
                max_y = max(indf[affcol].tolist())
                cur_ax.set_ylim([min_y, max_y])
            if i == numcol*numrow:
                pdf.savefig(fig)
                plt.close()
                i = 0
        pdf.savefig(fig)
    plt.clf()
    plt.close()

def plot_ori_inconsistency(indivsum_df, twosites_df, lbldf=False, namecol="Name", oricol='ori',
                affcol="affinity",  allplot=True, log=False, fixed_ax=False,
                prefix_path = "", thresline=False):
    """
    Args:
        indivsum_df: data frame of Name, orientation, affinity of the individual sites sum
        twosites_df: data frame of Name, orientation, affinity of the two sites
        lbldf: data frame of Name, label in first orientation, label in second orientation
        cutoffline: float, plot cutoff line if specified
    """
    o1, o2 = sorted(indivsum_df[oricol].unique()) # we assume we can only have 2 ori here
    indivsum, twosites = indivsum_df.copy(), twosites_df.copy()
    if log:
        indivsum[affcol] = np.log(indivsum[affcol])
        twosites[affcol] = np.log(twosites[affcol])
    indivsum["type"] = "one_" + indivsum[oricol]
    twosites["type"] = "two_" + twosites[oricol]

    numcol = 4
    numrow = 4
    if lbldf:
        label_names = set(lbldf[o1]) | set(lbldf[o2])
        perms = [(x,y) for x in label_names for y in label_names]
    else:
        perms = [["NA","NA"]]

    for perm in perms:
        if lbldf:
            cur_lbls = lbldf[(lbldf[o1] == perm[0]) & (lbldf[o2] == perm[1])]
            if cur_lbls.empty:
                continue
            cur_indiv = indivsum.merge(cur_lbls[[namecol]], on=namecol)
            cur_two = twosites.merge(cur_lbls[[namecol]], on=namecol)
        else:
            cur_indiv = indivsum
            cur_two = twosites
        curdf = pd.concat([cur_indiv, cur_two])
        #print(perm, curdf[namecol].unique().shape[0])
        if len(curdf[affcol].tolist()) == 0:
            continue

        all_lbl = "%s%s_%s_%s_%s" % (prefix_path,perm[0],o1,perm[1],o2)
        po = ["one_%s"%o1,"two_%s"%o1,"one_%s"%o2,"two_%s"%o2]
        co = ["red","red","blue","blue"]
        plot_multi_scatterbox("%s.pdf" % all_lbl, curdf, po, co, namecol=namecol, oricol=oricol,
                               allplot=allplot, affcol=affcol, thresline=thresline, fixed_ax=fixed_ax)
    return 0
