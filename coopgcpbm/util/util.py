import numpy as np
import matplotlib.pyplot as plt
import itertools

from matplotlib.backends.backend_pdf import PdfPages

def merge_listdict(ld1, ld2):
    if len(ld1) > 0 and len(ld2) > 0 and len(ld1) != len(ld2):
        print("Error:list length is not the same")
        return 0
    if len(ld1) == 0:
        return ld2
    elif len(ld2) == 0:
        return ld1
    else:
        ld_ret = []
        for i in range(0,len(ld1)):
            ld_ret.append({**ld1[i], **ld2[i]})
        return ld_ret

# =========

def dictlist2file(inputdict,filepath,listval=False):
    with open(filepath,'w') as f:
        for key in inputdict:
            f.write(">%s\n"%key)
            if listval:
                f.write(",".join(str(x) for x in inputdict[key]) + "\n")
            else:
                f.write(inputdict[key] + "\n") # f.write(inputdict[key] + "\n")

def read_dictlist_file(filepath, as_int=False):
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except IOError:
        print("Error: Unable to open file: " + filepath)
        exit(0)

    categories = {}
    i = 0
    while i < len(lines):
        # get the key
        cur = lines[i].strip()
        # need to make sure this line and next line is not empty
        i += 1
        if cur and cur[0] == '>':
            key = cur[1:]
            curlist = []
            while i < len(lines) and lines[i].strip() and lines[i][0] != '>':
                for x in lines[i].strip().split(","):
                    if as_int:
                        append = int(x) if x != 'NA' else np.NaN
                    else:
                        append = float(x) if x != 'NA' else np.NaN
                    curlist.append(append)
                i += 1
            categories[key] = curlist
    return categories

def print_dictlist_count(dictlist):
    for key in dictlist:
        print("Count of %s: %d" % (key,len(dictlist[key])))

def multiple_scatter_boxplots(listofdict,filepath="scatterbox.pdf",ylabel=""):
    n = 0
    nrow = 3
    ncol = 1
    with PdfPages(filepath) as pdf:
        for labeldict in listofdict:
            if n == 0:
                fig = plt.figure(figsize=(12,12))
                fig.subplots_adjust(hspace=0.4,wspace=0.5)
            n += 1

            ax = fig.add_subplot(nrow,ncol,n)
            keys = labeldict.keys()
            listrep = [labeldict[key] for key in keys]
            pos = np.linspace(1,1+len(listrep)*0.5-0.5,len(listrep))

            bp = ax.boxplot(listrep, positions=pos, widths=0.4)
            ax.set_xticks(pos)
            ax.set_xticklabels(keys,rotation=15, horizontalalignment='right')
            plt.setp(bp['boxes'], color='black')
            plt.setp(bp['caps'], color='black')

            for i in range(0,len(listrep)):
                y = listrep[i]
                x = np.random.normal(1+i*0.5, 0.02, size=len(y))
                ax.plot(x, y, 'r.', alpha=0.4,c='red')

            ax.set_ylabel(ylabel)
            if n == ncol * nrow:
                pdf.savefig(fig)
                plt.close()
                n = 0
        pdf.savefig(fig)
        plt.close()
        #plt.clf() # clear canvas


def scatter_boxplot_dict(groupdict, filepath="scatterbox.png",ylabel="",title=""):
    keys = groupdict.keys()
    listrep = [groupdict[key] for key in keys]
    pos = np.linspace(1,1+len(listrep)*0.5-0.5,len(listrep))

    bp = plt.boxplot(listrep,positions=pos,widths=0.4)
    plt.xticks(pos,keys,rotation=0) #horizontalalignment='right'
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['caps'], color='black')
    plt.title(title)

    for i in range(0,len(listrep)):
        y = listrep[i]
        x = np.random.normal(1+i*0.5, 0.02, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.4,c='red')

    plt.ylabel(ylabel)

    #print("Save distribution of row %s to %s" % (rownum,plotfilename))
    plt.tight_layout()
    plt.savefig(filepath,positions=[0, 1])
    plt.clf() # clear canvas
