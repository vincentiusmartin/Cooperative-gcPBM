'''
This file contains SitesPlotter class used to generate a pdf of plots
Authors: Vincentius Martin, Farica Zhuang
Created on Jul 19, 2019
'''

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class SitesPlotter(object):
    """SitesPlotter class plots escores and cores of each sequence"""

    def align_sequences(self, sequences):
    # TODO: check subset here
        maxseq = max(sequences, key=len)
        minseq = min(sequences, key=len)

        if not minseq in maxseq:
            raise Exception('sequences between objects are not subset of each other')

        pos = maxseq.index(minseq)
        return {"indices":[*range(-pos,len(maxseq)-pos)], "maxseq":maxseq, "minseq":minseq}

    def align_sequences(self, sequences):
    # TODO: check subset here
        maxseq = max(sequences, key=len)
        minseq = min(sequences, key=len)

        if not minseq in maxseq:
            raise Exception('sequences between objects are not subset of each other')

        pos = maxseq.index(minseq)
        return {"indices":[*range(-pos,len(maxseq)-pos)], "maxseq":maxseq, "minseq":minseq}

    def plot_seq_combine(self, plotlist, filepath = "plot.pdf" ,
                         numcol=4, numrow=4, bottom_cutoff=0, top_cutoff=1):
        plt.close('all')
        n = 0
        plot_count = 0
        if type(plotlist) != list:
            raise Exception('Input must be a list of dictionaries. Got {}'.format(type(plotlist)))
        for plot in plotlist:
            if type(plot) != dict:
                raise Exception('Input must be a dictionary of predictions for each protein. Got {}'.format(type(plot)))
                for protein in plot:
                    if type(plot[protein]) != list:
                        raise Exception('Values must be lists of model predictions. Got {}'.format(type(plot[protein])))

        if not plotlist:
            return 0

        # extract the model predictions
        # plotlist = [list(d.values()) for d in plotlist]
        # plotlist = sum(plotlist,[])

        print("Making plot to %s" % filepath)
        fig = plt.figure(figsize=(18,18))
        with PdfPages(filepath) as pdf:
            # we can just take the first object since each object has the same key
            lenall = len(plotlist[0])
            iter = 1
            for key in plotlist[0]:
                #print("Processing %d/%d" % (iter,lenall))
                iter += 1
                if n == 0:
                    fig = plt.figure(figsize=(18,18))
                    fig.subplots_adjust(hspace=0.4,wspace=0.5)

                # set skip_seq to false if we want to plot this sequence
                skip_seq = False
                # # check if we are plotting this sequence
                # for siteobj in plotlist: # for each imads object , escore, etc
                #     # if there is no imads rectangle, then skip this sequence
                #     if siteobj[key] == None or len(siteobj[key]['plt'])==0:
                #         # set skip_seq to true to skip this sequence
                #         skip_seq = True
                #         break
                ### ==========================================
                if not skip_seq:
                    n += 1
                    plot_count += 1
                    ax = fig.add_subplot(numrow,numcol,n) # how many plots are in a col,row
                    # ======= plot each element in plotdict =======
                    for siteobj in plotlist:
                        cmds = siteobj[key]["plt"] # get the commands for this key
                        for cmd in cmds:
                            getattr(ax,cmd["func"])(*cmd["args"],**cmd["kwargs"])
                    ax.axhline(y=0,color='gray')
                    ax.set_ylim(bottom=bottom_cutoff, top=top_cutoff) # this is based on the pwm, ,

                    # custom the first x axis, we put negative axis if one sequence is longer
                    # that other
                    sequences = [plotlist[i][key]["sequence"] for i in range(len(plotlist))]
                    align = self.align_sequences(sequences)
                    ax.set_xticks(align["indices"])
                    ax.set_xticklabels(align["maxseq"],fontdict={'fontsize':5})
                    seqlen = len(align["minseq"])
                    for xtick,index in zip(ax.get_xticklabels(),align["indices"]):
                        #xtick.set_color(color))
                        if index not in range(0,seqlen):
                            xtick.set_color("indianred")

                    ax.xaxis.set_label_text('sequence')

                    # Make the second axis:
                    ax2 = ax.twiny()
                    ax2.set_xlim(ax.get_xlim())

                    # ================ FINALIZE ================
                    #ax.yaxis.set_label_text('PWM log score')
                    ax.tick_params(direction="in")
                    ax2.tick_params(direction="in")

                    ax.set_title("%s"%key,pad=20)
                    if n == numcol*numrow:
                        pdf.savefig(fig)
                        plt.close()
                        n = 0
            pdf.savefig(fig)
            plt.close()
        return 0
