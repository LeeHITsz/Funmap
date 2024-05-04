import numpy as np
import matplotlib.pyplot as plt


def Manhattan_plot(pval=None, PIP=None, sets=None, pos=None):

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = ['r', 'g', 'c', 'm', 'y']

    if pval is not None:

        if pos is not None:

            plt.scatter(pos, -np.log10(pval), c='darkblue', s=12)
            plt.axhline(y=5, color='r', linestyle='--', linewidth=1)

        else:

            plt.scatter(np.arange(pval.shape[0]), -np.log10(pval), c='darkblue', s=12)
            plt.axhline(y=5, color='r', linestyle='--', linewidth=1)

        plt.xlabel('Position on Chromosome', fontsize=14)
        plt.ylabel(r'$-\log_{10}(p)$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        return

    elif PIP is not None:

        if pos is not None:

            plt.scatter(pos, PIP, c='darkblue', s=24)

            if sets is not None:

                for i, group in enumerate(sets.values()):

                    color = colors[i]
                    for idx in group:
                        plt.scatter(pos[idx], PIP[idx], c='darkblue', edgecolors=color, s=24)

        else:

            plt.scatter(np.arange(PIP.shape[0]), PIP, c='darkblue', s=24)

            if sets is not None:

                for i, group in enumerate(sets.values()):

                    color = colors[i]

                    for idx in group:
                        plt.scatter(idx, PIP[idx], c='darkblue', edgecolors=color, s=24)

        plt.xlabel('Position on Chromosome', fontsize=14)
        plt.ylabel('PIP', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        return


