import numpy as np
import matplotlib.pyplot as plt


def Manhattan_plot(PIP=None, pval=None, sets=None, pos=None):
    """
    Generates a Manhattan plot for visualizing genomic data.

    Args:
        PIP (numpy.ndarray): An array of Phenotype Impact Prediction (PIP) scores.
        pval (numpy.ndarray, optional): An array of p-values. If provided, a -log10(p) plot will be generated.
        sets (dict, optional): A dictionary containing sets of indices to be highlighted in different colors.
        pos (numpy.ndarray, optional): An array of position values on the chromosome.

    Returns:
        None

    This function generates a Manhattan plot for visualizing genomic data. If `pval` is provided, it will plot
    the -log10(p) values against the position on the chromosome. If `PIP` is provided, it will plot the PIP scores
    against the position on the chromosome. If `sets` is provided, it will highlight the specified sets of indices
    with different colors. If `pos` is provided, it will use the given position values on the x-axis; otherwise,
    it will use the index values.
    """

    colors = ['g', 'c', 'y', 'pink', 'm', 'purple', 'orange', 'grey', 'darkblue', 'darkcyan',
              'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen',
              'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue',
              'darkslategray', 'darkturquoise', 'darkviolet']

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

