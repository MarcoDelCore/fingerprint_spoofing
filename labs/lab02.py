from lib import utils, data_visualization as dv


def lab02_analysis(D, L):
    utils.compute_means(D, L)
    utils.compute_variances(D, L)

    dv.plotHist(D, L)
    dv.plotScatter(D, L)

