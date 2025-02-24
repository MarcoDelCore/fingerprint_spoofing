import numpy
import matplotlib.pyplot as plt
from lib import utils, gaussian_model as gm


def compute_pca(D, m):
    mu, C = gm.mu_C_matrices(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P


def apply_pca(P, D):
    return P.T @ D


def HistsPCA_allDirections(D, L, U):
    figure, axis = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(6):
        P = utils.vcol(U[:, i])
        DP = numpy.dot(P.T, D)
        D0 = DP[:, L == 0]
        D1 = DP[:, L == 1]
        axis[i // 3, i % 3].hist(D0.flatten(), bins=10, alpha=0.4, label='Counterfit', density=True)
        axis[i // 3, i % 3].hist(D1.flatten(), bins=10, alpha=0.4, label='Genuine', density=True)
        axis[i // 3, i % 3].set_title("Projected features for direction %d (PCA)" % (i + 1))
        axis[i // 3, i % 3].legend()

    plt.savefig("output_data/lab03/hists_PCA.png")
    plt.show()


def HistPCA_projected(DP, L):
    D0 = DP[:, L == 0]
    D1 = DP[:, L == 1]

    plt.figure()
    plt.title("PCA projection")

    plt.hist(D0[0, :], bins=5, density=True, alpha=0.4, label='Counterfit')
    plt.hist(D1[0, :], bins=5, density=True, alpha=0.4, label='Genuine')

    plt.legend()
    plt.tight_layout()
    plt.savefig('output_data/lab03/histogram_PCA.png')
    plt.show()


def PCA_analysis(D, L, m=2):
    mu = utils.vcol(D.mean(1))
    DC = D - mu
    C = (DC @ DC.T) / float(D.shape[1])
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    DP = numpy.dot(P.T, D)
    HistsPCA_allDirections(D, L, U)
    HistPCA_projected(DP, L)
    return P
