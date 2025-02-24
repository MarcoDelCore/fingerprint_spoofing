import numpy
import scipy
from lib import utils, gaussian_model as gm


def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    for g in range(len(gmm)):
        S[g, :] = numpy.log(gmm[g][0]) + gm.logpdf_GAU_ND(X, gmm[g][1], gmm[g][2])
    return S, scipy.special.logsumexp(S, axis=0)


def psi_covariance_matrix(C, psi):
    U, s, Vh = numpy.linalg.svd(C)
    s[s < psi] = psi
    CUpd = U @ (utils.vcol(s) * U.T)
    return CUpd


def train_GMM_EM_Iteration(X, gmm, covType='Full', psi=None):
    # E-step
    S, logdens = logpdf_GMM(X, gmm)
    gamma = numpy.exp(S - logdens)

    # M-step
    gmmUpdated = []
    for gIdx in range(len(gmm)):
        g = gamma[gIdx]
        Z = numpy.sum(g)
        F = utils.vcol((utils.vrow(g)*X).sum(1))
        S = (utils.vrow(g) * X) @ X.T
        muUpdated = F / Z
        if covType == 'Full' or covType == 'Tied':
            CUpdated = S / Z - muUpdated @ muUpdated.T
        else:
            CUpdated = numpy.diag(numpy.diag(S / Z - muUpdated @ muUpdated.T))
        wUpdated = Z / X.shape[1]
        gmmUpdated.append((wUpdated, muUpdated, CUpdated))

    if covType == 'Tied':
        CTied = 0
        for w, mu, C in gmmUpdated:
            CTied += w * C
        gmmUpdated = [(w, mu, CTied) for w, mu, C in gmmUpdated]

    if psi is not None:
        gmmUpdated = [(w, mu, psi_covariance_matrix(C, psi)) for w, mu, C in gmmUpdated]

    return gmmUpdated


def train_GMM_EM(X, gmm, covType='Full', psi=None, eps=1e-6, verbose=False):
    llOld = logpdf_GMM(X, gmm)[1].mean()
    llDelta = None

    if verbose:
        print("GMM - it %3d - average ll %.8e" % (0, llOld))

    it = 1
    while llDelta is None or llDelta > eps:
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType, psi)
        llUpd = logpdf_GMM(X, gmmUpd)[1].mean()
        llDelta = llUpd - llOld
        if verbose:
            print('GMM - it %3d - average ll %.8e' % (it, llUpd))
        gmm = gmmUpd
        llOld = llUpd
        it = it + 1

    if verbose:
        print("GMM - it %3d - average ll %.8e (eps = %e)" % (it, llUpd, eps))

    return gmm


def split_GMM(gmm, alpha=0.1, verbose=False):

    gmmNew = []

    if verbose:
        print("Splitting GMM from %d to %d components" % (len(gmm), 2*len(gmm)))

    for w, mu, C in gmm:
        U, s, Vh = numpy.linalg.svd(C)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        gmmNew.append((w/2, mu + d, C))
        gmmNew.append((w/2, mu - d, C))

    return gmmNew


def train_LBG_EM(X, num_components, covType='Full', psi=None, eps=1e-6, alpha=0.1, verbose=False):

    mu, C = gm.mu_C_matrices(X)
    if covType == 'Diagonal':
        C = numpy.diag(numpy.diag(C))

    if psi is not None:
        gmm = [(1.0, mu, psi_covariance_matrix(C, psi))]
    else:
        gmm = [(1, mu, C)]

    while len(gmm) < num_components:
        if verbose:
            print('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm)[1].mean())
        gmm = split_GMM(gmm, alpha)
        if verbose:
            print('Average ll after LBG: %.8e' % logpdf_GMM(X, gmm)[1].mean())
        gmm = train_GMM_EM(X, gmm, covType, psi, eps)

    return gmm
