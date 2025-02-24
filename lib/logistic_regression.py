import numpy as np
import scipy.optimize as opt
from lib import utils


def trainLogReg(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0

    def logreg_obj_with_grad(v):
        w = v[:-1]
        b = v[-1]
        s = np.dot(utils.vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        GW = (utils.vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()

        J = loss.mean() + l / 2 * np.linalg.norm(w) ** 2
        grad = np.hstack([GW, np.array(Gb)])
        return J, grad

    xt = opt.fmin_l_bfgs_b(logreg_obj_with_grad, x0=np.zeros(DTR.shape[0]+1))[0]
    return xt[:-1], xt[-1]


def trainWeightedLogReg(DTR, LTR, l, pT):
    ZTR = LTR * 2.0 - 1.0

    wTrue = pT / (ZTR > 0).sum()
    wFalse = (1 - pT) / (ZTR < 0).sum()

    def logreg_obj_with_grad(v):
        w = v[:-1]
        b = v[-1]
        s = np.dot(utils.vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)
        loss[ZTR > 0] *= wTrue
        loss[ZTR < 0] *= wFalse

        J = loss.sum() + l / 2 * np.linalg.norm(w) ** 2

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        G[ZTR > 0] *= wTrue
        G[ZTR < 0] *= wFalse

        GW = (utils.vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        grad = np.hstack([GW, np.array(Gb)])

        return J, grad

    xt = opt.fmin_l_bfgs_b(logreg_obj_with_grad, x0=np.zeros(DTR.shape[0] + 1))[0]
    return xt[:-1], xt[-1]


def expand_features(X):
    return np.concatenate(((X @ X.T).reshape(-1, 1).flatten(), X.flatten()))
