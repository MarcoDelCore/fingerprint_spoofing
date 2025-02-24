import numpy as np
import scipy.optimize as opt
from lib import utils


def create_Dcap(D, k):
    return np.vstack([D, np.ones((1, D.shape[1])) * k])


def compute_Hcap(Dcap, L):
    Z = 2 * L - 1
    return np.dot(Dcap.T, Dcap) * utils.vcol(Z) * utils.vrow(Z)


def trainSVM(DTR, LTR, C, k):
    bounds = [(0, C) for _ in range(DTR.shape[1])]
    z = np.ones(LTR.shape)
    z[LTR == 0] = -1
    DTRcap = create_Dcap(DTR, k)
    Hcap = compute_Hcap(DTRcap, LTR)

    def obj_func(a):
        J = 0.5 * np.dot(a.T, np.dot(Hcap, a)) - np.dot(a.T, np.ones(a.shape))
        grad = np.dot(Hcap, a) - np.ones(a.shape)
        return J, grad

    alpha, _, _ = opt.fmin_l_bfgs_b(obj_func, np.zeros(DTR.shape[1]), bounds=bounds, factr=1.0)
    print(f"Primal loss: {(-obj_func(alpha)[0]):.6e}")

    w_star = np.dot(alpha * z, DTRcap.T)

    def primal_obj(w):
        return 0.5 * np.linalg.norm(w) ** 2 + C * np.sum(np.maximum(0, 1 - z * np.dot(w, DTRcap)))

    duality_gap = primal_obj(w_star) + (obj_func(alpha)[0])

    return w_star, alpha, duality_gap, primal_obj(w_star)


def polyKernel(degree, c):
    def polyKernelFunc(D1, D2):
        return (np.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc


def rbfKernel(gamma):
    def rbfKernelFunc(D1, D2):
        D1Norms = (D1 ** 2).sum(0)
        D2Norms = (D2 ** 2).sum(0)
        Z = utils.vcol(D1Norms) + utils.vrow(D2Norms) - 2 * utils.numpy.dot(D1.T, D2)
        return np.exp(-gamma * Z)

    return rbfKernelFunc


def trainSVM_kernel(DTR, LTR, C, kernel_func, eps=1.0):
    bounds = [(0, C) for _ in LTR]
    ZTR = LTR * 2.0 - 1.0

    k = kernel_func(DTR, DTR) + eps

    H = utils.vcol(ZTR) * utils.vrow(ZTR) * k

    def obj_func(a):
        Ha = H @ utils.vcol(a)
        loss = 0.5 * (utils.vrow(a) @ Ha).ravel() - a.sum()
        grad = Ha.ravel() - utils.numpy.ones(a.size)
        return loss, grad

    alpha, _, _ = opt.fmin_l_bfgs_b(obj_func, np.zeros(DTR.shape[1]), bounds=bounds, factr=1.0)

    print('SVM (kernel) - C %e - dual loss %e' % (C, -obj_func(alpha)[0]))

    def fScore(DTE):
        K = kernel_func(DTR, DTE) + eps
        H = utils.vcol(alpha) * utils.vcol(ZTR) * K
        return H.sum(0)

    return fScore
