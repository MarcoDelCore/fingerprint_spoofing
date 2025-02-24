import numpy
import lib.utils as utils
import matplotlib.pyplot as plt


def logpdf_GAU_ND(X, mu, C):
    M = C.shape[0]
    logdet = numpy.linalg.slogdet(C)[1]
    X_mu = X - mu
    C_inv = numpy.linalg.inv(C)
    res = -0.5 * (M * numpy.log(2 * numpy.pi) + logdet + numpy.sum((X_mu.T @ C_inv) * X_mu.T, axis=1))
    return res


def plot_pdf(X, m_ML, C_ML, place):
    place.hist(X.ravel(), bins=50, density=True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.xlim(-8, 12)
    place.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(utils.vrow(XPlot), m_ML, C_ML)), 'r')


def plot_pdf_both_classes(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    for dIdx in range(6):
        D0_t = utils.vrow(D0[dIdx, :])
        D1_t = utils.vrow(D1[dIdx, :])
        m_0, C_0 = mu_C_matrices(D0_t)
        m_1, C_1 = mu_C_matrices(D1_t)
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature {dIdx + 1}")
        plt.hist(D0_t.ravel(), bins=50, alpha=0.6, label="Counterfeit", density=True)
        plt.hist(D1_t.ravel(), bins=50, alpha=0.6, label="Genuine", density=True)
        XPlot = numpy.linspace(-8, 12, 1000)
        plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(utils.vrow(XPlot), m_0, C_0)), 'b', label="Counterfeit",
                 linewidth=2)
        plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(utils.vrow(XPlot), m_1, C_1)), 'r', label="Genuine",
                 linewidth=2)
        plt.xlim(-8, 12)
        plt.legend()
        plt.savefig(f"output_data/lab04/hist_feature_{dIdx + 1}.png")
        plt.show()


def loglikelihood(XND, m_ML, C_ML):
    res = numpy.zeros(1)
    for i in range(XND.shape[1]):
        res += logpdf_GAU_ND(utils.vcol(XND[:, i]), m_ML, C_ML)
    return float(res[0])


def loglikelihood_ratios(XND, m0, m1, C0, C1):
    logpdf0 = logpdf_GAU_ND(XND, m0, C0)
    logpdf1 = logpdf_GAU_ND(XND, m1, C1)
    return logpdf1 - logpdf0


def mu_C_matrices(D):
    mu = utils.vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C


def ML_estimates(D, L):
    mu_0 = utils.vcol(numpy.mean(D[:, L == 0], axis=1))
    mu_1 = utils.vcol(numpy.mean(D[:, L == 1], axis=1))

    C_0 = numpy.zeros((D.shape[0], D.shape[0]))
    C_1 = numpy.zeros((D.shape[0], D.shape[0]))
    for i in range(D.shape[1]):
        if L[i] == 0:
            C_0 += (utils.vcol(D[:, i]) - mu_0) @ (utils.vcol(D[:, i]) - mu_0).T
        else:
            C_1 += (utils.vcol(D[:, i]) - mu_1) @ (utils.vcol(D[:, i]) - mu_1).T
    C_0 /= numpy.sum(L == 0)
    C_1 /= numpy.sum(L == 1)

    return mu_0, mu_1, C_0, C_1


def make_predictions(llrs, LVAL):
    predicted_labels = numpy.zeros(LVAL.shape)
    predicted_labels[llrs > 0] = 1
    predicted_labels[llrs <= 0] = 0
    return predicted_labels


def gaussian_predictions(D, L, covTipe="Full"):
    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
    mu_0, mu_1, C_0, C_1 = ML_estimates(DTR, LTR)
    if covTipe == "Full":
        LLRs = loglikelihood_ratios(DVAL, mu_0, mu_1, C_0, C_1)
    elif covTipe == "Tied":
        C_tied = C_0 * numpy.sum(LTR == 1) / DTR.shape[1] + C_1 * numpy.sum(LTR == 2) / DTR.shape[1]
        LLRs = loglikelihood_ratios(DVAL, mu_0, mu_1, C_tied, C_tied)
    elif covTipe == "Diag":
        C_0_diag = numpy.diag(numpy.diag(C_0))
        C_1_diag = numpy.diag(numpy.diag(C_1))
        LLRs = loglikelihood_ratios(DVAL, mu_0, mu_1, C_0_diag, C_1_diag)
    else:
        raise ValueError("Invalid covariance type")
    predicted_labels = numpy.zeros(LVAL.shape)
    predicted_labels[LLRs > 0] = 1
    predicted_labels[LLRs <= 0] = 0
    accuracy = numpy.mean(predicted_labels == LVAL)
    print(f"Accuracy: {accuracy*100:.2f} %")
    print(f"Error rate: {(1-accuracy)*100:.2f} %")
