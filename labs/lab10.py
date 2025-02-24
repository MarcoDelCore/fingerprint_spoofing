from lib import utils, gmm, bayes_risk as br, svm, logistic_regression as lr
import numpy
import matplotlib.pyplot as plt


def plotDCFvsComponents(C, actDCFs, minDCFs):
    plt.plot(C, actDCFs, label="actDCF", linestyle="-", marker="o", color="r", linewidth=2)
    plt.plot(C, minDCFs, label="minDCF", linestyle="-", marker="o", color="b", linewidth=2)
    plt.xlabel("Number of components")
    plt.ylabel("DCF")
    plt.grid()
    plt.legend()


def lab10_analysis(D, L):
    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)

    for covType in ["Full", "Diagonal"]:
        actDCFs = []
        minDCFs = []
        for numC in [1, 2, 4, 8, 16, 32]:
            gmm0 = gmm.train_LBG_EM(DTR[:, LTR == 0], numC, covType=covType, verbose=False, psi=0.01)
            gmm1 = gmm.train_LBG_EM(DTR[:, LTR == 1], numC, covType=covType, verbose=False, psi=0.01)

            SLLR = gmm.logpdf_GMM(DVAL, gmm1)[1] - gmm.logpdf_GMM(DVAL, gmm0)[1]

            confMat = br.predictions(SLLR, LVAL, 0.1, 1, 1)
            DCF, actDCF = br.bayes_risk(0.1, 1, 1, confMat)
            minDCF = br.compute_min_cost(SLLR, LVAL, 0.1, 1, 1)
            print(f"Cov Type: {covType} - {numC} components")
            print(f"actDCF: {actDCF:.4f}")
            print(f"minDCF: {minDCF:.4f}\n")
            actDCFs.append(actDCF)
            minDCFs.append(minDCF)

        plt.figure(figsize=(10, 6))
        plt.title("DCF vs number of components - covType: " + covType)
        plotDCFvsComponents([1, 2, 4, 8, 16, 32], actDCFs, minDCFs)
        plt.savefig(f"output_data/lab10/DCF_vs_C_{covType}.png")
        plt.show()

    effPriorLogOdds = numpy.linspace(-4, 4, 100)
    gmm0 = gmm.train_LBG_EM(DTR[:, LTR == 0], 8, covType="Diagonal", verbose=False, psi=0.01)
    gmm1 = gmm.train_LBG_EM(DTR[:, LTR == 1], 8, covType="Diagonal", verbose=False, psi=0.01)
    SLLR = gmm.logpdf_GMM(DVAL, gmm1)[1] - gmm.logpdf_GMM(DVAL, gmm0)[1]
    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 8 components")
    br.compute_Bayes_error(effPriorLogOdds, SLLR, LVAL)
    plt.savefig("output_data/lab10/Bayes_error_plot_GMM_8.png")
    plt.show()

    fScore = svm.trainSVM_kernel(DTR, LTR, 10, kernel_func=svm.rbfKernel(numpy.exp(-1)), eps=1.0)
    S = fScore(DVAL)
    plt.figure(figsize=(10, 6))
    plt.title("Bayes error SVM - RBF Kernel - C = 10 - $\\gamma = e^{-1}$")
    br.compute_Bayes_error(effPriorLogOdds, S, LVAL)
    plt.savefig("output_data/lab10/Bayes_error_plot_SVM.png")
    plt.show()

    D_expanded = numpy.array([lr.expand_features(D[:, i].reshape(6, 1)) for i in range(D.shape[1])]).T
    (DTR_expanded, LTR), (DVAL_expanded, LVAL) = utils.split_db_2to1(D_expanded, L)
    w, b = lr.trainLogReg(DTR_expanded, LTR, 0.0316)
    S = w.T @ DVAL_expanded + b
    p_emp = numpy.mean(LTR)
    S_llr = (S - numpy.log(p_emp / (1 - p_emp))).ravel()
    plt.figure(figsize=(10, 6))
    plt.title("Bayes error Logistic Regression - $\\lambda = 0.0316$")
    br.compute_Bayes_error(effPriorLogOdds, S_llr, LVAL)
    plt.savefig("output_data/lab10/Bayes_error_plot_LR.png")
    plt.show()