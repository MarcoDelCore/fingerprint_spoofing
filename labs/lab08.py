import numpy
from lib import utils, logistic_regression as lr, bayes_risk as br, pca
import matplotlib.pyplot as plt


def plotDCFvsLambda(lamdas, actDCFs, minDCFs):
    plt.plot(lamdas, actDCFs, label='actDCF', marker='o', linestyle='-', color='r', linewidth=2)
    plt.plot(lamdas, minDCFs, label='minDCF', marker='o', linestyle='-', color='b', linewidth=2)
    plt.xscale('log', base=10)
    plt.xlabel('lambda')
    plt.legend()
    plt.grid()


def lab08_analysis(D, L):
    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
    lamdas = numpy.logspace(-4, 2, 13)
    pi_T = 0.1
    p_emp = numpy.mean(LTR)
    actDCFs = []
    minDCFs = []
    print("\n******** All samples ********\n")
    for l in lamdas:
        print(f"\nLambda: {l}")
        w, b = lr.trainLogReg(DTR, LTR, l)
        S = w.T @ DVAL + b
        S_llr = (S - numpy.log(p_emp/(1-p_emp))).ravel()

        confMat = br.predictions(S_llr, LVAL, pi_T, 1, 1)
        DCF, actDCF = br.bayes_risk(pi_T, 1, 1, confMat)
        minDCF = br.compute_min_cost(S_llr, LVAL, pi_T, 1, 1)
        print(f"actDCF: {actDCF:.4f}")
        print(f"minDCF: {minDCF:.4f}")
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)

    plt.figure(figsize=(10, 6))
    plt.title("DCF vs lambda - All samples")
    plotDCFvsLambda(lamdas, actDCFs, minDCFs)
    plt.savefig("output_data/lab08/DCF_vs_lambda_all.png")
    plt.show()

    actDCFs = []
    minDCFs = []
    print("\n******** 1/50 samples ********\n")
    for l in lamdas:
        print(f"\nLambda: {l}")
        w, b = lr.trainLogReg(DTR[:, ::50], LTR[::50], l)
        S = w.T @ DVAL + b
        S_llr = (S - numpy.log(p_emp / (1 - p_emp))).ravel()

        confMat = br.predictions(S_llr, LVAL, pi_T, 1, 1)
        DCF, actDCF = br.bayes_risk(pi_T, 1, 1, confMat)
        minDCF = br.compute_min_cost(S_llr, LVAL, pi_T, 1, 1)
        print(f"actDCF: {actDCF:.4f}")
        print(f"minDCF: {minDCF:.4f}")
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)

    plt.figure(figsize=(10, 6))
    plt.title("DCF vs lambda - 1/50 samples")
    plotDCFvsLambda(lamdas, actDCFs, minDCFs)
    plt.savefig("output_data/lab08/DCF_vs_lambda_50.png")
    plt.show()

    print("\n******** prior-weighted ********\n")
    actDCFs = []
    minDCFs = []
    for l in lamdas:
        print(f"\nLambda: {l}")
        w, b = lr.trainWeightedLogReg(DTR, LTR, l, pi_T)
        S = w.T @ DVAL + b
        S_llr = (S - numpy.log(pi_T / (1 - pi_T))).ravel()

        confMat = br.predictions(S_llr, LVAL, pi_T, 1, 1)
        DCF, actDCF = br.bayes_risk(pi_T, 1, 1, confMat)
        minDCF = br.compute_min_cost(S_llr, LVAL, pi_T, 1, 1)
        print(f"actDCF: {actDCF:.4f}")
        print(f"minDCF: {minDCF:.4f}")
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)

    plt.figure(figsize=(10, 6))
    plt.title("DCF vs lambda - prior-weighted")
    plotDCFvsLambda(lamdas, actDCFs, minDCFs)
    plt.savefig("output_data/lab08/DCF_vs_lambda_prior_weighted.png")
    plt.show()

    print("\n******** quadratic logistic regression ********\n")

    D_expanded = numpy.array([lr.expand_features(D[:, i].reshape(6, 1)) for i in range(D.shape[1])]).T

    (DTR_expanded, LTR), (DVAL_expanded, LVAL) = utils.split_db_2to1(D_expanded, L)
    actDCFs = []
    minDCFs = []
    for l in lamdas:
        print(f"\nLambda: {l}")
        w, b = lr.trainLogReg(DTR_expanded, LTR, l)
        S = w.T @ DVAL_expanded + b

        S_llr = (S - numpy.log(p_emp / (1 - p_emp))).ravel()

        confMat = br.predictions(S_llr, LVAL, pi_T, 1, 1)
        DCF, actDCF = br.bayes_risk(pi_T, 1, 1, confMat)
        minDCF = br.compute_min_cost(S_llr, LVAL, pi_T, 1, 1)
        print(f"actDCF: {actDCF:.4f}")
        print(f"minDCF: {minDCF:.4f}")
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)

    plt.figure(figsize=(10, 6))
    plt.title("DCF vs lambda - Expanded features")
    plotDCFvsLambda(lamdas, actDCFs, minDCFs)
    plt.savefig("output_data/lab08/DCF_vs_lambda_expanded.png")
    plt.show()

    print("\n******** Centered data ********\n")
    mu_TR = numpy.mean(DTR, axis=1, keepdims=True)
    DTR_centered = DTR - mu_TR
    DVAL_centered = DVAL - mu_TR
    actDCFs = []
    minDCFs = []
    for l in lamdas:
        print(f"\nLambda: {l}")
        w, b = lr.trainLogReg(DTR_centered, LTR, l)
        S = w.T @ DVAL_centered + b
        S_llr = (S - numpy.log(p_emp / (1 - p_emp))).ravel()

        confMat = br.predictions(S_llr, LVAL, pi_T, 1, 1)
        DCF, actDCF = br.bayes_risk(pi_T, 1, 1, confMat)
        minDCF = br.compute_min_cost(S_llr, LVAL, pi_T, 1, 1)
        print(f"actDCF: {actDCF:.4f}")
        print(f"minDCF: {minDCF:.4f}")
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)
    plt.figure(figsize=(10, 6))
    plt.title("DCF vs lambda - Centered")
    plotDCFvsLambda(lamdas, actDCFs, minDCFs)
    plt.savefig("output_data/lab08/DCF_vs_lambda_centered.png")
    plt.show()

    print("\n******** Center and normalize - Z-norm ********\n")
    std_TR = numpy.std(DTR_centered, axis=1, keepdims=True)
    DTR_normalized = DTR_centered / std_TR
    DVAL_normalized = DVAL_centered / std_TR
    actDCFs = []
    minDCFs = []
    for l in lamdas:
        print(f"\nLambda: {l}")
        w, b = lr.trainLogReg(DTR_normalized, LTR, l)
        S = w.T @ DVAL_normalized + b
        S_llr = (S - numpy.log(p_emp / (1 - p_emp))).ravel()

        confMat = br.predictions(S_llr, LVAL, pi_T, 1, 1)
        DCF, actDCF = br.bayes_risk(pi_T, 1, 1, confMat)
        minDCF = br.compute_min_cost(S_llr, LVAL, pi_T, 1, 1)
        print(f"actDCF: {actDCF:.4f}")
        print(f"minDCF: {minDCF:.4f}")
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)
    plt.figure(figsize=(10, 6))
    plt.title("DCF vs lambda - Z-norm")
    plotDCFvsLambda(lamdas, actDCFs, minDCFs)
    plt.savefig("output_data/lab08/DCF_vs_lambda_znorm.png")
    plt.show()

    print("\n******** Center and normalize - PCA ********\n")
    P = pca.compute_pca(DTR_centered, 2)
    DTR_pca = pca.apply_pca(P, DTR_centered)
    DVAL_pca = pca.apply_pca(P, DVAL_centered)
    actDCFs = []
    minDCFs = []
    for l in lamdas:
        print(f"\nLambda: {l}")
        w, b = lr.trainLogReg(DTR_pca, LTR, l)
        S = w.T @ DVAL_pca + b
        S_llr = (S - numpy.log(p_emp / (1 - p_emp))).ravel()

        confMat = br.predictions(S_llr, LVAL, pi_T, 1, 1)
        DCF, actDCF = br.bayes_risk(pi_T, 1, 1, confMat)
        minDCF = br.compute_min_cost(S_llr, LVAL, pi_T, 1, 1)
        print(f"actDCF: {actDCF:.4f}")
        print(f"minDCF: {minDCF:.4f}")
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)
    plt.figure(figsize=(10, 6))
    plt.title("DCF vs lambda - PCA")
    plotDCFvsLambda(lamdas, actDCFs, minDCFs)
    plt.savefig("output_data/lab08/DCF_vs_lambda_pca.png")
    plt.show()
