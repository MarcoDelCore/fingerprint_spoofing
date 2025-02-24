import numpy

from lib import utils, svm, bayes_risk as br
import numpy as np
import matplotlib.pyplot as plt


def plotDCFvsC(C, actDCFs, minDCFs):
    plt.plot(C, actDCFs, label="actDCF", linestyle="-", marker="o", color='r', linewidth=2)
    plt.plot(C, minDCFs, label="minDCF", linestyle="-", marker='o', color='b', linewidth=2)
    plt.xscale("log", base=10)
    plt.xlabel("C")
    plt.ylabel("DCF")
    plt.legend()
    plt.grid()


def lab09_analysis(D, L):
    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
    C = np.logspace(-5, 0, 11)
    actDCFs = []
    minDCFs = []
    pi_T = 0.1
    Cp = 1
    Cn = 1
    print("\n******** Linear kernel ********\n")
    for c in C:
        print(f"c = {c}")
        w_star, alpha, duality_gap, primal_objective = svm.trainSVM(DTR, LTR, c, 1)
        S = np.dot(w_star.T, svm.create_Dcap(DVAL, 1))
        Lpred = (S > 0).astype(int)
        error_rate = (1 - np.mean(LVAL == Lpred)) * 100
        print(f"Error rate: {error_rate:.1f}%")
        DCF, actDCF = br.bayes_risk(pi_T, Cp, Cn, br.compute_confusion_matrix(Lpred, LVAL))
        minDCF = br.compute_min_cost(S, LVAL, pi_T, Cp, Cn)
        print(f"micDCF: {minDCF:.4f}")
        print(f"actDCF: {actDCF:.4f}\n")
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)

    plt.figure(figsize=(10, 6))
    plt.title("DCF vs C")
    plotDCFvsC(C, actDCFs, minDCFs)
    plt.savefig("output_data/lab09/DCF_vs_C.png")
    plt.show()

    print("\n******** Linear kernel - centered data ********\n")

    mu_TR = numpy.mean(DTR, axis=1, keepdims=True)
    DTR_centered = DTR - mu_TR
    DVAL_centered = DVAL - mu_TR
    actDCFs = []
    minDCFs = []
    for c in C:
        print(f"c = {c}")
        w_star, alpha, duality_gap, primal_objective = svm.trainSVM(DTR_centered, LTR, c, 1)
        S = np.dot(w_star.T, svm.create_Dcap(DVAL_centered, 1))
        Lpred = (S > 0).astype(int)
        error_rate = (1 - np.mean(LVAL == Lpred)) * 100
        print(f"Error rate: {error_rate:.1f}%")
        DCF, actDCF = br.bayes_risk(pi_T, Cp, Cn, br.compute_confusion_matrix(Lpred, LVAL))
        minDCF = br.compute_min_cost(S, LVAL, pi_T, Cp, Cn)
        print(f"micDCF: {minDCF:.4f}")
        print(f"actDCF: {actDCF:.4f}\n")
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)

    plt.figure(figsize=(10, 6))
    plt.title("DCF vs C - centered data")
    plotDCFvsC(C, actDCFs, minDCFs)
    plt.savefig("output_data/lab09/DCF_vs_C_centered.png")
    plt.show()

    print("\n******** Polynomial kernel ********\n")
    actDCFs = []
    minDCFs = []
    for c in C:
        print(f"c = {c}")
        fScore = svm.trainSVM_kernel(DTR, LTR, c, kernel_func=svm.polyKernel(2, 1), eps=0)
        S = fScore(DVAL)
        Lpred = (S > 0) * 1
        error_rate = (1 - np.mean(LVAL == Lpred)) * 100
        print(f"Error rate: {error_rate:.1f}%")
        DCF, actDCF = br.bayes_risk(pi_T, Cp, Cn, br.compute_confusion_matrix(Lpred, LVAL))
        minDCF = br.compute_min_cost(S, LVAL, pi_T, Cp, Cn)
        print(f"micDCF: {minDCF:.4f}")
        print(f"actDCF: {actDCF:.4f}\n")
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)

    plt.figure(figsize=(10, 6))
    plt.title("DCF vs C - kernel polynomial")
    plotDCFvsC(C, actDCFs, minDCFs)
    plt.savefig("output_data/lab09/DCF_vs_C_kernel.png")
    plt.show()

    print("\n******** RBF kernel ********\n")

    C = np.logspace(-3, 2, 11)
    for gamma in [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]:
        actDCFs = []
        minDCFs = []
        exponent = int(np.log(gamma))
        for c in C:
            print(f"c = {c}, gamma = {gamma}")
            fScore = svm.trainSVM_kernel(DTR, LTR, c, kernel_func=svm.rbfKernel(gamma), eps=1.0)
            S = fScore(DVAL)
            Lpred = (S > 0).astype(int)
            error_rate = (1 - np.mean(LVAL == Lpred)) * 100
            print(f"Error rate: {error_rate:.1f}%")
            DCF, actDCF = br.bayes_risk(pi_T, Cp, Cn, br.compute_confusion_matrix(Lpred, LVAL))
            minDCF = br.compute_min_cost(S, LVAL, pi_T, Cp, Cn)
            print(f"micDCF: {minDCF:.4f}")
            print(f"actDCF: {actDCF:.4f}\n")
            actDCFs.append(actDCF)
            minDCFs.append(minDCF)

        plt.figure(figsize=(10, 6))
        plt.title("DCF vs C - kernel RBF with $ \\gamma  = e^{" + str(exponent) + "}$")
        plotDCFvsC(C, actDCFs, minDCFs)
        plt.savefig("output_data/lab09/DCF_vs_C_kernel_rbf_%d.png" % -exponent)
        plt.show()

    print("\n******** Polynomial kernel - d = 4 ********\n")
    C = np.logspace(-5, 0, 11)
    actDCFs = []
    minDCFs = []
    for c in C:
        print(f"c = {c}")
        fScore = svm.trainSVM_kernel(DTR, LTR, c, kernel_func=svm.polyKernel(4, 1), eps=0)
        S = fScore(DVAL)
        Lpred = (S > 0).astype(int)
        error_rate = (1 - np.mean(LVAL == Lpred)) * 100
        print(f"Error rate: {error_rate:.1f}%")
        DCF, actDCF = br.bayes_risk(pi_T, Cp, Cn, br.compute_confusion_matrix(Lpred, LVAL))
        minDCF = br.compute_min_cost(S, LVAL, pi_T, Cp, Cn)
        print(f"micDCF: {minDCF:.4f}")
        print(f"actDCF: {actDCF:.4f}\n")
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)

    plt.figure(figsize=(10, 6))
    plt.title("DCF vs C - kernel polynomial d = 4")
    plotDCFvsC(C, actDCFs, minDCFs)
    plt.savefig("output_data/lab09/DCF_vs_C_kernel_d_4.png")
    plt.show()
