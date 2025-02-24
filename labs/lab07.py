import numpy
from lib import utils, gaussian_model as gm, pca, bayes_risk as br
import matplotlib.pyplot as plt


def evaluate_model(model, DVAL, LVAL, mu_0, mu_1, C_0, C_1, pi, Cp, Cn):
    if model == 'MVG':
        LLRs = gm.loglikelihood_ratios(DVAL, mu_0, mu_1, C_0, C_1)
    elif model == 'Tied':
        C_tied = C_0 * numpy.sum(LVAL == 0) / DVAL.shape[1] + C_1 * numpy.sum(LVAL == 1) / DVAL.shape[1]
        LLRs = gm.loglikelihood_ratios(DVAL, mu_0, mu_1, C_tied, C_tied)
    else:  # Naive Bayes
        C_0_diag = numpy.diag(numpy.diag(C_0))
        C_1_diag = numpy.diag(numpy.diag(C_1))
        LLRs = gm.loglikelihood_ratios(DVAL, mu_0, mu_1, C_0_diag, C_1_diag)

    confMat = br.predictions(LLRs, LVAL, pi, Cp, Cn)
    DCF, actDCF = br.bayes_risk(pi, Cp, Cn, confMat)
    minDCF = br.compute_min_cost(LLRs, LVAL, pi, Cp, Cn)

    return actDCF, minDCF


def pca_and_evaluate_model(model, DTR, LTR, DVAL, LVAL, m, pi, Cp, Cn):
    UPCA = pca.compute_pca(DTR, m)
    DTR_projected = pca.apply_pca(UPCA, DTR)
    DVAL_projected = pca.apply_pca(UPCA, DVAL)
    mu_0_projected, mu_1_projected, C_0_projected, C_1_projected = gm.ML_estimates(DTR_projected, LTR)

    actDCF_pca, minDCF_pca = evaluate_model(model, DVAL_projected, LVAL, mu_0_projected, mu_1_projected, C_0_projected,
                                            C_1_projected, pi, Cp, Cn)

    return actDCF_pca, minDCF_pca


def lab07_analysis(D, L):
    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
    mu_0, mu_1, C_0, C_1 = gm.ML_estimates(DTR, LTR)

    applications = [
        (0.5, 1.0, 1.0),
        (0.9, 1.0, 1.0),
        (0.1, 1.0, 1.0),
    ]
    for i, (pi, Cn, Cp) in enumerate(applications):
        print(f"\nApplication {i + 1}: pi={pi}, Cn={Cn}, Cp={Cp}")
        for model in ['MVG', 'Tied', 'Naive']:
            actDCF, minDCF = evaluate_model(model, DVAL, LVAL, mu_0, mu_1, C_0, C_1, pi, Cp, Cn)
            print(f"actDCF_{model}: {actDCF}")
            print(f"minDCF_{model}: {minDCF}")

        for m in range(1, D.shape[0] + 1):
            print(f"\nApplication {i + 1}: pi={pi}, Cn={Cn}, Cp={Cp}, PCA with m={m}")
            for model in ['MVG', 'Tied', 'Naive']:
                actDCF_pca, minDCF_pca = pca_and_evaluate_model(model, DTR, LTR, DVAL, LVAL, m, pi, Cp, Cn)
                print(f"actDCF_{model}: {actDCF_pca}")
                print(f"minDCF_{model}: {minDCF_pca}")

    plt.figure()
    plt.title("Bayes error plot MVG Model")
    effPriorLogOdds = numpy.linspace(-4, 4, 100)
    LLRs = gm.loglikelihood_ratios(DVAL, mu_0, mu_1, C_0, C_1)
    br.compute_Bayes_error(effPriorLogOdds, LLRs, LVAL)
    plt.savefig("output_data/lab07/MVG_Bayes_error_plot.png")
    plt.show()

    plt.figure()
    plt.title("Bayes error plot Tied Model")
    C_tied = C_0 * numpy.sum(LTR == 0) / DTR.shape[1] + C_1 * numpy.sum(LTR == 1) / DTR.shape[1]
    LLRs_tied = gm.loglikelihood_ratios(DVAL, mu_0, mu_1, C_tied, C_tied)
    br.compute_Bayes_error(effPriorLogOdds, LLRs_tied, LVAL)
    plt.savefig("output_data/lab07/Tied_Bayes_error_plot.png")
    plt.show()

    plt.figure()
    plt.title("Bayes error plot Naive Model")
    C_0_diag = numpy.diag(numpy.diag(C_0))
    C_1_diag = numpy.diag(numpy.diag(C_1))
    LLRs_nb = gm.loglikelihood_ratios(DVAL, mu_0, mu_1, C_0_diag, C_1_diag)
    br.compute_Bayes_error(effPriorLogOdds, LLRs_nb, LVAL)
    plt.savefig("output_data/lab07/Naive_Bayes_Bayes_error_plot.png")
    plt.show()

    LLRs = gm.loglikelihood_ratios(DVAL, mu_0, mu_1, C_0, C_1)
    plt.figure()
    plt.title("ROC Curve MVG Model")
    br.compute_ROC_curve(LLRs, LVAL)
    plt.savefig("output_data/lab07/MVG_ROC_curve.png")
    plt.show()

    C_tied = C_0 * numpy.sum(LTR == 0) / DTR.shape[1] + C_1 * numpy.sum(LTR == 1) / DTR.shape[1]
    LLRs_tied = gm.loglikelihood_ratios(DVAL, mu_0, mu_1, C_tied, C_tied)
    plt.figure()
    plt.title("ROC Curve Tied Model")
    br.compute_ROC_curve(LLRs_tied, LVAL)
    plt.savefig("output_data/lab07/Tied_ROC_curve.png")
    plt.show()

    C_0_diag = numpy.diag(numpy.diag(C_0))
    C_1_diag = numpy.diag(numpy.diag(C_1))
    LLRs_nb = gm.loglikelihood_ratios(DVAL, mu_0, mu_1, C_0_diag, C_1_diag)
    plt.figure()
    plt.title("ROC Curve Naive Model")
    br.compute_ROC_curve(LLRs_nb, LVAL)
    plt.savefig("output_data/lab07/Naive_ROC_curve.png")
    plt.show()
