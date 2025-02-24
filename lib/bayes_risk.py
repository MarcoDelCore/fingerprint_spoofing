import numpy
import matplotlib.pyplot as plt


def effective_prior(pi, Cfn, Cfp):
    return pi * Cfp / (pi * Cfp + (1 - pi) * Cfn)


def compute_confusion_matrix(predicted_labels, true_labels):
    TP = numpy.sum((predicted_labels == 1) & (true_labels == 1))
    FN = numpy.sum((predicted_labels == 0) & (true_labels == 1))
    FP = numpy.sum((predicted_labels == 1) & (true_labels == 0))
    TN = numpy.sum((predicted_labels == 0) & (true_labels == 0))
    return numpy.array([[TP, FN], [FP, TN]])


def predictions(llr_ratios, labels, pi, Cp, Cn):
    threshold = -numpy.log(pi * Cn / (1 - pi) / Cp)
    predicted_labels = numpy.zeros(labels.shape, dtype=numpy.int32)
    predicted_labels[llr_ratios > threshold] = 1
    res = compute_confusion_matrix(predicted_labels, labels)
    return res


def bayes_risk(pi, Cp, Cn, confusion_matrix):
    TP = confusion_matrix[0, 0]
    FN = confusion_matrix[0, 1]
    FP = confusion_matrix[1, 0]
    TN = confusion_matrix[1, 1]

    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    FNR = FN / (TP + FN) if (TP + FN) != 0 else 0

    DCF = pi * Cn * FNR + (1 - pi) * Cp * FPR
    B_dummy = min(pi * Cn, (1 - pi) * Cp)
    actDCF = DCF / B_dummy

    return DCF, actDCF


def compute_min_cost(llr_ratios, labels, pi, Cp, Cn):
    sorted_scores = numpy.sort(llr_ratios)
    minDCF = float('inf')

    for threshold in sorted_scores:
        predicted_labels = numpy.zeros(labels.shape, dtype=numpy.int32)
        predicted_labels[llr_ratios > threshold] = 1
        confusion_mat = compute_confusion_matrix(predicted_labels, labels)
        DCF, actDCF = bayes_risk(pi, Cp, Cn, confusion_mat)

        if actDCF < minDCF:
            minDCF = actDCF

    return minDCF


def compute_ROC_curve(llr_ratios, labels):
    sorted_scores = numpy.sort(llr_ratios)
    TPR = []
    FPR = []

    for threshold in sorted_scores:
        predicted_labels = numpy.zeros(labels.shape, dtype=numpy.int32)
        predicted_labels[llr_ratios > threshold] = 1
        confusion_mat = compute_confusion_matrix(predicted_labels, labels)
        TPR.append(confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])
                   if (confusion_mat[0, 0] + confusion_mat[0, 1]) != 0 else 0)
        FPR.append(confusion_mat[1, 0] / (confusion_mat[1, 0] + confusion_mat[1, 1])
                   if (confusion_mat[1, 0] + confusion_mat[1, 1]) != 0 else 0)

    plt.plot(FPR, TPR, color='b', linewidth=2)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    return TPR, FPR


def compute_Bayes_error(effPriorLogOdds, llr_ratios, labels, calScores=None, calLabels=None):
    effPrior = 1 / (1 + numpy.exp(-effPriorLogOdds))
    actDCFs = []
    minDCFs = []
    calActMinDCF = []
    for pi in effPrior:
        app = predictions(llr_ratios, labels, pi, 1, 1)
        actDCFs.append(bayes_risk(pi, 1, 1, app)[1])
        minDCFs.append(compute_min_cost(llr_ratios, labels, pi, 1, 1))
        if calScores is not None:
            calActMinDCF.append(compute_min_cost(calScores, calLabels, pi, 1, 1))

    if calScores is not None:
        plt.plot(effPriorLogOdds, calActMinDCF, label="calibrated min DCF", color='g', linewidth=2)
        plt.plot(effPriorLogOdds, actDCFs, label="DCF", color='r', linewidth=2, linestyle='--')
        plt.plot(effPriorLogOdds, minDCFs, label="min DCF", color='b', linewidth=2, linestyle='--')
    else:
        plt.plot(effPriorLogOdds, actDCFs, label="DCF", color='r', linewidth=2)
        plt.plot(effPriorLogOdds, minDCFs, label="min DCF", color='b', linewidth=2)
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.ylabel("DCF value")
    plt.xlabel("prior log-odds")
    plt.tick_params(axis='both', which='both', direction='in', length=3, width=1, right='on', top='on')
    plt.legend()
