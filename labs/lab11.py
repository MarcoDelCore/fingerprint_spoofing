from lib import utils, gmm, logistic_regression as lr, svm, score_calibration as sc, bayes_risk as br
import numpy
import matplotlib.pyplot as plt


def lab11_analysis(D, L):
    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
    KFOLD = 5
    pT = 0.1
    effpriorlogodds = numpy.linspace(-4, 4, 100)

    print("******** GMM - Diag Covariance - 8 components *******")

    gmm0 = gmm.train_LBG_EM(DTR[:, LTR == 0], 8, covType="Diagonal", verbose=False, psi=0.01)
    gmm1 = gmm.train_LBG_EM(DTR[:, LTR == 1], 8, covType="Diagonal", verbose=False, psi=0.01)

    SLLR = gmm.logpdf_GMM(DVAL, gmm1)[1] - gmm.logpdf_GMM(DVAL, gmm0)[1]

    print(" K-Fold Cross Validation - GMM - Diag Covariance - 8 components - 5 folds")

    confMat = br.predictions(SLLR, LVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(SLLR, LVAL, pT, 1, 1)
    print(f"Before calibration: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}")

    scores_gmm = []
    labels_gmm = []
    for dIdx in range(KFOLD):
        S_CAL, S_VAL = sc.extract_train_val_folds(SLLR, dIdx, KFOLD)
        L_CAL, L_VAL = sc.extract_train_val_folds(LVAL, dIdx, KFOLD)
        w, b = lr.trainWeightedLogReg(utils.vrow(S_CAL), L_CAL, 0, pT)

        cal_SVAL = (w.T @ utils.vrow(S_VAL) + b - numpy.log(pT / (1 - pT))).ravel()

        scores_gmm.append(cal_SVAL)
        labels_gmm.append(L_VAL)

    scores_gmm = numpy.hstack(scores_gmm)
    labels_gmm = numpy.hstack(labels_gmm)

    confMat = br.predictions(scores_gmm, labels_gmm, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(scores_gmm, labels_gmm, pT, 1, 1)
    print(f"After calibration: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 8 components - calibrated")
    br.compute_Bayes_error(effpriorlogodds, scores_gmm, labels_gmm)
    plt.savefig("output_data/lab11/Bayes_error_plot_GMM_8_calibrated.png")
    plt.show()

    print("******** Quadratic Logistic Regression - lamda = 0.031622 ********")

    D_expanded = numpy.array([lr.expand_features(D[:, i].reshape(6, 1)) for i in range(D.shape[1])]).T
    (DTR_expanded, LTR), (DVAL_expanded, LVAL) = utils.split_db_2to1(D_expanded, L)
    lamda = numpy.logspace(-4, 2, 13)[5]
    w, b = lr.trainLogReg(DTR_expanded, LTR, lamda)
    S = w.T @ DVAL_expanded + b
    p_emp = numpy.mean(LTR)
    S_llr = (S - numpy.log(p_emp / (1 - p_emp))).ravel()

    print(" K-Fold Cross Validation -  Quadratic Logistic Regression - 5 folds")
    scores_lr = []
    labels_lr = []
    for dIdx in range(KFOLD):
        S_CAL, S_VAL = sc.extract_train_val_folds(S_llr, dIdx, KFOLD)
        L_CAL, L_VAL = sc.extract_train_val_folds(LVAL, dIdx, KFOLD)
        w, b = lr.trainWeightedLogReg(utils.vrow(S_CAL), L_CAL, 0, pT)

        cal_SVAL = (w.T @ utils.vrow(S_VAL) + b - numpy.log(pT / (1 - pT))).ravel()

        scores_lr.append(cal_SVAL)
        labels_lr.append(L_VAL)

    scores_lr = numpy.hstack(scores_lr)
    labels_lr = numpy.hstack(labels_lr)

    confMat = br.predictions(S_llr, LVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(S_llr, LVAL, pT, 1, 1)
    print(f"Before calibration: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}")

    confMat = br.predictions(scores_lr, labels_lr, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(scores_lr, labels_lr, pT, 1, 1)
    print(f"After calibration: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error Quadratic Logistic Regression - calibrated")
    br.compute_Bayes_error(effpriorlogodds, scores_lr, labels_lr)
    plt.savefig("output_data/lab11/Bayes_error_plot_LR_calibrated.png")
    plt.show()

    print("******** SVM - RBF Kernel - C = 10 - gamma = 0.367879 ********")

    fScore = svm.trainSVM_kernel(DTR, LTR, 10, kernel_func=svm.rbfKernel(numpy.exp(-1)), eps=1.0)
    S = fScore(DVAL)

    print(" K-Fold Cross Validation -  SVM - 5 folds")
    scores_svm = []
    labels_svm = []
    for dIdx in range(KFOLD):
        S_CAL, S_VAL = sc.extract_train_val_folds(S, dIdx, KFOLD)
        L_CAL, L_VAL = sc.extract_train_val_folds(LVAL, dIdx, KFOLD)
        w, b = lr.trainWeightedLogReg(utils.vrow(S_CAL), L_CAL, 0, pT)

        cal_SVAL = (w.T @ utils.vrow(S_VAL) + b - numpy.log(pT / (1 - pT))).ravel()

        scores_svm.append(cal_SVAL)
        labels_svm.append(L_VAL)

    scores_svm = numpy.hstack(scores_svm)
    labels_svm = numpy.hstack(labels_svm)

    Lpred = (S > 0).astype(int)
    CF, actDCF = br.bayes_risk(pT, 1, 1, br.compute_confusion_matrix(Lpred, LVAL))
    minDCF = br.compute_min_cost(S, LVAL, pT, 1, 1)
    print(f"Before calibration: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}")

    confMat = br.predictions(scores_svm, labels_svm, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(scores_svm, labels_svm, pT, 1, 1)
    print(f"After calibration: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error SVM - RBF Kernel - calibrated")
    br.compute_Bayes_error(effpriorlogodds, scores_svm, labels_svm)
    plt.savefig("output_data/lab11/Bayes_error_plot_SVM_calibrated.png")
    plt.show()

    print("\n ********** System fusion **********\n")

    fused_scores = []
    fused_labels = []

    for dIdx in range(KFOLD):
        S_CAL_gmm, S_VAL_gmm = sc.extract_train_val_folds(SLLR, dIdx, KFOLD)
        S_CAL_lr, S_VAL_lr = sc.extract_train_val_folds(S_llr, dIdx, KFOLD)
        S_CAL_svm, S_VAL_svm = sc.extract_train_val_folds(S, dIdx, KFOLD)
        L_CAL, L_VAL = sc.extract_train_val_folds(LVAL, dIdx, KFOLD)

        S_CAL = numpy.vstack([S_CAL_gmm, S_CAL_lr, S_CAL_svm])

        w, b = lr.trainWeightedLogReg(S_CAL, L_CAL, 0, pT)

        SVAL = numpy.vstack([S_VAL_gmm, S_VAL_lr, S_VAL_svm])

        cal_SVAL = (w.T @ SVAL + b - numpy.log(pT / (1 - pT))).ravel()

        fused_scores.append(cal_SVAL)
        fused_labels.append(L_VAL)

    fused_scores = numpy.hstack(fused_scores)
    fused_labels = numpy.hstack(fused_labels)
    confMat = br.predictions(fused_scores, fused_labels, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(fused_scores, fused_labels, pT, 1, 1)
    print(f"minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error System Fusion - calibrated")
    br.compute_Bayes_error(effpriorlogodds, fused_scores, fused_labels)
    plt.savefig("output_data/lab11/Bayes_error_plot_fusion_calibrated.png")
    plt.show()
