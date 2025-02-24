from lib import utils, gmm, bayes_risk as br, logistic_regression as lr, svm
import numpy
import matplotlib.pyplot as plt

def evaluation_analysis(D, L, DEVAL, LEVAL):
    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
    pT = 0.1

    print("******** GMM - 8 components - Diagonal covariance ********")
    gmm0 = gmm.train_LBG_EM(DTR[:, LTR == 0], 8, covType="Diagonal", verbose=False, psi=0.01)
    gmm1 = gmm.train_LBG_EM(DTR[:, LTR == 1], 8, covType="Diagonal", verbose=False, psi=0.01)
    SLLR = gmm.logpdf_GMM(DVAL, gmm1)[1] - gmm.logpdf_GMM(DVAL, gmm0)[1]

    SLLR_eval = gmm.logpdf_GMM(DEVAL, gmm1)[1] - gmm.logpdf_GMM(DEVAL, gmm0)[1]

    w, b = lr.trainWeightedLogReg(utils.vrow(SLLR), LVAL, 0, pT)

    cal_eval = (w.T @ utils.vrow(SLLR_eval) + b - numpy.log(pT / (1 - pT))).ravel()

    confMat = br.predictions(SLLR_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(SLLR_eval, LEVAL, pT, 1, 1)
    print(f"Not calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}")

    confMat = br.predictions(cal_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(cal_eval, LEVAL, pT, 1, 1)
    print(f"Calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 8 components - Diagonal Covariance - evaluation")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), SLLR_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_GMM_8_diag_evaluation.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 8 components - Diagonal Covariance - evaluation calibrated")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), cal_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_GMM_8_diag_evaluation_cal.png")
    plt.show()

    print("******** Quadratic Logistic Regression - lamda = 0.031622 ********")

    D_expanded = numpy.array([lr.expand_features(D[:, i].reshape(6, 1)) for i in range(D.shape[1])]).T
    (DTR_expanded, LTR), (DVAL_expanded, LVAL) = utils.split_db_2to1(D_expanded, L)
    D_EVAL_expanded = numpy.array([lr.expand_features(DEVAL[:, i].reshape(6, 1)) for i in range(DEVAL.shape[1])]).T
    lamda = numpy.logspace(-4, 2, 13)[5]
    w, b = lr.trainLogReg(DTR_expanded, LTR, lamda)
    S = w.T @ DVAL_expanded + b
    p_emp = numpy.mean(LTR)
    S_llr = (S - numpy.log(p_emp / (1 - p_emp))).ravel()

    S_eval = w.T @ D_EVAL_expanded + b
    S_llr_eval = (S_eval - numpy.log(p_emp / (1 - p_emp))).ravel()

    w, b = lr.trainWeightedLogReg(utils.vrow(S_llr), LVAL, 0, pT)

    cal_eval = (w.T @ utils.vrow(S_llr_eval) + b - numpy.log(pT / (1 - pT))).ravel()

    confMat = br.predictions(S_llr_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(S_llr_eval, LEVAL, pT, 1, 1)
    print(f"Not calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}")

    confMat = br.predictions(cal_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(cal_eval, LEVAL, pT, 1, 1)
    print(f"Calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("Quadratic Logistic Regression - lamda = 0.031622")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), S_llr_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_lr_evaluation.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("Quadratic Logistic Regression - lamda = 0.031622 - calibrated")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), cal_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_lr_evaluation_cal.png")
    plt.show()

    print("******** SVM - RBF Kernel - C = 10 - gamma = e^-1 ********")
    fScore = svm.trainSVM_kernel(DTR, LTR, 10, kernel_func=svm.rbfKernel(numpy.exp(-1)), eps=1.0)
    S = fScore(DVAL)

    S_eval = fScore(DEVAL)
    Lpred_eval = (S_eval > 0).astype(int)

    w, b = lr.trainWeightedLogReg(utils.vrow(S), LVAL, 0, pT)
    cal_eval = (w.T @ utils.vrow(S_eval) + b - numpy.log(pT / (1 - pT))).ravel()
    cal_Lpred_eval = (cal_eval > 0).astype(int)

    confMat = br.compute_confusion_matrix(Lpred_eval, LEVAL)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(S_eval, LEVAL, pT, 1, 1)
    print(f"Not calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}")

    confMat = br.compute_confusion_matrix(cal_Lpred_eval, LEVAL)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(cal_eval, LEVAL, pT, 1, 1)
    print(f"Calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("SVM - RBF Kernel - C = 10 - $\\gamma = e^{-1}$")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), S_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_svm_evaluation.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("SVM - RBF Kernel - C = 10 - $\\gamma = e^{-1}$ - calibrated")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), cal_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_svm_evaluation_cal.png")
    plt.show()

    print("******** Fusion Model ********")

    SMatrix = numpy.vstack([SLLR, S_llr, S])
    w, b = lr.trainWeightedLogReg(SMatrix, LVAL, 0, pT)

    SMatrix_eval = numpy.vstack([SLLR_eval, S_llr_eval, S_eval])

    fused_eval_scores = (w.T @ SMatrix_eval + b - numpy.log(pT / (1 - pT))).ravel()
    confMat = br.predictions(fused_eval_scores, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(fused_eval_scores, LEVAL, pT, 1, 1)
    print(f"Calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("Fusion Model")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), fused_eval_scores, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_fusion_evaluation.png")
    plt.show()

    print("******** GMM - 8 components - Full ********")
    gmm0 = gmm.train_LBG_EM(DTR[:, LTR == 0], 8, covType="Full", verbose=False, psi=0.01)
    gmm1 = gmm.train_LBG_EM(DTR[:, LTR == 1], 8, covType="Full", verbose=False, psi=0.01)
    SLLR = gmm.logpdf_GMM(DVAL, gmm1)[1] - gmm.logpdf_GMM(DVAL, gmm0)[1]

    SLLR_eval = gmm.logpdf_GMM(DEVAL, gmm1)[1] - gmm.logpdf_GMM(DEVAL, gmm0)[1]

    w, b = lr.trainWeightedLogReg(utils.vrow(SLLR), LVAL, 0, pT)

    cal_eval = (w.T @ utils.vrow(SLLR_eval) + b - numpy.log(pT / (1 - pT))).ravel()

    confMat = br.predictions(SLLR_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(SLLR_eval, LEVAL, pT, 1, 1)
    print(f"Not calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}")

    confMat = br.predictions(cal_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(cal_eval, LEVAL, pT, 1, 1)
    print(f"Calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 8 components - Full Covariance - evaluation")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), SLLR_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_GMM_8_full_evaluation.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 8 components - Full Covariance - evaluation calibrated")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), cal_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_GMM_8_full_evaluation_cal.png")
    plt.show()

    print("******** GMM - 4 components ********")
    gmm0 = gmm.train_LBG_EM(DTR[:, LTR == 0], 4, covType="Diagonal", verbose=False, psi=0.01)
    gmm1 = gmm.train_LBG_EM(DTR[:, LTR == 1], 4, covType="Diagonal", verbose=False, psi=0.01)
    SLLR = gmm.logpdf_GMM(DVAL, gmm1)[1] - gmm.logpdf_GMM(DVAL, gmm0)[1]

    SLLR_eval = gmm.logpdf_GMM(DEVAL, gmm1)[1] - gmm.logpdf_GMM(DEVAL, gmm0)[1]

    w, b = lr.trainWeightedLogReg(utils.vrow(SLLR), LVAL, 0, pT)

    cal_eval = (w.T @ utils.vrow(SLLR_eval) + b - numpy.log(pT / (1 - pT))).ravel()

    confMat = br.predictions(SLLR_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(SLLR_eval, LEVAL, pT, 1, 1)
    print(f"Not calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}")

    confMat = br.predictions(cal_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(cal_eval, LEVAL, pT, 1, 1)
    print(f"Calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 4 components - Diagonal Covariance - evaluation")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), SLLR_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_GMM_4_diag_evaluation.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 4 components - Diagonal Covariance - evaluation calibrated")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), cal_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_GMM_4_diag_evaluation_cal.png")
    plt.show()

    print("******** GMM - 16 components - Full Covariance ********")
    gmm0 = gmm.train_LBG_EM(DTR[:, LTR == 0], 16, covType="Full", verbose=False, psi=0.01)
    gmm1 = gmm.train_LBG_EM(DTR[:, LTR == 1], 16, covType="Full", verbose=False, psi=0.01)
    SLLR = gmm.logpdf_GMM(DVAL, gmm1)[1] - gmm.logpdf_GMM(DVAL, gmm0)[1]

    SLLR_eval = gmm.logpdf_GMM(DEVAL, gmm1)[1] - gmm.logpdf_GMM(DEVAL, gmm0)[1]

    w, b = lr.trainWeightedLogReg(utils.vrow(SLLR), LVAL, 0, pT)

    cal_eval = (w.T @ utils.vrow(SLLR_eval) + b - numpy.log(pT / (1 - pT))).ravel()

    confMat = br.predictions(SLLR_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(SLLR_eval, LEVAL, pT, 1, 1)
    print(f"Not calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}")

    confMat = br.predictions(cal_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(cal_eval, LEVAL, pT, 1, 1)
    print(f"Calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 16 components - Full Covariance - evaluation")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), SLLR_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_GMM_16_full_evaluation.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 16 components - Full Covariance - evaluation calibrated")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), cal_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_GMM_16_full_evaluation_cal.png")
    plt.show()

    print("******** GMM - 16 components - Diagonal Covariance ********")
    gmm0 = gmm.train_LBG_EM(DTR[:, LTR == 0], 16, covType="Diagonal", verbose=False, psi=0.01)
    gmm1 = gmm.train_LBG_EM(DTR[:, LTR == 1], 16, covType="Diagonal", verbose=False, psi=0.01)
    SLLR = gmm.logpdf_GMM(DVAL, gmm1)[1] - gmm.logpdf_GMM(DVAL, gmm0)[1]

    SLLR_eval = gmm.logpdf_GMM(DEVAL, gmm1)[1] - gmm.logpdf_GMM(DEVAL, gmm0)[1]

    w, b = lr.trainWeightedLogReg(utils.vrow(SLLR), LVAL, 0, pT)

    cal_eval = (w.T @ utils.vrow(SLLR_eval) + b - numpy.log(pT / (1 - pT))).ravel()

    confMat = br.predictions(SLLR_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(SLLR_eval, LEVAL, pT, 1, 1)
    print(f"Not calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}")

    confMat = br.predictions(cal_eval, LEVAL, pT, 1, 1)
    DCF, actDCF = br.bayes_risk(pT, 1, 1, confMat)
    minDCF = br.compute_min_cost(cal_eval, LEVAL, pT, 1, 1)
    print(f"Calibrated: minDCF: {minDCF:.3f} - actDCF: {actDCF:.3f}\n")

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 16 components - Diagonal Covariance - evaluation")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), SLLR_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_GMM_16_diag_evaluation.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("Bayes error GMM - 16 components - Diagonal Covariance - evaluation calibrated")
    br.compute_Bayes_error(numpy.linspace(-4, 4, 100), cal_eval, LEVAL)
    plt.savefig("output_data/evaluation/Bayes_error_plot_GMM_16_diag_evaluation_cal.png")
    plt.show()

