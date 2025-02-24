from lib import utils, gaussian_model as gm, lda, pca


def lab05_analysis(D, L):
    for covType in ["Full", "Tied", "Diag"]:
        print("\nCovariance type: %s" % covType)
        gm.gaussian_predictions(D, L, covType)

    print("\n*********** Model LDA ***********\n")
    lda.LDA_classify(D, L)

    print("\n********* Analyze covariance matrix **********\n")
    _, _, C_0, C_1 = gm.ML_estimates(D, L)
    print("Covariance matrix of class 0:\n", C_0)
    print("\nCovariance matrix of class 1:\n", C_1)
    Corr_0 = C_0 / (utils.vcol(C_0.diagonal() ** 0.5) * utils.vrow(C_0.diagonal() ** 0.5))
    Corr_1 = C_1 / (utils.vcol(C_1.diagonal() ** 0.5) * utils.vrow(C_1.diagonal() ** 0.5))
    print("\nCorrelation matrix of class 0:\n", Corr_0)
    print("\nCorrelation matrix of class 1:\n", Corr_1)

    print("\n********* Repeat claffification without last 2 features **********\n")
    for covType in ["Full", "Tied", "Diag"]:
        print("\nCovariance type: %s" % covType)
        gm.gaussian_predictions(D[:-2, :], L, covType)

    print("\n*********** Model LDA without last 2 features ***********\n")
    lda.LDA_classify(D[:-2, :], L)

    print("\n********* MVG model for features 1 and 2 **********\n")
    for covType in ["Full", "Tied"]:
        print("\nCovariance type: %s" % covType)
        gm.gaussian_predictions(D[:2, :], L, covType)

    print("\n********* MVG model for features 3 and 4 **********\n")
    for covType in ["Full", "Tied"]:
        print("\nCovariance type: %s" % covType)
        gm.gaussian_predictions(D[2:4, :], L, covType)

    print("\n********* PCA pre-processing **********\n")
    for m in range(1, D.shape[0] + 1):
        print(f"\nNumber of features: {m}")
        P = pca.compute_pca(D, m)
        DP = pca.apply_pca(P, D)
        for covType in ["Full", "Tied", "Diag"]:
            print("\nCovariance type: %s" % covType)
            gm.gaussian_predictions(DP, L, covType)
        print("\nLDA calssification")
        lda.LDA_classify(DP, L)
