from lib import utils, pca, lda
import numpy
import matplotlib.pyplot as plt


def lab03_analysis(D, L):
    pca.PCA_analysis(D, L)
    plt.figure()
    plt.title("LDA Projection")
    lda.LDA_analysis(D, L)
    plt.savefig("output_data/lab03/LDA_projection.png")
    plt.show()

    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
    print("\n********* Apply LDA as classifier **********\n")
    plt.figure()
    plt.title("Model training set with LDA (DTR, LTR)")
    W = lda.LDA_analysis(DTR, LTR)
    plt.savefig("output_data/lab03/LDA_training_set.png")
    plt.show()
    DVAL_lda = numpy.dot(W.T, DVAL)
    plt.figure()
    plt.title("Model validation set with LDA (DVAL, LVAL)")
    lda.HistLDA(DVAL_lda, LVAL)
    plt.savefig("output_data/lab03/LDA_validation_set.png")
    plt.show()
    DTR_lda = numpy.dot(W.T, DTR)
    threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] > threshold] = 0
    PVAL[DVAL_lda[0] <= threshold] = 1
    accuracy_LDA = numpy.mean(PVAL == LVAL)
    print("Accuracy (LDA): %.2f" % (accuracy_LDA * 100) + " %")
    print("Error rate (LDA): %.2f" % ((1 - accuracy_LDA) * 100) + " %")

    print("\n********* Change the value of the threshold **********\n")
    for i in range(-10, 10):
        threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0 + 0.1 * i
        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL[DVAL_lda[0] >= threshold] = 0
        PVAL[DVAL_lda[0] < threshold] = 1
        accuracy_LDA = numpy.mean(PVAL == LVAL)
        print(f"Threshold: {threshold:.2f}")
        print("Accuracy (LDA): %.2f" % (accuracy_LDA * 100) + " %")
        print("Error rate (LDA): %.2f" % ((1 - accuracy_LDA) * 100) + " %")
        print()

    print("\n********* Pre-process with PCA and apply LDA as classifier **********\n")
    for m in range(1, D.shape[0]+1):
        print(f"Number of features: {m}")
        P = pca.compute_pca(D, m)
        DTR_pca = pca.apply_pca(P, DTR)
        DVAL_pca = pca.apply_pca(P, DVAL)
        plt.figure()
        plt.title(f"Model training set with PCA (DTR_pca, LTR) with {m} features")
        W2 = lda.LDA_analysis(DTR_pca, LTR)
        plt.savefig("output_data/lab03/PCA_LDA_training_set_%d.png" % m)
        plt.show()
        DVAL_lda2 = numpy.dot(W2.T, DVAL_pca)
        plt.figure()
        plt.title(f"Model validation set with LDA after PCA (DVAL, LVAL) with {m} features")
        lda.HistLDA(DVAL_lda2, LVAL)
        plt.savefig("output_data/lab03/PCA_LDA_validation_set_%d.png" % m)
        plt.show()
        DTR_lda2 = numpy.dot(W2.T, DTR_pca)
        threshold = (DTR_lda2[0, LTR == 0].mean() + DTR_lda2[0, LTR == 1].mean()) / 2.0
        PVAL2 = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL2[DVAL_lda2[0] >= threshold] = 0
        PVAL2[DVAL_lda2[0] < threshold] = 1
        accuracy_LDA = numpy.mean(PVAL2 == LVAL)
        print("Accuracy (LDA): %.2f" % (accuracy_LDA * 100) + " %")
        print("Error rate (LDA): %.2f" % ((1 - accuracy_LDA) * 100) + " %")
        print()
