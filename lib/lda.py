import matplotlib.pyplot as plt
import numpy
import scipy.linalg as spy
import lib.utils as utils


def HistLDA(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    plt.hist(D0.flatten(), bins=10, alpha=0.4, label='Counterfit', density=True)
    plt.hist(D1.flatten(), bins=10, alpha=0.4, label='Genuine', density=True)
    plt.legend()
    plt.tight_layout()


def LDA_matrices(D, L):
    sumMatrixW = numpy.zeros((D.shape[0], D.shape[0]))
    sumMatrixB = numpy.zeros((D.shape[0], D.shape[0]))
    dataset_mean = utils.vcol(D.mean(1))
    for i in range(max(L) + 1):
        D_class = D[:, L == i]
        class_mean = utils.vcol(D_class.mean(1))
        centered_mean = class_mean - dataset_mean
        D_class_W = D_class - class_mean
        Swc = D_class_W.shape[1] * (D_class_W @ D_class_W.T) / float(D_class_W.shape[1])
        sumMatrixW = sumMatrixW + Swc
        sumMatrixB += (centered_mean @ centered_mean.T) * D_class.shape[1]
    return sumMatrixB / D.shape[1], sumMatrixW / D.shape[1]


def LDA_analysis(D, L):
    Sb, Sw = LDA_matrices(D, L)
    s, U = spy.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:1]
    DW = numpy.dot(W.T, D)
    if DW.mean() < 0:
        W *= -1
        DW = numpy.dot(W.T, D)
    HistLDA(DW, L)
    return W


def LDA_classify(D, L):
    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
    plt.figure()
    plt.title("Model training set with LDA (DTR, LTR)")
    W = LDA_analysis(DTR, LTR)
    plt.savefig("output_data/lab03/LDA_training_set.png")
    plt.show()
    DVAL_lda = numpy.dot(W.T, DVAL)
    plt.figure()
    plt.title("Model validation set with LDA (DVAL, LVAL)")
    HistLDA(DVAL_lda, LVAL)
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
