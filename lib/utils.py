import numpy


def vcol(v):
    return v.reshape((v.shape[0], 1))


def vrow(v):
    return v.reshape((1, v.shape[0]))


def compute_means(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    for i in range(6):
        print("Mean for feature %d:" % (i + 1))
        print("Counterfeit: ", D0[i].mean())
        print("Genuine: ", D1[i].mean())
        print()


def compute_variances(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    for i in range(6):
        print("Variance for feature %d:" % (i + 1))
        print("Counterfeit: ", D0[i].var())
        print("Genuine: ", D1[i].var())
        print()


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)
