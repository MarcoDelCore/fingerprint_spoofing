import numpy


def extract_train_val_folds(X, val_idx, k_folds=3):
    train_set = numpy.hstack([X[i::k_folds] for i in range(k_folds) if i != val_idx])
    val_set = X[val_idx::k_folds]
    return train_set, val_set
