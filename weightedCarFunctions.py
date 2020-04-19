import numpy as np


def getRobustError(errorVector, flag=False):
    assert type(errorVector) == np.ndarray
    assert errorVector.shape[1] == 1
    var = np.var(errorVector)
    sd = np.sqrt(var)
    mean = np.mean(errorVector)
    r, c = errorVector.shape
    q = np.zeros((r, c))
    r1, c1 = np.where(np.abs(mean - errorVector) <= var)
    q[r1, c1] = 1
    r2, c2 = np.where(np.abs(mean - errorVector) > var)
    q[r2, c2] = 1

    errorVector = q * errorVector
    if flag:
        print(q)

    return errorVector
