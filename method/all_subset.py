import numpy as np
import itertools as it
from method import least_squared
import math

def reg(x_train, y_train, x_test, y_test):
    #print("all subset")

    n = x_train.shape[1]
    #print(x_train)
    #print(x_train.shape)

    count = 0
    for i in range(n):
        count = count + math.comb(n, i+1)

    out_other = np.zeros((count, 3))
    out_coef = np.zeros((count, n))

    count_index = 0
    for i in range(n):
        for each in list(it.combinations(range(n), i+1)):
            #print("------------------")
            reg_result = least_squared.reg(x_train[:, each],
                                           y_train,
                                           x_test[:, each],
                                           y_test)
            out_other[count_index] = reg_result[0:3]
            out_coef[count_index,each] = reg_result[3:]
            #print(each)
            #print(out_other[count_index])
            #print(out_coef[count_index])
            if reg_result[0] != np.count_nonzero(out_coef[count_index]):
                exit()

            count_index += 1
    if count != count_index:
        exit()

    out = np.concatenate((out_other, out_coef), axis=1)

    return out