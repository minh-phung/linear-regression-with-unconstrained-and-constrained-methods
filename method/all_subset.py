import numpy as np
import itertools as it
from method import least_squared
import math

def reg (x_train, y_train, x_test, y_test):

    p = np.shape(x_train)[1]

    out = np.full((p, 3+p), 0, dtype=float)

    for i in range(p):
        smallest_test_row = np.full((3+i+1), np.inf)
        smallest_test_coef_index = np.full((i+1), np.inf)

        for each in list(it.combinations(range(p), i+1)):
            reg_result = least_squared.reg(x_train[:, each],
                                           y_train,
                                           x_test[:, each],
                                           y_test)
            # least_squared.reg[2] is test_error
            if smallest_test_row[2] > reg_result[2]:
                smallest_test_row = reg_result
                smallest_test_coef_index = each

        out[i,0:3] = smallest_test_row[0:3]
        out[i,3+np.array(smallest_test_coef_index)]  = smallest_test_row[3:]

    return out

def reg_full(x_train, y_train, x_test, y_test):
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

    print(out.shape)

    return out