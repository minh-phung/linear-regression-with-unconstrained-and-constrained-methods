import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import numpy as np

def reg(x_train, y_train, x_test, y_test):

    p = x_train.shape[1]

    out = np.full((p, p+3), 0, dtype=float)

    for i in range(p):
        regression = PLSRegression(n_components=i+1).fit(x_train, y_train)

        y_hat_train = regression.predict(x_train)
        train_error = sum((y_hat_train - y_train) ** 2) / len(y_train)

        y_hat_test = regression.predict(x_test)
        test_error = sum((y_hat_test - y_test) ** 2) / len(y_test)

        out[i, 0] = i+1
        out[i, 1] = train_error[0]
        out[i, 2] = test_error[0]
        out[i, 3:] = regression.coef_.flatten()

    return out