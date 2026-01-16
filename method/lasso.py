import numpy as np
import sympy as sp
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

# following 3.4.2 hastie et al
# pdf page 87
# lambda (hastie) = alpha (scikit learn)
def reg(x_train, y_train, x_test, y_test, lambd):

    out = np.zeros((len(x_test), x_train.shape[1]+3))

    for i in range(len(lambd)):
        if lambd[i] != 0:
            regression = Lasso(alpha=lambd[i])
        else:
            regression = LinearRegression

        regression.fit(x_train, y_train)

        y_hat_train = regression.predict(x_train).reshape(-1, 1)
        train_error = sum((y_hat_train - y_train) ** 2) / len(y_train)

        y_hat_test = regression.predict(x_test).reshape(-1, 1)
        test_error = sum((y_hat_test - y_test) ** 2) / len(y_test)

        out[i, 0]  = lambd[i]
        out[i, 1]  = train_error
        out[i, 2]  = test_error
        out[i, 3:] = regression.coef_.flatten()

    return out