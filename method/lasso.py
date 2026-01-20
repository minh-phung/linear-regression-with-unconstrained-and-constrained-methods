import numpy as np
import sympy as sp
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import cvxpy as cp

# following 3.4.2 hastie et al
# pdf page 87
# lambda (hastie) = alpha (scikit learn)
def reg(x_train, y_train, x_test, y_test, lambd):

    out = np.zeros((len(lambd), x_train.shape[1]+3))

    for i in range(len(lambd)):
        if lambd[i] != 0:
            regression = Lasso(alpha=lambd[i], fit_intercept=False)
        else:
            regression = LinearRegression(fit_intercept=False)
        regression.fit(x_train, y_train)

        y_hat_train = regression.predict(x_train).reshape(-1, 1)
        train_error = sum((y_hat_train - y_train) ** 2) / len(y_train)

        y_hat_test = regression.predict(x_test).reshape(-1, 1)
        test_error = sum((y_hat_test - y_test) ** 2) / len(y_test)

        out[i, 0]  = lambd[i]
        out[i, 1]  = train_error[0]
        out[i, 2]  = test_error[0]
        out[i, 3:] = regression.coef_.flatten()

    print(out.shape)

    return out

def reg1 (x_train, y_train, x_test, y_test, lambd):

    #print(x_train.shape)
    y_train = y_train.flatten()

    t_0_val = t_0(x_train, y_train)
    #print(t_0_val)

    p = x_train.shape[1]

    beta = cp.Variable(p)
    objective = cp.Minimize(cp.sum_squares(x_train @ beta - y_train))
    constraint = [cp.norm1(beta) <= t_0_val]

    prob = cp.Problem(objective, constraint)

    result = prob.solve(solver=cp.OSQP, verbose=False)

    print(beta.value)

    return

def t_0 (x_train, y_train):
    # pdf page 88 hastie et al

    regression = LinearRegression(fit_intercept=False)
    regression.fit(x_train, y_train)

    out = 0
    for each in regression.coef_:
        print(each)
        out = out + abs(each)

    return out