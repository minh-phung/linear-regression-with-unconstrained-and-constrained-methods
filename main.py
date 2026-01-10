import pandas as pd
import numpy as np
import method
from sklearn.model_selection import StratifiedKFold
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300     # For saved figures



data = pd.read_csv("Walmart_Sales.csv").dropna()
data.drop("Store", axis=1, inplace=True)

data["Date - datetime"] = pd.to_datetime(data["Date"], format="%d-%m-%Y")
data.drop("Date", axis=1, inplace=True)

data["Quarter"] = ['']*data.shape[0]
data['Quarter 1'] = np.zeros(data.shape[0])
data['Quarter 2'] = np.zeros(data.shape[0])
data['Quarter 3'] = np.zeros(data.shape[0])
data['Quarter 4'] = np.zeros(data.shape[0])

print(data.shape)

for i in range(data.shape[0]):
    each = data.loc[i, "Date - datetime"].month
    out = ""
    if each <= 3:
        out = "1st"
        data.loc[i,'Quarter 1'] = 1
    elif each <= 6:
        out = "2st"
        data.loc[i,'Quarter 2'] = 1
    elif each <= 9:
        out = "3st"
        data.loc[i,'Quarter 3'] = 1
    else :
        out = "4th"
        data.loc[i,'Quarter 4'] = 1
    data.loc[i, "Quarter"] = out

data['Intercept'] = np.ones(data.shape[0])

print(data.head())
print(data.columns)

predictor = ['Intercept', 'Holiday_Flag', 'Temperature',
             'Fuel_Price', 'CPI', 'Unemployment',
             'Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4']

target    = ['Weekly_Sales']

data_predictor = np.array(data[predictor])
data_target    = np.array(data[target])

# #----------------------------------------------------------------------------------------------------------
return_field = ["fold", "dof", "train_error", "test_error"]
return_field = np.concatenate((return_field, predictor))

# k-fold validation - equal stratum for categorical predictor (quarter)
k = 5

folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
strata = data["Quarter"]

#----------------------------------------------------------------------------------------------------------
# least_squared
# return(dof, train_error, test_error, coefficient)
# row - dof: (1) * k folds
least_squared_result = pd.DataFrame(np.nan, index = range(k), columns= return_field)

#-------------------------------------------
# all_subset
# return(dof, train_error, test_error, coefficient)
# row - dof: (1, 2, .., num(predictor) ) * k folds
row_count_each_fold = 0
for i in range(0, len(predictor)):
    row_count_each_fold += math.comb(len(predictor), i)

all_subset_result = pd.DataFrame(np.nan, index = range(row_count_each_fold*k), columns= return_field)

#-------------------------------------------
# ridge
# return(dof, train_error, test_error, coefficient)
# row - dof: (1, 2, .., num(predictor) ) * k folds



#-------------------------------------------
# lasso
# return(dof, train_error, test_error, coefficient)
# row - dof: discrete dof (step size tbd) (continuous?)

#-------------------------------------------
# pcr
# return(dof, train_error, test_error, coefficient)
# row - dof: 1, 2, .., num(predictor)

#----------------------------------------------------------------------------------------------------------
# for each k-fold
# method(train_predict, train_target, test_predict, test_target)
for i, (train_index, test_index) in enumerate(folds.split(data, strata)):
    print("-------------------------------------------")
    print("fold ", i)
    x_train, x_test = data_predictor[train_index], data_predictor[test_index]
    y_train, y_test = data_target[train_index],    data_target[test_index]

    #---------------------------------------------------
    # least_square
    least_squared_result.iloc[i] = np.append(i, method.least_squared.reg(x_train, y_train, x_test, y_test))

    #--------------------------------------------------
    # all_subset
    #start_index = i*row_count_each_fold
    #end_index   = (i+1)*row_count_each_fold
    #all_subset_result.iloc[start_index:end_index, 0] = np.full(row_count_each_fold, i)
    #all_subset_result.iloc[start_index:end_index, 1:] = method.all_subset.reg(x_train, y_train, x_test, y_test)

    #--------------------------------------------------
    # ridge
    method.ridge.reg(x_train, y_train, x_test, y_test)



print("-------------------------------------------")
#print(least_squared_result)
#print(all_subset_result)


'''
print(all_subset_result[["dof", "train_error", "test_error"]])
plt.scatter(all_subset_result["dof"], all_subset_result["train_error"],
            color = 'blue', alpha = 0.1, s = 1)

plt.scatter(all_subset_result["dof"], all_subset_result["test_error"],
            color = 'red' , alpha = 0.1, s = 1)
plt.savefig("method_all_subset.png")
'''
#----------------------------------------------------------------------------------------------------------
# match by dof, avg test_error - cv_error, avg_coef
# compare them with table from page 82 - least dof within 1 sd of minimum per method (all except least_squared)


# plot dof vs cv_error for all method (with the chosen dof for each method)
# select best model also by using least dof within 1 sd of minimum

