import pandas as pd
import numpy as np
import method
from sklearn.model_selection import StratifiedKFold
import math
import miscellaneous

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
row_count_each_fold_all_subset = len(predictor)

all_subset_result = pd.DataFrame(np.nan, index = range(row_count_each_fold_all_subset*k), columns= return_field)

#-------------------------------------------
# ridge
# return(dof, train_error, test_error, coefficient)
# row - dof: (1, 2, .., num(predictor) ) * k folds
row_count_each_fold_ridge = len(predictor)

ridge_result = pd.DataFrame(np.nan, index = range(row_count_each_fold_ridge*k), columns= return_field)

#-------------------------------------------
# lasso
# return(lambda, train_error, test_error, coefficient)
# row: between 0 and inf (choose linspace) - no clear dof
return_field_lasso = ["fold", "lambda", "train_error", "test_error"]
return_field = np.concatenate((return_field, predictor))

lasso_const_val = np.linspace(0, 100, 10)

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
    #least_squared_result.iloc[i] = np.append(i, method.least_squared.reg(x_train, y_train, x_test, y_test))

    #--------------------------------------------------
    # all_subset
    #start_index_all_subset = i*row_count_each_fold_ridge
    #end_index_all_subset = (i+1)*row_count_each_fold_ridge
    #all_subset_result.iloc[start_index_all_subset:end_index_all_subset, 0] = np.full(row_count_each_fold_all_subset, i)
    #all_subset_result.iloc[start_index_all_subset:end_index_all_subset, 1:] = method.all_subset.reg(x_train, y_train, x_test, y_test)

    #--------------------------------------------------
    # ridge
    #start_index_ridge = i*row_count_each_fold_ridge
    #end_index_ridge   = (i+1)*row_count_each_fold_ridge
    #ridge_result.iloc[start_index_ridge:end_index_ridge, 0] = np.full(row_count_each_fold_ridge, i)
    #ridge_result.iloc[start_index_ridge:end_index_ridge, 1:] = method.ridge.reg(x_train, y_train, x_test, y_test)

    #--------------------------------------------------
    # lasso
    method.lasso.reg(x_train, y_train, x_test, y_test, lasso_const_val)





print("-------------------------------------------")
#print(least_squared_result)
#print(all_subset_result)
#print(ridge_result)


#----------------------------------------------------------------------------------------------------------
# across folds, group by dof, search for minimum test_error

# least squared
# (not applicable)

#--------------------------------------------------
# all subset
#all_subset_grouped = all_subset_result.groupby("dof").agg(["mean", "std"])
#print(all_subset_grouped)

#--------------------------------------------------
ridge_result_grouped = ridge_result.groupby("dof").agg(["mean", "std"])
print(ridge_result_grouped)

#----------------------------------------------------------------------------------------------------------
# search for dof within 1 degree of freedom of minimum (of test error) (chosen dof per method)
# plot test and train error as a function dof
# grid chose dof vs test_error


# least squared


#--------------------------------------------------
# all subset




#--------------------------------------------------
#--------------------------------------------------

#'''

data_plot = ridge_result_grouped

x = data_plot.index
y = data_plot["test_error","mean"]
y_error = data_plot["test_error","std"]


#plt.ylim(0.2e12, 1.2e12)

plt.errorbar(x, y, yerr=y_error,
             fmt='-o', ecolor='r', capsize=3, markersize=3)

plt.savefig("test.png")
#'''


# match by dof, avg test_error - cv_error, avg_coef
# compare them with table from page 82 - least dof within 1 sd of minimum per method (all except least_squared)


# plot dof vs cv_error for all method (with the chosen dof for each method)
# select best model also by using least dof within 1 sd of minimum

