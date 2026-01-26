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
return_info = ["fold", "dof", "train_error", "test_error"]
return_field = np.concatenate((return_info, predictor))

# k-fold validation - equal stratum for categorical predictor (quarter)
k = 5

folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
strata = data["Quarter"]

#----------------------------------------------------------------------------------------------------------
# least_squared
# return(dof, train_error, test_error, coefficient)
# row - dof: (1) * k folds
least_squared_result = pd.DataFrame(np.nan,
                                    index = range(k),
                                    columns= return_field)

#-------------------------------------------
# all_subset
# return(dof, train_error, test_error, coefficient)
# row - dof: (1, 2, .., num(predictor) ) * k folds
row_count_each_fold_all_subset = len(predictor)

all_subset_result = pd.DataFrame(np.nan,
                                 index = range(row_count_each_fold_all_subset*k),
                                 columns= return_field)

#-------------------------------------------
# ridge
# return(dof, train_error, test_error, coefficient)
# row - dof: (1, 2, .., num(predictor) ) * k folds
row_count_each_fold_ridge = len(predictor)

ridge_result = pd.DataFrame(np.nan,
                            index = range(row_count_each_fold_ridge*k),
                            columns= return_field)

#-------------------------------------------
# lasso
# return(s_val, train_error, test_error, coefficient)
# row: between 0 and 1 (choose linspace) for s value (definition 3.4.2 hastie et al)
return_info_lasso = ["fold", "s_val", "train_error", "test_error"]
return_field_lasso = np.concatenate((return_info_lasso, predictor))

lasso_s_val = np.linspace(0.1, 1, 10)
row_count_each_fold_lasso = len(lasso_s_val)

lasso_result = pd.DataFrame(np.nan,
                            index = range(row_count_each_fold_lasso*k),
                            columns= return_field_lasso)

#-------------------------------------------
# partial least squared
# return(dof, train_error, test_error, coefficient)
# row - dof: 1, 2, .., num(predictor)
row_count_each_fold_pls = len(predictor)

pls_result = pd.DataFrame(np.nan,
                          index = range(row_count_each_fold_pls*k),
                          columns = return_field)

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
    start_index_all_subset = i*row_count_each_fold_ridge
    end_index_all_subset   = (i+1)*row_count_each_fold_ridge
    all_subset_result.iloc[start_index_all_subset:end_index_all_subset, 0] = np.full(row_count_each_fold_all_subset, i)
    all_subset_result.iloc[start_index_all_subset:end_index_all_subset, 1:] = method.all_subset.reg(x_train, y_train, x_test, y_test)

    #--------------------------------------------------
    # ridge
    start_index_ridge = i*row_count_each_fold_ridge
    end_index_ridge   = (i+1)*row_count_each_fold_ridge
    ridge_result.iloc[start_index_ridge:end_index_ridge, 0] = np.full(row_count_each_fold_ridge, i)
    ridge_result.iloc[start_index_ridge:end_index_ridge, 1:] = method.ridge.reg(x_train, y_train, x_test, y_test)

    #--------------------------------------------------
    # lasso
    start_index_lasso = i*row_count_each_fold_lasso
    end_index_lasso   = (i+1)*row_count_each_fold_lasso
    lasso_result.iloc[start_index_lasso:end_index_lasso, 0] = np.full(row_count_each_fold_lasso, i)
    lasso_result.iloc[start_index_lasso:end_index_lasso, 1:] = method.lasso.reg_norm_ball(x_train, y_train, x_test, y_test, lasso_s_val)

    #--------------------------------------------------
    # pls
    start_index_pls = i*row_count_each_fold_pls
    end_index_pls   = (i+1)*row_count_each_fold_pls
    pls_result.iloc[start_index_pls:end_index_pls, 0] = np.full(row_count_each_fold_pls, i)
    pls_result.iloc[start_index_pls:end_index_pls, 1:] = method.pls.reg(x_train, y_train, x_test, y_test)


print("-------------------------------------------")
#print(least_squared_result)
#print(all_subset_result)
#print(ridge_result)
#print(lasso_result)

#----------------------------------------------------------------------------------------------------------
# across folds, group by dof, search for minimum test_error
print("-------------------------------------------")
print("grouping - mean and std")

#--------------------------------------------------
# least squared
least_squared_grouped = least_squared_result.groupby("dof").agg(["mean", "std"])
least_squared_grouped.drop(columns=["fold"], inplace=True)
print("least squared")
print(least_squared_grouped)

#--------------------------------------------------
# all subset
all_subset_grouped = all_subset_result.groupby("dof").agg(["mean", "std"])
all_subset_grouped.drop(columns=["fold"], inplace=True)
print("all subset")
print(all_subset_grouped)

#--------------------------------------------------
# ridge
ridge_result_grouped = ridge_result.groupby("dof").agg(["mean", "std"])
ridge_result_grouped.drop(columns=["fold"], inplace=True)
print("ridge")
print(ridge_result_grouped)

#--------------------------------------------------
# lasso
lasso_result_grouped = lasso_result.groupby("s_val").agg(["mean", "std"])
lasso_result_grouped.drop(columns=["fold"], inplace=True)
print("lasso")
print(lasso_result_grouped)

#--------------------------------------------------
# pls
pls_result_grouped = pls_result.groupby("dof").agg(["mean", "std"])
pls_result_grouped.drop(columns=["fold"], inplace=True)
print("pls")
print(pls_result_grouped)
pls_result_grouped_filtered = pls_result_grouped.iloc[pls_result_grouped.index <= 7]

#----------------------------------------------------------------------------------------------------------
# search for dof within 1 degree of freedom of minimum (of test error) (chosen dof per method)
# plot test and train error as a function dof
# grid chose dof vs test_error
print("-------------------------------------------")
print("choose dof per method")

least_squared_chosen = miscellaneous.within_1_std_of_min(least_squared_grouped)
least_squared_chosen["method"] = ["least_squared"]

all_subset_chosen = miscellaneous.within_1_std_of_min(all_subset_grouped)
all_subset_chosen["method"] = ["all_subset"]

ridge_chosen = miscellaneous.within_1_std_of_min(ridge_result_grouped)
ridge_chosen["method"] = ["ridge"]

lasso_chosen = miscellaneous.within_1_std_of_min(lasso_result_grouped)
lasso_chosen["method"] = ["lasso"]

pls_chosen = miscellaneous.within_1_std_of_min(pls_result_grouped)
pls_chosen["method"] = ["pls"]
#--------------------------------------------------
#--------------------------------------------------

'''
x = pls_result_grouped_filtered.index
y = pls_result_grouped_filtered["test_error","mean"]
y_error = pls_result_grouped_filtered["test_error","std"]

#plt.ylim(0.2e12, 1.2e12)

plt.errorbar(x, y, yerr=y_error,
             fmt='-o', ecolor='r', capsize=3, markersize=3)

plt.savefig("z_pls_filtered_test.png")
'''

comparison = pd.concat([least_squared_chosen,
                        all_subset_chosen,
                        ridge_chosen,
                        lasso_chosen,
                        pls_chosen], axis=0)
comparison.to_csv("result.csv")

print(miscellaneous.within_1_std_of_min(comparison))

