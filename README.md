Predict weekly Walmart sales data (https://www.kaggle.com/datasets/mikhail1681/walmart-sales) using various unconstrainted and constrained linear methods (ordinary least squared, all subsets, ridge, lasso, partial least squared).

ordinary least squared, all subset: scikit-learn/sklearn.linear_model.LinearRegression  
ridge: scikit-learn/sklearn.linear_model.Ridge, sympy (polynomial solver for conversion between dof and constraint coefficient)  
lasso: cvxpy (quadratic convexity solver to have solution in norm-ball form)  
partial least squared: scikit-learn/cross_decomposition.PLSRegression  

Predictors were identitified to be 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', each quarters, and intercept term. A k = 5 folds cross validation methods where chosen. Each method was computed on each fold, as a function of the effective degree of freedom (dof). Averages and standard deviation across folds were computed.  

The best dof, per methods, were chosen to be within 1 standard deviation of the minimum test error. A list of the chosen dof per method is saved in result.csv  

The best method overal were chosen to be within 1 standard deviation of the minimum test error, across all chosen dof per method: Lasso, with a s-value of 0.3.  

References: Hastie et al, Elements of Statistical Learning, 2009.
