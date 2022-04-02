# Project5

## Comparison of Different Regularization and Variable Selection Techniques
### By Larry Feng

In this project, I will apply and compare different regularization techniques including Ridge, Lasso, Square Root Lasso, Elastic Net, and SCAD.

I will be creating my own sklearn compliant functions for Square Root Lasso and SCAD to use in conjunction with GridSearchCV to find the optimal hyper-parameters given some kind of data. I will simulate 100 data sets, each of them with 1200 features, 200 observations, and a toeplitz correlation structure with p = 0.8. Lastly, I will apply variable selection methods(Ridge, Lasso, ...) with GridSearchCV and record the final results, measured by accuracy in reconstructing the sparsity pattern. 

