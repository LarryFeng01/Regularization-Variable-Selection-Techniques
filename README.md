# Project 5: Comparison of Different Regularization and Variable Selection Techniques
### By Larry Feng

## Introduction
In this project, I will apply and compare different regularization techniques including Ridge, Lasso, Square Root Lasso, Elastic Net, and SCAD.

I will be creating my own sklearn compliant functions for Square Root Lasso and SCAD to use in conjunction with GridSearchCV to find the optimal hyper-parameters given some kind of data. I will simulate 100 data sets, each of them with 1200 features, 200 observations, and a toeplitz correlation structure with p = 0.8. Lastly, I will apply variable selection methods(Ridge, Lasso, ...) with GridSearchCV and record the final results, measured by accuracy in reconstructing the sparsity pattern. 

## Methods
Below are two functions, Square Root Lasso and SCAD, that are sklearn compliant in which we are able to use concurrently to find the optimal hyper-parameters for some data. 
```
from sklearn.base import BaseEstimator, RegressorMixin
from numba import njit
class SQRTLasso(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
  
    def fit(self, x, y):
        alpha=self.alpha
        @njit
        def f_obj(x,y,beta,alpha):
          n =len(x)
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.sqrt(1/n*np.sum((y-x.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
          return output
        @njit
        def f_grad(x,y,beta,alpha):
          n=x.shape[0]
          p=x.shape[1]
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = (-1/np.sqrt(n))*np.transpose(x).dot(y-x.dot(beta))/np.sqrt(np.sum((y-x.dot(beta))**2))+alpha*np.sign(beta)
          return output.flatten()
        
        def objective(beta):
          return(f_obj(x,y,beta,alpha))
        
        def gradient(beta):
          return(f_grad(x,y,beta,alpha))
        
        beta0 = np.ones((x.shape[1],1))
        output = minimize(objective, beta0, method='L-BFGS-B', jac=gradient,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
```
This is the sklearn compliant function for Square root Lasso. It is important to remember to import BaseEstimator, RegressorMixin, and njit to make this function work. 

```
from sklearn.base import BaseEstimator, RegressorMixin
class SCAD(BaseEstimator, RegressorMixin):
    def __init__(self, a=2,lam=1):
        self.a, self.lam = a, lam
  
    def fit(self, x, y):
        a = self.a
        lam   = self.lam

        @njit
        def scad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          return 1/n*np.sum((y-x.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))

        @njit  
        def dscad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          output = -2/n*np.transpose(x).dot(y-x.dot(beta))+scad_derivative(beta,lam,a)
          return output.flatten()
          
        beta0 = np.zeros(p)
        output = minimize(scad, beta0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 50,'disp': False})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
```
Once again, importing BaseEstimator and RegressorMixin is very important to making this a sklearn compliant function.

## Simulation

To simulate our datasets, we will make use of the numpy library and some lists. 

```
n = 200
p = 1200
beta_star = np.concatenate(([1]*7,[0]*25,[.25]*5,[0]*50,[.7]*15,[0]*1098))

v = [] #to be used for toeplitz()
for i in range(p):
    v.append(0.8**i)
```
According to the project guidelines, each dataset will have 1200 features (*p* = 1200), 200 observations (*n* = 200), coefficients of a specific order, and a toeplitz correlation structure. Now let's generate some random samples
```
mu = [0]*p
sigma = 3.5
np.random.seed(123)
x = np.random.multivariate_normal(mu, toeplitz(v), size=n)
y = np.matmul(x,beta_star).reshape(-1,1) + sigma*np.random.normal(0,1,size=(n,1))
```
Now that we are able to generate data, we can apply some methods on them.

## Application

