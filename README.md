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

First we have to find the optimal hyper-parameters in order to run the model and then later run some KFold cross-validation to validate our answers. To get the optimal hyper-parameters, we are using a library called GridSearchCV. Below is the implementation for one model:
```
model = Lasso(alpha = 0.1, fit_intercept=False, max_iter = 10000)
model.fit(x,y)
grid = GridSearchCV(estimator=Lasso(),cv=10,scoring='neg_mean_squared_error',param_grid={'alpha': np.linspace(0,1,20)})
grid.fit(x,y)
print("The best parameters for Lasso is:\n",grid.best_params_)
```
First we define a model and fit; then we run GridSearchCV to find the best parameter for this specific model in a given range. After fitting the grid, we can call its functions to return the best parameter. For this code, it is only for the Lasso model, but the same syntax applies for the other models. After this ran for a long time, we get the optimal hyper-parameters. Below are the results. A few things that I could improve here is looking into more on the parameters for Ridge and Elastic Net as GridSearchCV returned that an alpha of 0 was the optimal parameter for the models. 
```
The best parameters for Lasso is:
 {'alpha': 0.21052631578947367}

The best parameters for Ridge is:
 {'alpha': 0.0}
 
The best parameters for Elastic Net is:
 {'alpha': 0.001, 'l1_ratio': 0.8572857142857143}
 
 The best parameters for SCAD is:
 {'a': 0.7777777777777777, 'lam': 0.3333333333333333}

 The best parameters for SQRTLasso is:
 {'alpha': 0.15789473684210525}
```

Next, we can run some validation for these models. Below is some code written to find the root mean squared error, L2 distance to the real coefficients, and how accurate the sparsity pattern is to the real one. 
```
def validate(model,x,y,rs,nfolds=5):
    kf = KFold(n_splits=nfolds, shuffle=True,random_state=rs)
    RMSE = [] #prediction error
    dist = [] #L2 distance to real coefficients
    acc = [] #accuracy to real sparsity pattern
    for idxtrain, idxtest in kf.split(x):
        xtrain = x[idxtrain]
        ytrain = y[idxtrain]
        xtest = x[idxtest]
        ytest = y[idxtest]
        model.fit(xtrain,ytrain)
        RMSE.append(np.sqrt(mean_squared_error(ytest,model.predict(xtest))))

        beta_hat = model.coef_
        dist.append(np.linalg.norm(model.coef_ - beta_star, ord=2))

        pos_model = np.where(beta_hat != 0)
        acc.append(np.intersect1d(pos,pos_model).shape[0]/np.array(pos).shape[1])

    print("The Root Mean Squared Error is: " + str(np.mean(RMSE)))
    print("The L2 distance to the real coefficients is: " + str(np.mean(dist)))
    print("The accuracy to the real sparisty pattern is: " + str(np.mean(acc)))
```
and the results are as follows:
 | Models      | Validations |
 | ----------- | ----------- |
 | Lasso       | The Root Mean Squared Error is: 4.090661139244214 |
 |             | The L2 distance to the real coefficients is: 3.5358850321178443 |
 |             | The accuracy to the real sparisty pattern is: 0.7555555555555555 |
 |             | |
 | Ridge       | The Root Mean Squared Error is: 6.276818151515922|
 |             | The L2 distance to the real coefficients is: 3.046021042560226|
 |             | The accuracy to the real sparisty pattern is: 1.0|
 |             | |
 | Elastic Net | The Root Mean Squared Error is: 4.411117331604949|
 |             | The L2 distance to the real coefficients is: 3.566104361846014|
 |             | The accuracy to the real sparisty pattern is: 0.8666666666666668|
 |             | |
 | SCAD        | The Root Mean Squared Error is: 7.267522760527325|
 |             | The L2 distance to the real coefficients is: 3.0505892470293468|
 |             | |
 |             | The accuracy to the real sparisty pattern is: 1.0|
 | SQRTLasso   | The Root Mean Squared Error is: 3.8585317937777077|
 |             | The L2 distance to the real coefficients is: 1.3930172410723514|
 |             | The accuracy to the real sparisty pattern is: 1.0|
 
 If we compare the models to each other, SQRTLasso has a significantly lower L2 distance, but a slightly lower root mean squared error compared to Lasso. One note I would like to make is that the "accuracy" of the sparsity pattern isn't really an accuracy. This merely measures how many values are the same in both sparsity patterns. One example is the Ridge model. Looking at the sparsity pattern returns every single feature, which is 1200 of them. So, the Ridge did get all of the correct positions for the real sparsity pattern but that is because it has all of the positions. 
 
 ## Conclusion
 For the above results, we conclude that SQRTLasso is the best model for our purposes based on these three validation measures. There are a couple things I would like to improve. The first would be the number of times I could simulate and validate our models. Due to the high runtime, I wasn't able to have many Kfold validations. Another thing is I would like to recheck the parameters for Ridge and Elastic Net since their alpha values are zero, or very close to zero. I would run a few more tests to see what the optimal alpha is (most likely is it zero, but double chekcking isn't bad). 
