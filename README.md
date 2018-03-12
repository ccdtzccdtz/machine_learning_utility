# machine_learning_utility
Utility for regression and classification
1. Currently only support regression models
2. `run_basic_models` method will train data on each model in the model list with default parameters
3. `Model_GridSearchCV` method will train each model in the model list while conducing a gridsearchcv with parameters provided
## Usage
```
from lib.ML_regression_util import UTIL
from lib.ML_regression_util import Model_GridSearchCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

data=train_test_split(X,y,test_size=0.15,random_state=42)

util=UTIL()
modellist=['lasso','rf','enet','gb','xtree']
util.run_basic_models(model_list=modellist,data=data)
parameters=util.parameters_dict
models=util.model_dict
cv=5
fitmodel_dict={}
for model in models:
    print("Fitting %s"%(model))
    fitmodel, best_params_=Model_GridSearchCV(models[model],parameters[model],cv,data,name=model,n_jobs=3,logtrans=False)
    fitmodel_dict[model]=fitmodel
```
>
    Avaiable models:
    
    krr:Kernel Ridge
    enet:Elastic Net
    lasso:Lasso
    rf:Random Forest
    gb:Gradient Boosting
    xtree:extra tree
    knn: K nearest neighbour regressor
