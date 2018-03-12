# machine_learning_utility
Utility for regression and classification
1. Currently only support regression models and classification models
2. `run_basic_models` method will train data on each model in the model list with default parameters
3. `Model_GridSearchCV` method will train each model in the model list while conducing a gridsearchcv with parameters provided
## Regression Utility Usage
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
## Classification Utility Usage
```
from ML_util.ML_classification_util import UTIL
from ML_util.ML_classification_util import Model_GridSearchCV, Model_Eval
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
import pandas as pd
from sklearn.preprocessing import LabelEncoder

                                                         
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=44)


cv=5
data=[train_x,test_x,train_y,test_y]
model=UTIL()
model.run_basic_models(data=data,model_list=['rf','gb','xgb','xtree','knn','nn'])
clf=model.model_dict['nn']
parameters=model.parameters_dict['nn']
parameters['gradientboostingclassifier__max_depth']=[2,3,4]
clf,param=Model_GridSearchCV(clf,parameters,cv,data)
Model_Eval(clf,5,data,n_jobs=3)


```
>
    Avaiable models:
    
    xgb:Extreme Gradient Boosting
    gnb:Gaussian Naive Bayes
    ada:Adapative Boosting
    nn:Nerual Net
    rf:Random Forest
    gb:Gradient Boosting
    xtree:extra tree
    knn: K nearest neighbour regressor
    svc: Support Vector Classifier
