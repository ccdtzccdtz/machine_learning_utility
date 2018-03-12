
# coding: utf-8

# In[ ]:

import warnings
warnings.filterwarnings("ignore")


# In[ ]:

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

#from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


# In[ ]:

class UTIL():
    """
    Avaiable models:

    krr:Kernel Ridge
    enet:Elastic Net
    lasso:Lasso
    rf:Random Forest
    gb:Gradient Boosting
    xtree:extra tree
    knn: K nearest neighbour regressor

    """
    model_dict={}
    parameters_dict={}
    metrics='mse'
    


# In[2]:

    def _create_models(self,model_list):
        if 'krr' in model_list:    
            KRR = KernelRidge()
            parameters={'kernel':['linear','polynomial','sigmoid'],'alpha':[1,0.5],'degree':[2,3],'gamma':[0.5,0.75,1]}
            self.model_dict['krr']=KRR
            self.parameters_dict['krr']=parameters 
            
        if 'enet' in model_list:
            enet = make_pipeline(RobustScaler(), ElasticNet())
            parameters={'elasticnet__alpha':[0.0001,0.01,1],'elasticnet__l1_ratio':[0.1,0.5,0.7]}
            self.model_dict['enet']=enet
            self.parameters_dict['enet']=parameters    

        if 'lasso' in model_list:
            lasso = make_pipeline(RobustScaler(), Lasso())
            parameters={'lasso__alpha':[0.0001,0.0005,0.01,0.05,0.1,0.5]}
            self.model_dict['lasso']=lasso
            self.parameters_dict['lasso']=parameters

        if 'rf' in model_list:
            rf = RandomForestRegressor(n_estimators=150,n_jobs=-1)
            parameters={'n_estimators':[100,200]}
            self.model_dict['rf']=rf
            self.parameters_dict['rf']=parameters 

        if 'gb' in model_list:
            gb = GradientBoostingRegressor(n_estimators=150)
            parameters={'n_estimators':[100,200]}
            self.model_dict['gb']=gb
            self.parameters_dict['gb']=parameters 
        
        if 'xtree' in model_list:
            xtree = ExtraTreesRegressor(n_estimators=15,n_jobs=-1)
            parameters={'n_estimators':[10,20]}
            self.model_dict['xtree']=xtree
            self.parameters_dict['xtree']=parameters

        if 'knn' in model_list:
            knn = make_pipeline(RobustScaler(), KNeighborsRegressor(n_jobs=-1, n_neighbors=4))
            parameters={'kneighborsregressor__n_neighbors':[3,4,5]}
            self.model_dict['knn']=knn
            self.parameters_dict['knn']=parameters
                 


# In[1]:

    def run_basic_models(self,model_list,data):
        train_x,test_x,train_y,test_y=data
        self._create_models(model_list)
        
        for model_k, model_v in self.model_dict.items():

            print("Working on model: %s"%(model_k))
            
            model_v.fit(train_x,train_y)
            train_preds=model_v.predict(train_x)
            test_preds=model_v.predict(test_x)

            if self.metrics=='mse':
                train_acc=mean_squared_error(train_preds, train_y)
                test_acc=mean_squared_error(test_preds, test_y)
            # if metrics=='rmsle':
            #     train_acc=mean_squared_error(train_preds, train_y)**0.5
            #     test_acc=mean_squared_error(test_preds, test_y)**0.5                
            
            print("Fitting model %s, train MSE:%f,test MSE:%f"%(model_k,train_acc,test_acc))
        

        
    
    


# In[ ]:

def Model_GridSearchCV(model,parameters,cv,data,name,n_jobs=3,logtrans=False):
    train_x,test_x,train_y,test_y=data
    if logtrans==True:
        train_y=np.log1p(train_y)
    clf = GridSearchCV(model, parameters,cv=cv,n_jobs=3)
    clf.fit(train_x.values, train_y)
    for mean, std, params in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score'], clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    model.set_params(**clf.best_params_)
    model.fit(train_x.values, train_y)
    train_preds=model.predict(train_x.values)
    test_preds=model.predict(test_x.values)
    
    if logtrans==True:
        train_preds=np.expm1(train_preds)
        train_y=np.expm1(train_y)
        test_preds=np.expm1(test_preds)
        
    print("trainset MSE= %0.3f" % mean_squared_error(train_preds, train_y))
    print("testset MSE= %0.3f" % mean_squared_error(test_preds, test_y))
    fig,axes=plt.subplots(1, 2, figsize=(11, 8))
    
    axes[0].set_title("train %s"%(name))
    axes[1].set_title("test %s"%(name))
    sns.regplot(train_preds,train_y,ax=axes[0])
    sns.regplot(test_preds, test_y,ax=axes[1])
    return model, clf.best_params_

