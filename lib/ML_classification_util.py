import warnings
warnings.filterwarnings("ignore")

# In[ ]:

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc,confusion_matrix
from sklearn import metrics
import itertools
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier

from sklearn.svm import SVC
from numpy import interp
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV,StratifiedKFold
from itertools import product

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

# In[ ]:

class UTIL():
    """
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

    """
    model_dict={}
    parameters_dict={}
    metrics='accuracy'
    
    
    def __create_nn(self,optimizer='rmsprop', init='glorot_uniform'):
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=21, kernel_initializer=init, activation='relu'))
        model.add(Dense(8, kernel_initializer=init, activation='relu'))
        model.add(Dense(4, kernel_initializer=init, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model


# In[2]:

    def _create_models(self,model_list):
        if 'xgb' in model_list:    
            xgb = make_pipeline(RobustScaler(),XGBClassifier(seed=2))
            parameters={'xgbclassifier__learning_rate':[0.1,0.3],'xgbclassifier__max_depth':[3,5,7],'xgbclassifier__gamma':[0,0.5],'xgbclassifier__n_estimators':[100,300]}
            self.model_dict['xgb']=xgb
            self.parameters_dict['xgb']=parameters 
            
        if 'gb' in model_list:
            gb = make_pipeline(RobustScaler(), GradientBoostingClassifier(random_state=42))
            parameters={'gradientboostingclassifier__max_depth':[3,5,7],'gradientboostingclassifier__n_estimators':[100,200,300]}
            self.model_dict['gb']=gb
            self.parameters_dict['gb']=parameters    

        if 'ada' in model_list:
            ada = make_pipeline(RobustScaler(), AdaBoostClassifier(random_state=42))
            parameters={'adaboostclassifier__n_estimators':[100,200,300]}
            self.model_dict['ada']=ada
            self.parameters_dict['ada']=parameters

        if 'nn' in model_list:
            seed = 7
            np.random.seed(seed)
            nn = KerasClassifier(build_fn=self.__create_nn, verbose=0)
            parameters={'optimizer':['rmsprop', 'adam'],'init' : ['glorot_uniform', 'normal'],'epochs' : [50, 100],'batch_size' :[5, 10]}
            self.model_dict['nn']=nn
            self.parameters_dict['nn']=parameters 

        if 'gnb' in model_list:
            gnb=make_pipeline(RobustScaler(),GaussianNB())
            parameters={}
            self.model_dict['gnb']=gnb
            self.parameters_dict['gnb']=parameters 
        
        if 'rf' in model_list:
            rf=make_pipeline(RobustScaler(),RandomForestClassifier(random_state=42))
            parameters={'randomforestclassifier__max_depth':[3,5,7],'randomforestclassifier__n_estimators':[100,200,300]}
            self.model_dict['rf']=rf
            self.parameters_dict['rf']=parameters

        if 'knn' in model_list:
            knn = make_pipeline(StandardScaler(),KNeighborsClassifier())
            parameters={'kneighborsregressor__n_neighbors':[3,4,5]}
            self.model_dict['knn']=knn
            self.parameters_dict['knn']=parameters
                 
        if 'xtree' in model_list:
            xtree = make_pipeline(RobustScaler(), ExtraTreesClassifier(n_jobs=-1,random_state=42))
            parameters={'extratreesclassifier__n_estimators':[200,500,700]}
            self.model_dict['xtree']=xtree
            self.parameters_dict['xtree']=parameters
        
        if 'svc' in model_list:
            svc=make_pipeline(RobustScaler(),SVC(random_state=42))
            parameters={'svc__C':[0.05,0.2,1]}
            self.model_dict['svc']=svc
            self.parameters_dict['svc']=parameters
# In[1]:

    def run_basic_models(self,model_list,data):
        train_x,test_x,train_y,test_y=data
        self._create_models(model_list)
        print(model_list)
        
        for model_k, model_v in self.model_dict.items():

            print("Working on model: %s"%(model_k))
            
            model_v.fit(train_x.values,train_y)
            train_preds=model_v.predict(train_x.values)
            test_preds=model_v.predict(test_x.values)

            if self.metrics=='accuracy':
                train_acc=accuracy_score(train_preds, train_y)
                test_acc=accuracy_score(test_preds, test_y)
            # if metrics=='rmsle':
            #     train_acc=mean_squared_error(train_preds, train_y)**0.5
            #     test_acc=mean_squared_error(test_preds, test_y)**0.5                
            
            print("Fitting model %s, train accuracy:%f,test accuracy:%f"%(model_k,train_acc,test_acc))
        

        
    
    


# In[ ]:

def Model_GridSearchCV(model,parameters,cv,data,n_jobs=3):
    train_x,test_x,train_y,test_y=data

    if n_jobs==0:
        clf = GridSearchCV(model, parameters,cv=cv)
    else:
        
        clf = GridSearchCV(model, parameters,cv=cv,n_jobs=n_jobs)
    clf.fit(train_x.values, train_y)
    for mean, std, params in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score'], clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    model.set_params(**clf.best_params_)
    model.fit(train_x.values, train_y)
    
    return model, clf.best_params_

def Model_Eval(model,n_folds,data,n_jobs=3,one_hot=False,binary=False):
    [train_x,test_x,train_y,test_y]=data
    if binary:
        plot_ROC_curve(model, train_x, train_y, pos_label=1, n_folds=n_folds)
    
    # Predict the values from the validation dataset
    y_pred = model.predict(test_x.values)

    if one_hot==True:
        #Convert predictions classes to one hot vectors 
        y_pred_classes = np.argmax(y_pred,axis=1) 
        #Convert validation observations to one hot vectors
        y_true = np.argmax(test_y,axis=1) 
        # compute the confusion matrix
        confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
        print('Auccuracy= %0.2f' % (sum(y_pred_classes - y_true == 0)/len(y_true)))
    else:
        confusion_mtx = confusion_matrix(test_y, y_pred) 
        print('Auccuracy= %0.2f' % (sum(test_y - y_pred == 0)/len(test_y)))
    # plot the confusion matrix
        
    __plot_confusion_matrix(confusion_mtx, classes = range(max(test_y)+1))
    
def __plot_ROC_curve(classifier, X, y, pos_label=1, n_folds=5):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    skf = StratifiedKFold(n_splits=n_folds)
    i=0
    for train, test in skf.split(X, y):
        i+=1
        #probas_ = classifier.fit(X.iloc[train].values, y.iloc[train].values).predict_proba(X.iloc[test].values)
        classifier.fit(X.iloc[train].values, y.iloc[train].values)
        probas_=classifier.predict_proba(X.iloc[test].values)
        print(probas_)
        print(y.iloc[test])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    mean_tpr /= n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def __plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
