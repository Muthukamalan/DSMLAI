# Do we've enough data??
'''
    - high amount of quality data
'''

from sklearn.model_selection import learning_curve
import dtreeviz
import os 
from hyperopt import fmin,hp,STATUS_OK,Trials,tpe
from typing import Dict,Any,Union,Callable,Sequence
import pandas as pd 
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn import preprocessing,pipeline,model_selection
from feature_engine import imputation,encoding
from xgboost import XGBClassifier,plot_tree
import seaborn as sns 
from matplotlib import pyplot as plt
from ch2 import get_rawX_y,TweakKagTransformer


if __name__=='__main__':
    ######################################################################
    #
    #                         Pre Processing
    # 
    ######################################################################
    
    raw_file = pd.read_csv(os.path.join(os.path.dirname(__file__), 'assets','multipleChoiceResponses.csv'))

    kaggle_question:pd.Series = raw_file.iloc[0]
    kaggle_df:pd.DataFrame = raw_file.iloc[1:]

    kaggle_X, kaggle_y = get_rawX_y(kaggle_df,y_col='Q6')

    kaggle_pl= pipeline.Pipeline([
        ('tweak', TweakKagTransformer()),
        ('cat', encoding.OneHotEncoder(top_categories=5, drop_last=True,  variables=['Q1', 'Q3', 'major'])),
        ('num_impute', imputation.MeanMedianImputer(imputation_method='median', variables=['education', 'years_exp']))
    ])

    X_train, X_test, y_train, y_test  = model_selection.train_test_split(kaggle_X,kaggle_y,random_state=42,shuffle=True,stratify=kaggle_y,train_size=0.7)
    X_train = kaggle_pl.fit_transform(X_train)
    X_test  = kaggle_pl.transform(X_test)

    
    
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(y_train)
    print(label_enc.classes_,label_enc.transform(label_enc.classes_))
    y_train   = label_enc.transform(y_train)
    y_test    = label_enc.transform(y_test) 


    params = {
            'learning_rate': 0.3,
            'max_depth': 2,
            'n_estimators': 200,
            'n_jobs': -1,
            'random_state': 42,
            'reg_lambda': 0,
            'subsample': 1
    }


    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    # Training Data VS Loss 
    fig, ax = plt.subplots(figsize=(8, 4))
    train_size_abs, train_scores, test_scores = learning_curve(XGBClassifier(**params),X_train, y_train,cv=3,n_jobs=-1,train_sizes=[.3,.6,.9])
    for train_size, cv_train_scores, cv_test_scores in zip( train_size_abs, train_scores, test_scores):
        print('*'*200)
        print(f"{train_size} samples were used to train the model")
        print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
        print(f"The average test accuracy is {cv_test_scores.mean():.2f}")
        