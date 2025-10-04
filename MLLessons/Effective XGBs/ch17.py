# Model Interpretation

from typing import Dict,Sequence,Callable,Tuple
from xgboost import XGBClassifier,plot_tree,plot_importance
import os 
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn import metrics,model_selection,preprocessing,pipeline,linear_model,logging,tree
from feature_engine import encoding,imputation
import dtreeviz
from hyperopt import fmin,hp,STATUS_OK,Trials,tpe

from ch2 import get_rawX_y,TweakKagTransformer
from ch12 import hparam_tuning




if __name__=='__main__':
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


    
    options = {
        'max_depth': hp.quniform('max_depth',1,8,1),
        'min_child_weight': hp.loguniform('min_child_weight',-2,3),
        'subsample':hp.uniform('subsample',.5,1),
        'colsample_bytree':hp.uniform('colsample_bytree',.5,1),
        'reg_alpha':hp.uniform('reg_alpha',0,10),
        'reg_lambda':hp.uniform('reg_lambda',1,10),
        'gamma':hp.loguniform('gamma',-10,10),
        'learning_rate':hp.loguniform('learning_rate',-7,0),
        'random_state':42
    }

    trails:Trials = Trials()
    best = fmin(
                fn= lambda space: hparam_tuning(
                                        space=space,
                                        X_train=X_train,
                                        y_train=y_train,
                                        X_test=X_test,
                                        y_test=y_test,
                                    ),
                space=options,
                algo=tpe.suggest,
                max_evals=2_000,
                trials=trails,
                show_progressbar=True
        )
    print(best)
    
    # best_hparams:Dict = {
    #     'colsample_bytree': 0.7470228692203221, 
    #     'gamma': 0.010002924743648106, 
    #     'learning_rate': 0.14810193973234734, 
    #     'max_depth': int(5.0), 
    #     'min_child_weight': 0.8923824614916529, 
    #     'reg_alpha': 2.6520368621936474, 
    #     'reg_lambda': 7.337429825972468, 
    #     'subsample': 0.9986344239601417
    # }

    best['max_depth'] = int(best['max_depth'])

    

    ################################################################
    #
    #               XGBoost :: black model
    #
    #################################################################

    xgb_model = XGBClassifier(
                            **best,
                            early_stopping_rounds=50,
                            n_estimators=500
                )
    xgb_model.fit(
                X_train,
                y_train, 
                eval_set=[(X_train,y_train),(X_test,y_test)],
                verbose=10
            )
    print(f"model eval score:: {xgb_model.score(X_test,y_test)}")      # 76.7                                  # 76.7955% better model
    

    fig, ax = plt.subplots(figsize=(8,4))
    # gain - This measure the total gain in model's performance that results from using a feature.It is calculated as the avergae gain of splits that use the feature.
    pd.Series(xgb_model.feature_importances_,index=X_train.columns).sort_values().plot.barh(ax=ax)
    plt.show()

    fig,ax = plt.subplots(figsize=(8,4))
    # cover - this measure the number of samples that are affected by a feature. It is calc as the avg coverage of splits that use the feature
    plot_importance(xgb_model,importance_type='cover',ax=ax)
    plt.show()

    fig,ax = plt.subplots(figsize=(8,4))
    # weight - number of times a feature is used in the model.
    plot_importance(xgb_model,importance_type='weight',ax=ax)
    plt.show()


    ################################################################
    #
    #               Linear Model :: White model
    #
    #################################################################

    std = preprocessing.StandardScaler()
    lr = linear_model.LogisticRegression(penalty='l2')
    lr.fit(std.fit_transform(X_train),y_train)
    print(f"logistic reg eval score: {lr.score(std.transform(X_test),y_test)}")
    
    feature_importance = pd.DataFrame(lr.coef_[0]).rename(columns={0:'coef'}).assign(cols=X_train.columns).sort_values(by='coef')
    feature_importance.plot.barh()
    plt.yticks(ticks=range(18),labels=feature_importance['cols'])
    plt.grid(True)
    plt.show()
    '''
    The wider the bar, the higher the impact of the feature. Positive values push towards the positive label (or Software Engineer). Negative labels push towards the negative label (or Data Scientist).
    '''


    ################################################################
    #
    #               Decision Tree :: White model
    #
    #################################################################

    tree7 = tree.DecisionTreeClassifier(max_depth=7)
    tree7.fit(X_train, y_train)
    print(f"Decision Tree eval score: {tree7.score(X_test, y_test)}")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    pd.Series(tree7.feature_importances_, index=X_train.columns).sort_values().plot.barh(ax=ax)
    plt.show()
    # not neccessarily as same as logistic regression

    dt3 = tree.DecisionTreeClassifier(max_depth=3)
    dt3.fit(X_train, y_train)
    viz = dtreeviz.model(dt3, X_train=X_train, y_train=y_train, feature_names=list(X_train.columns), target_name='Job',class_names=['DS', 'SE'])
    viz.view(fontname='Monospace').show()

    #TODO: Surrogate Model
    # make interpretable model by (X_test, model_fn.predict_proba(X_test))