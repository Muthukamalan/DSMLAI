# SHAP

import shap
from typing import Dict,Sequence,Callable,Tuple
from xgboost import XGBClassifier,plot_tree,plot_importance
import os 
import numpy as np
import pandas as pd 
from matplotlib import (
                    cm,
                    pyplot as plt
)
import seaborn as sns
from sklearn import metrics,model_selection,preprocessing,pipeline
from feature_engine import encoding,imputation
from ch2 import get_rawX_y,TweakKagTransformer

shap.initjs()

'''
    Model Explainabilty
    - Model Based:: 
        - Attention
        - gradient saliency
        - Integrated gradients
    - Model Agnostic:
        - SHAP
        - LIME
        - Pertubation

    
    SHAP (SHapley Additive exPlanations) is a game-theoretic approach that can provide both global and local explanations for a model's behavior. 
    It can model non-linear relationships and rank the importance of features while also indicating the direction of their impact.

    - global(scope within dataset) & local Explainability


    shapely values = Average of marginal contributions over all possible permutations of subset of players  
    
    $\phi_{i}= f_{x}(\hat{Z}) - f_{x}(\hat{Z_{-i}})$
    
    i.e) $2^{F} combination$
    if 
        2 players = 2^2  combinations
        .
        .
        16 players = 65536 commbinations

    But in XGBoost features are fixed, we just randomzied the feature it'll loss prediective power

    Illusion of explaination, it'll tell whihc drives model, but why it does don't know

    
    Local::
        - Waterfall
        - forceplot (single/multiple)
        - scatter (interaction -multiple)
    Global::
        - beeswarm
'''
shap.initjs()



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

    
    best_hparams:Dict = {
        'colsample_bytree': 0.7470228692203221, 
        'gamma': 0.010002924743648106, 
        'learning_rate': 0.14810193973234734, 
        'max_depth': int(5.0), 
        'min_child_weight': 0.8923824614916529, 
        'reg_alpha': 2.6520368621936474, 
        'reg_lambda': 7.337429825972468, 
        'subsample': 0.9986344239601417
    }

    xgb_model = XGBClassifier(
                            **best_hparams,
                            early_stopping_rounds=50,
                            n_estimators=500
                )
    xgb_model.fit(
                X_train,
                y_train, 
                eval_set=[(X_train,y_train),(X_test,y_test)],
                verbose=10
            )
    print(f"model eval score:: {xgb_model.score(X_test,y_test)}")                                      # 76.7955% better model
    

    shap_explainer = shap.TreeExplainer(xgb_model)
    print(shap_explainer)
    vals           = shap_explainer(X_test)
    shap_df = pd.DataFrame(vals.values, columns=X_test.columns)
    print('*'*20,"SHAP DF,'*"*20)
    print(shap_df)
    print('*'*20,'*'*20)
    print(
        pd.concat([
            shap_df.sum(axis='columns').rename('pred') + vals.base_values ,
            pd.Series(y_test,name='true') 
        ],axis='columns').assign(
            prob = lambda adf: (np.exp(adf.pred)/(1+np.exp(adf.pred)) )
        )
    )

    print(
        f"base:: {shap_explainer.expected_value}", # default is Data Scientist if "0", else SWE "1"

        f"valed 0th:: {shap_explainer.expected_value + vals.values[0].sum()}"
    )



    # Local
    
    ## - waterfall
    fig = plt.figure(figsize=(8,4))
    shap.plots.waterfall(vals[0],show=True)
    plt.show()
    
    ## - barh 
    pd.Series(vals.values[0],index=X_test.columns).sort_values(key=np.abs).plot.barh();
    plt.show()

    ## - forceplot
    shap.plots.force(base_value=vals.base_values,shap_values=vals.values[24,:],features=X_test.iloc[24],matplotlib=True,show=True)
    print(f"24th index:: {y_test[24]}")
    plt.show()

    ## multiple-forceplot
    n=100
    shap.plots.force(
                base_value=vals.base_values, 
                shap_values = vals.values[:n,:],
                features = X_test.iloc[:n],
                matplotlib=False,
                show=True
    )
    plt.show()

    ## - interaction multiple plots
    shap.plots.scatter(vals[:, 'education'], color=vals[:, 'major_cs'], x_jitter=1, hist=False,alpha=.5) #'major_cs'
    plt.show()


    # HeatMap
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
                X_test.assign(software_eng=y_test)\
                    .corr(method='spearman')\
                    .loc[:, ['age', 'education', 'years_exp','compensation', 'r', 'major_cs','software_eng']],
                cmap='RdBu', 
                annot=True, 
                fmt='.2f', 
                vmin=-1, 
                vmax=1, 
                ax=ax
    )
    plt.show()

    # Heatmap for Shap values
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
            shap_df.assign(software_eng=y_test).corr(method='spearman').loc[:, ['age', 'education', 'years_exp', 'compensation', 'r', 'major_cs','software_eng']],
            cmap='RdBu', 
            annot=True, 
            fmt='.2f', 
            vmin=-1, 
            vmax=1, 
            ax=ax
    )
    plt.show() # ignoring the self-correlation values


    # Global
    fig = plt.figure(figsize=(8, 4))
    shap.plots.beeswarm(vals,alpha=.5,max_display=10)

    shap.plots.beeswarm(vals, max_display=len(X_test.columns), color=cm.autumn_r,plot_size=(8,6))



    # No interaction model
    no_int_params = {'random_state': 42,'max_depth': 1}
    xg_no_int = XGBClassifier(**no_int_params, early_stopping_rounds=50,n_estimators=500)
    xg_no_int.fit(X_train, y_train,eval_set=[(X_train, y_train),(X_test, y_test)])
    print(f"eval score:: {xg_no_int.score(X_test, y_test)}")
    shap_ind = shap.TreeExplainer(xg_no_int)
    shap_ind_vals = shap_ind(X_test)
    shap.plots.beeswarm(shap_ind_vals, max_display=len(X_test.columns))