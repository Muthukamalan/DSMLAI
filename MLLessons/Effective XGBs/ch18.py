# Feature Importance
import xgbfir


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


    
    # options = {
    #     'max_depth': hp.quniform('max_depth',1,8,1),
    #     'min_child_weight': hp.loguniform('min_child_weight',-2,3),
    #     'subsample':hp.uniform('subsample',.5,1),
    #     'colsample_bytree':hp.uniform('colsample_bytree',.5,1),
    #     'reg_alpha':hp.uniform('reg_alpha',0,10),
    #     'reg_lambda':hp.uniform('reg_lambda',1,10),
    #     'gamma':hp.loguniform('gamma',-10,10),
    #     'learning_rate':hp.loguniform('learning_rate',-7,0),
    #     'random_state':42
    # }

    # trails:Trials = Trials()
    # best = fmin(
    #             fn= lambda space: hparam_tuning(
    #                                     space=space,
    #                                     X_train=X_train,
    #                                     y_train=y_train,
    #                                     X_test=X_test,
    #                                     y_test=y_test,
    #                                 ),
    #             space=options,
    #             algo=tpe.suggest,
    #             max_evals=2_000,
    #             trials=trails,
    #             show_progressbar=True
    #     )
    # print(best)
    
    best:Dict = {
        'colsample_bytree': 0.7470228692203221, 
        'gamma': 0.010002924743648106, 
        'learning_rate': 0.14810193973234734, 
        'max_depth': int(5.0), 
        'min_child_weight': 0.8923824614916529, 
        'reg_alpha': 2.6520368621936474, 
        'reg_lambda': 7.337429825972468, 
        'subsample': 0.9986344239601417
    }

    best['max_depth'] = int(best['max_depth'])

    

    ################################################################
    #
    #               XGBoost :: black model
    #
    #################################################################

    xgb_model = XGBClassifier(
                            **best,
                            early_stopping_rounds=50,
                            n_estimators=500,
                            n_jobs=-1
                )
    xgb_model.fit(
                X_train,
                y_train, 
                eval_set=[(X_train,y_train),(X_test,y_test)],
                verbose=10
            )
    print(f"model eval score:: {xgb_model.score(X_test,y_test)}")      # 76.7                                  # 76.7955% better model
    

    xgbfir.saveXgbFI(xgb_model,feature_names=X_train.columns, OutputXlsxFile=os.path.join(os.getcwd(),'assets','features.xlsx'))


    # This sheet contains information about indiiviual features, including their importance, gain & coverage
    interaction_one = pd.read_excel('./assets/features.xlsx',sheet_name='Interaction Depth 0')


    # contains info about pairwise feature interaction
    interaction_two = pd.read_excel('./assets/features.xlsx',sheet_name='Interaction Depth 1')

    # Contains info about higher-order features interaction (2 or 2+)
    interaction_two_more = pd.read_excel('./assets/features.xlsx',sheet_name='Interaction Depth 2')



    '''
        Xgboost can limit feature iinteraction
        Experiment with Pick top interaction and model needs for simple or regulation model.
        `constraints`


        The purpose of using interaction constraints in a machine learning model is to restrict the possible interactions between features. This can be useful when domain knowledge suggests that certain interactions are not meaningful or when interpretability is important.
        
        
        # DRAWBACK
        The `potential drawbacks` of using interaction constraints in a machine learning model include reduced flexibility and potentially reduced predictive performance if important interactions are excluded.

    '''


    COLUMNS = ['age', 'education', 'years_exp', 'compensation', 'python', 'r', 'sql', 'Q1_Male', 'Q1_Female', 'Q1_Prefer not to say', 'Q1_Prefer to self-describe', 'Q3_United States of America', 'Q3_India', 'Q3_China', 'major_cs', 'major_other', 'major_eng', 'major_stat']
    CONSTRAINTS = set(['age', 'compensation', 'education', 'major_cs', 'major_stat', 'r', 'years_exp'])
    
    xgb_model = XGBClassifier(
                            # **best,
                            # early_stopping_rounds=50,
                            # n_estimators=500,
                            interaction_constraints = CONSTRAINTS
                )
        

    #FIXME: ValueError: Constrained features are not a subset of training data feature names
    # X_train =X_train.loc[:,['age', 'compensation', 'education', 'major_cs', 'major_stat', 'r', 'years_exp']]
    # X_test = X_test.loc[:,['age', 'compensation', 'education', 'major_cs', 'major_stat', 'r', 'years_exp']]
    # xgb_model.fit(
    #             X_train,
    #             y_train, 
    #             eval_set=[(X_train,y_train),(X_test,y_test)],
    #             verbose=10
            # )
    # print(f"model eval score:: {xgb_model.score(X_test.loc[:,['age', 'compensation', 'education', 'major_cs', 'major_stat', 'r', 'years_exp']],y_test)}") 