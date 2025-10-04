# XGBoost

'''
This ensemble is created by training a `weak model` and then training anther model to correct the resiual

Metaphor: Golfing

- Decision Tree: getting one swing to hit the ball and put it in hole.
- RF:            getting bunch of diff attempts at teeing off (swing, pace, etc ) and placing ball by taking avg of each those swing.
- Boosting:      hitting the ball once, and then going to landed & hitting again (correct from prev. mistake ).
                 They can correct the model's error as they progress
'''

import os 
import pandas as pd 
from sklearn import pipeline,model_selection,base,compose,preprocessing
from feature_engine import imputation,encoding
from matplotlib import pyplot as plt 
import xgboost as xgb 
import dtreeviz 
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

    
    # combined data
    X:pd.DataFrame  = pd.concat([X_train,X_test],axis='index')
    y:pd.DataFrame  = pd.Series([*y_train,*y_test],index=X.index)    


    ######################################################################
    #
    #                         XGBoost
    # 
    ######################################################################

    model_xgb:xgb.XGBClassifier = xgb.XGBClassifier()
    model_xgb.fit(X_train,y_train)
    print(f"default model eval score:: {model_xgb.score(X_test,y_test)}")



    shallow_model:xgb.XGBClassifier = xgb.XGBClassifier(max_depth=2,n_estimators=2)
    shallow_model.fit(X_train,y_train)
    print(f"shallow model eval score:: {shallow_model.score(X_test,y_test)}")



    dtreeviz.model(
                shallow_model,
                X_train=X_train,
                y_train=y_train,
                feature_names=X_train.columns.tolist(),
                class_names=['DS','SWE'],
                tree_index=1,
                target_name='JOb'
    ).view(depth_range_to_display=[0,3],precision=5,fontname='Monospace').show()


    xgb.plot_tree(shallow_model,num_trees=1)
    plt.show()


    # software-engineer data
    infer_data:pd.DataFrame = pd.DataFrame(
                                {'age': {2671: 50},
                                    'education': {2671: 18.0},
                                    'years_exp': {2671: 25.0},
                                    'compensation': {2671: 0},
                                    'python': {2671: 0},
                                    'r': {2671: 0},
                                    'sql': {2671: 1},
                                    'Q1_Male': {2671: 1},
                                    'Q1_Female': {2671: 0},
                                    'Q1_Prefer not to say': {2671: 0},
                                    'Q1_Prefer to self-describe': {2671: 0},
                                    'Q3_United States of America': {2671: 1},
                                    'Q3_India': {2671: 0},
                                    'Q3_China': {2671: 0},
                                    'major_cs': {2671: 0},
                                    'major_other': {2671: 0},
                                    'major_eng': {2671: 1},
                                    'major_stat': {2671: 0}})
    
    print(f"predicted proba of model DS, SWE respectively:: {model_xgb.predict_proba(infer_data)}")
