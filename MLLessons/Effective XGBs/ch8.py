# Random Forest 

# Ensembles = collection of model act as one model. (Bagging/Boosting/stacking/cascading)

# Bagging
# Collection of overfitted model and aggregated results. By assuring, different model column & row sample.
# Every estimator is a TREE
# mean::regression and majority-voting::classification

import os 
import pandas  as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline,model_selection
from feature_engine import imputation,encoding
from rich import print
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
import xgboost as xgb
import dtreeviz
from ch2 import get_rawX_y,TweakKagTransformer


rf:RandomForestClassifier = RandomForestClassifier(random_state=42,n_jobs=-1,verbose=1)
xgb_rf:xgb.XGBRFClassifier = xgb.XGBRFClassifier(
                                    random_state=42,
                                    n_jobs=-1,
                                    verbosity=1,
                                    learning_rate=0.3,
                                    num_parallel_tree=30,
                                    device='cuda',
                                    max_depth=6,
                                    grow_policy='depthwise',
                                    tree_method='approx',
                                    eval_metric='auc',
                                )
# set `grow_policy=lossguide` + `max_depth=0` => LightGBM behaviour 

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

    ######################################################################
    #
    #                         SKlearn
    # 
    ######################################################################
    rf.fit(X_train,y_train)
    print(f"random forest eval score:: {rf.score(X_test,y_test)}")

    print(rf.get_params())


    # Peeking only one Decision Tree Out of 100
    fig,ax = plt.subplots(figsize=(20,20))
    features = X_train.columns.tolist()
    plot_tree(decision_tree=rf.estimators_[99],ax=ax,feature_names=features,filled=True,class_names=rf.classes_,max_depth=5,fontsize=6)
    plt.show()




    ######################################################################
    #
    #                         XGBoost
    # 
    ######################################################################

    
    xgb_rf.fit(X_train,y_train=='Data Scientist')

    print(f"XGB RF eval score: {xgb_rf.score(X_test,y_test=='Data Scientist')}")
    print(xgb_rf.get_params())


    ### Hard to Visualize
    # fig,ax = plt.subplots(figsize=(20,20),dpi=600)
    # xgb.plot_tree(xgb_rf,num_trees=0,ax=ax,size='1,1')
    # plt.show()

    viz = dtreeviz.model(
                      xgb_rf,
                      X_train=X_train,
                      y_train=y_train=='Data Scientist',
                      target_name='Job',
                      feature_names=features,
                      class_names=['DS','SWD'],
                      tree_index=0
        )
    viz.view(depth_range_to_display=[0,3],precision=5,fontname='Monospace').show()
