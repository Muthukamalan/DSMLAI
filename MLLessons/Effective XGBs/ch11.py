# XGBoost Hparams
'''
    max_depth
    max_leaves
    min_child_weight
    grow_policy
    tree_method
    colsample_bytree
    colsample_bylevel
    colsample_bynode
    subsample
    sampling_method
    max_cat_to_onehot
    max_cat_threshold
    n_estimators
    early_stopping_rounds
    eval_metric
    objective
    learning_rate
    gamma
    regg_alpha
    reg_lambda
    scale_pos_weight
    max_delta_step


    .get_params()  # getter params
'''

# max_depth:: How many features interaction you can have. Large is more complex(likely to overfit)

# max_leaves = Number of leaves in a tree.

# min_child_weight = 1 - minimum sum of hessian needed in a child

# tree_method = `exact` small-data; `approx` large-data; `hist` for histogram (limit bins with max_bins)

# colsample_bytree = 1-subsample columns at each tree

# colsample_bynode = 1 - subsample column at each node split

# subsample = 1 -sample portion of training data.

# sampling_method = 'uniform

# max_cat_to_onehot = used when number of categories is less than this number

# max_cat_threshold = maximum number of categories to considder for each split

# n_estimators = number of trees

# early_stopping

# eval_metic = `logloss` or `auc`

# objective = `binary:logistic` or `multi:softmax`

# learning_rate

# gamma: L0 Regularization prune tree to remove splits that don't meet the given regularization recommmended-search::(0,1,10,100,1000..)

# reg_alpha: L1 Regularization

# reg_lambda: L2 Regularization

'''
# Boosting Hparams
# learning rate

# Tree Hparams
# max_depth: Explicitly controls the depth of the individual trees. Recommendation: Uniformly search across values ranging from 1-10 but be willing to increase the high value range for larger datasets.
# min_child_weight: Implicitly controls the complexity of each tree by requiring the minimum number of instances (measured by hessian within XGBoost) to be greater than a certain value for further partitioning to occur. Recommendation: Uniformly search across values ranging from near zero-20 but be willing to increase the high value range for larger datasets./

# Stochastic Hparams
#  Subsampling rows before creating each tree. Useful when there are dominating features in your dataset. Recommendation: Uniformly search across values ranging from 0.5-1.0.
#  Colsample_bytree: Subsampling of columns before creating each tree (i.e. mtry in random forests). Useful for large datasets or when multicollinearity exists. Recommendation: Uniformly search across values ranging from 0.5-1.0
#  Colsample_bylevel & colsample_bynode: Additional procedures for sampling columns as you build a tree. Useful for datasets with many highly correlated features. Recommendation: Uniformly search across values ranging from 0.5-1.0


# Regularization hyperparameters
# gamma: Controls the complexity of a given tree by growing the tree to the max depth but then pruning the tree to find and remove splits that do not meet the specified gamma. Recommendation: Search across values ranging from 0-some large number on a log scale (i.e. 0, 1, 10, 100, 1000, etc.).
# alpha: Provides an L2 regularization to the loss function, which is similar to the Ridge penalty commonly used for regularized regression. Recommendation: Search across values ranging from 0-some large number on a log scale (i.e. 0, 1, 10, 100, 1000, etc.).
# lambda: Provides an L1 regularization to the loss function, which is similar to the Lasso penalty commonly used for regularized regression. Recommendation: Search across values ranging from 0-some large number on a log scale (i.e. 0, 1, 10, 100, 1000, etc.).
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

    model_xgb:xgb.XGBClassifier = xgb.XGBClassifier(early_stopping_rounds=20)
    model_xgb.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test,y_test)])
    print(f"model+early-stopping eval score:: {model_xgb.score(X_test,y_test)}")                                       # 76.022% accuracy
    
    results:dict = model_xgb.evals_result()
    # Validation 0 for training-data;   Validation 1 for testing-data
    fig,ax = plt.subplots(figsize=(8,4))
    ax = (
        pd.DataFrame({
                    'training':results['validation_0']['logloss'],
                    'testing': results['validation_1']['logloss']
                }).assign(
                    ntrees=lambda dff: range(1,len(dff)+1) 
                ).set_index('ntrees').plot(
                    figsize=(5,4),
                    ax=ax,
                    title="eval results with early stopping"
                )
    )
    ax.annotate("best results with early stopping",xy=(13,.498),xytext=(20,.42),arrowprops={'color':'k'})
    ax.set_xlabel('ntrees')
    plt.show()
    


    model_xgb:xgb.XGBClassifier = xgb.XGBClassifier()
    model_xgb.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test,y_test)])
    print(f"model+NO-early-stopping eval score:: {model_xgb.score(X_test,y_test)}")                                       # 75.027%



    # GRID SEARCH
    '''
        calibrating one hparam takes some time, there are multiple hparams.., there are better ways to do
    '''

    params = {
        'reg_lambda':[0],
        'learning_rate':[0.03,.1,.3],
        'subsample':[.7,1],
        'max_depth':[2,3,4,6,10],
        'random_state':[42],
        'n_jobs':[-1],
        'n_estimators':[100,200,500]
    }

    model_xgb2 = xgb.XGBClassifier(early_stopping_rounds=5)
    cv = model_selection.GridSearchCV(model_xgb2, param_grid=params, cv=5,n_jobs=-1).fit(X=X_train,y=y_train,eval_set=[(X_test,y_test)],verbose=50)

    print(cv.best_params_)

    xgb_grid_bparams = {
        'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 200, 'n_jobs': -1, 'random_state': 42, 'reg_lambda': 0, 'subsample': 0.7
    }

    xgb_model = xgb.XGBClassifier(**xgb_grid_bparams,early_stopping_rounds=10)
    xgb_model.fit(X=X_train,y=y_train,eval_set=[(X_test,y_test)],verbose=10)
    print(f"model+grid+kFold search eval score:: {xgb_model.score(X_test,y_test)}")                                       # 75.138% accuracy
    





    # Default+earlystop -76 
    # k-Fold - 75.138

    # K Fold an help us understand how consistent the model is and whether some parts of the data might easier to model. you will be more confident in a consistent model