# Early Stopping
'''
    works by monitoring validation loss and automatically soft stop model training no-longer improves. helps for generalization
    most common metric:
        - logloss
        - auc             # measure binary classification model, `Area Under receiver Operating characteristic curve`
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
