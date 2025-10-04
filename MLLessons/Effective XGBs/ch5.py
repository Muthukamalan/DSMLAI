# Stumps on Real Data

'''
Thr first spliit should feature one of the most critical feature because it is the value that best sepearates into classes
'''




import numpy as np 
import numpy.random as rn  
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.tree import DecisionTreeClassifier,plot_tree
import xgboost as xgb
import dtreeviz
from ch2 import get_rawX_y,TweakKagTransformer
from feature_engine import imputation,encoding
from sklearn import pipeline,model_selection,preprocessing
import os


def calc_gini_fn(df:pd.DataFrame, val_col:str, label_col:str, pos_val:str, split_point:int, debug=False)->float:
    ge_split = df[val_col]>= split_point
    eq_pos = df[label_col]== pos_val

    tp = df[ge_split & eq_pos].shape[0]
    fp = df[ge_split & ~eq_pos].shape[0]
    tn = df[~ge_split & ~eq_pos].shape[0]
    fn = df[~ge_split & eq_pos].shape[0]

    pos_size = tp+fp 
    neg_size = tn+fn 

    total_size = len(df)

    if pos_size==0: gini_pos = 0
    else:           gini_pos = 1-(tp/pos_size)**2 - (fp/pos_size)**2

    if neg_size==0: gini_neg = 0
    else:           gini_neg = 1-(tn/neg_size)**2 - (fn/neg_size)**2

    weighted_avg =  gini_pos*(pos_size/total_size)  + gini_neg*(neg_size/total_size)
    if debug:
        print(f"{gini_pos=:.3} {gini_neg=:.3}  {weighted_avg=:.3}")
    return weighted_avg


#  $$entropy={\sum P(x) }\dot{ \log({{1}\over{P(x)}})}$$



def inv_logits_fn(p:float)->float:
    '''
        it returns probability
        fn(p) = exp(p) / (1+exp(p))
    '''
    return np.exp(p) / (1+np.exp(p))


# SKLEARN
stump_dt = DecisionTreeClassifier(max_depth=1)


# XGBoost(eXtreme Gradient Boosting) 

## uses boosting and gradient Descent
## Boosting = combining multiple models, subsequent models are trained from the error of the previous model with the goal to reduce error
## Gradient Descent = process of min the error is specified into objective function.
        ##   objective_fn = training_loss + regularization_term

stump_xgb:xgb.XGBClassifier = xgb.XGBClassifier(n_estimators=1,max_depth=1)

'''
    # Algo
    Trees are created by splitting on features that move a small step in the direction of negative gradient, which moves them closer to the global minimum of the loss function.

    # Many parameters::
    - n_estimators (number of trees)
'''




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

    print(pd.concat([X_train,y_train],axis='columns').head(5))


    ######################################################################
    #
    #                         SKlearn
    # 
    ######################################################################
    

    # Training
    stump_dt.fit(X_train,y_train)


    # "r" column best sepearates the two profession adequately good.
    # people who less prefer "r language" are "software engineers"
    fig,ax = plt.subplots(figsize=(8,4))
    features = X_train.columns.tolist()
    plot_tree(
        decision_tree=stump_dt,
        feature_names=features, 
        filled=True,
        class_names=stump_dt.classes_,
        ax=ax
    )
    plt.show()

    # Evaluate
    print(f"evaluate test-data:: {stump_dt.score(X_test,y_test)}") #0.624% is good enough?



    ######################################################################
    #
    #                         XGBoost
    # 
    ######################################################################
    
    label_enc = preprocessing.LabelEncoder()                # replace numerical value instead of string
    y_train = label_enc.fit_transform(y_train)
    y_test  = label_enc.transform(y_test)

    print(X_train.head(5),label_enc.inverse_transform(y_train[:5]), y_train[:5])

    # Training
    stump_xgb.fit(X_train,y_train)

    # Values in the leaves are propabilities/ log of the odds of the positive value (a.k.a logits)
    xgb.plot_tree(stump_xgb,num_trees=0)
    plt.show()

    # Evaluate
    print(f"evaluate on test data:: {stump_xgb.score(X_test,y_test)}")

    # if r<1: logits=0.128297 else:  logits=-0.3035
    print(f"if r<1 prob of SWD::{inv_logits_fn(0.128297)*100:.3}%, \telse prob of DS::{inv_logits_fn(-0.30355)*100:.3}%")

    viz = dtreeviz.model(stump_xgb,X_train=X_train,y_train=y_train,feature_names=features,class_names=['DS',"SWE"],target_name='Job',tree_index=0)
    viz.view(precision=5,fontname='Monospace').show()