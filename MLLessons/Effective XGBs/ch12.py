# Hyperopt  (Tree-Structed parzen Estimator- Expected Improvement)
## https://optunity.readthedocs.io/en/latest/user/solvers/TPE.html

'''
Problems with Grid Search It'll search entire space (Brute Force). Even if it's not worth trying anymore.

Hyperopt 
- way to smarter things, optimizing both `discrete` and `continous` hparams
- uses `Bayesian optimization`
- uses a probabilitic model to select the next set of hparams to try


Trials:
    store the results of the hparams tuning process
'''

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

def hparam_tuning(
        space:Dict[str,Union[float,int]],
        X_train:pd.DataFrame,
        y_train:pd.Series,
        X_test:pd.DataFrame,
        y_test:pd.Series,
        early_stopping_rounds:int=50,
        metic:Callable=accuracy_score   #metric to maximize:: accuracy_score
    )->Dict[str,Any]:
    int_vals = ['max_depth','reg_alpha']
    space = {
        k:int(val)  if k in int_vals else val for k,val in space.items()
    }
    space['early_stopping_rounds']= early_stopping_rounds
    model:XGBClassifier = XGBClassifier(**space)
    evaluation = [(X_train,y_train),(X_test,y_test)]
    model.fit(X_train,y_train,eval_set= evaluation,verbose=False)
    pred = model.predict(X_test)
    score = metic(y_test,pred)

    return {
        'loss': -score,
        'status':STATUS_OK,
        'model':model
    }


def trail2df(trail:Sequence[Dict[str,Any]])->pd.DataFrame:
    vals = []
    for t in trails:
        val = {k:v[0] if isinstance(v,list) else v for k,v in t['misc'].get('vals').items()}
        val['loss'] = t['result'].get('loss')
        val['tid']  = t['tid']
        vals.append(val) 
    return pd.DataFrame(vals)



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


    # Rather than limiting, we can leverage hyeropt to pick (we can provide distribution instead of range)

    print(trail2df(trail=trails).head(5).to_string())
    trails_df = trail2df(trail=trails)


    fig, ax = plt.subplots(figsize=(8, 4))
    trails_df.corr(method='spearman').style.background_gradient(cmap='RdBu', vmin=-1, vmax=1)
    sns.heatmap(trails_df.corr(method='spearman'),cmap='RdBu', annot=True, fmt='.2f', vmin=-1, vmax=1) 
    plt.show()    


    # Iterations VS loss
    fig, ax = plt.subplots(figsize=(8, 4))
    trails_df.plot.scatter(x='tid', y='loss', alpha=.1, color='purple', ax=ax)
    plt.show()

    # MAX_DEPTH Vs Loss
    fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(8, 4))
    trails_df.plot.scatter(x='max_depth', y='loss', alpha=1, color='purple', ax=ax1)
    sns.violinplot(x='max_depth', y='loss', data=trails_df,ax=ax2)
    plt.show()


    # loss curve VS gamma
    fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(8,4))
    trails_df.plot.scatter(x='gamma', y='loss', alpha=.1,ax=ax1)
    trails_df.plot.scatter(x='gamma', y='loss', alpha=.1,logx=True, ax=ax2)
    plt.show()