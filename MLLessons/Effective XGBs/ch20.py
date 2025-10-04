# Better Model
## - Partial Dependence Plots 2001
## - ICE (Individual Conditional Expectation) Plot 2024
## - Monotonic Constraints
## - calibration


# PDP Plot
''' Usage:
        a selected predictor variable contribution to an outcome is calculated by calculating its avg marginal effect whih ignores the effect of other variables present in the model.
    Pro:
        PDPs illustrate the average behavior of the model for a particular input variable while holding all other variables constant.
    cons:
        selected variable are uncorrelated
        line close to flat indicates no significant change in outcome value but it reality the data can be equal value with opposite signs
'''

# ICE Plot
''' 
    an ICE plot displays the model's output for a fixed instance while incrementally changing one input feature's value.
effect of single input variable to output of ML modeld'''

import shap
from typing import Dict,Sequence,Callable,Tuple
from xgboost import XGBClassifier,plot_tree,plot_importance
import os 
import numpy as np
import pandas as pd 
from matplotlib import (
                    cm,
                    pyplot as plt,
                    gridspec
)
import seaborn as sns
from sklearn import metrics,model_selection,preprocessing,pipeline,inspection,calibration
from feature_engine import encoding,imputation
from ch2 import get_rawX_y,TweakKagTransformer




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
    

    fig,axs = plt.subplots(ncols=5,figsize=(8,4),tight_layout=True)
    # ['age', 'education', 'years_exp', 'compensation', 'python', 'r', 'sql', 'Q1_Male', 'Q1_Female', 'Q1_Prefer not to say','Q1_Prefer to self-describe', 'Q3_United States of America', 'Q3_India', 'Q3_China', 'major_cs', 'major_other', 'major_eng', 'major_stat']
    # assume 0 as Data Scientist, 1 as SWE
    inspection.PartialDependenceDisplay.from_estimator(xgb_model,X_train,features=['r','education','python', 'r', 'sql'],kind='individual',ax=axs,centered=True,ice_lines_kw={'zorder':10})   
    plt.show()

    fig,axs = plt.subplots(ncols=2,figsize=(8,4),tight_layout=True)
    ax_h0 = axs[0].twinx()
    ax_h0.hist(X_train.r, zorder=0)
    ax_h1 = axs[1].twinx()
    ax_h1.hist(X_train.education, zorder=0)
    inspection.PartialDependenceDisplay.from_estimator(xgb_model,X_train,features=['r','education'],kind='individual',ax=axs,centered=True,ice_lines_kw={'zorder':10})#kind='both'   
    plt.show()


    # # If you set the quantile to 1 in the code above, you create a Partial Dependence Plot. It is theaverage of the ICE plots.
    fig, ax = plt.subplots(figsize=(8,4))
    shap.partial_dependence_plot(
                            ind='education',
                            model=lambda rows: xgb_model.predict_proba(rows)[:,-1],
                            data=X_train.iloc[0:1000], 
                            ice=True,                                                                   # partial_dependence_plot=False
                            npoints=(X_train.education.nunique()),
                            pd_linewidth=0, 
                            show=False, 
                            ax=ax
    )   
    fig.tight_layout()
    ax.set_title('ICE plot (from SHAP)')
    plt.show()


    fig, ax = plt.subplots(figsize=(8,4))
    shap.partial_dependence_plot(
                    ind='education',
                    model=lambda rows: xgb_model.predict_proba(rows)[:,-1],
                    data=X_train.iloc[0:1000], 
                    ice=False,                                                                         # partial_dependence_plot=True
                    npoints=(X_train['education'].nunique()),
                    pd_linewidth=2, 
                    show=False,
                    ax=ax)
    ax.set_title('PDP plot (from SHAP)')


    # MONOTONIC Constraints
    fig, ax = plt.subplots(figsize=(8,4))
    X_test.assign(target=y_test).corr(method='spearman').iloc[:-1].loc[:,'target'].sort_values(key=np.abs).plot.barh(title="spearman corr with target",ax=ax)
    plt.show()


    # deciding whether to add monotonic constraint to XGBoost depends on the specific context and goals of the mdoel.
    # ensure model to particular realtionship  where we expect clear cause-and-effect relationship (model predicts the increasing/decreasing trend for that feature)
    # increasing maps to 1 / decreasing maps to -1

    xgb_model = XGBClassifier(random_state=123,)
    xgb_model.fit(X_train,y_train)
    print("eval score without constraints:: ",xgb_model.score(X_test,y_test))

    xgb_model = XGBClassifier(random_state=123,monotone_constraints={'years_exp':1,'education':-1})
    xgb_model.fit(X_train,y_train)
    print("eval score with constraints:: ",xgb_model.score(X_test,y_test))



    # calibration
    ## plot scaling
    ## Isotonic regression
    calibrated_model = calibration.CalibratedClassifierCV(estimator=xgb_model,method='sigmoid',cv='prefit')# prefit= already fitted model
    calibrated_model.fit(X_test,y_test)


    fig = plt.figure(figsize=(8,6))
    gs  = gridspec.GridSpec(4,3)
    axes= fig.add_subplot(gs[:2,:3])
    dis = calibration.CalibrationDisplay.from_estimator(xgb_model,X_test,y_test,n_bins=20,ax=axes)
    dis_cal = calibration.CalibrationDisplay.from_estimator(xgb_model,X_test,y_test,n_bins=20,ax=axes,name='sigmoid')
    dis_cal_iso = calibration.CalibrationDisplay.from_estimator(xgb_model,X_test,y_test,n_bins=20,ax=axes,name='isotonic')

    row=2
    col=0
    ax = fig.add_subplot(gs[row,col])
    ax.hist(dis.y_prob,range=(0,1),bins=20)
    ax.set_title("Default")
    ax2 = fig.add_subplot(gs[row,1])
    ax2.hist(dis_cal.y_prob,range=(0,1),bins=20)
    ax2.set_title("Sigmoid")
    ax3 = fig.add_subplot(gs[row,2])
    ax3.hist(dis_cal_iso.y_prob,range=(0,1),bins=20)
    ax3.set_title("Isotonic")
    fig.tight_layout()
    plt.show()
    

    print(calibrated_model.score(X_test,y_test))
    