# Model Eval Metrics
## so many metrics for different purpose - model_performance, generalization , robostness
'''
- Accuracy
- confusion metric
- precision & recall
- F1 Score
- Threshold metrics
- cumulative gain curve
- lift curves
'''
from typing import Dict,Sequence,Callable,Tuple
from xgboost import XGBClassifier,plot_tree,plot_importance
import os 
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn import metrics,model_selection,preprocessing,pipeline
from feature_engine import encoding,imputation
from ch2 import get_rawX_y,TweakKagTransformer

def calc_precision_recall_fn(y_true:pd.Series, y_pred:pd.Series)->Tuple[float,float]:

    # both are in matching index
    y_pred = y_pred.reset_index(drop=True)
    y_true = y_true.reset_index(drop=True)

    # Instance counter
    TP = 0 
    FP = 0
    FN = 0 

    # Determin whether each prediction is Tp,Fp,Fn
    for i in y_true.index:
        if y_true[i]==y_pred[i]==1:
            TP+=1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
            FP+=1
        if y_pred[i]==0 and y_test[i]!=y_pred[i]:
            FN+=1
        
    # calc
    try:
        precision = TP/(TP+FP)
    except Exception as e:
        precision = 1
    
    try:
        recall = TP/(TP+FN)
    except Exception as e:
        recall = 1

    return (precision,recall)



def youden_fn(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true,y_score=y_pred)
    index =  np.argmax(tpr-fpr)
    return thresholds[index]



# ThresholdMetric 
class ThresholdXGBClassificer(XGBClassifier):
    def __init__(self,threshold=.5,**kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def predict(self,X,*args,**kwargs):
        proba = self.predict_proba(X,*args,**kwargs)
        return (proba[:,1]>self.threshold).astype(int)


def get_tpr_fpr(probs, y_truth):
    """
        Calculates true positive rate (TPR) and false positive rate
        (FPR) given predicted probabilities and ground truth labels.
        Parameters:
        probs (np.array): predicted probabilities of positive class
        y_truth (np.array): ground truth labels
        Returns:
        tuple: (tpr, fpr)
    """
    tp = (probs == 1) & (y_truth == 1)
    tn = (probs < 1) & (y_truth == 0)
    fp = (probs == 1) & (y_truth == 0)
    fn = (probs < 1) & (y_truth == 1)
    tpr = tp.sum() / (tp.sum() + fn.sum())
    fpr = fp.sum() / (fp.sum() + tn.sum())
    return tpr, fpr

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
    
    # Accuracy score
    acc_score = metrics.accuracy_score(
                                    y_true=y_test, 
                                    y_pred=xgb_model.predict(X_test)
                    )
    print(f"acc_score on test data:: {acc_score}")

    # CONFUSION Matrix
    # Type 1 Error: Actually negative, but model prediected as positive.
    # Type 2 Error: Actually positive, but model prediected as negative.
    conf_matrix = metrics.confusion_matrix(
                    y_true=y_test, 
                    y_pred=xgb_model.predict(X_test)
    )
    print(f"confusion matrix on test data::\n{conf_matrix}")
    metrics.ConfusionMatrixDisplay(conf_matrix).plot()
    plt.show()

    # Precision & Recall
    # TODO:
    '''
        For each threshold make confusion matrix and calc precision
    '''
    prec_score = metrics.precision_score(
                    y_true=y_test, 
                    y_pred=xgb_model.predict(X_test)
    )
    print(f"precision score on test data:: {prec_score}")

    rec_score = metrics.recall_score(
                    y_true=y_test, 
                    y_pred=xgb_model.predict(X_test)
    )
    print(f"recall score on test data:: {rec_score}")


    # Where does list of points comes from in curve??
    ## for each threshold what's the precision & recall  (.prediect_proba)
    precision, recall, thresholds = metrics.precision_recall_curve(
                                                    y_true=y_test,
                                                    probas_pred=xgb_model.predict(X_test),
                                                    pos_label=xgb_model.classes_[1]
                                    )
    metrics.PrecisionRecallDisplay(precision=precision,recall=recall).plot()
    plt.show()



    # F1 Score

    f1_score = metrics.f1_score(y_true=y_test, y_pred=xgb_model.predict(X_test))
    print(f"F1 score on test data:: {f1_score}")

    # F2 Score
    # fbeta_score = metrics.fbeta_score(y_true=y_test, y_pred=xgb_model.predict(X_test))
    # print(f"Fbeta_score score on test data:: {fbeta_score}")

    # Classification Report
    print(metrics.classification_report(y_true=y_test, y_pred=xgb_model.predict(X_test),target_names=['DS','SWE']))



    # ROC
    '''
        ROC curve (AUC) provides an aggregated measure of performance across all possible classification threshold

        
        The area under the ROC curve (AUC) can also be used as a single-number summary of a model's performance.
        AUC < 0.5 model is "bad"

        Good for Datascience doesn't hold meaning for business meaning, comparing models

        Diagonal ~ represent mean/random model.
        more skew ~ targeting more to be classified
    '''
    fpr,tpr,_ = metrics.roc_curve(
                            y_true=y_test,
                            y_score=xgb_model.predict(X_test),
                            pos_label=xgb_model.classes_[1]
                )
    metrics.RocCurveDisplay(fpr=fpr,tpr=tpr).plot()
    plt.show()


    fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(8,4))
    metrics.RocCurveDisplay.from_estimator(xgb_model,X=X_test,y=y_test,ax=ax1)   # Test ROC
    metrics.RocCurveDisplay.from_estimator(xgb_model,X=X_train,y=y_train,ax=ax2) # Train ROC
    plt.show()


    # Threshold metrics
    vals = []
    for thresh in np.arange(0,1,step=.05):
        probs = xgb_model.predict_proba(X_test)[:,1]
        tpr,fpr = get_tpr_fpr(probs>thresh,y_test)
        val     = [thresh,tpr,fpr]
        for metr in [metrics.accuracy_score, metrics.precision_score, metrics.recall_score, metrics.f1_score, metrics.roc_auc_score]:
            val.append(metr(y_test,probs>thresh))
            vals.append(val)

    fig,ax = plt.subplots(figsize=(8,4))
    pd.DataFrame(vals,columns=['thresh','tpr', 'fpr','acc','prec','rec','f1','auc']).set_index('thresh').plot(ax=ax,title='ThresholdMetric')
    plt.show()


    # Youden Index
    # print(f"youden index `threshold`: {youden_fn(y_pred=xgb_model.predict(X_test), y_true=y_test)}")
    

    
    # cumulative gain curve & Lift curve
    '''
        #FIXME: thread is still open in GitHub
        https://github.com/scikit-learn/scikit-learn/issues/10003 


        https://www.youtube.com/watch?v=meZ5qhr3nV0&pp=ygUXIEN1bXVsYXRpdmUgR2FpbnMgQ3VydmU%3D

        commits business value. plots againt ordered samples. 
        gain represents the proportion of positive instances that were correctly identified by the model up to a certain point in the ordered list of predictions

    ROC curves, precision-recall curves, and lift plots can be used to improve, optimize, and maximize the performance of a machine learning model by providing visual representations of its performance at different classification thresholds or operating points. These plots can help identify optimal thresholds or operating points for a given problem or context, allowing for fine-tuning of the model's performance.
    '''



    '''
    Bias Variance Trade off
    
    https://youtu.be/tUs0fFo7ki8?si=mPhbjKztjIYjTQae


    As Model Complexity Grows, (Overfit)
        - variance increase (low error)  (so sensitive to each point)
        - bias decrease

    As a shallow complexity model,
        - variance decrease 
        - high bias 
    '''