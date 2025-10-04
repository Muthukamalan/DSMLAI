# Model Complexity & Hyperparameters

## concept of UnderFit & Overfit


'''
Underfit
- model yet to learn more (simple model)
- add more features to learn pattern, use complex model
Overfit
- model is very sensitive ( high variance)  to too much variations (memorize index of the pages too)
- prune, regularizes(constraints)
'''


import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.tree import DecisionTreeClassifier,plot_tree
from ch2 import get_rawX_y,TweakKagTransformer
from feature_engine import imputation,encoding
from sklearn import pipeline,model_selection
import os

underfit_model = DecisionTreeClassifier(max_depth=1)  # << simple model 
overfit_model  = DecisionTreeClassifier(max_depth=None)



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

    X_train:pd.DataFrame
    X_test:pd.DataFrame
    y_train:pd.Series
    y_test:pd.Series
    X_train, X_test, y_train, y_test  = model_selection.train_test_split(kaggle_X,kaggle_y,random_state=42,shuffle=True,stratify=kaggle_y,train_size=0.7)
    X_train = kaggle_pl.fit_transform(X_train)
    X_test  = kaggle_pl.transform(X_test)



    print(X_train.sample(5).to_string())                       # << complex data

    ######################################################
    #
    #           Underfit Model
    #
    ######################################################

    underfit_model.fit(X=X_train,y=y_train)
    print(f"underfit model score:: {underfit_model.score(X=X_test,y=y_test)}, and respective leaves:: {underfit_model.get_n_leaves()}")

    fig,ax = plt.subplots(figsize=(20,20))
    plot_tree(
            underfit_model,
            feature_names=X_train.columns.tolist(), 
            filled=True,
            class_names=underfit_model.classes_,
            max_depth=4,
            fontsize=6
    )
    plt.show()
    ######################################################
    #
    #           Overfit Model
    #
    ######################################################
    overfit_model.fit(X=X_train,y=y_train)
    print(f"overfit_model model score:: {overfit_model.score(X=X_test,y=y_test)}, and respective leaves:: {overfit_model.get_n_leaves()}")

    fig,ax = plt.subplots(figsize=(20,20))
    plot_tree(
            overfit_model,
            feature_names=X_train.columns.tolist(), 
            filled=True,
            class_names=overfit_model.classes_,
            max_depth=4,
            fontsize=6
    )
    plt.show()
