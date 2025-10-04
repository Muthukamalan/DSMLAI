# Tree Hyperparameters

'''
Hyperparameters
# Probs and Knobs to adjust model complexity
'''


import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.tree import DecisionTreeClassifier,plot_tree
from ch2 import get_rawX_y,TweakKagTransformer
from feature_engine import imputation,encoding
from sklearn import pipeline,model_selection
import os
from rich import print
from tqdm import tqdm

############################################
#
#           Sklearn
#
# - criterion: Literal['gini', 'entropy', 'log_loss'] = "gini",
# - splitter: Literal['best', 'random'] = "best",
# - max_depth: Int | None = None,                                                     # default is to keep splitting until all nodes are pure (or) fewer than `min_samples_split`
# - max_features: float | int | Literal['auto', 'sqrt', 'log2'] | None = None,        # amount of features to examine for the split
# - max_leaf_nodes: Int | None = None,                                                # number of leaves in a tree
# - random_state: Int | RandomState | None = None,
# - min_samples_split: float | int = 2,
# - min_samples_leaf: float | int = 1,
# - min_weight_fraction_leaf: Float = 0,
# - min_impurity_decrease: Float = 0,
# - class_weight: Mapping | str | Sequence[Mapping] | None = None,
# - ccp_alpha: float = 0
############################################

underfit_model = DecisionTreeClassifier(max_depth=1)  # << simple model 
overfit_model  = DecisionTreeClassifier(max_depth=None) # << complex model



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

    print(overfit_model.get_params())

    
    #############################################
    #
    #  Experiments with `max_depth` param
    #
    #############################################
    experiments = list(range(1,15))
    val_accuracies:list = []
    train_accuracies:list = []
    for exp in tqdm(experiments,desc='experiments',total=len(experiments)):
        tre = DecisionTreeClassifier(max_depth=exp)
        tre.fit(X=X_train,y=y_train) 
        val_accuracies.append(tre.score(X=X_test,y=y_test))
        train_accuracies.append(tre.score(X_train,y_train))
    print(val_accuracies,sep=' ',end=' ') 


    # changes while happening in 
    fig,ax = plt.subplots(figsize=(10,4))
    pd.Series(val_accuracies,name='accuracy',index=experiments).plot(ax=ax,title='Accuracy at a given Tree Depth')
    ax.axvline(7, linewidth=2, color='r')
    ax.annotate(text='.735',xy=(7,0.73))
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('max_depth')
    plt.show()


    fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(8,4),tight_layout=True)
    pd.Series(val_accuracies,name='val_accuracy',index=experiments).plot(ax=ax1,title='valdation accuracy')
    pd.Series(train_accuracies,name='train_accuracy',index=experiments).plot(ax=ax2,title='Training accuracy')
    plt.show()



    # Limitation of validation curve is track single hyperparameter.
    # Grid/Random Search in one tool that allow us to experiment across many hyperparameter


    params = {
        'max_depth':[3,5,7,8],
        'min_samples_leaf':[1,3,4,5,6],
        'min_samples_split':[2,3,4,5,6],
    }

    grid_search:model_selection.GridSearchCV = model_selection.GridSearchCV(
                                                                    estimator=DecisionTreeClassifier(),
                                                                    param_grid=params,
                                                                    cv=4,
                                                                    n_jobs=-1,
                                                                    verbose=1,
                                                                    scoring='accuracy'
    )

    grid_search.fit(
                  X= pd.concat([X_train,X_test]),
                  y= pd.concat([y_train,y_test])
    )

    print(grid_search.best_params_,grid_search.best_score_,grid_search.best_estimator_)
    print(pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score').filter(regex='score').head(10).to_string())#.style.background_gradient(axis='columns'))



    # When Evaluating ML model is that you are comparing apples to apples.
    # validation curve to visualiz the impact of a single hyperparameter change.