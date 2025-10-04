# Correlation: understanding the relationship between features ranges from [-1,1]

## spearman:  correlates the ranks or monotonicity
## pearson:   metric assumes relation as linear 

# Starts with X_train data
# create data_scientist columns and spearman corr
# systematic approach to EDA

import pandas as pd 
from matplotlib import pyplot as plt
import os 
from ch2 import get_rawX_y,TweakKagTransformer
from feature_engine import imputation,encoding
from sklearn import pipeline,model_selection
import seaborn.objects as so 


# assets/EDA Process.png

if __name__=='__main__':
    raw_file:pd.DataFrame = pd.read_csv(os.path.join(os.path.dirname(__file__), 'assets','multipleChoiceResponses.csv'),low_memory=False)
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


    print((X_train.assign(data_scientist=y_train=='Data Scientist').corr(method='spearman')))

    X_train.assign(data_scientist=y_train=='Data Scientist').corr(method='spearman').style.background_gradient(cmap='RdBu').set_sticky(axis='index').to_html(os.path.join(os.getcwd(),"assets",'correlation_plot.html'))


    X_train.assign(data_scientist=y_train=='Data Scientist').groupby('r').data_scientist.value_counts().unstack().plot.bar();

    pd.crosstab(index=X_train['major_cs'],columns=y_train).plot.bar()
    plt.show()

    X_train.plot.scatter(x='years_exp',y='compensation',alpha=.3,c='purple')
    plt.show()
    
    so.Plot(X_train.assign(title=y_train),x='years_exp',y='compensation',color='title').add(so.Dots(alpha=.3,pointsize=2),so.Jitter(x=.5,y=10000)).add(so.Line(),so.PolyFit()).plot(pyplot=True)
    plt.show()




    so.Plot(
        #.query('compensation < 200_000 and years_exp < 16')
        X_train.assign(
                    title=y_train,
                    country=(X_train.loc[:, 'Q3_United States of America': 'Q3_China'].idxmax(axis='columns'))
        ),
        x='years_exp', 
        y='compensation', 
        color='title'
    ).facet('country').add(
        so.Dots(alpha=.01, pointsize=2, color='grey'), so.Jitter(x=.5, y=10_000),col=None
    ).add(
        so.Dots(alpha=.5, pointsize=1.5), so.Jitter(x=.5, y=10_000)
    ).add(
        so.Line(pointsize=1), so.PolyFit(order=2)
    ).scale(
        x=so.Continuous().tick(at=[0,1,2,3,4,5])
    ).limit(
        y=(-10_000, 200_000), x=(-1, 6)  # zoom in effect
    ).plot(pyplot=True)
    plt.show()