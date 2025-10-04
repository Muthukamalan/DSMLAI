from typing import TYPE_CHECKING, Optional,AnyStr,Tuple

import pandas as pd 
import os 
from sklearn import base,pipeline,model_selection
from feature_engine import encoding,imputation

url:AnyStr = 'https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip'
fname:AnyStr = 'kaggle-survey-2018.zip'
member_name:AnyStr = 'multipleChoiceResponses.csv'

def topn(ser:pd.Series, n:int=5, default:AnyStr='other'):
    """
    Replace all values in a Pandas Series that are not among 
    the top `n` most frequent values with a default value.

    This function takes a Pandas Series and returns a new 
    Series with the values replaced as described above. The 
    top `n` most frequent values are determined using the 
    `value_counts` method of the input Series.
    """
    counts = ser.value_counts()
    return ser.where(ser.isin(counts.index[:n]), default)


def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
    """
    - age:: Pull off the first 2 chars of Q2 column and convert them into Int
    - education:: Replace the education string witg numeric values
    - major:: Consider Top 3 and rest='other'
    - YOE:: Q8 `replace (+)->('') ==> split ('-) ==> first-value ==> take left-side and convert into float`
    - compensation:: Q9 column by `replace (',')->('')  ==> replace string with 0 ==> split ==> left-side ==> missing-zero ==> mul(1k)` 
    - python:: Q16_part_1 with zero and convert INT
    - r::  16_part_2 with zero and convert INT
    - sql::  16_part_2 with zero and convert INT

    - rename columns
    - pick columns with ['Q1', 'Q3', 'age', 'education', 'major', 'years_exp', 'compensation', 'python', 'r', 'sql']
    """    
    return (df_
            .assign(
                age=df_.Q2.str.slice(0,2).astype(int),
                education=df_.Q4.replace({"Master’s degree": 18,  'Bachelor’s degree': 16, 'Doctoral degree': 20,'Some college/university study without earning a bachelor’s degree': 13, 'Professional degree': 19,'I prefer not to answer': None, 'No formal education past high school': 12}),
                major=(df_.Q5.pipe(topn, n=3).replace({ 'Computer science (software engineering, etc.)': 'cs', 'Engineering (non-computer focused)': 'eng', 'Mathematics or statistics': 'stat'}) ),
                years_exp=(df_.Q8.str.replace('+','', regex=False).str.split('-', expand=True).iloc[:,0].astype(float)),
                compensation=(df_.Q9.str.replace('+','', regex=False).str.replace(',','', regex=False).str.replace('500000', '500', regex=False).str.replace('I do not wish to disclose my approximate yearly compensation', '0', regex=False).str.split('-', expand=True).iloc[:,0].fillna(0).astype(int).mul(1_000) ),
                python=df_.Q16_Part_1.fillna(0).replace({'Python':1}),
                r=df_.Q16_Part_2.fillna(0).replace('R', 1),
                sql=df_.Q16_Part_3.fillna(0).replace('SQL', 1)
               )#assign
        .rename(columns=lambda col:col.replace(' ', '_')) #rename
        .loc[:, ['Q1', 'Q3', 'age', 'education', 'major', 'years_exp', 'compensation', 'python', 'r', 'sql']]   
    )


class TweakKagTransformer(base.BaseEstimator, base.TransformerMixin):
    '''
    BaseEstimator
    - fit
    - predict

    TransformerMixin
    - fit
    - transform
    '''
    def __init__(self,ycol=None) -> None:
        self.ycol = ycol
        
    def transform(self, X):
        ''' used to transform data by using learned params on training '''
        return tweak_kag(X)
    
    def fit(self, X, y=None):
        ''' used to train model on training datas '''
        return self
    

def get_rawX_y(df:pd.DataFrame,y_col:AnyStr)->Tuple[pd.DataFrame,pd.Series]:
    _raw = df.query('Q3.isin(["United States of America", "China", "India"]) and Q6.isin(["Data Scientist", "Software Engineer"])')
    return _raw.drop(columns=[y_col]), _raw[y_col]



if __name__=='__main__':

    raw_file:pd.DataFrame = pd.read_csv(os.path.join(os.path.dirname(__file__), 'assets','multipleChoiceResponses.csv'),low_memory=False)
    kaggle_question:pd.Series = raw_file.iloc[0]
    kaggle_df:pd.DataFrame = raw_file.iloc[1:]

    print(kaggle_df.shape) #(23860, 395)


    kaggle_X:pd.DataFrame
    kaggle_y:pd.Series
    kaggle_X, kaggle_y = get_rawX_y(kaggle_df,y_col='Q6')


    kaggle_pl= pipeline.Pipeline([
        ('tweak', TweakKagTransformer()),
        ('cat', encoding.OneHotEncoder(top_categories=5, drop_last=True,  variables=['Q1', 'Q3', 'major'])),
        ('num_impute', imputation.MeanMedianImputer(imputation_method='median', variables=['education', 'years_exp']))
    ])

    print(kaggle_pl)

    X_train:pd.DataFrame
    X_test:pd.DataFrame
    y_train:pd.Series
    y_test:pd.Series
    X_train, X_test, y_train, y_test  = model_selection.train_test_split(kaggle_X,kaggle_y,random_state=42,shuffle=True,stratify=kaggle_y,train_size=0.7)


    X_train = kaggle_pl.fit_transform(X_train)
    X_test  = kaggle_pl.transform(X_test)

    # kaggle_pl.fit(X_test)
    print(X_train.sample(5,random_state=1).to_string())
    print(y_train.sample(5,random_state=1).to_string())