# Tree Creation
# handles both categorical and numerical data (not-linear) (if-else)


'''
# High Level
- Loop through all of the columns and
    - Find the split point that is best at seperating the different labels
    - Create a node with this information
- Repeat the above process for results of each node until each node is pure

- First split should feature one of the most critical features because it is the best seperates the data into classes.

# To check randomness:: Gini/Entropy
'''



import numpy as np 
import numpy.random as rn  
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.tree import DecisionTreeClassifier,plot_tree
import xgboost as xgb
import dtreeviz


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


if __name__=='__main__':
    pos_center = 12
    pos_count  = 100

    neg_center = 7
    neg_count  = 100

    rs = rn.RandomState(rn.MT19937(rn.SeedSequence(42)))


    gini = pd.DataFrame({
        'value':np.append(
            (pos_center)+rs.randn(pos_count),
            (neg_center)+rs.randn(neg_count)
        ),
        'label': ['pos']*pos_count + ['neg']*neg_count
    })

    print(f"DataFrame::\n{gini.sample(5).to_string()}")


    fig, ax = plt.subplots(figsize=(8,4))
    _ = (
        gini.groupby('label')[['value']].plot.hist(bins=30,alpha=.5,ax=ax,edgecolor='black')
    )
    ax.legend(['Negative','Positive'])
    ax.annotate(text='purely classified as\n negative',xy=(6,11))
    ax.annotate(text='purely classified as\n positive',xy=(11,11),annotation_clip=False)
    ax.annotate(text="confused\nportion",xy=(9,2.5),annotation_clip=True)
    ax.set_title("Distribution PLot",fontdict={'size':'medium','fontweight':20,'color':'red'},loc='center')
    plt.show()

    values:np.array  = np.arange(5,15,.1)
    ginis:list       = [ calc_gini_fn(
                                df = gini,
                                val_col='value',
                                label_col="label",
                                pos_val='pos',
                                split_point=i,
                                debug=False
                            ) for i in values
                        ]
    
    # low Score found around 9.1 <-> 9,9
    fig,ax =  plt.subplots(figsize=(8,4))
    ax.plot(values,ginis)
    ax.set_title("check Randomness Score")
    ax.set_ylabel("Gini Coefficient")
    ax.set_xlabel('split point')    
    ax.annotate(text='purely classified as\nnegative',xy=(5,.3))
    ax.annotate(text='purely classified as\npositive',xy=(13,.3),annotation_clip=False)
    plt.show()
    
    print(pd.DataFrame({'gini':ginis,'split_point':values}).query('gini<=gini.min()'))

    # Decision Stump (tree has one node)
    stump:DecisionTreeClassifier = DecisionTreeClassifier(
                                            criterion="gini",
                                            splitter="best",
                                            max_depth=1,
                                            min_samples_split=2,
                                            min_samples_leaf=1,
                                            min_weight_fraction_leaf=0.0,
                                            max_features=None,
                                            random_state=None,
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            class_weight=None,
                                            ccp_alpha=0.0,
                                            monotonic_cst=None,
                                    ) 
    stump.fit(X=gini[['value']],y=gini['label'], sample_weight=None, check_input=True)
    


    # It shows single decision, if the value<=9.7 then decision is true 'neg' label. else 'pos' label
    fig,ax = plt.subplots(figsize=(8,8))
    plot_tree(stump,feature_names=['value'],filled=True,class_names=stump.classes_,ax=ax)
    plt.show()

    # XGB doesn't use Gini, SKlearn does by default.
    # XGB goes through a tree-building process
    # After training retuns (g) and (h). similar to gini but behaviour as loss fn
    #  $g_{i} first derivative$   gradient of loss function
    #  $h_{i} second derivative$  curvature of loss fn
    # Split point as same as Sklearn
    xg_stump = xgb.XGBClassifier(n_estimators=1,max_depth=1,n_jobs=-1,device='cuda')
    xg_stump.fit(X=gini['value'],y=gini['label']=='pos')      # y << as value
    print(f"xg_stumps params:\n{xg_stump.get_params()}")
    xgb.plot_tree(xg_stump,num_trees=0) 
    plt.show()

    viz = dtreeviz.model(
                      model=xg_stump,
                      X_train=gini[['value']],
                      y_train=gini['label']=='pos',        # << True, False
                      target_name='label',
                      feature_names=['value'],
                      class_names=['negative','positive'],
                      tree_index=0
            )
    viz.view(precision=5,fontname='Monospace').show()
    