import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif,chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split, GridSearchCV



def impute_NA_with_avg(data,strategy='mean',NA_col=[]):
    """
    replacing the NA with mean/median/most frequent values of that variable. 
    Note it should only be performed over training set and then propagated to test set.
    """
    
    data_copy = data.copy(deep=True)
    for i in NA_col:
        if data_copy[i].isnull().sum()>0:
            if strategy=='mean':
                data_copy[i+'_impute_mean'] = data_copy[i].fillna(data[i].mean())
            elif strategy=='median':
                data_copy[i+'_impute_median'] = data_copy[i].fillna(data[i].median())
            elif strategy=='mode':
                data_copy[i+'_impute_mode'] = data_copy[i].fillna(data[i].mode()[0])
        else:
            warn("Column %s has no missing" % i)
    return data_copy  

def chi_square_test(X,y,select_k=10):
   
    """
    Compute chi-squared stats between each non-negative feature and class.
    This score should be used to evaluate categorical variables in a classification task
    """
    if select_k >= 1:
        sel_ = SelectKBest(chi2, k=select_k).fit(X,y)
        col = X.columns[sel_.get_support()]
    elif 0 < select_k < 1:
        sel_ = SelectPercentile(chi2, percentile=select_k*100).fit(X,y)
        col = X.columns[sel_.get_support()]   
    else:
        raise ValueError("select_k must be a positive number")  
    
    return col




data_path = "data/train.csv"
label_path = "data/train_target.csv"
predict_data_path = "data/test.csv"

df = pd.read_csv(data_path)
df_label  = pd.read_csv(label_path)
df_predict  = pd.read_csv(predict_data_path)

print (df) 
column_headers = list( df.columns.values )
# print(column_headers)


train_data = pd.merge(df, df_label, how='left', on=['id'])

# 全为缺失值删除
train_data.dropna(how='all') 
# 缺失值进行填充
train_data = impute_NA_with_avg(train_data, column_headers) 

df_predict = impute_NA_with_avg(df_predict, column_headers) 

train_data.fillna(0, inplace = True) 
df_predict.fillna(0, inplace = True) 


print (train_data)
column_headers = list( train_data.columns.values )
print(column_headers)

# x_columns = [x for x in train_data.columns if x not in ["target", "id"]]
x_columns = ['certId', 'loanProduct', 'gender', 'age', 'dist', 'edu', 'job', 'lmt', 'basicLevel', 'x_14', 'x_16', 'x_20', 'x_29', 'x_33', 'x_34', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_49', 'x_51', 'x_52', 'x_54', 'x_55', 'x_61', 'x_62', 'x_63', 'x_67', 'x_68', 'x_71', 'x_72', 'x_75', 'x_76', 'certValidBegin', 'certValidStop', 'bankCard', 'ethnic', 'residentAddr', 'highestEdu', 'linkRela', 'setupHour', 'weekday']
# AUC Score (Train): 

X = train_data[x_columns]
y = train_data["target"] 



print( "corr:", X.corr() )
c = np.linalg.cond(X,p=None)
print (c)
#d = np.linalg.inv(X)
#print (d)
'''
np.linalg.matrix_rank(data_x) #矩阵的秩
np.linalg.inv (data_x)# 求逆矩阵
np.linalg.eigvals(data_x) # 求特征值
np.linalg.eig(data_x)   #特征向量
np.linalg.svd(data_x)  #singular value decomposition 奇异值分解
'''

'''
# 选用卡方检验选取特征
x_columns = chi_square_test(X, y, 10)
X = train_data[x_columns]
y = train_data["target"] 
'''

X_predict = df_predict[x_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
'''
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'num_leaves': 25,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf':5,
        'max_bin':200,
        'verbose': 0,
    }
'''
# num_boost_round=2000
gbm = lgb.LGBMClassifier(boosting_type='gbdt',  min_data_in_leaf=5, max_bin=200, num_leaves=25)
gbm_model = gbm.fit(X_train, y_train) 
importances = gbm_model.feature_importances_ 
indices = np.argsort(importances)[::-1]
feat_labels = X_train.columns
print("Feature ranking:") 
#    l1,l2,l3,l4 = [],[],[],[]
for f in range(X_train.shape[1]):
    print("%d. feature no:%d feature name:%s (%f)" % (f + 1, indices[f], feat_labels[indices[f]], importances[indices[f]]))
print (">>>>>", importances)
# 默认参数                        :   AUC Score (Train): 0.703644
# max_depth=6, n_estimators=200  :   AUC Score (Train): 0.688059
# max_depth=6, n_estimators=100  :   AUC Score (Train): 0.702864


y_pred = gbm_model.predict(X_test) 
y_predprob = gbm_model.predict_proba(X_test)[:, 1] 
print (y_predprob)

print("Accuracy : %.4g" % metrics.accuracy_score(y_test.values, y_pred))  # Accuracy : 0.9852
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))


y_pp = gbm_model.predict_proba(X_predict)[:, 1] 
# print (y_pp)
c ={ "target" : y_pp }
data_lable = DataFrame(c)#将字典转换成为数据框
# print ( data_lable )
id_list = df_predict["id"].tolist()
# print ( id_list )


d ={ "id" : id_list, "target" : y_pp  }
res = DataFrame(d)#将字典转换成为数据框
print (">>>>", res)

res.to_csv("res/res_lgb.csv")
