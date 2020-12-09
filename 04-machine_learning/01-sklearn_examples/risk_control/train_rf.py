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
# x_columns = ['certId', 'loanProduct', 'gender', 'age', 'dist', 'job', 'lmt', 'basicLevel', 'x_0', 'x_1', 'x_2', 'x_4', 'x_6', 'x_7', 'x_8', 'x_10', 'x_11', 'x_12', 'x_14', 'x_16', 'x_17', 'x_20', 'x_21', 'x_22', 'x_23', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_33', 'x_34', 'x_35', 'x_38', 'x_39', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_59','x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'certValidBegin', 'certBalidStop', 'bankCard', 'ethnic', 'residentAddr', 'highestEdu', 'linkRela', 'setupHour', 'weekday', 'ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan']
# RF: n_estimators=90, max_depth=4  # AUC Score (Train): 0.681259 
# x_columns = ['certId', 'loanProduct', 'gender', 'age', 'dist', 'job', 'lmt', 'basicLevel', 'x_1', 'x_2', 'x_4', 'x_6', 'x_8', 'x_12', 'x_14', 'x_16', 'x_17', 'x_20', 'x_21', 'x_23', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_33', 'x_34', 'x_35', 'x_39', 'x_41', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_57','x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'certValidBegin', 'bankCard', 'ethnic', 'residentAddr', 'highestEdu', 'linkRela', 'setupHour', 'weekday', 'ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan']
# RF: n_estimators=90, max_depth=4  # AUC Score (Train): 0.677848 
x_columns = ['certId', 'loanProduct', 'gender', 'age', 'dist', 'job', 'lmt', 'basicLevel', 'x_1', 'x_2', 'x_6', 'x_8', 'x_12', 'x_14', 'x_16', 'x_17', 'x_20', 'x_23', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_33', 'x_34', 'x_35', 'x_39', 'x_41', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55','x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'certValidBegin', 'bankCard', 'ethnic', 'residentAddr', 'highestEdu', 'linkRela', 'setupHour', 'weekday', 'ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan']
# RF: n_estimators=90, max_depth=4  # AUC Score (Train): 0.690318
# x_columns = ['certId', 'loanProduct', 'gender', 'age', 'dist', 'job', 'lmt', 'basicLevel', 'x_2', 'x_8', 'x_12', 'x_14', 'x_16', 'x_17', 'x_20', 'x_23', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_33', 'x_34', 'x_35', 'x_41', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55','x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'certValidBegin', 'bankCard', 'ethnic', 'residentAddr', 'highestEdu', 'linkRela', 'setupHour', 'weekday', 'ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan']

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)


# gbdt = GradientBoostingClassifier(n_estimators=20, random_state=10) 
# 不同n_estimators数量下的结果比较。
# n_estimators=20  # AUC Score (Train): 0.645734
# n_estimators=30  # AUC Score (Train): 0.652648
# n_estimators=40  # AUC Score (Train): 0.652760

# gbdt = LogisticRegression() 

'''
param_dict = {'max_depth':[2,4,6], 'n_estimators':[50, 100, 200]}
 
rgs = GridSearchCV(gbdt, param_dict)
 
rgs.fit(X_boston, y_boston)
 
print(rgs.best_score_)
 
print(rgs.best_params_)

'''
# gbdt = RandomForestClassifier() 
gbdt = RandomForestClassifier(n_estimators=90, max_depth=4,random_state=10) 
# n_estimators=100  # AUC Score (Train): 0.632956
# n_estimators=90   # AUC Score (Train): 0.638696
# n_estimators=90, max_depth=4  # AUC Score (Train): 0.687838
# n_estimators=90, max_depth=6  # AUC Score (Train): 0.685170
# n_estimators=90 max_depth=8   # AUC Score (Train): AUC Score (Train): 0.653320
# n_estimators=90 max_depth=10  # AUC Score (Train): AUC Score (Train): 0.636410
# n_estimators=80   # AUC Score (Train): 0.633332


gbdt.fit(X_train, y_train) 

importances = gbdt.feature_importances_ 
indices = np.argsort(importances)[::-1]
feat_labels = X_train.columns
std = np.std([tree.feature_importances_ for tree in gbdt.estimators_],
                axis=0) #  inter-trees variability. 
print("Feature ranking:") 
#    l1,l2,l3,l4 = [],[],[],[]
for f in range(X_train.shape[1]):
    print("%d. feature no:%d feature name:%s (%f)" % (f + 1, indices[f], feat_labels[indices[f]], importances[indices[f]]))

print( importances )


y_pred = gbdt.predict(X_test) 
y_predprob = gbdt.predict_proba(X_test)[:, 1] 
print (y_predprob)

print("Accuracy : %.4g" % metrics.accuracy_score(y_test.values, y_pred))  # Accuracy : 0.9852
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))


y_pp = gbdt.predict_proba(X_predict)[:, 1] 
# print (y_pp)
c ={ "target" : y_pp }
data_lable = DataFrame(c)#将字典转换成为数据框
# print ( data_lable )
id_list = df_predict["id"].tolist()
# print ( id_list )


d ={ "id" : id_list, "target" : y_pp  }
res = DataFrame(d)#将字典转换成为数据框
print (">>>>", res)

res.to_csv("res/res_rf.csv")
