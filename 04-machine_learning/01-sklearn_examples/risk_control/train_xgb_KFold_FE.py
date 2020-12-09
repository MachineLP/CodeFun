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
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from reportgen.utils.preprocessing import Discretization 
from reportgen.utils.preprocessing import WeightOfEvidence 
from sklearn.decomposition import PCA


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


#--------------------------------------------------------------------------------------------------------------------------#
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

#--------------------------------------------------------------------------------------------------------------------------#
# 训练集进行扩增
# train_data_temp = train_data[train_data['target']==1]
# train_data = pd.concat([train_data,train_data_temp,train_data_temp,train_data_temp,train_data_temp,train_data_temp,train_data_temp,train_data_temp,train_data_temp,train_data_temp],axis=0,ignore_index = True)

#--------------------------------------------------------------------------------------------------------------------------#
# 全为缺失值删除
train_data.dropna(how='all') 
# 缺失值进行填充
#train_data = impute_NA_with_avg(train_data, column_headers) 
#df_predict = impute_NA_with_avg(df_predict, column_headers)
train_data.fillna(0, inplace = True) 
df_predict.fillna(0, inplace = True) 

#--------------------------------------------------------------------------------------------------------------------------#

print (train_data)
column_headers = list( train_data.columns.values )
print(column_headers)
# ['id', 'certId', 'loanProduct', 'gender', 'age', 'dist', 'edu', 'job', 'lmt', 'basicLevel', 'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_69', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'x_77', 'x_78', 'certValidBegin', 'certValidStop', 'bankCard', 'ethnic', 'residentAddr', 'highestEdu', 'linkRela', 'setupHour', 'weekday', 'ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan', 'isNew', 'target']
# x_columns = [x for x in train_data.columns if x not in ["target", "id"]]
# 删除特征 bankCard、weekday
x_columns = ["unpayOtherLoan", "x_45", "loanProduct", "x_20", "x_43", "unpayIndvLoan", "x_28", "x_47", "x_68", "x_25", "x_61","x_65", "x_46", "lmt", "x_75", "x_41", "x_27", "linkRela", "x_51", "x_74", "highestEdu", "bankCard", "gender", "x_50", "x_52", "basicLevel", "job", "certId", "x_14", "setupHour", "x_34", "x_63", "weekday", "ethnic", "x_72", "age", "residentAddr", "dist","certValidBegin", "certValidStop", "x_33", "x_62", "x_53", "x_67", "isNew"]
# AUC Score (Train): 
X = train_data[x_columns]
y = train_data["target"] 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
X_train, y_train = X, y
X_predict = df_predict[x_columns]

print (X_train.shape)
print (X_predict.shape)

#--------------------------------------------------------------------------------------------------------------------------#
# 训练与测试数据进行拼接
train_test_data = pd.concat([X_train,X_predict],axis=0,ignore_index = True)
#--------------------------------------------------------------------------------------------------------------------------#
# 数据转换
train_test_data['certBeginDt'] = pd.to_datetime(train_test_data["certValidBegin"] * 1000000000) - pd.offsets.DateOffset(years=70)
print ("time >>>", train_test_data['certBeginDt'])
train_test_data = train_test_data.drop(['certValidBegin'], axis=1)
train_test_data['certStopDt'] = pd.to_datetime(train_test_data["certValidStop"] * 1000000000) - pd.offsets.DateOffset(years=70)
train_test_data = train_test_data.drop(['certValidStop'], axis=1)

#--------------------------------------------------------------------------------------------------------------------------#
# 特征组合 
train_test_data["certStopDt"+"certBeginDt"] = train_test_data["certStopDt"] - train_test_data["certBeginDt"]
print ("train_test_data>>>>>>", train_test_data["certStopDt"+"certBeginDt"])
#--------------------------------------------------------------------------------------------------------------------------#
# 缺失值填充众数
print ("缺失值进行填充")
#for per_x in x_columns:
#    train_test_data[per_x].replace(-999, train_test_data[per_x].mode().iloc[0], inplace = True)

#--------------------------------------------------------------------------------------------------------------------------#
print ("进行分箱")
# 等频／等距／卡方等
train_test_data["age_bin"] = pd.cut(train_test_data["age"],20,labels=False)
train_test_data = train_test_data.drop(['age'], axis=1)
train_test_data["dist_bin"] = pd.qcut(train_test_data["dist"],60,labels=False)
train_test_data = train_test_data.drop(['dist'], axis=1)
train_test_data["lmt_bin"] = pd.qcut(train_test_data["lmt"],50,labels=False)
train_test_data = train_test_data.drop(['lmt'], axis=1)
train_test_data["setupHour_bin"] = pd.qcut(train_test_data["setupHour"],10,labels=False)
train_test_data = train_test_data.drop(['setupHour'], axis=1)
train_test_data["certStopDtcertBeginDt_bin"] = pd.cut(train_test_data["certStopDtcertBeginDt"],30,labels=False)
train_test_data = train_test_data.drop(['certStopDtcertBeginDt'], axis=1)
# 'certValidBegin', 'certValidStop'
train_test_data["certBeginDt_bin"] = pd.cut(train_test_data["certBeginDt"],30,labels=False)
train_test_data = train_test_data.drop(['certBeginDt'], axis=1)
train_test_data["certStopDt_bin"] = pd.cut(train_test_data["certStopDt"],30,labels=False)
train_test_data = train_test_data.drop(['certStopDt'], axis=1)
X_train = train_test_data.iloc[:X_train.shape[0],:]
X_predict = train_test_data.iloc[X_train.shape[0]:,:]

print (">>>>>>", X_train)



#--------------------------------------------------------------------------------------------------------------------------#
# https://www.biaodianfu.com/categorical-features.html
# 大小有序的数据像年龄等，不建议用onehot
print ("进行onehot")
train_data = X_train
test_data = X_predict
# 选择要做onehot的列['id', 'certId', 'loanProduct', 'gender', 'age', 'dist', 'edu', 'job', 'lmt', 'basicLevel', 'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_69', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'x_77', 'x_78', 'certValidBegin', 'certValidStop', 'bankCard', 'ethnic', 'residentAddr', 'highestEdu', 'linkRela', 'setupHour', 'weekday', 'ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan', 'isNew', 'target']
# ["gender", "edu", "job", 'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_69', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'x_77', 'x_78']
# edu
dummy_fea = ["gender","job", "loanProduct", "basicLevel","ethnic"] #'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_69', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'x_77', 'x_78']
train_test_data = pd.concat([train_data,test_data],axis=0,ignore_index = True) 
dummy_df = pd.get_dummies(train_test_data.loc[:,dummy_fea], columns=train_test_data.loc[:,dummy_fea].columns)
dunmy_fea_rename_dict = {}
for per_i in dummy_df.columns.values:
    dunmy_fea_rename_dict[per_i] = per_i + '_onehot'
print (">>>>>",  dunmy_fea_rename_dict)
dummy_df = dummy_df.rename( columns=dunmy_fea_rename_dict )
train_test_data = pd.concat([train_test_data,dummy_df],axis=1)
column_headers = list( train_test_data.columns.values )
print(column_headers)
train_test_data = train_test_data.drop(dummy_fea,axis=1)
column_headers = list( train_test_data.columns.values )
print(column_headers)
train_train = train_test_data.iloc[:train_data.shape[0],:]
test_test = train_test_data.iloc[train_data.shape[0]:,:]
X_train = train_train
X_predict = test_test

print (X_train.shape)
print (X_predict.shape)

#--------------------------------------------------------------------------------------------------------------------------#
'''
XX = pd.concat([X_train,X_predict])
pca = PCA(n_components=50)
newXX = pca.fit_transform(XX)
# X_train.iloc
X_train = newXX[:X_train.shape[0]]
X_predict = newXX[X_train.shape[0]:]
'''

#--------------------------------------------------------------------------------------------------------------------------#
# 下面开始训练。。。。。。。。。
# 0.7311476390229732
# 
# 
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
max_auc = 0
for train_index, test_index in kf.split(X_train):
    #--------------------------------------------------------------------------------------------------------------------------#
    # 训练集进行扩增
    train_data = X_train.iloc[train_index]
    train_y = y_train.iloc[train_index]
    train_data_temp = train_data[train_y==1]
    train_y_temp = train_y[train_y==1]
    # label==1的样本复制N份。
    for i in range(0):
        train_data = pd.concat([train_data,train_data_temp],axis=0,ignore_index = True)
        train_y = pd.concat([train_y,train_y_temp],axis=0,ignore_index = True)
    print ( ">>>", train_index ) 
    other_params = {'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 100, 'seed': 123, 'eta':1,
                    'subsample': 0.8, 'colsample_bytree': 0.5, 'gamma': 0, 'lambda':1, 'reg_alpha': 1, 'reg_lambda': 1, 'scale_pos_weight':5, 'eval_metric': 'auc'}
    xgboost = xgb.XGBClassifier(**other_params) 
    xgboost_model = xgboost.fit(train_data, train_y) 
    # xgboost_model = xgboost.fit(X_train.iloc[train_index], y_train.iloc[train_index]) 
    # xgboost_model = xgboost.fit(X_train[train_index], y_train[train_index]) 
    '''
    importances = xgboost_model.feature_importances_ 
    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns
    print("Feature ranking:") 
    #    l1,l2,l3,l4 = [],[],[],[]
    for f in range(X_train.shape[1]):
        print("%d. feature no:%d feature name:%s (%f)" % (f + 1, indices[f], feat_labels[indices[f]], importances[indices[f]]))
    '''


    y_pred = xgboost_model.predict(X_train.iloc[test_index]) 
    y_predprob = xgboost_model.predict_proba(X_train.iloc[test_index])[:, 1] 
    # y_pred = xgboost_model.predict(X_train[test_index]) 
    # y_predprob = xgboost_model.predict_proba(X_train[test_index])[:, 1] 
    print (y_predprob)

    print("Accuracy : %.4g" % metrics.accuracy_score(y_train.iloc[test_index].values, y_pred))  
    auc = metrics.roc_auc_score(y_train.iloc[test_index], y_predprob)
    #print("Accuracy : %.4g" % metrics.accuracy_score(y_train[test_index].values, y_pred))  
    #auc = metrics.roc_auc_score(y_train[test_index], y_predprob)
    print("AUC Score (Train): %f" % auc) 

    if auc > max_auc:
        max_auc = auc
        print ("auc>>>>>>", auc)
        y_pp = xgboost_model.predict_proba(X_predict)[:, 1] 
        # print (y_pp)
        c ={ "target" : y_pp }
        data_lable = DataFrame(c)#将字典转换成为数据框
        # print ( data_lable )
        id_list = df_predict["id"].tolist()
        # print ( id_list )


        d ={ "id" : id_list, "target" : y_pp  }
        res = DataFrame(d)#将字典转换成为数据框
        print (">>>>", res)
        csv_file = 'xgb_res/res_xgb_kfold_cross' + str(n_splits) + '_'  + str(auc) + '.csv'
        res.to_csv( csv_file ) 
'''
n_splits = 5
cv_params = {'max_depth': [4, 6, 8, 10], 'min_child_weight': [3, 4, 5, 6], 'scale_pos_weight':[5,8,10]}
other_params = {'learning_rate': 0.1, 'n_estimators': 4, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 1, 'reg_alpha': 1, 'reg_lambda': 1}
xgboost = xgb.XGBClassifier()
optimized_GBM = GridSearchCV(estimator=xgboost, param_grid=cv_params, scoring='roc_auc', cv=n_splits, verbose=1, n_jobs=4)
xgboost_model = optimized_GBM.fit(X_train, y_train) 
y_pp = xgboost_model.predict_proba(X_predict)[:, 1] 
# print (y_pp)
c ={ "target" : y_pp }
data_lable = DataFrame(c)#将字典转换成为数据框
# print ( data_lable )
id_list = df_predict["id"].tolist()
# print ( id_list )


d ={ "id" : id_list, "target" : y_pp  }
res = DataFrame(d)#将字典转换成为数据框
print (">>>>", res)
csv_file = 'xgb_res/res_xgb_kfold_cross' + str(n_splits) + '_'   + '.csv'
res.to_csv( csv_file ) 

'''



