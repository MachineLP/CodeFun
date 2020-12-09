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
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder



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

x_columns = [x for x in train_data.columns if x not in ["target", "id"]]
# x_columns = ['certId', 'loanProduct', 'gender', 'age', 'dist', 'edu', 'job', 'lmt', 'basicLevel', 'x_14', 'x_16', 'x_20', 'x_29', 'x_33', 'x_34', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_49', 'x_51', 'x_52', 'x_54', 'x_55', 'x_61', 'x_62', 'x_63', 'x_67', 'x_68', 'x_71', 'x_72', 'x_75', 'x_76', 'certValidBegin', 'certValidStop', 'bankCard', 'ethnic', 'residentAddr', 'highestEdu', 'linkRela', 'setupHour', 'weekday']
# AUC Score (Train): 

X = train_data[x_columns]
y = train_data["target"] 


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
X_train, y_train = X, y
X_predict = df_predict[x_columns]

# X_train: (132029, 103)
# X_predict: (23561, 103)
print( "X_train:", X_train.shape )
print( "X_predict:", X_predict.shape )


# XX = pd.concat([X_train,X_predict])

train_data = X_train
test_data = X_predict
# 选择要做onehot的列['id', 'certId', 'loanProduct', 'gender', 'age', 'dist', 'edu', 'job', 'lmt', 'basicLevel', 'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_69', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'x_77', 'x_78', 'certValidBegin', 'certValidStop', 'bankCard', 'ethnic', 'residentAddr', 'highestEdu', 'linkRela', 'setupHour', 'weekday', 'ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan', 'isNew', 'target']
# ["gender", "edu", "job", 'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_69', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'x_77', 'x_78']
dummy_fea = ["gender", "edu", "job", 'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_69', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'x_77', 'x_78']
train_test_data = pd.concat([train_data,test_data],axis=0,ignore_index = True)
train_test_data = train_test_data.fillna(0)
dummy_df = pd.get_dummies(train_test_data.loc[:,dummy_fea], columns=train_test_data.loc[:,dummy_fea].columns)
train_test_data = pd.concat([train_test_data,dummy_df],axis=1)
train_test_data = train_test_data.drop(dummy_fea,axis=1)
train_train = train_test_data.iloc[:train_data.shape[0],:]
test_test = train_test_data.iloc[train_data.shape[0]:,:]

X_train = train_train
X_predict = test_test

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
max_auc = 0
for train_index, test_index in kf.split(X_train):
    print ( ">>>", train_index )
    xgboost = xgb.XGBClassifier()
    xgboost_model = xgboost.fit(X_train.iloc[train_index], y_train.iloc[train_index]) 


    y_pred = xgboost_model.predict(X_train.iloc[test_index]) 
    y_predprob = xgboost_model.predict_proba(X_train.iloc[test_index])[:, 1] 
    print (y_predprob)

    print("Accuracy : %.4g" % metrics.accuracy_score(y_train.iloc[test_index].values, y_pred))  
    auc = metrics.roc_auc_score(y_train.iloc[test_index], y_predprob)
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
        csv_file = 'xgb_res/res_xgb_kfold_onehot' + str(n_splits) + '_'  + str(auc) + '.csv'
        res.to_csv( csv_file ) 

