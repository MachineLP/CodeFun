from efficient_apriori import apriori
import sqlalchemy as sql
import pandas as pd

# 数据加载
engine = sql.create_engine('mysql+mysqlconnector://root:passwd@localhost/wucai')
query = 'SELECT * FROM bread_basket'
data = pd.read_sql_query(query, engine)

# 统一小写
data['Item'] = data['Item'].str.lower()
# 去掉none项
data = data.drop(data[data.Item == 'none'].index)

# 得到一维数组orders_series，并且将Transaction作为index, value为Item取值
orders_series = data.set_index('Transaction')['Item']
# 将数据集进行格式转换
transactions = []
temp_index = 0
for i, v in orders_series.items():
	if i != temp_index:
		temp_set = set()
		temp_index = i
		temp_set.add(v)
		transactions.append(temp_set)
	else:
		temp_set.add(v)

# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.5)
print('频繁项集：', itemsets)
print('关联规则：', rules)
