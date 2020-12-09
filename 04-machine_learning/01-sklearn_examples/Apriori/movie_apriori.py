# -*- coding: utf-8 -*-
# 分析某导演的电影的 频繁项集和关联规则
from efficient_apriori import apriori
import csv
director = u'宁浩'
file_name = './'+director+'.csv'
lists = csv.reader(open(file_name, 'r', encoding='utf-8-sig'))
# 数据加载
data = []
for names in lists:
	name_new = []
	for name in names:
		# 去掉演员数据中的空格
		name_new.append(name.strip())
	data.append(name_new[1:])

# 挖掘频繁项集和关联规则
#itemsets, rules = apriori(data, min_support=0.05,  min_confidence=1)
itemsets, rules = apriori(data, min_support=0.5,  min_confidence=1)
print('频繁项集：', itemsets)
print('关联规则：', rules)
