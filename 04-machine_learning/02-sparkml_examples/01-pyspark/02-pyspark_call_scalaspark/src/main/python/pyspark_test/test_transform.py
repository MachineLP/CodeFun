#!/usr/bin/env python
# -*- coding:utf8 -*-

"""
-------------------------------------------------
   Description :  模型预测接口
   Author :       liupeng
   Date :         2019/7/23
-------------------------------------------------

"""

import os
import sys
import json
import pandas as pd
import numpy as np
#下面这些目录都是你自己机器的Spark安装目录和Java安装目录
os.environ['SPARK_HOME']="/di_software/emr-package/spark-2.4.3-bin-hadoop2.7"

sys.path.append("/di_software/emr-package/spark-2.4.3-bin-hadoop2.7/bin")
sys.path.append("/di_software/emr-package/spark-2.4.3-bin-hadoop2.7/python")
sys.path.append("/di_software/emr-package/spark-2.4.3-bin-hadoop2.7/pyspark")
sys.path.append("/di_software/emr-package/spark-2.4.3-bin-hadoop2.7/python/lib")
sys.path.append("/di_software/emr-package/spark-2.4.3-bin-hadoop2.7/python/lib/pyspark.zip")
sys.path.append("/di_software/emr-package/spark-2.4.3-bin-hadoop2.7/lib/py4j-0.9-src.zip")
# sys.path.append("/Library/Java/JavaVirtualMachines/jdk1.8.0_144.jdk/Contents/Home")
os.environ['JAVA_HOME'] = "/usr/lib/jdk1.8.0_171"

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local[2]").setAppName("My App").set("spark.jars", "/home/rd/machinelp/test_example/qdspark-1.0.0-jar-with-dependencies.jar")
sc = SparkContext(conf = conf)
spark = SparkSession.builder.appName('CalculatingGeoDistances').getOrCreate()
sqlContext = SQLContext(sparkContext=sc)

from pyspark.sql.types import *

def pandas_df2arr(pandas_df, loc_name = 'prediction'):
    return np.array( [ np.array ( per_pd ) for per_pd in pandas_df[ loc_name ].values ] )



schema = StructType(
  [StructField("sepal length", DoubleType()),
    StructField("sepal width", DoubleType()),
    StructField("petal length", DoubleType()),
    StructField("petal width", DoubleType()),
    StructField("class", StringType()),
  ])

dataset = spark.read.option("header", "true").schema(schema).csv("/tmp/rd/lp/iris.data")
df_raw = dataset._jdf

print ( ">>>>", df_raw )
import time
start_time = time.time()
h = sc._jvm.com.qudian.qdspark.test.model.TestPysparkTransform("./model")
df_res = h.test( df_raw )
# df_res.show()
# 两种方式 转pyspark dataframe
data_pandas = DataFrame( df_res, dataset.sql_ctx).select("prediction").toPandas()
print ( "res:", data_pandas)
data_pandas = DataFrame( df_res, sqlContext).toPandas()
print ( "res:", data_pandas )
print ("res np:", pandas_df2arr(data_pandas) )

print (">>>>", type( df_res ) )
print ("time:", time.time()- start_time)

'''  json转df
newJson = '{"Name":"something","Url":"https://stackoverflow.com","Author":"jangcy","BlogEntries":100,"Caller":"jangcy"}'
# df = spark.read.json(sc.parallelize([newJson]))
json_data = json.loads( newJson )
pandas_df = pd.DataFrame([json_data])
df = spark.createDataFrame(pandas_df)
df.show(truncate=False)
df = df._jdf
'''
