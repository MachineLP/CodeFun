#!/usr/bin/env python
# -*- coding:utf8 -*-

"""
-------------------------------------------------
   Description :  pyspark测试
   Author :       liupeng
   Date :         2019/7/23
-------------------------------------------------

"""

import os
import sys
import time
import pandas as pd
import numpy as np
from start_pyspark import spark, sc, sqlContext
import pyspark.sql.types as typ
import pyspark.ml.feature as ft
from pyspark.sql.functions import isnan, isnull


import os
# os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars xgboost4j-spark-0.72.jar,xgboost4j-0.72.jar pyspark-shell'
# import findspark
# findspark.init()

import pyspark
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# spark.sparkContext.addPyFile("hdfs:///tmp/rd/lp/sparkxgb.zip")


schema = StructType(
  [StructField("PassengerId", DoubleType()),
    StructField("Survived", DoubleType()),
    StructField("Pclass", DoubleType()),
    StructField("Name", StringType()),
    StructField("Sex", StringType()),
    StructField("Age", DoubleType()),
    StructField("SibSp", DoubleType()),
    StructField("Parch", DoubleType()),
    StructField("Ticket", StringType()),
    StructField("Fare", DoubleType()),
    StructField("Cabin", StringType()),
    StructField("Embarked", StringType())
  ])


df_raw = spark\
  .read\
  .option("header", "true")\
  .schema(schema)\
  .csv("hdfs:///tmp/rd/lp/titanic/train.csv")

df_raw.show(20)
df = df_raw.na.fill(0)

sexIndexer = StringIndexer()\
  .setInputCol("Sex")\
  .setOutputCol("SexIndex")\
  .setHandleInvalid("keep")

cabinIndexer = StringIndexer()\
  .setInputCol("Cabin")\
  .setOutputCol("CabinIndex")\
  .setHandleInvalid("keep")

embarkedIndexer = StringIndexer()\
  .setInputCol("Embarked")\
  .setHandleInvalid("keep")

# .setOutputCol("EmbarkedIndex")\

vectorAssembler = VectorAssembler()\
  .setInputCols(["Pclass", "Age", "SibSp", "Parch", "Fare"])\
  .setOutputCol("features")


from sparkxgb import XGBoostClassifier
xgboost = XGBoostClassifier(
    featuresCol="features",
    labelCol="Survived",
    predictionCol="prediction",
    missing = 0.0
)
pipeline = Pipeline(stages=[vectorAssembler, xgboost])


trainDF, testDF = df.randomSplit([0.8, 0.2], seed=24)
trainDF.show(2)
model = pipeline.fit(trainDF)

print (88888888888888888888)
model.transform(testDF).select(col("PassengerId"), col("prediction")).show()
print (9999999999999999999)

# Write model/classifier
model.write().overwrite().save("/tmp/rd/lp/titanic/xgboost_class_test")

model.load( "/tmp/rd/lp/titanic/xgboost_class_test" )
model.transform(testDF).show()
