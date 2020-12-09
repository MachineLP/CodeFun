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

from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local[2]").setAppName("My App").set("spark.jars", "/home/rd/machinelp/test_example/qdspark-1.0.0-jar-with-dependencies.jar")
sc = SparkContext(conf = conf)


h = sc._jvm.com.qudian.qdspark.test.model.TestPysparkTrain
h.test( "/tmp/rd/lp/iris.data", "./modelna", "./model" )
