
### pyspark-xgboost-jupyter-example

Spark version: spark-2.4.3-bin-hadoop2.7
Pyspark-xgboost client lib: https://github.com/dmlc/xgboost/files/3384356/pyspark-xgboost_0.90_261ab52e07bec461c711d209b70428ab481db470.zip

run:
```
/***/spark-2.4.3-bin-hadoop2.7/bin/spark-submit --master local[2] --jars /***/xgboost4j-0.90.jar,/***/xgboost4j-spark-0.90.jar /***/test.py
```
