HOW TO BUILD A SPARK FAT JAR IN SCALA AND SUBMIT A JOB ？

### scala版本下xgboost-spark。


``` 
mvn clean package
```

```
spark-2.4.3-bin-hadoop2.7/bin/spark-submit  --class ml.dmlc.xgboost4j.scala.example.spark.SparkMLlibPipeline --jars /***/scala_workSpace/test/xgboost4j-example_2.11-1.0.0-jar-with-dependencies.jar /***/scala_workSpace/test/xgboost4j-example_2.11-1.0.0.jar /tmp/rd/lp/iris.data /***/scala_workSpace/test/nativeModel /tmp/rd/lp/pipelineModel
```

