/*
-------------------------------------------------
   Description :  json解析测试
   Author :       liupeng
   Date :         2019-09-24
-------------------------------------------------
*/

package com.qudian.qdspark.test.model

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostClassificationModel}

// h = sc._jvm.com.qudian.qdspark.test.model.TestPysparkTrain
// h.test( "/tmp/rd/lp/iris.data", "/tmp/rd/lp/modelna", "/tmp/rd/lp/model" )

object TestPysparkTrain{

  def test(inputPath:String, nativeModelPath:String, pipelineModelPath:String): Unit = {
    
    val spark = SparkSession
      .builder()
      .appName("XGBoost4J-Spark Pipeline Example")
      .getOrCreate()

    // Load dataset
    val schema = new StructType(Array(
      StructField("sepal length", DoubleType, true),
      StructField("sepal width", DoubleType, true),
      StructField("petal length", DoubleType, true),
      StructField("petal width", DoubleType, true),
      StructField("class", StringType, true)))

    val rawInput = spark.read.schema(schema).csv(inputPath)

    // Split training and test dataset
    val Array(training, test) = rawInput.randomSplit(Array(0.9, 0.1), 123)

    // Build ML pipeline, it includes 4 stages:
    // 1, Assemble all features into a single vector column.
    // 2, From string label to indexed double label.
    // 3, Use XGBoostClassifier to train classification model.
    // 4, Convert indexed double label back to original string label.
    val assembler = new VectorAssembler()
      .setInputCols(Array("sepal length", "sepal width", "petal length", "petal width"))
      .setOutputCol("features")
    val labelIndexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("classIndex")
      .fit(training)
    val booster = new XGBoostClassifier(
      Map("eta" -> 0.1f,
        "max_depth" -> 2,
        "objective" -> "binary:logistic",
        // "num_class" -> 3,                        //  会出现概率相反的情况。。。
        "num_round" -> 100,
        "num_workers" -> 2
      )
    )
    booster.setFeaturesCol("features")
    booster.setLabelCol("classIndex")
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("realLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(assembler, labelIndexer, booster, labelConverter))
    val model = pipeline.fit(training)

    // Batch prediction
    val prediction = model.transform(test)
    prediction.show(false)

    // Model evaluation
    val evaluator = new MulticlassClassificationEvaluator()
    evaluator.setLabelCol("classIndex")
    evaluator.setPredictionCol("prediction")
    val accuracy = evaluator.evaluate(prediction)
    println("The model accuracy is : " + accuracy)

    // Tune model using cross validation
    val paramGrid = new ParamGridBuilder()
      .addGrid(booster.maxDepth, Array(3, 8))
      .addGrid(booster.eta, Array(0.2, 0.6))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(training)

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(2)
      .asInstanceOf[XGBoostClassificationModel]
    println("The params of best XGBoostClassification model : " +
      bestModel.extractParamMap())
    println("The training summary of best XGBoostClassificationModel : " +
      bestModel.summary)

    // Export the XGBoostClassificationModel as local XGBoost model,
    // then you can load it back in local Python environment.
    bestModel.nativeBooster.saveModel(nativeModelPath)

    // ML pipeline persistence
    model.write.overwrite().save(pipelineModelPath)

    // Load a saved model and serving
    val model2 = PipelineModel.load(pipelineModelPath)
    model2.transform(test).show(false)
  }
}