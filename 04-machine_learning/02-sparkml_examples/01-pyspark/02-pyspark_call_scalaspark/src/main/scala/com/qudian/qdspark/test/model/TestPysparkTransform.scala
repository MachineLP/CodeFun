/*
-------------------------------------------------
   Description :  json解析测试
   Author :       liupeng
   Date :         2019-09-24
-------------------------------------------------
*/

package com.qudian.qdspark.test.model

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

// h = sc._jvm.com.qudian.qdspark.test.model.TestPysparkTransform("/tmp/rd/lp/model")
// h.test( inputDF )

class TestPysparkTransform(pipelineModelPath:String){

  val model2 = PipelineModel.load(pipelineModelPath)

  def test(inputDF:DataFrame): DataFrame = {

    model2.transform(inputDF)

  }
}

