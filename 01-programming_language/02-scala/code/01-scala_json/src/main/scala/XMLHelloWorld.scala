import scala.util.parsing.json.JSON._
import scala.io.Source


object XMLHelloWorld {
  def main(args: Array[String]): Unit = {

    def regJson(json:Option[Any]) = json match {
      case Some(map: Map[String, Any]) => map
      //      case None => "erro"
      //      case other => "Unknow data structure : " + other
      }

    // 解析List
    def regList(tmp:Option[Any]) = tmp match {
        case Some( list: List[String] ) => list
    }

    // 解析String
    def regString(tmp:Option[Any]) = tmp match {
        case Some( str: String ) => str
    }


    var tt =  Map.empty[String, Any] 

    val tree = parseFull(Source.fromFile("config_params.json").mkString)
    
    val first = regJson(tree)
    println(first.get("experiment_name"))
    println( regString( first.get("experiment_name") ) )

    val dev = first.get("model_monitor")
    println( dev )
    val sec = regJson(dev)
    println( sec.get("evaluator") )
    val evList = regList( sec.get("evaluator") )
    println( enList )


    //tt = tree match {
    //  case Some(map: Map[String, Any]) => map
    //}
    //println(tt.getClass.getSimpleName)
    //println(tt.get("experiment_name"))
    //println(tt.get("model_monitor"))


  }
}
 

/*
// config_params.json
{
    "experiment_name": "qdml_command_test",
    "experiment_id": "16",
    "problem_type": "classifier",
    "estimator": "xgradient_boosting",
    "fit_gird": false,
    "eval_metric": "auc",
    "model_monitor": {
        "evaluator": [
            "auc",
            "precision_score",
            "recall_score",
            "ks_value"
        ],
        "threshold": 0.5,
        "is_vaild": true
    }
}
*/
