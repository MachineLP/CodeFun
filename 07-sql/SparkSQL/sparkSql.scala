import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.hive.HiveContext
 
 
object SparkConfTrait {
 
    val conf = new SparkConf( ).setAppName( "TestSpark Pipeline" )
    val sparkContext = new SparkContext( conf )
    val hiveContext = new HiveContext(sparkContext)
    val sqlContext = new SQLContext(sparkContext)
    val spark = SparkSession.builder().enableHiveSupport.appName("TestSpark").getOrCreate()
 
}
 
 
 
object SparkSQL{
 
    def sqlFromFile( dataSqlFile:String ): DataFrame = {
        val sqlQuery = Source.fromFile( dataSqlFile ).mkString
        val dataSqlFrame = SparkConfTrait.spark.sql( sqlQuery )
        dataSqlFrame
    }
 
 
    // 测试    
    def main(args: Array[String]): Unit = {
        // val sqlQuery = Source.fromFile("path/to/data.sql").mkString //read file
        
        val trainDataSqlFrame = sqlFromFile( "path/to/data.sql"  )
        trainDataSqlFrame.show()
 
    }
}
 
 
 
 
object HiveQL{
 
    def sqlFromFile( dataSqlFile:String ): DataFrame = {
        val sqlQuery = Source.fromFile( dataSqlFile ).mkString
        val dataSqlFrame = SparkConfTrait.hiveContext.sql( sqlQuery )
        dataSqlFrame
    }
 
 
 
    // 测试    
    def main(args: Array[String]): Unit = {
        // val sqlQuery = Source.fromFile("path/to/data.sql").mkString //read file
        
        val trainDataSqlFrame = sqlFromFile( "path/to/data.sql" )
        trainDataSqlFrame.show()
 
    }
}
 


