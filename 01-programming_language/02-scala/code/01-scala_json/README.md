
``` 
mvn clean package
```

```
cd target
java -jar scala-module-dependency-sample-1.0-SNAPSHOT.jar
```


```
结果：
Some(qdml_command_test)
Some(Map(evaluator -> List(auc, precision_score, recall_score, ks_value), threshold -> 0.5, is_vaild -> true))
Some(List(auc, precision_score, recall_score, ks_value))
```

