#importing Libraries
#import findspark
#findspark.init()

import time
from pyspark.sql import DataFrame
from pyspark import SparkContext, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import RandomForestClassificationModel

######################### Creating Spark Session ########################
spark = SparkSession.builder.master("local").appName("wineClasssification").getOrCreate()

######################### Reading Dataset########################
testDf = spark.read.csv('TestDataset.csv',header='true', inferSchema='true', sep=';')
#testDf = spark.read.csv('hdfs://ip-172-31-19-75.ec2.internal:8020/TestDataset.csv',header='true', inferSchema='true', sep=';')
feature = [c for c in testDf.columns if c != 'quality']
assembler_test = VectorAssembler(inputCols=feature, outputCol="features")
test_trans = assembler_test.transform(testDf)
#test_trans.printSchema() 

######################### Loading Model ############################
model= RandomForestClassificationModel.load("wine_train_model")

######################### Predicting ##########################
predictions = model.transform(test_trans)
##Value inside show this is just for printing number of value
#predictions.select("quality", "features").show(1000)

######################### Printing Accuracy ##########################
eval = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = eval.evaluate(predictions)
print("accuracy test Error = %g" % (1.0 - accuracy))

from pyspark.mllib.evaluation import MulticlassMetrics
transformed_data = model.transform(test_trans)
print(eval.getMetricName(), 'accuracy:', eval.evaluate(transformed_data))

######################### F1 Score ##########################
eval1 = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="f1")
accuracy = eval1.evaluate(predictions)
print("f1 score test Error = %g" % (1.0 - accuracy))
transformed_data = model.transform(test_trans)
print(eval1.getMetricName(), 'accuracy :', eval1.evaluate(transformed_data))
