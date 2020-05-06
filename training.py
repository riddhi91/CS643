#importing Libraries
#import findspark
#findspark.init()

from pyspark.sql import DataFrame
from pyspark import SparkContext, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics


######################### Creating Spark Session ########################
spark = SparkSession.builder.master("local").appName("WineQualityPrediction").config("spark.some.config.option","some-value").getOrCreate()

######################### Reading Dataset   Training Dataset ###########################
trainDf = spark.read.csv('TrainingDataset.csv',header='true', inferSchema='true', sep=';')
 ########### Validation dataset ###############
valDf = spark.read.csv('ValidationDataset.csv',header='true', inferSchema='true', sep=';')                                      


######################### Creating Feature column #########################
featureColumns = [c for c in trainDf.columns if c != 'quality']
assembler_t = VectorAssembler(inputCols=featureColumns, outputCol="features")
train_trans = assembler_t.transform(trainDf)
train_trans.cache()
#train_trans.printSchema() #remove comment to see scheama

######################### Converting for ValidationDataset ##########################
feature = [c for c in valDf.columns if c != 'quality']
assembler_v = VectorAssembler(inputCols=feature, outputCol="features")
val_trans = assembler_v.transform(valDf)
#val_trans.printSchema()    #remove comment to see scheama

######################### creating Model with Random Forest ###########################
from pyspark.ml.classification import RandomForestClassifier
random_forest = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=10)
model = random_forest.fit(train_trans)
model.save("wine_train_model")

print("Model Trained Sucessfully")

