from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
from pyspark.sql.functions import concat
from pyspark.sql.functions import lit

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics


import json
from collections import OrderedDict
import pandas as pd
import numpy as np


# loading data from an HBase Table
def loadTableFromHBase(givenCatalog):
    result = spark.read.format("org.apache.hadoop.hbase.spark") \
        .options(catalog=givenCatalog) \
        .option("hbase.spark.use.hbasecontext", False) \
        .load()

    return result

# putting data into an HBase Table
def putTableIntoHBase(df, givenCatalog):
    df.write.format("org.apache.hadoop.hbase.spark") \
        .options(catalog=givenCatalog, newTable=5) \
        .option("hbase.spark.use.hbasecontext", False) \
        .save()

    print("Added " + json.loads(givenCatalog)["table"]["name"] + " to HBase!")


# BASELINE LINEAR REGRESSION
def build_model(training):
    training.cache()
    columns = training.columns
    columns.remove("Occupancy")

    assembler = VectorAssembler(inputCols=columns, outputCol="featureVec")
    lr = LogisticRegression(featuresCol="featureVec", labelCol="Occupancy")

    pipeline = Pipeline(stages=[assembler, lr])

    param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.0001, 0.001, 0.01, 0.1, 1.0]) \
        .build()

    evaluator = BinaryClassificationEvaluator(labelCol="Occupancy")

    validator = TrainValidationSplit(estimator=pipeline,
                                     estimatorParamMaps=param_grid,
                                     evaluator=evaluator,
                                     trainRatio=0.9)

    validator_model = validator.fit(training)
    
    return validator_model.bestModel


# DECISION TREES
def build_model2(training):
    training.cache()
    columns = training.columns
    columns.remove("Occupancy")

    assembler = VectorAssembler(inputCols=columns, outputCol="featureVec")
    dt = DecisionTreeClassifier(featuresCol="featureVec", labelCol="Occupancy")

    pipeline = Pipeline(stages=[assembler, dt])

    param_grid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [2, 5, 10, 20, 30]) \
        .addGrid(dt.maxBins, [10, 20, 40, 80, 100]) \
        .build()

    evaluator = BinaryClassificationEvaluator(labelCol="Occupancy")

    validator = TrainValidationSplit(estimator=pipeline,
                                     estimatorParamMaps=param_grid,
                                     evaluator=evaluator,
                                     trainRatio=0.9)

    validator_model = validator.fit(training)
    
    return validator_model.bestModel

# RANDOM FOREST
def build_model3(training):
    training.cache()
    columns = training.columns
    columns.remove("Occupancy")

    assembler = VectorAssembler(inputCols=columns, outputCol="featureVec")
    rf = RandomForestClassifier(featuresCol="featureVec", labelCol="Occupancy")

    pipeline = Pipeline(stages=[assembler, rf])

    param_grid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 50, num = 3)]) \
            .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 3)]) \
            .build()

    evaluator = BinaryClassificationEvaluator(labelCol="Occupancy")

    validator = TrainValidationSplit(estimator=pipeline,
                                     estimatorParamMaps=param_grid,
                                     evaluator=evaluator,
                                     trainRatio=0.9)

    validator_model = validator.fit(training)
    
    return validator_model.bestModel

# GRADIENT BOOSTED TREES
def build_model4(training):
    training.cache()
    columns = training.columns
    columns.remove("Occupancy")

    assembler = VectorAssembler(inputCols=columns, outputCol="featureVec")
    gbt = GBTClassifier(featuresCol="featureVec", labelCol="Occupancy")

    pipeline = Pipeline(stages=[assembler, gbt])

    param_grid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, [2, 4, 6]) \
            .addGrid(gbt.maxBins, [20, 60]) \
            .addGrid(gbt.maxIter, [10, 20]) \
            .build()

    evaluator = BinaryClassificationEvaluator(labelCol="Occupancy")

    validator = TrainValidationSplit(estimator=pipeline,
                                     estimatorParamMaps=param_grid,
                                     evaluator=evaluator,
                                     trainRatio=0.9)

    validator_model = validator.fit(training)
    
    return validator_model.bestModel


def convert_to_row(d: dict) -> Row:
    return Row(**OrderedDict(sorted(d.items())))


def classify(input, model):
    target_columns = input.columns + ["Prediction"]
    # input = input.select("Temperature","Humidity","Light","CO2", "HumidityRatio", "Occupancy")
    # target_columns = ["prediction"]
    return model.transform(input).select(target_columns)


def createBatchScoreTable():
    listOfTemps = [19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5]
    listOfHums = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    listOfLigh = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    listOfRats = [.002, .0025, .003, .0035, .004, .0045, .005, .0055, .006, .0065]
    listOfCO2 = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

    p = sqlContext.createDataFrame(listOfTemps, FloatType()).toDF('Temperature')
    q = sqlContext.createDataFrame(listOfHums, IntegerType()).toDF('Humidity')
    r = sqlContext.createDataFrame(listOfLigh, IntegerType()).toDF('Light')
    s = sqlContext.createDataFrame(listOfCO2, IntegerType()).toDF('CO2')
    t = sqlContext.createDataFrame(listOfRats, FloatType()).toDF('HumidityRatio')

    fullouter = p.crossJoin(q).crossJoin(r).crossJoin(s).crossJoin(t)
    return fullouter


def classifyBatchScoreTable(model, batchScoreDF):
    withPredictions = classify(batchScoreDF, actualModel)
    withPredictions = withPredictions.withColumn("Key", concat(withPredictions["Temperature"], lit(','),
                                                               withPredictions["Humidity"], lit(','),
                                                               withPredictions["Light"], lit(','),
                                                               withPredictions["CO2"], lit(','),
                                                               withPredictions["HumidityRatio"]))
    return withPredictions


def getTrainingDataCatalog():
    catalog = ''.join("""{
                   "table":{"namespace":"default", "name":"trainingDataFinal", "tableCoder":"PrimitiveType"},
                   "rowkey":"key",
                   "columns":{
                     "Key":{"cf":"rowkey", "col":"key", "type":"string"},
                     "Temperature":{"cf":"weather", "col":"Temperature", "type":"double"},
                     "Humidity":{"cf":"weather", "col":"Humidity", "type":"double"},
                     "Light":{"cf":"weather", "col":"Light", "type":"double"},
                     "CO2":{"cf":"weather", "col":"CO2", "type":"double"},
                     "HumidityRatio":{"cf":"weather", "col":"HumidityRatio", "type":"double"},
                     "Occupancy":{"cf":"weather", "col":"Occupancy", "type":"double"}
                   }
                 }""".split())
    return catalog


def getBatchScoreTableCatalog():
    BatchScoreTableCatalog = ''.join("""{
                 "table":{"namespace":"default", "name":"BatchTable", "tableCoder":"PrimitiveType"},
                 "rowkey":"key",
                 "columns":{
                     "Key":{"cf":"rowkey", "col":"key", "type":"string"},
                     "Temperature":{"cf":"weather", "col":"Temperature", "type":"double"},
                     "Humidity":{"cf":"weather", "col":"Humidity", "type":"double"},
                     "Light":{"cf":"weather", "col":"Light", "type":"double"},
                     "CO2":{"cf":"weather", "col":"CO2", "type":"double"},
                     "HumidityRatio":{"cf":"weather", "col":"HumidityRatio", "type":"double"},
                     "Prediction":{"cf":"weather", "col":"Prediction", "type":"double"}
                 }
               }""".split())
    return BatchScoreTableCatalog


# Main Method
if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Room Occupancy") \
        .getOrCreate()

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    # grabs TrainingData from HBase
    trainDataCtlg = getTrainingDataCatalog()
    # trainingData = loadTableFromHBase(trainDataCtlg)

    # for non-hbase storage
    fullData = pd.read_csv('data/datatraining.txt').to_csv('data/datatraining.csv', index=None)
    trainingData = spark.read.csv('data/datatraining.csv', header ='true')
    
    trainingData = trainingData.drop('date')
    
    trainingData = trainingData.withColumn("Temperature", trainingData["Temperature"].cast(FloatType()))
    trainingData = trainingData.withColumn("Humidity", trainingData["Humidity"].cast(FloatType()))
    trainingData = trainingData.withColumn("CO2", trainingData["CO2"].cast(FloatType()))
    trainingData = trainingData.withColumn("Light", trainingData["Light"].cast(FloatType()))
    trainingData = trainingData.withColumn("HumidityRatio", trainingData["HumidityRatio"].cast(FloatType()))
    trainingData = trainingData.withColumn("Occupancy", trainingData["Occupancy"].cast(FloatType()))

    
    trainingData.show()


    # dropping key column since we dont need it for the model
    trainingData = trainingData.drop('Key')


    trainingData, test = trainingData.randomSplit([0.8, 0.2], seed=12345)

    # --------
    print("baseline linear regression")

    # builds and saves the model
    actualModel = build_model(trainingData)
    target_path = 'models/linearReg'
    actualModel.write().overwrite().save(target_path)

    predictionAndTarget = actualModel.transform(test).select("Occupancy", "prediction")

    metrics_binary = BinaryClassificationMetrics(predictionAndTarget.rdd.map(tuple))
    metrics_multi = MulticlassMetrics(predictionAndTarget.rdd.map(tuple))

    acc = metrics_multi.accuracy
    f1 = metrics_multi.fMeasure(1.0)
    precision = metrics_multi.precision(1.0)
    recall = metrics_multi.recall(1.0)
    auc = metrics_binary.areaUnderROC

    print("accuracy: " + str(acc))
    print("f1: " + str(f1))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("auc: " + str(auc))
    
    print(actualModel.stages[-1].extractParamMap())


    # --------
    print("decision")
    
    # decision trees
    actualModel2 = build_model2(trainingData)
    target_path = 'models/decisiontrees'
    actualModel2.write().overwrite().save(target_path)

    predictionAndTarget = actualModel2.transform(test).select("Occupancy", "prediction")

    metrics_binary = BinaryClassificationMetrics(predictionAndTarget.rdd.map(tuple))
    metrics_multi = MulticlassMetrics(predictionAndTarget.rdd.map(tuple))

    acc = metrics_multi.accuracy
    f1 = metrics_multi.fMeasure(1.0)
    precision = metrics_multi.precision(1.0)
    recall = metrics_multi.recall(1.0)
    auc = metrics_binary.areaUnderROC

    print("accuracy: " + str(acc))
    print("f1: " + str(f1))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("auc: " + str(auc))

    print(actualModel2.stages[-1].extractParamMap())



    # --------
    print("random forest")
    
    # random forest
    actualModel3 = build_model3(trainingData)
    target_path = 'models/randomforest'
    actualModel3.write().overwrite().save(target_path)

    predictionAndTarget = actualModel3.transform(test).select("Occupancy", "prediction")

    metrics_binary = BinaryClassificationMetrics(predictionAndTarget.rdd.map(tuple))
    metrics_multi = MulticlassMetrics(predictionAndTarget.rdd.map(tuple))

    acc = metrics_multi.accuracy
    f1 = metrics_multi.fMeasure(1.0)
    precision = metrics_multi.precision(1.0)
    recall = metrics_multi.recall(1.0)
    auc = metrics_binary.areaUnderROC

    print("accuracy: " + str(acc))
    print("f1: " + str(f1))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("auc: " + str(auc))

    print(actualModel3.stages[-1].extractParamMap())




    # --------
    print("gradient boosted trees")
    
    # random forest
    actualModel4 = build_model4(trainingData)
    target_path = 'models/gbt'
    actualModel4.write().overwrite().save(target_path)

    predictionAndTarget = actualModel4.transform(test).select("Occupancy", "prediction")

    metrics_binary = BinaryClassificationMetrics(predictionAndTarget.rdd.map(tuple))
    metrics_multi = MulticlassMetrics(predictionAndTarget.rdd.map(tuple))

    acc = metrics_multi.accuracy
    f1 = metrics_multi.fMeasure(1.0)
    precision = metrics_multi.precision(1.0)
    recall = metrics_multi.recall(1.0)
    auc = metrics_binary.areaUnderROC

    print("accuracy: " + str(acc))
    print("f1: " + str(f1))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("auc: " + str(auc))

    print(actualModel4.stages[-1].extractParamMap())





    # code for storing in batch score table

    # #creates, scores, and stores the batch score table

    # batchTable = createBatchScoreTable()
    # scoredBatchTable = classifyBatchScoreTable(actualModel, batchTable)
    # putTableIntoHBase(scoredBatchTable, getBatchScoreTableCatalog())

