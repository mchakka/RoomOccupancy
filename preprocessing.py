# !hdfs dfs -rm -r /tmp/hdfsTrainingData.csv
# !hdfs dfs -rm -r /tmp/hdfsTrainingData.txt
# !hdfs dfs -put data/hdfsTrainingData.txt /tmp

from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import concat
from pyspark.sql.functions import lit
import json
import pandas as pd
from pyspark.sql import SQLContext
import hdfs


#Tells spark to read a csv from HDFS and return the the resulting dataframe
def grabDataFromHDFS(local_path):
  csv_path = local_path.replace(".txt", ".csv")
  hdfs_full_path = 'hdfs://{0}'.format(csv_path)
    
  spark.sparkContext.textFile('hdfs://{0}'.format(local_path)) \
    .map(lambda l: l if l.startswith('"date"') else ','.join(l.split(','))[1:]) \
    .saveAsTextFile(hdfs_full_path)
    
  dataframe = spark.read.csv(hdfs_full_path,
                 inferSchema=True, header=True)  
  return dataframe


#Puts dataframe into HBase based on a given catalog
def loadingIntoHBase(data, givenCatalog):
  data.write.format("org.apache.hadoop.hbase.spark") \
      .options(catalog=givenCatalog, newTable = 5) \
      .option("hbase.spark.use.hbasecontext", False) \
      .save()
  
  print("Added " + json.loads(givenCatalog)["table"]["name"] + " to HBase!")


#returns a catalog for the Training Data
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

def replaceDateColumn(trainingData):
  trainingData = trainingData.drop("date")
  trainingData = trainingData.withColumn("Key",concat(trainingData["Temperature"],lit(','), trainingData["Humidity"],lit(','), trainingData["Light"],lit(','), trainingData["CO2"], lit(','), trainingData["HumidityRatio"]))  
  
  return trainingData


if __name__ == "__main__":
  spark = SparkSession\
  .builder\
  .appName("Pre-Processing")\
  .getOrCreate()
 
  sc = spark.sparkContext
  sqlContext = SQLContext(sc)

  #HBase Training Data
  trainingDataHBase = pd.read_csv('data/hbase.csv')
  trainingDataHBase = sqlContext.createDataFrame(trainingDataHBase)
  trainingDataHBase = replaceDateColumn(trainingDataHBase)
  
  loadingIntoHBase(trainingDataHBase, getTrainingDataCatalog())
  print('yay')
  
  #HDFS Training Data
  trainingDataHDFS = grabDataFromHDFS("/tmp/hdfsTrainingData.txt")
  trainingDataHDFS = replaceDateColumn(trainingDataHDFS)
  
  trainingDataHDFS.show()
  #Putting HDFS Training Data in the HBase Training Table Data
  #This let's have all the training Data all in one place
  loadingIntoHBase(trainingDataHDFS, getTrainingDataCatalog())



