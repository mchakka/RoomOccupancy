from flask import Flask, jsonify, render_template, request
from pyspark.sql import SparkSession
import os
import logging
from pyspark.ml.pipeline import PipelineModel
from csv import writer

log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)

MASTER = 'local'
APPNAME = 'simple-ml-serving'
MODEL_PATH = 'hdfs:///tmp/spark-model'

spark = SparkSession\
    .builder\
    .appName("WebApp!")\
    .getOrCreate()


#Catalogs
def getBatchScoreTableCatalog():
    modelResultsCatalog = ''.join("""{
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
    return modelResultsCatalog
  
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
                     "Occupancy":{"cf":"weather", "col":"Occupancy", "type":"int"}
                   }
                 }""".split())
  return catalog



# webapp
app = Flask(__name__)


def grabPredictionFromBatchScoreTable(keyToUse, modelResultsCatalog):
  
  persistedModel = PipelineModel.load('models/randomforest')
  print(keyToUse)
  splitKey = keyToUse.split(',')
  splitKey = [float(i) for i in splitKey]
  
  splitKey.insert(0, keyToUse)
  
  listToConvert = [tuple(splitKey)]
  listOfColumns = ['Key', 'Temperature', 'Humidity', 'Light', "CO2", 'HumidityRatio']
  keyToUse = spark.createDataFrame(listToConvert, listOfColumns)
  output = persistedModel.transform(keyToUse)
  
  # output.show()
  # statement ="SELECT * FROM sampleView WHERE Key = '"+keyToUse +"'"
  # result = spark.sql(statement)
  t = output.collect()[0]["prediction"]
  
  if t is None:
    return "N/A"
  else:
    return t  


def addToTrainingTable(key, prediction):
  #Making the row to add to the Training Table
  splitKey = key.split(',')
  splitKey = [float(i) for i in splitKey]
  
  splitKey.insert(0, key)
  splitKey.append(int(prediction))
  
  # listToConvert = [splitKey]
  
  listOfColumns = ['Key', 'Temperature', 'Humidity', 'Light', "CO2", 'HumidityRatio', "Occupancy"]
  #data = spark.createDataFrame(listToConvert, listOfColumns)
  
  # data.write.format("org.apache.hadoop.hbase.spark") \
  #   .options(catalog=getTrainingDataCatalog(), newTable = 5) \
  #   .option("hbase.spark.use.hbasecontext", False) \
  #   .save()
  
  #data.show()
  splitKey[0] = '2015-02-10 09:33:00'
  print(splitKey)


  # non-hbase implementation 
  with open('data/datatraining.csv', 'a') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(splitKey)
    f_object.close()

  print("This is now added to HBase Training Data Table")
      


@app.route('/api', methods=['POST', 'GET'])
def predict():
  keyToUse = request.form['temp6']

  output = grabPredictionFromBatchScoreTable(keyToUse, getBatchScoreTableCatalog())

  if request.form['status'] == "Added":
    addToTrainingTable(request.form['temp6'], output)
  
  if output == 1:
    output = "Occupied"
  elif output == 0:
    output = "Not Occupied"
    
  return render_template("index.html", 
                         output=output, 
                         temp=request.form['temp'],
                        temp2=request.form['temp2'],
                        temp3=request.form['temp3'],
                        temp4=request.form['temp4'],
                        temp5=request.form['temp5'],
                        temp6=request.form['temp6'],
                        status=request.form['status'])

@app.route('/')
def main():
    output = 'No Inputs Yet'
    return render_template("index.html", output=output)


if __name__ == '__main__':
  df = spark.read.format("org.apache.hadoop.hbase.spark") \
    .options(catalog=getBatchScoreTableCatalog()) \
    .option("hbase.spark.use.hbasecontext", False) \
    .load()
  df.createOrReplaceTempView("sampleView")
  
  app.run(port=os.environ["CDSW_APP_PORT"])
    
    
    
# curl -v -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"Temperature":23.7,"Humidity":26.272,"Light":585.2,"CO2":749.2,"HumidityRatio":0.00476416302416414}' http://localhost:8100/api/predict