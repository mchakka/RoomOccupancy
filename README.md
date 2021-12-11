# Predicting Room Occupancy

## What Is This Repository?

This project consists of example code of how to perform operations (put, get, scan) in PySpark on HBase Tables.
Examples are in the `code-examples` folder of this repository

In addition, this repository contains a demo application of building, training, and serving a simple PySpark ML Model using data from HBase.
For more information about the demo application, please refer to this [link](https://docs.google.com/document/d/1f4Pe6ggRkD2R1XzK8C8e2E2Qut8_UraZDn3PgTf_Mm4/edit?usp=sharing)

## How To Run This Demo Application through CDSW


1. Make sure PySpark and HBase are configured - For reference look at Part 1
   - To make sure all these steps are completed, Python3 should installed and the PySpark Enviornment Variables must be set correctly
   - HBase has the right bindings by adding the jar paths to the RegionServer Enviornment Variable through Cloudera Manager (CM)
   - spark-defaults.conf is already included in this repo, double check the path to the "hbase-connectors" jars is correct

2. Make a new project on CDSW and select “Git” under the “Initial Setup” section
   - Use “https://github.com/mchakka/PySpark-HBaseDemoApp.git” for the Git URL

3. Create a new session with Python3

4. Run preprocessing.py on your CDSW project
   - This will put all training data into HBase
   
5. Run main.py on your CDSW project
   - Creates the model
   - Builds and Scores batch score table
   - Stores batch score table in HBase
   
6. Run app.py on your CDSW project
   - In order to view the web application, go to http://<$CDSW_ENGINE_ID>.<$CDSW_DOMAIN>
