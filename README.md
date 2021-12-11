# Predicting Room Occupancy

## What Is This Repository?

This project is to optimize room occupancy prediction. The novelty of our project is to utilize an active learning pipeline to stream prediction in real-time and improve accuracy at the same time. 

Through the demo application, a user can tell the model if a prediction is "correct" or not - from there, these correct data points are then appended to the training dataset and used for retraining. 

## How To Run This Demo Application without HBase/HDFS

1. Install python packages through 
   - ```pip install -r requirements.txt```

2. Run main.py to recreate the models and get accuracy results

3. Run ```flask run``` to deploy the web application

4. If you want to retrain the models with the extra data points from active learning, re-run main.py
