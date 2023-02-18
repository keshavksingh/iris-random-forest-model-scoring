# Import Uvicorn & the necessary modules from FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
# Import other necessary packages
from dotenv import load_dotenv
import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient
# Load the environment variables from the .env file into the application
load_dotenv() 
# Initialize the FastAPI application
app = FastAPI()
# Create a class to load the Model from MLFLOW Registry & use it for prediction
class Model:
    def __init__(self, tracking_uri, model_uri,model_name,experiment_name,model_version,stage,input):
        """
        To initalize the model Details
        """
        self.tracking_uri = mlflow.set_tracking_uri(tracking_uri)
        self.model_uri = model_uri
        self.model_name = model_name
        self.experiment_name = mlflow.set_experiment(experiment_name)
        self.model_version = model_version
        self.stage = stage
        self.input = input
        client = MlflowClient()

    def predict(self):
        """
        To use the loaded model to make predictions on the data
        data: Pandas DataFrame to perform predictions
        """
        stage = self.stage
        model_registry_path = f'models:/{self.model_name}/{self.stage}'
        production_model = mlflow.pyfunc.load_model(model_registry_path)
        #Score Value
        target_names=['setosa' ,'versicolor' ,'virginica']
        predVal = self.input #[[5.9,3.,5.1,1.8]] #Sepal Lenght, Sepal Width, Petal Lenght, Petal Width
        prediction = production_model.predict(predVal)
        prediction = [round(p) for p in prediction]
        prediction = [target_names[p] for p in prediction]
        # Return the predictions  
        return prediction

# Create the POST endpoint with path '/predict'
@app.post("/predict")
async def create_score_input(input: str):
    #input =[5.9,3.,5.1,1.8]
    tracking_uri = os.environ.get('tracking_uri')
    model_uri = os.environ.get('model_uri')
    model_name = os.environ.get('model_name')
    experiment_name = os.environ.get('experiment_name')
    model_version = os.environ.get('model_version')
    stage = os.environ.get('stage')
    input = [input.strip('][').split(',')]
    model =  Model(tracking_uri, model_uri,model_name,experiment_name,model_version,stage,input)
    return {
        str(model.predict())
    }

if __name__ == '__main__':
    app.run(debug=True)