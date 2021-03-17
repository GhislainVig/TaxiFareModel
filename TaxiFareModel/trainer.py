# imports
from TaxiFareModel.utils import *
from TaxiFareModel.data import *
from TaxiFareModel.encoders import *
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from google.cloud import storage
from TaxiFareModel.constant import *

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self, model):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_calc', DistanceTransformer()),('scaler', RobustScaler())])
        time_pipe = Pipeline([('time_calc', TimeFeaturesEncoder('pickup_datetime')),
                            ('encoder', OneHotEncoder(sparse = False, handle_unknown='ignore'))])
        preprocessor = ColumnTransformer([('dist_transformer', dist_pipe, 
                                ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']),
                                        ('time_transformer', time_pipe, ['pickup_datetime'])])
        pipe = Pipeline([('preprocessor', preprocessor),('model', model)])
        self.pipeline = pipe
        return self

    def run(self, model):
        """set and train the pipeline"""
        self.set_pipeline(model)
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)
    
    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, f'model{MODEL_VERSION}.joblib')
        print(f"saved model{MODEL_VERSION}.joblib locally")

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename(filename=f'model{MODEL_VERSION}.joblib')

        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")
        return self
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    y = df.fare_amount
    X = df.drop(columns='fare_amount')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    trainer = Trainer(X_train, y_train)
    trainer.run(RandomForestRegressor())
    result = (trainer.evaluate(X_test, y_test))
    print(result)
    #For mlflow :
    '''trainer.mlflow_log_metric("rmse", result)
    trainer.mlflow_log_param("model", 'RandomTree')'''
    #For gds :
    trainer.save_model()
