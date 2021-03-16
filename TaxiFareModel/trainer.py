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

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_calc', DistanceTransformer()),('scaler', RobustScaler())])
        time_pipe = Pipeline([('time_calc', TimeFeaturesEncoder('pickup_datetime')),
                            ('encoder', OneHotEncoder(sparse = False, handle_unknown='ignore'))])
        preprocessor = ColumnTransformer([('dist_transformer', dist_pipe, 
                                ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']),
                                        ('time_transformer', time_pipe, ['pickup_datetime'])])
        pipe = Pipeline([('preprocessor', preprocessor),('model', XGBRegressor())])
        self.pipeline = pipe
        return self

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)

if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    y = df.fare_amount
    X = df.drop(columns='fare_amount')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    trainer = Trainer(X_train, y_train)
    trainer.run()
    result = trainer.evaluate(X_test, y_test)
    print(result)
