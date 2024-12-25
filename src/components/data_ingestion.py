import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        

    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
            data=pd.read_csv("https://raw.githubusercontent.com/atalamahardika/Diabetes-Dataset/refs/heads/main/diabetes.csv")
            logging.info(" reading a data")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info(" i have saved the raw dataset in artifact folder")
            
            logging.info("here i have remove an outliers")
            # Calculate quartiles and IQR for each column
            quartile_data = {}
            for col in data.columns:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                quartile_data[col] = {'q1': q1, 'q3': q3, 'iqr': iqr,
                                    'lower_bound': lower_bound, 'upper_bound': upper_bound}

            for col in data.columns:
                data = data[(data[col] >= quartile_data[col]['lower_bound']) & (data[col] <= quartile_data[col]['upper_bound'])]

            logging.info("outlier removed")

            logging.info("here i have add ROS + SMOTE + ENN")
            x = data.drop(columns='Outcome')
            y = data['Outcome']
            # for imbalance data
            ros = RandomOverSampler(random_state=40)
            x_ros, y_ros = ros.fit_resample(x, y)
            # SMOTE + ENN
            smote_enn = SMOTEENN(random_state=42)
            X_resampled_enn, y_resampled_enn = smote_enn.fit_resample(x_ros, y_ros)

            # Buat DataFrame baru dari data yang telah di-resample
            resampled_data = pd.concat([X_resampled_enn, y_resampled_enn], axis=1)
            logging.info("dataframe has been resampled")

            logging.info("here i have performed train test split")
            train_data,test_data=train_test_split(resampled_data,test_size=0.2, random_state=42)
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("data ingestion part completed")
            
            return (
                 
                
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            logging.info()
            raise customexception(e,sys)


if __name__=="__main__":
    obj=DataIngestion()

    obj.initiate_data_ingestion()