import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.logger.logging import logging
from src.exception. exception import customexception

class ModelEvaluation:
    def __init__(self):
        logging.info("evaluation started")

    def eval_metrics(self,actual,pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        # rmse = np.sqrt(mean_squared_error(actual, pred))# here is RMSE
        # mae = mean_absolute_error(actual, pred)# here is MAE
        # r2 = r2_score(actual, pred)# here is r3 value
        logging.info("evaluation metrics captured")
        return accuracy, precision, recall, f1

    def initiate_model_evaluation(self,train_array,test_array):
        try:
             X_test,y_test=(test_array[:,:-1], test_array[:,-1])

             model_path=os.path.join("artifacts","model.pkl")
             model=load_object(model_path)

             #mlflow.set_registry_uri("")
             
             logging.info("model has register")

             tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

             print(tracking_url_type_store)



             with mlflow.start_run():

                prediction=model.predict(X_test)

                (accur, precis, recal, f1) = self.eval_metrics(y_test,prediction)

                mlflow.log_metric("Accuracy", accur)
                mlflow.log_metric("Precision", precis)
                mlflow.log_metric("Recall", recal)
                mlflow.log_metric("F1", f1)

                 # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")


        except Exception as e:
            raise customexception(e,sys)