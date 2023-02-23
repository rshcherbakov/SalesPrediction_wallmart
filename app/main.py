"""
Basic container entrypoint without argparse params added

The aim is provide an example how to

1. load data from ... somewhere - the details will be in data_loader module
2. transform data with the helper objects and and methods from the data_transformer module 
!IMPORTANT huggen fase provide models called transformer, so in order to not be 
confused we called method data_transformer
3. finetune superparams and train model using model package 
!None model wrapper helps unify models interfaces in case we will decide to use 
some deep learning methods wich not provides sklearn interface 
4. All, best params, experiment result, models and dataset should be tracked via Mlflow

Mlflow service alongside with infrastructure: S3 like bucket (minio) and PostgreSQL 
instance added via docker copmouser 
    
"""

import mlflow
from loguru import logger

from data_loader.data_loader import SalesDataLoader
from data_transformer.data_transformer import SalesDataTransformer
from model.sales_model import LightGBMRegressor 



# Note Idealy experiment name will be better to set via some expernat 
# UI interface, MLflow web UI or set in the container's virtualenv  
experiment_name = "LightGBM experiments"

experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment = mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name=experiment_name)

if __name__ == "__main__":
    with mlflow.start_run():
        
        artifact_uri = mlflow.get_artifact_uri()
        data_loader = SalesDataLoader(f"{artifact_uri}/sales_data.csv")
        sales_df = data_loader.load_data() # attempt to load data from the Mlflow storage
        #TODO: needs to be tested 
        
        #Order of transformation:
        # TBD
        # 1. clean/transform data
        # 2. order by date
        # 3. split train/val/test by slices
        # 4. add aggregation transformation and timelag features 
        # !Important data from one chank could be used in other chank
        # only in agg, not as a separate lag feature 
        data_transformer = SalesDataTransformer()
        monthly_sales = data_transformer.transform_data(sales_df)
        
        data_size = monthly_sales.shape[0]
        train_size = data_size * 0.50
        valid_size = data_size * 0.25
        test_part_start = train_size + valid_size

        train_data, valid_data, test_data = monthly_sales[:train_size], \
                                          monthly_sales[train_size:test_part_start], \
                                          monthly_sales[test_part_start:]

        fearure_columns = ["features",]
        target_column = "revenue"

        predictor = LightGBMRegressor()
        model = predictor.train_model(train_data[fearure_columns], 
                                      train_data[target_column],
                                      valid_data[fearure_columns], 
                                      valid_data[target_column])

        mlflow.sklearn.log_model(predictor.model, "sales_prediction_model")
        predictions = predictor.predict(test_data)

        # TODO: predictor.validation method should be used
        mae = mean_absolute_error(test_data['revenue'],  predictions)
        mlflow.log_metric("MAE", mae)

        # Log post-processed predictions to MLflow as a CSV file
        # TODO: if there will be a big file it could be packed in the parquet or avro data scheme 
        post_processed_predictions.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")
