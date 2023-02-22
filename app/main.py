import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn

class SalesDataLoader:
    def __init__(self, filename):
        self.filename = filename
    
    def load_data(self):
        sales_df = pd.read_csv(self.filename)
        return sales_df

class SalesDataTransformer:
    def __init__(self):
        pass
    
    def transform_data(self, sales_df):
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        sales_df['revenue'] = sales_df['units_sold'] * sales_df['price']
        features_df = sales_df[['name', 'description', 'category', 'store', 'date', 'revenue']]
        monthly_sales = features_df.groupby(['name', 'description', 'category', 'store', pd.Grouper(key='date', freq='M')]).sum()
        return monthly_sales.reset_index()

class SalesModelTrainer:
    def __init__(self):
        pass
    
    def train_model(self, train_data):
        model = LinearRegression()
        model.fit(train_data.index.factorize()[0].reshape(-1, 1), train_data['revenue'])
        return model

class SalesPredictor:
    def __init__(self, model):
        self.model = model
    
    def optimize_hyperparameters(self, train_data):
        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "l1",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 4, 64),
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 0.1),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1.0),
                "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "verbose": -1
            }
            dtrain = lgb.Dataset(train_data['date'].factorize()[0].reshape(-1, 1), label=train_data['revenue'])
            cv_results = lgb.cv(
                params,
                dtrain,
                num_boost_round=100,
                nfold=3,
                metrics=["l1"],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            return cv_results['l1-mean'][-1]
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)
        best_params = study.best_params
        
        with mlflow.start_run():
            for param, value in best_params.items():
                mlflow.log_param(param, value)
                
        return best_params
    
    def predict(self, test_data):
        predictions = self.model.predict(test_data['date'].factorize()[0].reshape(-1, 1))
        mae = mean_absolute_error(test_data['revenue'], predictions)
        
        with mlflow.start_run():
            mlflow.lightgbm.log_model(self.model, "model")
            mlflow.log_metric("mae", mae)
        
        return predictions

# Example usage
with mlflow.start_run(run_name="Sales Prediction Experiment"):
    # Load data from MLflow artifact store
    with mlflow.start_run(run_id="12345"):
        artifact_uri = mlflow.get_artifact_uri()
        data_loader = SalesDataLoader(f"{artifact_uri}/sales_data.csv")
        sales_df = data_loader.load_data()
    
    data_transformer = SalesDataTransformer()
    monthly_sales = data_transformer.transform_data(sales_df)

    # Aggregate sales data by month
    monthly_sales_agg = monthly_sales.groupby('date').sum()

    train_size = int(len(monthly_sales_agg) * 0.8)
    train_data, test_data = monthly_sales_agg[:train_size], monthly_sales_agg[train_size:]

    model_trainer = SalesModelTrainer()
    model = model_trainer.train_model(train_data)

    predictor = SalesPredictor(model)
    predictions = predictor.predict(test_data)
    post_processed_predictions = predictor.post_process(predictions)

    # Compute evaluation metrics
    mae = mean_absolute_error(test_data['revenue'], post_processed_predictions)

    # Log metrics to MLflow
    mlflow.log_metric("MAE", mae)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "sales_prediction_model")

    # Log post-processed predictions to MLflow as a CSV file
    post_processed_predictions.to_csv("post_processed_predictions.csv", index=False)
    mlflow.log_artifact("post_processed_predictions.csv")
