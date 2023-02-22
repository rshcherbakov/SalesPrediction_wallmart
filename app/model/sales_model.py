
import mlflow
from mlflow import lightgbm as lgb
import optuna 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from loguru import  logger

class LightGBMRegressor:
    def __init__(self):
        self.params = None
        self.model = None
        self.metrics = {}
    
    def objective(self, trial):
        """
        Optuna objective method
        no additional description here
        """
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-8, 100.0),
            "subsample": trial.suggest_loguniform("subsample", 0.1, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 100.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 100.0),
            "max_depth": trial.suggest_int("max_depth", 1, 20),
            "min_split_gain": trial.suggest_loguniform("min_split_gain", 1e-8, 100.0),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000)
        }
        
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        model = lgb.train(params, train_data)
        y_pred = model.predict(self.X_valid)
        
        rmse = mean_squared_error(self.y_valid, y_pred, squared=False)
        return rmse

    def optimize_params(self, X_train, y_train, X_valid, y_valid, n_trials=100):
        """Params optimizer

        Args:
            X_train (pd.Dataset): dataset for training
            y_train (pd.Dataset): target vector for training
            X_valid (pd.Dataset): dataset for validation
            y_valid (pd.Dataset): target for validation
            n_trials (int, optional): Number of iteration. Defaults to 100.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        self.params = study.best_params
        #TODO: log params if it will be required, currently Mlflow should be used
        logger.info("Hyperparams optimysed! the best set of params is {study.best_params}")
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Method for model training

        Args:
            X_train (pd.Dataset): dataset for training
            y_train (pd.Dataset): target vector for training
            X_valid (pd.Dataset): dataset for validation
            y_valid (pd.Dataset): target for validation

        Raises:
            ValueError: throws if no model trained yet
        """
        if self.params is None:
            raise ValueError("No hyperparameters specified. Please run 'optimize_params' method first.")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            self.model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=False)
        else:
            self.model = lgb.train(params, train_data)
    
    def predict(self, X):
        """Prediction wrapper

        Args:
            X (numpy 2d array or pandas dataframe): dataset for prediction 

        Raises:
            ValueError: raised error if there is no model  

        Returns:
            numpy vector with result
        """
        if self.model is None:
            raise ValueError("No model trained. Please run 'train' method first.")
        return self.model.predict(X)
    
    def validate(self, X_test, y_test):
        """ Validatin method 
            !!!! important Validation set and Test set should be checked 

        Args:
            X_test (pd.DataFrame): test set 
            y_test (pd.DataFrame): test target

        Raises:
            ValueError: model is not trained
        """
        if self.model is None:
            raise ValueError('Model is not trained. Please run `train` method first.')
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.debug(f'MAE: {mae:.2f}')
        logger.debug(f'RMSE: {rmse:.2f}')

        self.metrics = {"MAE": mae, 
                        "RMSE": rmse}